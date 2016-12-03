extern crate bytevec;
extern crate rand;
extern crate alumina;
extern crate clap;
extern crate image;

//mod suppliers;
mod network;


use std::fs::*;
use std::path::Path;
use std::io::{Write, Read};

use std::cmp;

use clap::{Arg, App, SubCommand, AppSettings, ArgMatches};
use bytevec::{ByteEncodable, ByteDecodable};
use image::GenericImage;

const IMAGENET_PARAMS: &'static [u8] = include_bytes!("res/imagenet.par");
const ANIME_PARAMS: &'static [u8] = include_bytes!("res/anime.par");

use network::*;
use alumina::opt::supplier::*;
use alumina::opt::supplier::imagefolder::*;
use alumina::graph::*;
use alumina::shape::*;
use alumina::opt::asgd::Asgd2;
use alumina::opt::*;

fn main() {
	let app_m = App::new("Rusty SR")
	.version("v0.0.1")
	.author("J. Millard <millard.jn@gmail.com>")
	.about("A convolutional neural network trained to upscale sRGB images")
	.settings(&[AppSettings::SubcommandsNegateReqs, AppSettings::VersionlessSubcommands])
	.arg(Arg::with_name("INPUT_FILE")
		.help("Sets the input image to upscale")
		.required(true)
		.index(1)
	)
	.arg(Arg::with_name("OUTPUT_FILE")
		.help("Sets the PNG output file to write/overwrite")
		.required(true)
		.index(2)
	)
	.arg(Arg::with_name("parameters")
		.help("Sets which built-in parameters to use with the neural net")
		.short("p")
		.long("parameters")
		.value_name("PARAMETERS")
		.possible_values(&["imagenet", "anime", "bilinear"])
		.takes_value(true)
	)
	.arg(Arg::with_name("custom")
		.conflicts_with("parameters")
		.short("c")
		.long("custom")
		.value_name("PARAMETER_FILE")
		.help("Sets a custom parameter file to use with the neural net")
		.takes_value(true)
	)
	.subcommand(SubCommand::with_name("train")
		.about("Train a new set of neural parameters on your own dataset")
		.arg(Arg::with_name("PARAMETER_FILE")
			.required(true)
			.index(1)
			.help("Learned network parameters will be (over)written to this file")
		)
		.arg(Arg::with_name("TRAINING_FOLDER")
			.required(true)
			.index(2)
			.help("Images from this folder(or sub-folders) will be used for training")
		)
		.arg(Arg::with_name("START_PARAMETERS")
			.short("s")
			.long("start")
			.help("Start training from from known parameters rather than random initialisation")
			.takes_value(true)
		)
		.arg(Arg::with_name("VALIDATION_FOLDER")
			.short("v")
			.long("val_folder")
			.help("Images from this folder(or sub-folders) will be used to evaluate training progress")
			.takes_value(true)
		)
		.arg(Arg::with_name("val_max")
			.requires("VALIDATION_FOLDER")
			.short("m")
			.long("val_max")
			.value_name("N")
			.help("Set upper limit on number of images used for each validation pass")
			.takes_value(true)
		) 
	)

	.get_matches();


	if let Some(sub_m) = app_m.subcommand_matches("train") {
		train(sub_m);
		return;
	}

	
	{
		//-- Sort out parameters and graph
		let (params, mut graph) = if let Some(file_str) = app_m.value_of("custom") {
			let mut param_file = File::open(Path::new(file_str)).expect("Error opening parameter file");
			let mut data = Vec::new();
			param_file.read_to_end(&mut data).expect("Reading parameter file failed");
			println!("Upscaling using custom neural net parameters");
			(<Vec<f32>>::decode::<u32>(&data).expect("ByteVec conversion failed"), sr_net(3, false, 0.0))
		} else {
			 match app_m.value_of("parameters") {
				Some("imagenet") | None => {
					println!("Upscaling using imagenet neural net parameters");
					(<Vec<f32>>::decode::<u32>(IMAGENET_PARAMS).expect("ByteVec conversion failed"), sr_net(3, false, 0.0))},
				Some("anime")           => {
					println!("Upscaling using anime neural net parameters");
					(<Vec<f32>>::decode::<u32>(ANIME_PARAMS).expect("ByteVec conversion failed"),    sr_net(3, false, 0.0))},
				Some("bilinear")        => {
					println!("Upscaling using bilinear interpolation");
					(Vec::new(), linear_net())},
				 _ => unimplemented!(),
			}
		};
		

		let input_image = image::open(Path::new(app_m.value_of("INPUT_FILE").expect("No input file given?"))).expect("Error opening image file");

		let mut out_file = File::create(app_m.value_of("OUTPUT_FILE").expect("No output file given?")).expect("Could not save output file");


		let mut input = NodeData::new_blank(DataShape::new(CHANNELS, &[input_image.dimensions().0 as usize, input_image.dimensions().1 as usize], 1));

		img_to_data(&mut input.values, &input_image);

		let output = graph.forward(1, vec![input], &params).remove(0);
		data_to_img(output).save(&mut out_file, image::ImageFormat::PNG).expect("Could not write output file");
	}

	println!("Done");
	
}


fn train(matches: &ArgMatches){
	


	let mut g = sr_net(3, true, 1e-5);

	let mut training_set = ImageFolderSupplier::<ShuffleRandom>::new(Path::new(matches.value_of("TRAINING_FOLDER").expect("No training folder?")), Cropping::Random{width:120, height:120});

	let start_params = if let Some(param_str) = matches.value_of("START_PARAMETERS") {
		let mut param_file = File::open(Path::new(param_str)).expect("Error opening parameter file");
		let mut data = Vec::new();
		param_file.read_to_end(&mut data).expect("Reading parameter file failed");
		<Vec<f32>>::decode::<u32>(&data).expect("ByteVec conversion failed")
	} else {
		g.init_params()
	};



	let mut solver = Asgd2::new(&mut g);
	
	let mut parameter_file = File::create(Path::new(matches.value_of("PARAMETER_FILE").expect("No parameter file?"))).expect("Could not make parameter file");

	if let Some(val_str) = matches.value_of("VALIDATION_FOLDER"){ // Add occasional test set evaluation as solver callback
		let mut g2 = sr_net(3, true, 0.0);
		let mut validation_set = ImageFolderSupplier::<Sequential>::new(Path::new(val_str), Cropping::Random{width:120, height:120});

		let n = if let Some(val_max) = matches.value_of("val_max"){
			cmp::min(validation_set.epoch_size() as usize, val_max.parse::<usize>().expect("-val_max N must be a positive integer"))
		} else {
			validation_set.epoch_size() as usize
		};

		solver.add_step_callback(move |_err, step, _evaluations, _graph, params|{

			if step % 100 == 0 {
				print!("Validation errors:\t");
				for _ in 0..n {

					let (input, training_input) = validation_set.next_n(1);
					// training_input.push(NodeData::new_blank(DataShape::new_flat(1000, 1)));

					let (batch_err, _, _) = g2.backprop(1, input, training_input, params);
					
					print!("{}\t", batch_err)
				}
				println!("");
			}

			if step % 100 == 0 || step == 1{
				let bytes = params.encode::<u32>().expect("ByteVec conversion failed");
				parameter_file.write_all(&bytes).expect("Could not save to parameter file");
			}


			true
		});
	}

	println!("Beginning Training");
	solver.set_max_evals(10_000_000);
	solver.optimise_from(&mut training_set, start_params);	
	println!("Done");
}