extern crate bytevec;
extern crate rand;
extern crate alumina;
extern crate clap;
extern crate image;

mod network;

use std::fs::*;
use std::path::Path;
use std::io::{stdout, Write, Read};
use std::cmp;

use clap::{Arg, App, SubCommand, AppSettings, ArgMatches};
use bytevec::{ByteEncodable, ByteDecodable};
use image::GenericImage;

use network::*;
use alumina::supplier::*;
use alumina::supplier::imagefolder::*;
use alumina::graph::*;
use alumina::shape::*;
use alumina::opt::cain::Cain;
use alumina::opt::*;

const IMAGENET_PARAMS: &'static [u8] = include_bytes!("res/imagenet.rsr");
const IMAGENETLINEAR_PARAMS: &'static [u8] = include_bytes!("res/imagenetlinear.rsr");
const ANIME_PARAMS: &'static [u8] = include_bytes!("res/anime.rsr");

// TODO: expose upscaling factor as argument.
const FACTOR: usize = 3;

fn main() {
	let app_m = App::new("Rusty SR")
	.version("v0.1.1")
	.author("J. Millard <millard.jn@gmail.com>")
	.about("A convolutional neural network trained to upscale images")
	.settings(&[AppSettings::SubcommandsNegateReqs, AppSettings::VersionlessSubcommands])
	.arg(Arg::with_name("INPUT_FILE")
		.help("Sets the input image to upscale")
		.required(true)
		.index(1)
	)
	.arg(Arg::with_name("OUTPUT_FILE")
		.help("Sets the output file to write/overwrite (.png recommended)")
		.required(true)
		.index(2)
	)
	.arg(Arg::with_name("parameters")
		.help("Sets which built-in parameters to use with the neural net")
		.short("p")
		.long("parameters")
		.value_name("PARAMETERS")
		.possible_values(&["imagenet", "imagenetlinear", "anime", "bilinear"])
		.takes_value(true)
	)
	.arg(Arg::with_name("custom")
		.conflicts_with("parameters")
		.short("c")
		.long("custom")
		.value_name("PARAMETER_FILE")
		.help("Sets a custom parameter file (.rsr) to use with the neural net")
		.takes_value(true)
	)
	.arg(Arg::with_name("downsample")
		.conflicts_with_all(&["parameters", "custom"])
		.short("d")
		.long("downsample")
		.help("Perform downscaling rather than upscaling")
		.takes_value(false)
	)
	.subcommand(SubCommand::with_name("train")
		.about("Train a new set of neural parameters on your own dataset")
		.arg(Arg::with_name("PARAMETER_FILE")
			.required(true)
			.index(1)
			.help("Learned network parameters will be (over)written to this parameter file (.rsr)")
		)
		.arg(Arg::with_name("TRAINING_FOLDER")
			.required(true)
			.index(2)
			.help("Images from this folder(or sub-folders) will be used for training")
		)
		.arg(Arg::with_name("LINEAR_LOSS")
			.short("l")
			.long("linearLoss")
			.help("Apply MSE loss to a linearised RGB output rather than sRGB values")
			.takes_value(false)
		)
		.arg(Arg::with_name("RECURSE_SUBFOLDERS")
			.short("r")
			.long("recurse")
			.help("Recurse into subfolders of training and validation folders looking for files")
			.takes_value(false)
		)
		.arg(Arg::with_name("START_PARAMETERS")
			.short("s")
			.long("start")
			.help("Start training from known parameters loaded from this .rsr file rather than random initialisation")
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
	).get_matches();


	if let Some(sub_m) = app_m.subcommand_matches("train") {
		train(sub_m);
		return;
	}

	
	upscale(&app_m);
	
}


fn upscale(app_m: &ArgMatches){
		
	//-- Sort out parameters and graph
	let (params, mut graph) = if let Some(file_str) = app_m.value_of("custom") {
		let mut param_file = File::open(Path::new(file_str)).expect("Error opening parameter file");
		let mut data = Vec::new();
		param_file.read_to_end(&mut data).expect("Reading parameter file failed");
		print!("Upscaling using custom neural net parameters...");
		(<Vec<f32>>::decode::<u32>(&data).expect("ByteVec conversion failed"), sr_net(FACTOR, None))
	} else if app_m.is_present("downsample"){
		print!("Downsampling using average pooling of linear RGB values...");
		(Vec::new(), downsample_net(FACTOR))
	} else {
		match app_m.value_of("parameters") {
			Some("imagenet") | None => {
				print!("Upscaling using imagenet neural net parameters...");
				(<Vec<f32>>::decode::<u32>(IMAGENET_PARAMS).expect("ByteVec conversion failed"), sr_net(FACTOR, None))},
			Some("imagenetlinear") => {
				print!("Upscaling using linear loss imagenet neural net parameters...");
				(<Vec<f32>>::decode::<u32>(IMAGENETLINEAR_PARAMS).expect("ByteVec conversion failed"), sr_net(FACTOR, None))},
			Some("anime") => {
				print!("Upscaling using anime neural net parameters...");
				(<Vec<f32>>::decode::<u32>(ANIME_PARAMS).expect("ByteVec conversion failed"),    sr_net(FACTOR, None))},
			Some("bilinear") => {
				print!("Upscaling using bilinear interpolation...");
				(Vec::new(), bilinear_net(FACTOR))},
			_ => unreachable!(),
		}
	};
	stdout().flush().ok();

	// TODO: ensure factor argument requires custom neural net file argument.
	assert_eq!(params.len(), graph.num_params(), "Parameters selected do not have the size required by the neural net. Ensure that the same sample factor is used for upscaling and training", );

	let input_image = image::open(Path::new(app_m.value_of("INPUT_FILE").expect("No input file given?"))).expect("Error opening input image file.");

	let out_path = Path::new(app_m.value_of("OUTPUT_FILE").expect("No output file given?"));

	let mut input = NodeData::new_blank(DataShape::new(CHANNELS, &[input_image.dimensions().0 as usize, input_image.dimensions().1 as usize], 1));

	img_to_data(&mut input.values, &input_image);
	let output = graph.forward(1, vec![input], &params).remove(0);

	print!(" Writing file...");
	stdout().flush().ok();
	data_to_img(output).to_rgba().save(out_path).expect("Could not write output file");
	
	println!(" Done");
}


fn train(app_m: &ArgMatches){
	
	let linear_loss = app_m.is_present("LINEAR_LOSS");

	let mut g = sr_net(FACTOR, Some((1e-6, linear_loss)));
	let recurse = app_m.is_present("RECURSE_SUBFOLDERS");
	let training_set = ImageFolderSupplier::<ShuffleRandom>::new(Path::new(app_m.value_of("TRAINING_FOLDER").expect("No training folder?")), recurse, Cropping::Random{width:192, height:192});
	let mut training_set = Buffer::new(training_set, 128);

	let start_params = if let Some(param_str) = app_m.value_of("START_PARAMETERS") {
		let mut param_file = File::open(Path::new(param_str)).expect("Error opening start parameter file");
		let mut data = Vec::new();
		param_file.read_to_end(&mut data).expect("Reading start parameter file failed");
		<Vec<f32>>::decode::<u32>(&data).expect("ByteVec conversion failed")
	} else {
		g.init_params()
	};



	let mut solver = Cain::new(&mut g)
		.num_subbatches(8)
		.target_err(0.85)
		.subbatch_increase_damping(0.15)
		.subbatch_decrease_damping(0.15)
		.aggression(0.5)
		.momentum(0.95)
		.initial_learning_rate(1e-4)
		.finish();
		
	let param_file_path = Path::new(app_m.value_of("PARAMETER_FILE").expect("No parameter file?")).to_path_buf();

	solver.add_step_callback(move |data|{
		if data.step_count % 100 == 0 || data.step_count == 1{
			let mut parameter_file = File::create(&param_file_path).expect("Could not make parameter file");
			let bytes = data.params.encode::<u32>().expect("ByteVec conversion failed");
			parameter_file.write_all(&bytes).expect("Could not save to parameter file");
		}
		CallbackSignal::Continue
	});


	if let Some(val_str) = app_m.value_of("VALIDATION_FOLDER"){ // Add occasional test set evaluation as solver callback
		let mut g2 = sr_net(FACTOR, Some((0.0, linear_loss)));
		let validation_set = ImageFolderSupplier::<Sequential>::new(Path::new(val_str), recurse, Cropping::None);
		
		let n = if let Some(val_max) = app_m.value_of("val_max"){
			cmp::min(validation_set.epoch_size() as usize, val_max.parse::<usize>().expect("-val_max N must be a positive integer"))
		} else {
			validation_set.epoch_size() as usize
		};
		let mut validation_set = Buffer::new(validation_set, n);

		solver.add_step_callback(move |data|{

			if data.step_count % 100 == 0 || data.step_count == 1{

				let mut err_sum = 0.0;
				let mut pix_sum = 0.0;
				for _ in 0..n {
					let (input, training_input) = validation_set.next_n(1);
					let pixels = input[0].shape.flat_size_single() as f32;
					let (batch_err, _, _) = g2.backprop(1, input, training_input, data.params);

					pix_sum += pixels;
					err_sum += batch_err * pixels;
				}
				let psnr = -10.0*(err_sum/pix_sum).log10();
				println!("Validation PSNR:\t{}", psnr);
			}

			CallbackSignal::Continue
		});
	}

	solver.add_boxed_step_callback(max_evals(10_000_000));

	println!("Beginning Training");
	solver.optimise_from(&mut training_set, start_params);	
	println!("Done");
}