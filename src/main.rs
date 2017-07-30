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
//use alumina::ops::activ::{SrgbToLinearFunc, LinearToSrgbFunc, ActivationFunc};

const L1NATURAL_PARAMS: &'static [u8] = include_bytes!("res/L1SplineNatural.rsr");
const L1ANIME_PARAMS: &'static [u8] = include_bytes!("res/L1SplineAnime.rsr");
const L1FACES_PARAMS: &'static [u8] = include_bytes!("res/L1SplineFaces.rsr");

// TODO: expose upsampling factor as argument.
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
		.help("Sets which built-in parameters to use with the neural net, default: L1imagenet")
		.short("p")
		.long("parameters")
		.value_name("PARAMETERS")
		.possible_values(&["natural", "anime", "faces", "bilinear"])
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
		.arg(Arg::with_name("L1_LOSS")
			.long("L1Loss")
			.help("Use a charbonneir loss with a scale of 0.01 to approximated the L1 reconstruction loss (default loss is L2).")
			.takes_value(false)
		)
		.arg(Arg::with_name("SRGB_DOWNSCALE")
			.long("srgbDown")
			.help("Perform downsampling for training in sRGB colorspace")
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
	)
	.subcommand(SubCommand::with_name("downsample")
	 	.about("Downsample images")
		.arg(Arg::with_name("INPUT_FILE")
			.help("Sets the input image to downscale")
			.required(true)
			.index(1)
		)
		.arg(Arg::with_name("OUTPUT_FILE")
			.help("Sets the output file to write/overwrite (.png recommended)")
			.required(true)
			.index(2)
		)
	 	.arg(Arg::with_name("COLOURSPACE")
	 		.help("colorspace in which to perform downsampling")
	 		.short("c")
	 		.long("colourspace")
	 		.value_name("COLOURSPACE")
	 		.possible_values(&["sRGB", "RGB"])
	 		.takes_value(true)
	 	)
	)
	// .subcommand(SubCommand::with_name("psnr")
	// 	.about("Train a new set of neural parameters on your own dataset")
	// 	.arg(Arg::with_name("IMAGE1")
	// 		.required(true)
	// 		.index(1)
	// 		.help("PSNR is calculated using the difference between this image and IMAGE2")
	// 	)
	// 	.arg(Arg::with_name("IMAGE2")
	// 		.required(true)
	// 		.index(2)
	// 		.help("PSNR is calculated using the difference between this image and IMAGE1")
	// 	)
	// 	.arg(Arg::with_name("channel")
	// 		.help("Sets which channels will be included in psnr calculation")
	// 		.short("c")
	// 		.long("channel")
	// 		.value_name("CHANNEL")
	// 		.possible_values(&["sRGB", "RGB", "luminance", "luma"])
	// 		.takes_value(true)
	// 	)
	// 	.arg(Arg::with_name("border")
	// 		.help("Sets which channels will be included in psnr calculation")
	// 		.short("b")
	// 		.long("border")
	// 		.value_name("BORDER")
	// 		.takes_value(true)
	// 		.validator(|s| {
	// 			s.parse::<usize>().map(|x| ()).map_err(|int_err| format!("border parameter must be a valid unsized integer: {}", int_err))
	// 		})
	// 	))
	.get_matches();
	
	if let Some(sub_m) = app_m.subcommand_matches("train") {
		train(sub_m);
	} if let Some(sub_m) = app_m.subcommand_matches("downsample") {
		downsample(sub_m);
	// } if let Some(sub_m) = app_m.subcommand_matches("psnr") {
	// 	psnr(sub_m);
	} else {
		upscale(&app_m);
	}
	
}

fn downsample(app_m: &ArgMatches){
		
	let (params, mut graph) = match app_m.value_of("colourspace") {
		Some("RGB") | None => {
			print!("Downsampling using average pooling of linear RGB values...");
			(Vec::new(), downsample_lin_net(FACTOR))},
		Some("sRGB")=> {
			print!("Downsampling using average pooling of sRGB values...");
			(Vec::new(), downsample_srgb_net(FACTOR))},
		_ => unreachable!(),
	};

	stdout().flush().ok();

	assert_eq!(params.len(), graph.num_params(), "Parameters selected do not have the size required by the neural net. Ensure that the same sample factor is used for upsampling and training", );

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

fn upscale(app_m: &ArgMatches){
		
	//-- Sort out parameters and graph
	let (params, mut graph) = if let Some(file_str) = app_m.value_of("custom") {
		let mut param_file = File::open(Path::new(file_str)).expect("Error opening parameter file");
		let mut data = Vec::new();
		param_file.read_to_end(&mut data).expect("Reading parameter file failed");
		print!("Upsampling using custom neural net parameters...");
		(<Vec<f32>>::decode::<u32>(&data).expect("ByteVec conversion failed"), sr_net(FACTOR, None))
	} else {
		match app_m.value_of("parameters") {
			Some("natural") | None => {
				print!("Upsampling using natural neural net parameters...");
				(<Vec<f32>>::decode::<u32>(L1NATURAL_PARAMS).expect("ByteVec conversion failed"), sr_net(FACTOR, None))},
			Some("anime")=> {
				print!("Upsampling using anime neural net parameters...");
				(<Vec<f32>>::decode::<u32>(L1ANIME_PARAMS).expect("ByteVec conversion failed"), sr_net(FACTOR, None))},
			Some("faces")=> {
				print!("Upsampling using faces neural net parameters...");
				(<Vec<f32>>::decode::<u32>(L1FACES_PARAMS).expect("ByteVec conversion failed"), sr_net(FACTOR, None))},
			Some("bilinear") => {
				print!("Upsampling using bilinear interpolation...");
				(Vec::new(), bilinear_net(FACTOR))},
			_ => unreachable!(),
		}
	};
	stdout().flush().ok();

	// TODO: ensure factor argument requires custom neural net file argument.
	assert_eq!(params.len(), graph.num_params(), "Parameters selected do not have the size required by the neural net. Ensure that the same sample factor is used for upsampling and training", );

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
	
	let srgb_downscale = app_m.is_present("SRGB_DOWNSCALE");
	if srgb_downscale {
		println!("Downsampling for training performed in sRGB");
	} else {
		println!("Downsampling for training performed in linear RGB");
	}

	let (power, scale) = if app_m.is_present("L1_LOSS") {
		println!("Training using the L1 reconstruction loss");
		(1.0, 0.01) // L1, smoothed
	} else {
		println!("Training using the L2 reconstruction loss");
		(2.0, 1.0) // L2
	};

	let mut g = sr_net(FACTOR, Some((1e-6, srgb_downscale, power, scale))); // at 1e-3 the error was 1e-1, at 1e-6 err is 1e-4
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
		.num_subbatches(4)
		.target_err(0.9)
		.subbatch_increase_damping(0.25)//25)
		.subbatch_decrease_damping(0.125)//125)
		.initial_subbatch_size(8.0)
		.rate_adapt_coefficient(1.0)
		.aggression(0.5)
		.momentum(0.95)
		.max_subbatch_size(8)
		.initial_learning_rate(1.0e-3)
		.finish();
		
	let param_file_path = Path::new(app_m.value_of("PARAMETER_FILE").expect("No parameter file?")).to_path_buf();

	solver.add_step_callback(move |data|{
		if data.step_count % 100 == 0 {
			let mut parameter_file = File::create(&param_file_path).expect("Could not make parameter file");
			let bytes = data.params.encode::<u32>().expect("ByteVec conversion failed");
			parameter_file.write_all(&bytes).expect("Could not save to parameter file");
		}
		CallbackSignal::Continue
	});


	if let Some(val_str) = app_m.value_of("VALIDATION_FOLDER"){ // Add occasional test set evaluation as solver callback
		let mut g2 = sr_net(FACTOR, Some((0.0, srgb_downscale, power, scale)));
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
				let mut y_err_sum = 0.0;
				let mut pix_sum = 0.0f32;

				let mut psnr_sum = 0.0;
				let mut y_psnr_sum = 0.0;


				for _ in 0..n {
					let (input, _training_input) = validation_set.next_n(1);
					let input_copy = input[0].clone();
					//let pixels = input[0].shape.flat_size_single() as f32;
					let output = g2.forward(1, input, data.params).remove(0);
					
					let mut err = 0.0;
					let mut y_err = 0.0;
					let mut pix = 0.0f32;
					for (o, i) in output.values.chunks(CHANNELS).zip(input_copy.values.chunks(CHANNELS)){
						let dr = o[0].max(0.0).min(1.0)-i[0].max(0.0).min(1.0);
						let dg = o[1].max(0.0).min(1.0)-i[1].max(0.0).min(1.0);
						let db = o[2].max(0.0).min(1.0)-i[2].max(0.0).min(1.0);


						//let yo = LinearToSrgbFunc::activ(SrgbToLinearFunc::activ(o[0].max(0.0).min(1.0)).0*0.212655 + SrgbToLinearFunc::activ(o[1].max(0.0).min(1.0)).0*0.715158 + SrgbToLinearFunc::activ(o[2].max(0.0).min(1.0)).0*0.072187).0;
						//let yi = LinearToSrgbFunc::activ(SrgbToLinearFunc::activ(i[0].max(0.0).min(1.0)).0*0.212655 + SrgbToLinearFunc::activ(i[1].max(0.0).min(1.0)).0*0.715158 + SrgbToLinearFunc::activ(i[2].max(0.0).min(1.0)).0*0.072187).0;


						let y_diff = dr*0.299 + dg*0.587 + db*0.114;

						y_err += y_diff*y_diff;//(yo - yi)*(yo - yi);
						err += (dr*dr + dg*dg + db*db)/3.0; // R G B
						pix += 1.0;
					}
					psnr_sum += -10.0*(err/pix).log10();
					y_psnr_sum += -10.0*(y_err/pix).log10();

					pix_sum += pix;
					err_sum += err;
					y_err_sum += y_err;
				}

				psnr_sum /= n as f32;
				y_psnr_sum /= n as f32;
				let psnr = -10.0*(err_sum/pix_sum).log10();
				let y_psnr = -10.0*(y_err_sum/pix_sum).log10();
				println!("Validation PixAvgPSNR:\t{}\tPixAvgY_PSNR:\t{}\tImgAvgPSNR:\t{}\tImgAvgY_PSNR:\t{}", psnr, y_psnr, psnr_sum, y_psnr_sum);
			}

			CallbackSignal::Continue
		});
	}

	solver.add_boxed_step_callback(max_evals(10_000_000));

	println!("Beginning Training");
	solver.optimise_from(&mut training_set, start_params);	
	println!("Done");
}

// fn psnr(app_m: &ArgMatches){


// }
