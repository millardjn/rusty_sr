extern crate alumina;
extern crate rand;
extern crate clap;
extern crate image;
extern crate ndarray;
extern crate serde;
extern crate bincode;
extern crate xz2;
extern crate byteorder;

extern crate rusty_sr;


use std::fs::*;
use std::path::Path;
use std::io::{stdout, Write, Read};
use std::cmp;
use std::iter;

use clap::{Arg, App, SubCommand, AppSettings, ArgMatches};

use alumina::opt::adam::Adam;
use alumina::data::image_folder::{ImageFolder, image_to_data};
use alumina::data::{DataSet, DataStream, Cropping};
use alumina::graph::{Result, GraphDef};
use alumina::opt::{Opt, UnboxedCallbacks, CallbackSignal};

use rusty_sr::psnr;
use rusty_sr::network::*;
use rusty_sr::{UpscalingNetwork, NetworkDescription};

fn main() {
	let app_m = App::new("Rusty SR")
	.version("v0.2.0")
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
	.arg(Arg::with_name("PARAMETERS")
		.help("Sets which built-in parameters to use with the neural net. Default: natural")
		.short("p")
		.long("parameters")
		.value_name("PARAMETERS")
		.possible_values(&["natural", "natural_L1", "natural_rgb", "anime", "anime_L1", "bilinear"])
		.empty_values(false)
	)
	.arg(Arg::with_name("CUSTOM")
		.conflicts_with("PARAMETERS")
		.short("c")
		.long("custom")
		.value_name("PARAMETER_FILE")
		.help("Sets a custom parameter file (.rsr) to use with the neural net")
		.empty_values(false)
	)
	.arg(Arg::with_name("BILINEAR_FACTOR")
		.short("f")
		.long("factor")
		.help("The integer upscaling factor used if bilinear upscaling is performed. Default 3")
		.empty_values(false)
	)
	.subcommand(SubCommand::with_name("train")
		.about("Train a new set of neural parameters on your own dataset")
		.arg(Arg::with_name("TRAINING_FOLDER")
			.required(true)
			.index(1)
			.help("Images from this folder(or sub-folders) will be used for training")
		)
		.arg(Arg::with_name("PARAMETER_FILE")
			.required(true)
			.index(2)
			.help("Learned network parameters will be (over)written to this parameter file (.rsr)")
		)
		.arg(Arg::with_name("LEARNING_RATE")
			.short("R")
			.long("rate")
			.help("The learning rate used by the Adam optimiser. Default: 3e-3")
			.empty_values(false)
		)
		.arg(Arg::with_name("QUANTISE")
			.short("q")
			.long("quantise")
			.help("Quantise the weights by zeroing the smallest 12 bits of each f32. Reduces parameter file size")
			.takes_value(false)
		)
		.arg(Arg::with_name("TRAINING_LOSS")
	 		.help("Selects whether the neural net learns to minimise the L1 or L2 loss. Default: L1")
	 		.short("l")
	 		.long("loss")
	 		.value_name("LOSS")
	 		.possible_values(&["L1", "L2"])
	 		.empty_values(false)
	 	)
		.arg(Arg::with_name("DOWNSCALE_COLOURSPACE")
	 		.help("Colourspace in which to perform downsampling. Default: sRGB")
	 		.short("c")
	 		.long("colourspace")
	 		.value_name("COLOURSPACE")
	 		.possible_values(&["sRGB", "RGB"])
	 		.empty_values(false)
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
			.empty_values(false)
		)
		.arg(Arg::with_name("FACTOR")
			.short("f")
			.long("factor")
			.help("The integer upscaling factor of the network the be trained. Default: 3")
			.empty_values(false)
		)
		.arg(Arg::with_name("LOG_DEPTH")
			.short("d")
			.long("log_depth")
			.help("There will be 2^(log_depth)-1 hidden layers in the network the be trained. Default: 4")
			.empty_values(false)
		)
		.arg(Arg::with_name("PATCH_SIZE")
			.short("p")
			.long("patch_size")
			.help("The integer patch_size of the training input after downsampling. Default: 48")
			.empty_values(false)
		)
		.arg(Arg::with_name("BATCH_SIZE")
			.short("b")
			.long("batch_size")
			.help("The integer batch_size of the training input. Default: 4")
			.empty_values(false)
		)
		.arg(Arg::with_name("VALIDATION_FOLDER")
			.short("v")
			.long("val_folder")
			.help("Images from this folder(or sub-folders) will be used to evaluate training progress")
			.empty_values(false)
		)
		.arg(Arg::with_name("VAL_MAX")
			.requires("VALIDATION_FOLDER")
			.short("m")
			.long("val_max")
			.value_name("N")
			.help("Set upper limit on number of images used for each validation pass")
			.empty_values(false)
		)
	)
	.subcommand(SubCommand::with_name("downscale")
	 	.about("Downscale images")
		.arg(Arg::with_name("FACTOR")
			.help("The integer downscaling factor")
			.required(true)
			.index(1)
		)
		.arg(Arg::with_name("INPUT_FILE")
			.help("Sets the input image to downscale")
			.required(true)
			.index(2)
		)
		.arg(Arg::with_name("OUTPUT_FILE")
			.help("Sets the output file to write/overwrite (.png recommended)")
			.required(true)
			.index(3)
		)
	 	.arg(Arg::with_name("COLOURSPACE")
	 		.help("colourspace in which to perform downsampling. Default: sRGB")
	 		.short("c")
	 		.long("colourspace")
	 		.value_name("COLOURSPACE")
	 		.possible_values(&["sRGB", "RGB"])
	 		.empty_values(false)
	 	)
	)
	.subcommand(SubCommand::with_name("quantise")
	 	.about("Quantise the weights of a network, reducing file size")
		.arg(Arg::with_name("INPUT_FILE")
			.help("The input network to be quantised")
			.required(true)
			.index(1)
		)
		.arg(Arg::with_name("OUTPUT_FILE")
			.help("The location at which the quantised network will be saved")
			.required(true)
			.index(2)
		)
	)
	.subcommand(SubCommand::with_name("psnr")
		.about("Print the PSNR value from the differences between the two images")
		.arg(Arg::with_name("IMAGE1")
			.required(true)
			.index(1)
			.help("PSNR is calculated using the difference between this image and IMAGE2")
		)
		.arg(Arg::with_name("IMAGE2")
			.required(true)
			.index(2)
			.help("PSNR is calculated using the difference between this image and IMAGE1")
		)
	)
	.get_matches();
	
	if let Some(sub_m) = app_m.subcommand_matches("train") {
		train(sub_m).unwrap_or_else(|err| println!("{}", err));
	} else if let Some(sub_m) = app_m.subcommand_matches("downscale") {
		downscale(sub_m).unwrap_or_else(|err| println!("{}", err));
	} else if let Some(sub_m) = app_m.subcommand_matches("psnr") {
		psnr(sub_m).unwrap_or_else(|err| println!("{}", err));
	} else if let Some(sub_m) = app_m.subcommand_matches("quantise") {
		quantise(sub_m).unwrap_or_else(|err| println!("{}", err));
	} else {
		upscale(&app_m).unwrap_or_else(|err| println!("{}", err));
	}
	
}

pub fn psnr(app_m: &ArgMatches)-> ::std::result::Result<(), String>{
	let image1 = image::open(Path::new(app_m.value_of("IMAGE1").expect("No input file given?"))).map_err(|e| format!("Error opening image1 file: {}", e))?;
	let image2 = image::open(Path::new(app_m.value_of("IMAGE2").expect("No input file given?"))).map_err(|e| format!("Error opening image2 file: {}", e))?;

	let image1 = image_to_data(&image1);
	let image2 = image_to_data(&image2);

	if image1.shape() != image2.shape() {
		println!("Image shapes will be cropped to the top left areas which overlap");
	}

	let (err, y_err, pix) = psnr::psnr_calculation(image1.view(), image2.view());

	println!("sRGB PSNR: {}\tLuma PSNR:{}", -10.0*(err/pix).log10(), -10.0*(y_err/pix).log10());
	Ok(())
}

fn quantise(app_m: &ArgMatches) -> ::std::result::Result<(), String>{
	let mut input_file = File::open(Path::new(app_m.value_of("INPUT_FILE").expect("No input file given?")))
		.map_err(|e| format!("Error opening input file: {}", e))?;
	let mut input_data = Vec::new();
	input_file.read_to_end(&mut input_data).map_err(|e| format!("Error reading input file: {}", e))?;
	let input_network = rusty_sr::network_from_bytes(&input_data)?;

	let mut output_file = File::create(Path::new(app_m.value_of("OUTPUT_FILE").expect("No output file given?")))
		.map_err(|e| format!("Error creating output file: {}", e))?;
	let output_data = rusty_sr::network_to_bytes(input_network, true).map_err(|e| e.to_string())?;
	output_file.write_all(&output_data).map_err(|e| format!("Error writing output file: {}", e))?;

	Ok(())
}

fn downscale(app_m: &ArgMatches) -> Result<()>{

	let factor = match app_m.value_of("FACTOR") {
		Some(string) => string.parse::<usize>().expect("Factor argument must be an integer"),
		_ => unreachable!(),
	};

	let srgb = match app_m.value_of("COLOURSPACE") {
		Some("sRGB") | None => true,
		Some("RGB")=> false,
		_ => unreachable!(),
	};

	let input = rusty_sr::read(&mut File::open(Path::new(app_m.value_of("INPUT_FILE").expect("No input file given?"))).map_err(|e| format!("Error opening input file: {}", e))?)
		.map_err(|e| format!("Error reading input file: {}", e))?;

	let output = rusty_sr::downscale(input, factor, srgb).map_err(|e| format!("Error while downscaling: {}", e))?;

	print!(" Writing file...");
	stdout().flush().ok();
	rusty_sr::save(output, &mut File::create(Path::new(app_m.value_of("OUTPUT_FILE").expect("No output file given?"))).map_err(|e| format!("Error creating output file: {}", e))?)
		.map_err(|e| format!("Error writing output file: {}", e))?;

	println!(" Done");
	Ok(())
}

fn upscale(app_m: &ArgMatches) -> ::std::result::Result<(), String>{
	let factor = match app_m.value_of("BILINEAR_FACTOR") {
		Some(string) => string.parse::<usize>().expect("Factor argument must be an integer"),
		_ => 3,
	};

	//-- Sort out parameters and graph
	let network: UpscalingNetwork = if let Some(file_str) = app_m.value_of("CUSTOM") {
		let mut param_file = File::open(Path::new(file_str)).expect("Error opening parameter file");
		let mut data = Vec::new();
		param_file.read_to_end(&mut data).expect("Reading parameter file failed");
		UpscalingNetwork::Custom(rusty_sr::network_from_bytes(&data)?)
	} else {
		app_m.value_of("PARAMETERS").unwrap_or("natural").parse::<UpscalingNetwork>().map_err(|e| format!("Error parsing PARAMETERS: {}", e))?
	};
	println!("Upsampling using {}...", network);


	let input = rusty_sr::read(&mut File::open(Path::new(app_m.value_of("INPUT_FILE").expect("No input file given?"))).map_err(|e| format!("Error opening input file: {}", e))?)
		.map_err(|e| format!("Error reading input file: {}", e))?;

	let output = rusty_sr::upscale(input, network, Some(factor)).map_err(|e| format!("Error while upscaling: {}", e))?;

	print!(" Writing file...");
	stdout().flush().ok();
	rusty_sr::save(output, &mut File::create(Path::new(app_m.value_of("OUTPUT_FILE").expect("No output file given?"))).map_err(|e| format!("Error creating output file: {}", e))?)
		.map_err(|e| format!("Error writing output file: {}", e))?;

	println!(" Done");
	Ok(())
}


fn train(app_m: &ArgMatches) -> Result<()> {

	println!("Training with:");

	let srgb_downscale = match app_m.value_of("DOWNSCALE_COLOURSPACE") {
		Some("sRGB") | None => {println!(" sRGB downscaling"); true},
		Some("RGB") => {println!(" RGB downscaling"); false},
		_ => unreachable!(),
	};
	
	let (power, scale) = match app_m.value_of("TRAINING_LOSS") {
		Some("L1") | None => {println!(" L1 loss"); (1.0, 1.0/255.0)},
		Some("L2") => {println!(" L2 loss"); (2.0, 1.0/255.0)},
		_ => unreachable!(),
	};

	let lr = app_m.value_of("LEARNING_RATE")
		.map(|string|string.parse::<f32>().expect("Learning rate argument must be a numeric value"))
		.unwrap_or(3e-3);
	if lr <= 0.0 {
		eprintln!("Learning_rate ({}) probably should be greater than 0.", lr);
	}
	println!(" learning rate: {}", lr);

	let patch_size = app_m.value_of("PATCH_SIZE")
		.map(|string| string.parse::<usize>().expect("Patch_size argument must be an integer"))
		.unwrap_or(48);
	assert!(patch_size > 0, "Patch_size ({}) must be greater than 0.", patch_size);
	println!(" patch_size: {}", patch_size);

	let batch_size = app_m.value_of("BATCH_SIZE")
		.map(|string| string.parse::<usize>().expect("Batch_size argument must be an integer"))
		.unwrap_or(4);
	assert!(batch_size > 0, "Batch_size ({}) must be greater than 0.", batch_size);
	println!(" batch_size: {}", patch_size);

	let quantise = app_m.is_present("QUANTISE");

	let recurse = app_m.is_present("RECURSE_SUBFOLDERS");

	let training_folder = app_m.value_of("TRAINING_FOLDER").expect("No training folder?");

	let param_file_path = Path::new(app_m.value_of("PARAMETER_FILE").expect("No parameter file?")).to_path_buf();

	let mut factor_option = None;
	match app_m.value_of("FACTOR") {
		Some(string) => factor_option = Some(string.parse::<usize>().expect("Factor argument must be an integer")),
		_ => {},
	}

	let mut log_depth_option = None;
	match app_m.value_of("LOG_DEPTH") {
		Some(string) => log_depth_option = Some(string.parse::<u32>().expect("Log_depth argument must be an integer")),
		_ => {},
	}

	let mut params_option = None;
	if let Some(param_str) = app_m.value_of("START_PARAMETERS") {
		println!(" initialising with parameters from: {}", param_file_path.to_string_lossy());
		let mut param_file = File::open(Path::new(param_str)).expect("Error opening start parameter file");
		let mut data = Vec::new();
		param_file.read_to_end(&mut data).expect("Reading start parameter file failed");
		let network_desc = rusty_sr::network_from_bytes(&data)?;
		if let Some(factor) = factor_option {
			if factor != network_desc.factor {
				eprintln!("Using factor from parameter file ({}) rather than factor from argument ({})", network_desc.factor, factor);
			}
		}
		if let Some(log_depth) = log_depth_option {
			if log_depth != network_desc.log_depth {
				eprintln!("Using log_depth from parameter file ({}) rather than log_depth from argument ({})", network_desc.log_depth, log_depth);
			}
		}
		params_option = Some(network_desc.parameters);
		factor_option = Some(network_desc.factor);
		log_depth_option = Some(network_desc.log_depth);
	}

	let factor = factor_option.unwrap_or(3);
	assert!(factor > 0, "factor ({}) must be greater than 0.", factor);
	println!(" factor: {}", factor);

	let log_depth = log_depth_option.unwrap_or(4);
	println!(" log_depth: {}", log_depth);

	let graph = training_sr_net(factor, log_depth, 1e-6, power, scale, srgb_downscale)?;
	
	let mut training_stream = ImageFolder::new(training_folder, recurse)
		.crop(0, &[patch_size*factor, patch_size*factor, 3], Cropping::Random)
		.shuffle_random()
		.batch(batch_size)
		.buffered(16);

	let mut solver = Adam::new(&graph)?
		.rate(lr)
		.beta1(0.9)
		.beta2(0.99)
		.bias_correct(false);

	let mut step_count = 0;
	solver.add_callback(move |data|{
		if step_count % 100 == 0 {
			let mut parameter_file = File::create(&param_file_path).expect("Could not make parameter file");
			let bytes = rusty_sr::network_to_bytes(NetworkDescription{factor: factor, log_depth: log_depth, parameters: data.params.to_vec()}, quantise).unwrap();
			parameter_file.write_all(&bytes).expect("Could not save to parameter file");
		}
		println!("step {}\terr:{}\tchange:{}", step_count, data.err, data.change_norm);
		step_count += 1;
		CallbackSignal::Continue
	});

	add_validation(app_m,  recurse, &mut solver, &graph)?;

	let params = params_option.unwrap_or_else(||graph.initialise_nodes(solver.parameters()).expect("Could not initialise parameters"));
	println!("Beginning Training");
	solver.optimise_from(&mut training_stream, params)?;	
	println!("Done");
	Ok(())
}

/// Add occasional validation set evaluation as solver callback
fn add_validation(app_m: &ArgMatches, recurse: bool, solver: &mut Opt, graph: &GraphDef) -> Result<()>{
	if let Some(val_folder) = app_m.value_of("VALIDATION_FOLDER"){

		let training_input_id = graph.node_id("training_input").value_id();
		let input_ids: Vec<_> = iter::once(training_input_id.clone()).chain(solver.parameters().iter().map(|node_id| node_id.value_id())).collect();
		let output_id = graph.node_id("output").value_id();
		let mut validation_subgraph = graph.subgraph(&input_ids, &[output_id.clone(), training_input_id.clone()])?;

		let validation_set =ImageFolder::new(val_folder, recurse);
		let epoch_size = validation_set.length();
		let mut validation_stream = validation_set
			.shuffle_random()
			.batch(1)
			.buffered(8);

		let n: usize = app_m.value_of("VAL_MAX").map(|val_max|{
			cmp::min(epoch_size, val_max.parse::<usize>().expect("-val_max N must be a positive integer"))
		}).unwrap_or(epoch_size);

		let mut step_count = 0;
		solver.add_boxed_callback(Box::new(move |data|{

			if step_count % 100 == 0 {

				let mut err_sum = 0.0;
				let mut y_err_sum = 0.0;
				let mut pix_sum = 0.0f32;

				let mut psnr_sum = 0.0;
				let mut y_psnr_sum = 0.0;

				for _ in 0..n {
					let mut training_input = validation_stream.next();
					training_input.extend(data.params.to_vec());

					let result = validation_subgraph.execute(training_input).expect("Could not execute upsampling graph");
					let output = result.get(&output_id).unwrap();
					let training_input = result.get(&training_input_id).unwrap();

					let (err, y_err, pix) = psnr::psnr_calculation(output, training_input);

					pix_sum += pix;
					err_sum += err;
					y_err_sum += y_err;

					psnr_sum += -10.0*(err/pix).log10();
					y_psnr_sum += -10.0*(y_err/pix).log10();
				}

				psnr_sum /= n as f32;
				y_psnr_sum /= n as f32;
				let psnr = -10.0*(err_sum/pix_sum).log10();
				let y_psnr = -10.0*(y_err_sum/pix_sum).log10();
				println!("Validation PixAvgPSNR:\t{}\tPixAvgY_PSNR:\t{}\tImgAvgPSNR:\t{}\tImgAvgY_PSNR:\t{}", psnr, y_psnr, psnr_sum, y_psnr_sum);
			}
			step_count += 1;
			CallbackSignal::Continue
		}));
	}
	Ok(())
}


