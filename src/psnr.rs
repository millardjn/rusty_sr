use std::path::Path;
use std::cmp;

use clap::ArgMatches;

use ndarray::{ArrayViewD, Zip, Si};

use alumina::data::image_folder::image_to_data;

use image;


pub fn psnr(app_m: &ArgMatches)-> Result<(), String>{
	let image1 = image::open(Path::new(app_m.value_of("IMAGE1").expect("No input file given?"))).expect("Error opening image1 file.");
	let image2 = image::open(Path::new(app_m.value_of("IMAGE2").expect("No input file given?"))).expect("Error opening image2 file.");

	let image1 = image_to_data(&image1);
	let image2 = image_to_data(&image2);

	if image1.shape() != image2.shape() {
		println!("Image shapes will be cropped to the areas which overlap");
	}

	let min_height = cmp::min(image1.shape()[0], image2.shape()[0]);
	let min_width = cmp::min(image1.shape()[1], image2.shape()[2]);

	let slice_arg: [Si; 3] = [Si(0, Some(min_height as isize), 1), Si(0, Some(min_width as isize), 1), Si(0, Some(3), 1)];
	let image1 = image1.slice(&slice_arg);
	let image2 = image2.slice(&slice_arg);

	let (err, y_err, pix) = psnr_calculation(image1, image2);

	println!("sRGB PSNR: {}\tLuma PSNR:{}", -10.0*(err/pix).log10(), -10.0*(y_err/pix).log10());
	Ok(())
}

/// returns the err, y_err and pixel count of a pair of images
pub fn psnr_calculation(image1: ArrayViewD<f32>, image2: ArrayViewD<f32>) -> (f32, f32, f32) {
	let mut err = 0.0;
	let mut y_err = 0.0;
	let mut pix = 0.0f32;

	Zip::from(image1.genrows())
		.and(image2.genrows())
		.apply(|o, i| {
			let dr = o[0].max(0.0).min(1.0)-i[0].max(0.0).min(1.0);
			let dg = o[1].max(0.0).min(1.0)-i[1].max(0.0).min(1.0);
			let db = o[2].max(0.0).min(1.0)-i[2].max(0.0).min(1.0);
			let y_diff = dr*0.299 + dg*0.587 + db*0.114; // BT.601
			//let y_diff = dr*0.2126 + dg*0.7152 + db*0.0722; // BT.709

			y_err += y_diff*y_diff;//(yo - yi)*(yo - yi);
			err += (dr*dr + dg*dg + db*db)/3.0; // R G B
			pix += 1.0;
		});

	(err, y_err, pix)
}