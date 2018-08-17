
use std::cmp;

use ndarray::{ArrayViewD, Axis, Zip};


/// Takes two tensors of shape [H, W, 3] and
/// returns the err, y_err and pixel count of a pair of images.
/// 
/// If a 4th dimension is present a subview at index 0 will be used.
pub fn psnr_calculation(image1: ArrayViewD<f32>, image2: ArrayViewD<f32>) -> (f32, f32, f32) {
	
	let image1 = if image1.ndim() == 4 {
		image1.subview(Axis(0), 0)
	} else {
		image1.view()
	};

	let image2 = if image2.ndim() == 4 {
		image2.subview(Axis(0), 0)
	} else {
		image2.view()
	};

	let min_height = cmp::min(image1.shape()[0], image2.shape()[0]);
	let min_width = cmp::min(image1.shape()[1], image2.shape()[2]);

	let image1 = image1.slice(s![0..min_height, 0..min_width, 0..3]);
	let image2 = image2.slice(s![0..min_height, 0..min_width, 0..3]);

	
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