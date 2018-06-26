use indexmap::{IndexMap, IndexSet};
use rand::{thread_rng, Rng};
use ndarray::{ArrayD, ArrayViewD, IxDyn, Si};
use smallvec::SmallVec;
use alumina::data::DataSet;

use std::mem;

use alumina::data::crop::Cropping;

pub trait AlignedCrop: DataSet {
	fn aligned_crop(self, component: usize, shape: &[usize], cropping: Cropping) -> AlignedCropSet<Self> where Self: Sized {
		AlignedCropSet::new(self, component, shape, cropping)
	}
}

impl<D: DataSet> AlignedCrop for D {}

/// For one component in each element of the dataset: apply a function.
///
/// Renaming the component is optional.
pub struct AlignedCropSet<S: DataSet> {
	set: S,
	fill: IndexMap<usize, f32>,
	main_crop: (usize, Vec<usize>, Cropping),
	other_crops: IndexSet<(usize, Option<usize>)>,
}

impl<S: DataSet> AlignedCropSet<S> {
	/// Crop the given component to the given shape. `shape` must have the same dimensionality as the component.
	pub fn new(set: S, component: usize, shape: &[usize], cropping: Cropping) -> Self {
		AlignedCropSet {
			set,
			fill: IndexMap::new(),
			main_crop: (component, shape.to_vec(), cropping),
			other_crops: IndexSet::new(),
		}
	}

	/// Crop another component
	pub fn and_crop(mut self, component: usize, factor: Option<usize>) -> Self {
		self.other_crops.insert((component, factor));
		self
	}

	/// Set what should be used to fill areas where the crop dimension is larger than the input dimension.
	///
	/// Default: 0.0
	pub fn fill(mut self, component: usize, fill: f32) -> Self {
		self.fill.insert(component, fill);
		self
	}

	/// Borrows the wrapped dataset.
	pub fn inner(&self) -> &S {
		&self.set
	}

	/// Returns the wrapped dataset.
	pub fn into_inner(self) -> S {
		let Self{set, ..} = self;
		set
	}
}

impl<S: DataSet> DataSet for AlignedCropSet<S> {
	fn get(&mut self, i: usize) -> Vec<ArrayD<f32>> {
		let mut data = self.set.get(i);


		let (component, ref crop_shape, ref cropping) = self.main_crop;

		let arr = mem::replace(&mut data[component], ArrayD::zeros(IxDyn(&[])));
		let arr_shape = arr.shape().to_vec();
		let (input_slice_arg, output_slice_arg) = slice_args(arr.view(), crop_shape, cropping);
		let fill = self.fill.get(&component).cloned().unwrap_or(0.0);
		mem::replace(&mut data[component], crop(arr, crop_shape, fill, input_slice_arg.clone(), output_slice_arg.clone())) ;


		for &(component, ref factor) in self.other_crops.iter() {
			let next_arr = mem::replace(&mut data[component], ArrayD::zeros(IxDyn(&[])));

			let (next_crop_shape, next_input_slice_arg, next_output_slice_arg) = secondary_slice_args(factor.clone(), &arr_shape, next_arr.shape(), crop_shape, input_slice_arg.clone(), output_slice_arg.clone());

			let fill = self.fill.get(&component).cloned().unwrap_or(0.0);
			mem::replace(&mut data[component], crop(next_arr, &next_crop_shape, fill, next_input_slice_arg.clone(), next_output_slice_arg.clone()));
		}

		data
	}

	fn length(&self) -> usize{
		self.set.length()
	}

	fn width(&self) -> usize {
		self.set.width()
	}

	fn components(&self) -> Vec<String>{
		self.set.components()
	}
}


fn crop(arr: ArrayD<f32>, crop_shape: &[usize], fill: f32, input_slice_arg: SmallVec<[Si; 6]>, output_slice_arg: SmallVec<[Si; 6]>) -> ArrayD<f32> {

	assert_eq!(crop_shape.len(), arr.ndim());

	let mut out_arr = ArrayD::from_elem(IxDyn(crop_shape), fill);

	{
		let in_slice = arr.slice(input_slice_arg.as_slice());
		let mut out_slice = out_arr.slice_mut(output_slice_arg.as_slice());
		out_slice.assign(&in_slice);
	}

	out_arr
}

fn secondary_slice_args(factor: Option<usize>, arr_shape: &[usize], next_arr_shape: &[usize], crop_shape: &[usize], input_slice_arg: SmallVec<[Si; 6]>, output_slice_arg: SmallVec<[Si; 6]>) -> (SmallVec<[usize; 6]>, SmallVec<[Si; 6]>, SmallVec<[Si; 6]>) {

	let mut next_crop_shape = SmallVec::new();
	let mut next_input_slice_arg: SmallVec<[Si; 6]> = SmallVec::new();
	let mut next_output_slice_arg: SmallVec<[Si; 6]> = SmallVec::new();

	for (i, (&arr_dim, &next_arr_dim)) in arr_shape.iter().zip(next_arr_shape).enumerate() {
		if let Some(factor) = factor {
			if i < arr_shape.len()-1 {
				//eprintln!("arr_shape:{:?} next_arr_shape:{:?}", arr_shape, next_arr_shape);
				assert_eq!(arr_dim, (next_arr_dim+factor-1)/factor);
			}
		}
		next_crop_shape.push(crop_shape[i]*next_arr_dim/arr_dim);

		let arr_dim = arr_dim as isize;
		let next_arr_dim = next_arr_dim as isize;

		if let Si(start, Some(end), 1) = input_slice_arg[i]{
			next_input_slice_arg.push(Si(start*next_arr_dim/arr_dim, Some(end*next_arr_dim/arr_dim), 1))
		}
		if let Si(start, Some(end), 1) = output_slice_arg[i]{
			next_output_slice_arg.push(Si(start*next_arr_dim/arr_dim, Some(end*next_arr_dim/arr_dim), 1))
		}
	}

	(next_crop_shape, next_input_slice_arg, next_output_slice_arg)
}

fn slice_args(arr: ArrayViewD<f32>, crop_shape: &[usize], cropping: &Cropping) -> (SmallVec<[Si; 6]>, SmallVec<[Si; 6]>) {
	let mut input_slice_arg: SmallVec<[Si; 6]> = SmallVec::new();
	let mut output_slice_arg: SmallVec<[Si; 6]> = SmallVec::new();
	for (&input_width, &output_width) in arr.shape().iter().zip(crop_shape) {
		let (in_si, out_si) = range(cropping, input_width as isize, output_width as isize);
		input_slice_arg.push(in_si);
		output_slice_arg.push(out_si);
	}
	(input_slice_arg, output_slice_arg)
}


// returns Si for input and output
fn range(cropping: &Cropping, input_width: isize, output_width: isize) -> (Si, Si) {
	match cropping {
		&Cropping::Centre{..} => {
			if input_width < output_width {
				let width = input_width;
				let output_start = (input_width - output_width)/2;
				(Si(0, Some(width), 1),
				Si(output_start, Some(output_start + width), 1))
			} else {
				let width = output_width;
				let input_start = (output_width - input_width)/2;
				(Si(input_start, Some(input_start + width), 1),
				Si(0, Some(width), 1))
			}
		},

		&Cropping::Random{..} => {
			if input_width < output_width {
				let width = input_width;
				let output_start = thread_rng().gen_range(0, output_width - input_width + 1);
				(Si(0, Some(width), 1),
				Si(output_start, Some(output_start + width), 1))
			} else {
				let width = output_width;
				let input_start = thread_rng().gen_range(0, input_width - output_width + 1);
				(Si(input_start, Some(input_start + width), 1),
				Si(0, Some(width), 1))
			}
		},
	}
}
