#[macro_use]
extern crate alumina;
extern crate rand;
extern crate clap;
extern crate image;
extern crate ndarray;
#[macro_use]
extern crate serde_derive;
extern crate serde;
extern crate bincode;
extern crate xz2;
extern crate byteorder;
extern crate indexmap;
extern crate smallvec;

pub mod network;
pub mod psnr;
pub mod aligned_crop;

use std::fs::*;
use std::io::{stdout, Write, Read};
use std::num::FpCategory;
use std::fmt;

use bincode::{serialize, deserialize, Infinite};
use image::{ImageFormat, ImageResult};

use network::*;

use ndarray::{ArrayD, IxDyn, Axis};
use byteorder::{BigEndian, ByteOrder};
use xz2::read::{XzEncoder, XzDecoder};

use alumina::data::image_folder::{image_to_data, data_to_image};
use alumina::graph::{GraphDef};


const L1_SRGB_NATURAL_PARAMS: &'static [u8] = include_bytes!("res/L1_x4_UCID_x1node.rsr");
const L1_SRGB_ANIME_PARAMS: &'static [u8] = include_bytes!("res/L1_x4_Anime_x1node.rsr");


/// A struct containing the network parameters and hyperparameters.
#[derive(Debug, Serialize, Deserialize)]
pub struct NetworkDescription {
	pub factor: u32,
	pub log_depth: u32,
	pub global_node_factor: u32,
	pub parameters: Vec<ArrayD<f32>>,
}

/// Decompresses and deserialises the NetworkDescription from the byte format used in .rsr. files
pub fn network_from_bytes(data: &[u8]) -> ::std::result::Result<NetworkDescription, String> {
	let decompressed = XzDecoder::new(data).bytes().collect::<::std::result::Result<Vec<_>, _>>().map_err(|e| e.to_string())?;
	let unshuffled = unshuffle(&decompressed, 4);
	let deserialized: NetworkDescription = deserialize(&unshuffled).map_err(|e| format!("NetworkDescription decoding failed: {}", e))?;
	Ok(deserialized)
}

/// Serialises and compresses the NetworkDescription returning the byte format used in .rsr files
/// If quantise = true, then the least significant 12 bits are zeroed to improve compression.
pub fn network_to_bytes(mut desc: NetworkDescription, quantise: bool) -> ::std::result::Result<Vec<u8>, String> {
	for arr in &mut desc.parameters {
		for e in arr.iter_mut() {
			if let FpCategory::Subnormal = e.classify(){
				*e = 0.0;
			}
			if quantise {
				let mut bytes = [0; 4];
				BigEndian::write_f32(&mut bytes, *e);
				bytes[2] &= 0xF0;
				bytes[3] &= 0x00;
				*e = BigEndian::read_f32(&bytes);
			}
		}
	}
	
	let serialized: Vec<u8> = serialize(&desc, Infinite).map_err(|e| format!("NetworkDescription encoding failed: {}", e))?;
	let shuffled = shuffle(&serialized, 4);
	let compressed = XzEncoder::new(shuffled.as_slice(), 7).bytes().collect::<::std::result::Result<Vec<_>, _>>().map_err(|e| e.to_string())?;
	Ok(compressed)
}

/// Shuffle f32 bytes so that all first bytes are contiguous etc
/// Improves compression of floating point data
fn shuffle(data: &[u8], stride: usize) -> Vec<u8> {
	let mut vec = Vec::with_capacity(data.len());
	for offset in 0..stride {
		for i in 0..(data.len()-offset+stride-1)/stride {
			vec.push(data[offset + i*stride])
		}
	}
	debug_assert_eq!(vec.len(), data.len());
	vec
}

/// Inverts `shuffle()`
fn unshuffle(data: &[u8], stride: usize) -> Vec<u8> {
	let mut vec = vec![0; data.len()];
	let mut inc = 0;
	for offset in 0..stride {
		for i in 0..(data.len()-offset+stride-1)/stride {
			vec[offset + i*stride] = data[inc];
			inc += 1;
		}
	}
	debug_assert_eq!(inc, data.len());
	vec
}


/// Returns an rgb image converted to a tensor of floats in the range of [0, 1] and of shape, [1, H, W, 3];
pub fn read(file: &mut File) -> ImageResult<ArrayD<f32>> {
	let mut vec = vec![];
	file.read_to_end(&mut vec).map_err(|err| image::ImageError::IoError(err))?;
	let input_image = image::load_from_memory(&vec)?;
	let input = image_to_data(&input_image);
	let shape = input.shape().to_vec();
	let input = input.into_shape(IxDyn(&[1, shape[0], shape[1], shape[2]])).unwrap();
	Ok(input)
}

/// Save tensor of shape [1, H, W, 3] as .png image. Converts floats in range of [0, 1] to bytes in range [0, 255].
pub fn save(image: ArrayD<f32>, file: &mut File) -> ImageResult<()> {
	stdout().flush().ok();
	data_to_image(image.subview(Axis(0), 0)).save(file, ImageFormat::PNG)
}


/// Takes an image tensor of shape [1, H, W, 3] and returns one of shape [1, H/factor, W/factor, 3].
/// 
/// If `sRGB` is `true` the pooling operation averages over the image values as stored,
/// if `false` then the sRGB values are temporarily converted to linear RGB, pooled, then converted back.
#[allow(non_snake_case)]
pub fn downscale(image: ArrayD<f32>, factor: usize, sRGB: bool) -> alumina::graph::Result<ArrayD<f32>> {
	let graph = if sRGB {
			downscale_srgb_net(factor)?
		} else {
			downscale_lin_net(factor)?
		};

	let input_id = graph.node_id("input").value_id();
	let output_id = graph.node_id("output").value_id();
	let mut subgraph = graph.subgraph(&[input_id.clone()], &[output_id.clone()])?;
	let result = subgraph.execute(vec![image])?;

	Ok(result.into_map().remove(&output_id).unwrap())
}

/// A container type for upscaling networks
#[derive(Clone, Debug)]
pub struct UpscalingNetwork {
	graph: GraphDef,
	parameters: Vec<ArrayD<f32>>,
	display: String,
}

impl UpscalingNetwork {

	pub fn new(desc: NetworkDescription, display: &str) -> ::std::result::Result<Self, String> {
		Ok(UpscalingNetwork {
			graph: inference_sr_net(desc.factor as usize, desc.log_depth, desc.global_node_factor as usize).map_err(|e| e.to_string())?,
			parameters: desc.parameters,
			display: display.to_string(),
		})
	}

	/// Accepts labels: [natural, natural_L1, natural_rgb, anime, anime_L1, bilinear]
	pub fn from_label(label: &str, bilinear_factor: Option<usize>) -> ::std::result::Result<Self, String> {
		match label {
			"natural" => {
				let network_desc = network_from_bytes(L1_SRGB_NATURAL_PARAMS)?;
				Ok(UpscalingNetwork {
					graph: inference_sr_net(network_desc.factor as usize, network_desc.log_depth, network_desc.global_node_factor as usize).map_err(|e| e.to_string())?,
					parameters: network_desc.parameters,
					display: "neural net trained on natural images with an L1 loss".to_string(),
				})
			},
			"anime" => {
				let network_desc = network_from_bytes(L1_SRGB_ANIME_PARAMS)?;
				Ok(UpscalingNetwork {
					graph: inference_sr_net(network_desc.factor as usize, network_desc.log_depth, network_desc.global_node_factor as usize).map_err(|e| e.to_string())?,
					parameters: network_desc.parameters,
					display: "neural net trained on animation images with an L1 loss".to_string(),
				})
			},
			"bilinear" => {
				Ok(UpscalingNetwork {
					graph: bilinear_net(bilinear_factor.unwrap_or(4)).map_err(|e| e.to_string())?,
					parameters: Vec::new(),
					display: "bilinear interpolation".to_string(),
				})
			},
			_ => Err(format!("Unsupported network type. Could not parse: {}", label)),
		}
	}

	pub fn borrow_network(&self) -> (&GraphDef, &[ArrayD<f32>]) {
		(&self.graph, &self.parameters)
	}
}

impl fmt::Display for UpscalingNetwork {
	fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
		write!(f, "{}", self.display)
	}
}


/// Takes an image tensor of shape [1, H, W, 3] and returns one of shape [1, H*factor, W*factor, 3], where factor is determined by the network definition.
/// 
/// The exact results of the upscaling depend on the content on which the network being used was trained, and what loss it was trained to minimise.
/// L2 loss maximises PSNR, where as L1 loss results in sharper edges.
/// `bilinear_factor` is ignored unless the network is Bilinear.
pub fn upscale(image: ArrayD<f32>, network: UpscalingNetwork) -> alumina::graph::Result<ArrayD<f32>> {
	let (graph, params) = network.borrow_network();

	let mut input_vec = vec![image];
	input_vec.extend(params.iter().cloned());
	let input_id = graph.node_id("input").value_id();
	let param_ids: Vec<_> = graph.parameter_ids().iter().map(|node_id| node_id.value_id()).collect();
	let mut subgraph_inputs = vec![input_id];
	subgraph_inputs.extend(param_ids);
	let output_id = graph.node_id("output").value_id();
	let mut subgraph = graph.subgraph(&subgraph_inputs, &[output_id.clone()])?;
	let result = subgraph.execute(input_vec).expect("Could not execute upsampling graph");

	Ok(result.into_map().remove(&output_id).unwrap())
}