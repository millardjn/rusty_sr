
use alumina::ops::Op;
use alumina::ops::shape::avg_pool::AvgPool;
use alumina::ops::nn::conv::Conv;
use alumina::ops::shape::linterp::Linterp;
use alumina::ops::shape::pixel_shuffle::Expand;
use alumina::ops::nn::bias::Bias;
use alumina::ops::activ::spline::Spline;
use alumina::ops::activ::srgb::{SrgbToLinear, LinearToSrgb};
use alumina::ops::shape::shape_constraint::ShapeConstraint;
use alumina::ops::regularisation::l2::L2;
use alumina::ops::loss::robust::Robust;

use alumina::graph::{GraphDef, Result};
use alumina::id::NodeTag;


const CHANNELS: usize = 3;

/// Creates the upscaling network used as a base for training and inference
///
/// The connection map in this network is similar to fractal-net, and ensures the following:
///  * The compute required increases linearly (amortised) with depth.
///  * The inference time peak memory increases O((log n)^2) (amortised) with depth.
///  * The training time memory increases linearly (amortised) with depth.
///  * The effective backprop depth (maximum jumps for gradient to get to any particular node) increases logarithmically with depth
pub fn sr_net_base(factor: usize, log_depth: u32) -> Result<GraphDef> {
	
	let hidden_layers = 2usize.pow(log_depth) - 1;// 2^x-1 if the prediction layer should connect directly to the input

	let mut g = GraphDef::new();
	let mut conv_nodes = vec![];
	let mut active_nodes = vec![];

	active_nodes.push(g.new_node(shape![Unknown, Unknown, Unknown, CHANNELS], "input", tag![])?);

	for i in 0..hidden_layers+1{
		let jumps = (0..).map(|x| 2usize.pow(x)).take_while(|&jump| (i+1)%jump == 0 && jump <= active_nodes.len()).collect::<Vec<_>>();
		let hidden_layer_channels = 16 * jumps.len();
		let (new_conv_node, init_weight) = if i < hidden_layers {
			(g.new_node(shape![Unknown, Unknown, Unknown, hidden_layer_channels], format!("conv{}", i), tag![])?,
			1.0)
		} else {
			(g.new_node(shape![Unknown, Unknown, Unknown, CHANNELS*factor*factor], "expand", tag![])?,
			0.001)
		};
		
		// connect each layer (hidden and output) to previous layers in a fractal-net like pattern.
		for jump in jumps.iter(){
			Conv::new(&active_nodes[active_nodes.len() - jump], &new_conv_node, &[3, 3])
				.init(Conv::msra(init_weight/jumps.len() as f32)).add_to(&mut g, tag![])?;
		}
		Bias::new(&new_conv_node).add_to(&mut g, tag![])?;

		// add activation only for hidden layers
		if i < hidden_layers{
			let new_active_node = g.new_node(shape![Unknown, Unknown, Unknown, hidden_layer_channels], format!("activ{}", i), tag![])?;
			Spline::new(&new_conv_node, &new_active_node).shared_axes(&[0, 1, 2]).init(Spline::swan()).add_to(&mut g, tag![])?;
			active_nodes.push(new_active_node);
		}
		conv_nodes.push(new_conv_node);
	}

	// Linterp and expansion of residuals
	let input = &active_nodes[0];
	let output = g.new_node(shape![Unknown, Unknown, Unknown, CHANNELS], "output", tag![])?;
	Expand::new(&conv_nodes[conv_nodes.len()- 1], &output, &[1, factor, factor]).add_to(&mut g, tag![])?;
	Linterp::new(input, &output, &[1, factor, factor, 1]).add_to(&mut g, tag![])?;

	Ok(g)
}

/// Just the base net plus the constraint that the output is exactly factor larger than the input
pub fn inference_sr_net(factor: usize, log_depth: u32) -> Result<GraphDef> {
	let mut g = sr_net_base(factor, log_depth)?;

	ShapeConstraint::new(&g.node_id("input"), &g.node_id("output"))
		.single(1, move |d| d*factor)
		.single(2, move |d| d*factor)
		.add_to(&mut g, tag![])?;	

	Ok(g)
}

pub fn training_sr_net(factor: usize, log_depth: u32, regularisation: f32, loss_power: f32, loss_scale: f32, srgb_downsample: bool) -> Result<GraphDef> {
	let mut g = sr_net_base(factor, log_depth)?;

	if regularisation != 0.0 {
		for node_id in g.node_ids(NodeTag::Parameter) {
			L2::new(&node_id).multiplier(regularisation).add_to(&mut g, tag![])?;
		}
	}	

	let input = g.node_id("input");
	let output = g.node_id("output");

	let training_input = g.new_node(shape![Unknown, Unknown, Unknown, CHANNELS], "training_input", tag![])?;

	if srgb_downsample {
		AvgPool::new(&training_input, &input, &[1, factor, factor, 1]).add_to(&mut g, tag![])?;
	} else {
		let training_input_lin = g.new_node(shape![Unknown, Unknown, Unknown, CHANNELS], "training_input_lin", tag![])?;
		let input_pool = g.new_node(shape![Unknown, Unknown, Unknown, CHANNELS], "input_pool", tag![])?;
		SrgbToLinear::new(&training_input, &training_input_lin).add_to(&mut g, tag![])?;
		AvgPool::new(&training_input_lin, &input_pool, &[1, factor, factor, 1]).add_to(&mut g, tag![])?;
		LinearToSrgb::new(&input_pool, &input).add_to(&mut g, tag![])?;
	}

	Robust::new(&output, &training_input, loss_scale, loss_power).multiplier(100.0*loss_scale*loss_power).mean_axes(&[0, 1, 2, 3]).add_to(&mut g, tag![])?;

	ShapeConstraint::new(&training_input, &output).single(1, |d| d).single(2, |d| d).add_to(&mut g, tag![])?;

	Ok(g)
}

pub fn bilinear_net(factor: usize) -> Result<GraphDef> {
	let mut g = GraphDef::new();

	let inp = g.new_node(shape![Unknown, Unknown, Unknown, CHANNELS], "input", tag![])?;
	let lin = g.new_node(shape![Unknown, Unknown, Unknown, CHANNELS], "linear_color", tag![])?;
	let upscale = g.new_node(shape![Unknown, Unknown, Unknown, CHANNELS], "upscaled", tag![])?;
	let out = g.new_node(shape![Unknown, Unknown, Unknown, CHANNELS], "output", tag![])?;

	SrgbToLinear::new(&inp, &lin).add_to(&mut g, tag![])?;
	Linterp::new(&lin, &upscale, &[1, factor, factor, 1]).add_to(&mut g, tag![])?;
	LinearToSrgb::new(&upscale, &out).add_to(&mut g, tag![])?;

	ShapeConstraint::new(&inp, &upscale).single(1, move |d| d*factor).single(2, move |d| d*factor).add_to(&mut g, tag![])?;
	Ok(g)
}

pub fn downscale_lin_net(factor: usize) -> Result<GraphDef> {
	let mut g = GraphDef::new();

	let input_hr = g.new_node(shape![Unknown, Unknown, Unknown, CHANNELS], "input", tag![])?;
	let input_hr_lin = g.new_node(shape![Unknown, Unknown, Unknown, CHANNELS], "linear_color", tag![])?;
	let input_pool = g.new_node(shape![Unknown, Unknown, Unknown, CHANNELS], "pool", tag![])?;
	let output = g.new_node(shape![Unknown, Unknown, Unknown, CHANNELS], "output", tag![])?;

	SrgbToLinear::new(&input_hr, &input_hr_lin).add_to(&mut g, tag![])?;
	AvgPool::new(&input_hr_lin, &input_pool, &[1, factor, factor, 1]).add_to(&mut g, tag![])?;
	LinearToSrgb::new(&input_pool, &output).add_to(&mut g, tag![])?;

	Ok(g)
}

pub fn downscale_srgb_net(factor: usize) -> Result<GraphDef> {
	let mut g = GraphDef::new();

	let input_hr = g.new_node(shape![Unknown, Unknown, Unknown, CHANNELS], "input", tag![])?;
	let output = g.new_node(shape![Unknown, Unknown, Unknown, CHANNELS], "output", tag![])?;
	
	AvgPool::new(&input_hr, &output, &[1, factor, factor, 1]).add_to(&mut g, tag![])?;

	Ok(g)
}
