
use std::sync::Arc;
use alumina::ops::activ::*;
use alumina::ops::basic::*;	
use alumina::ops::loss::*;
use alumina::ops::conv::*;
use alumina::ops::reshape::*;
use alumina::ops::broadcast::*;
use alumina::ops::*;
use alumina::ops::spline_activation::Spline;

use alumina::graph::*;	

const CHANNELS: usize = 3;

pub fn sr_net(factor: usize, training: Option<(f32, bool, f32, f32)>) -> Graph {
	
	let mut g = Graph::new();
	
	let mut conv_nodes = vec![];
	let mut conv_broadcast_nodes = vec![];
	let mut active_nodes = vec![];
	let mut active_avg_nodes = vec![];
	let mut ops: Vec<Box<Operation>> = vec![];

	if training.is_some() {
		active_nodes.push(g.add_node(Node::new_shaped(CHANNELS, 2, "input")))
	} else {
		active_nodes.push(g.add_input_node(Node::new_shaped(CHANNELS, 2, "input")))
	};
	let output = g.add_output_node(Node::new_shaped(CHANNELS, 2, "output"));

	active_avg_nodes.push(g.add_node(Node::new_flat(CHANNELS, "input_avg")));
	ops.push(GlobalAvg::new(&active_nodes[0], &active_avg_nodes[0], "input_avg"));

	let hidden_layers = 13usize;//29usize; // 2^x-1 if the prediction layer should connect directly to the input
	let hidden_layer_channels = 32;

	
	for i in 0..hidden_layers+1{

		let (new_conv_node, new_conv_broadcast_node, init_weight) = if i < hidden_layers {
			(g.add_node(Node::new_shaped(hidden_layer_channels, 2, &format!("conv{}", i))),
			g.add_node(Node::new_flat(hidden_layer_channels, &format!("conv_broadcast{}", i))),
			1.0)
		} else {
			(g.add_node(Node::new_shaped(CHANNELS*factor*factor, 2, "expand")),
			g.add_node(Node::new_flat(CHANNELS*factor*factor, "expand_broadcast")),
			0.01)
		};
		
		//let jumps = (0..).map(|x| 2usize.pow(x)).take_while(|&jump| jump <= active_nodes.len()).collect::<Vec<_>>();
		let jumps = (0..).map(|x| 2usize.pow(x+1)-1).take_while(|&jump| jump <= active_nodes.len()).collect::<Vec<_>>();


		// connect each layer (hidden and output) to previous layers which are a power of 2 from it.
		for jump in jumps.iter(){
			ops.push(Convolution::new(&active_nodes[active_nodes.len() - jump], &new_conv_node, &[3, 3], Padding::Same, &format!("conv{}", i), Convolution::init_msra(init_weight/jumps.len() as f32)));
			ops.push(LinearMap::new(&active_avg_nodes[active_avg_nodes.len() - jump], &new_conv_broadcast_node, &format!("linearmap{}", i), LinearMap::init_msra(init_weight/jumps.len() as f32)));
		}
		ops.push(Broadcast::new(&new_conv_broadcast_node, &new_conv_node, &format!("broadcast{}", i)));
		ops.push(Bias::new(&new_conv_node, ParamSharing::Spatial, &format!("conv_bias{}", i), init_fill(0.0)));

		
		// add activation only for hidden layers
		if i < hidden_layers{
			let new_active_node = g.add_node(Node::new_shaped(hidden_layer_channels, 2, &format!("activ{}", i)));
			let new_active_avg_node = g.add_node(Node::new_flat(hidden_layer_channels, &format!("activ_avg{}", i)));

			//ops.push(activ::Tanh::new(&new_convnode, &new_active_node, "activation"));
			//ops.push(activ::ELU::new(&new_conv_node, &new_active_node, &format!("activ{}", i)));
			//ops.push(activ::BeLU::new(&new_conv_node, &new_active_node, ParamSharing::Spatial, &format!("activation{}", i), activ::BeLU::init_porque_no_los_dos()));
			ops.push(Spline::new(&new_conv_node, &new_active_node, ParamSharing::Spatial, &format!("activation{}", i), Spline::init_swan()));

			ops.push(GlobalAvg::new(&new_active_node, &new_active_avg_node, &format!("average{}", i)));
			active_nodes.push(new_active_node);
			active_avg_nodes.push(new_active_avg_node);
		}

		conv_nodes.push(new_conv_node);
		conv_broadcast_nodes.push(new_conv_broadcast_node);
		
	}
	let input = &active_nodes[0];

	ops.push(Expand::new(&conv_nodes[conv_nodes.len()- 1], &output, &[factor, factor], "expand"));
	ops.push(LinearInterp::new(input, &output, &[factor, factor], "linterp"));

	let op_inds = g.add_operations(ops);



	if let Some((regularisation, srgb_downsample, power, scale)) = training {
		if regularisation != 0.0 {
			for op_id in &op_inds {
				if op_id.num_params == 0 {continue};
				g.add_secondary_operation(L2Regularisation::new(op_id, regularisation, "L2"), op_id);
			}
		}	

		//let _dummy_training_node = g.add_training_input_node(Node::new_flat(1000, "label")); //imagenet only
		let input_hr = g.add_input_node(Node::new_shaped(CHANNELS, 2, "input_hr"));


		if srgb_downsample {
			g.add_operation(Pooling::new(&input_hr, input, &[factor, factor], "input_pooling"));
		} else {
			let input_hr_lin = g.add_node(Node::new_shaped(CHANNELS, 2, "input_hr_lin"));
			g.add_operation(SrgbToLinear::new(&input_hr, &input_hr_lin,"srgb2lin"));
			let input_pool = g.add_node(Node::new_shaped(CHANNELS, 2, "input_pool"));
			g.add_operation(Pooling::new(&input_hr_lin, &input_pool, &[factor, factor], "input_pooling"));
			g.add_operation(LinearToSrgb::new(&input_pool, input, "lin2srgb"));
		}


		g.add_operation(GeneralLoss::new(&output, &input_hr, 100.0*scale*power, scale, power, "loss"));
		//g.add_operation(GeneralLoss::new(&output, &input_hr, 0.01, 0.01, 1.0, "loss")); // L1 smoothed
		//g.add_operation(GeneralLoss::new(&output, &input_hr, 1.0, 1.0, 2.0, "loss")); // L2

		g.add_operation(ShapeConstraint::new(&input_hr, &output, &[Arc::new(|d| d), Arc::new(|d| d)], "output_shape"));

	} else {
		g.add_operation(ShapeConstraint::new(input, &output, &[Arc::new(move|d| d*factor), Arc::new(move|d| d*factor)], "output_shape"));	
	}

	g
}

pub fn bilinear_net(factor: usize) -> Graph{
	let mut g = Graph::new();
	let inp = g.add_input_node(Node::new_shaped(CHANNELS, 2, "input"));
	let lin = g.add_node(Node::new_shaped(CHANNELS, 2, "linear"));
	let upscale = g.add_node(Node::new_shaped(CHANNELS, 2, "upscale"));
	let out = g.add_output_node(Node::new_shaped(CHANNELS, 2, "output"));
	g.add_operation(SrgbToLinear::new(&inp, &lin, "lin"));
	g.add_operation(LinearInterp::new(&lin, &upscale, &[factor, factor], "linterp"));
	g.add_operation(LinearToSrgb::new(&upscale, &out, "srgb"));
	g.add_operation(ShapeConstraint::new(&inp, &upscale, &[Arc::new(move |d| d*factor), Arc::new(move |d| d*factor)], "triple"));

	g
}

pub fn downsample_lin_net(factor: usize) -> Graph{
	let mut g = Graph::new();

	let input_hr = g.add_input_node(Node::new_shaped(CHANNELS, 2, "input_hr"));
	let input_hr_lin = g.add_node(Node::new_shaped(CHANNELS, 2, "input_hr_lin"));
	let input_pool = g.add_node(Node::new_shaped(CHANNELS, 2, "input_pool"));
	let output = g.add_output_node(Node::new_shaped(CHANNELS, 2, "output"));

	g.add_operation(SrgbToLinear::new(&input_hr, &input_hr_lin,"srgb2lin"));
	g.add_operation(Pooling::new(&input_hr_lin, &input_pool, &[factor, factor], "input_pooling"));
	g.add_operation(LinearToSrgb::new(&input_pool, &output, "lin2srgb"));

	g
}

pub fn downsample_srgb_net(factor: usize) -> Graph{
	let mut g = Graph::new();

	let input_hr = g.add_input_node(Node::new_shaped(CHANNELS, 2, "input_hr"));
	let output = g.add_output_node(Node::new_shaped(CHANNELS, 2, "output"));
	
	g.add_operation(Pooling::new(&input_hr, &output, &[factor, factor], "input_pooling"));

	g
}
