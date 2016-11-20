


use std::path::{PathBuf, Path};
use std::io::{self, Write};


use image::{GenericImage, DynamicImage, Pixel};
use image;


use rand::*;
use alumina::graph::*;
use alumina::opt::*;
use alumina::shape::*;

pub const CHANNELS: usize = 3;

pub struct ImageFolderSupplier{
	count: u64,
	epoch: usize,
	paths: Vec<PathBuf>,
	order: ShuffleRandomiser,
	crop: Option<(u32,u32, Cropping)>
}

impl Supplier for ImageFolderSupplier{



	fn next_n(&mut self, n: usize) -> (Vec<NodeData>, Vec<NodeData>){
		
		if let Some(crop) = self.crop {
			let (width, height, _) = crop;
			let mut data = NodeData::new_blank(DataShape::new(CHANNELS, vec![width as usize, height as usize], n));
			let data_len = data.values.len();


			for data in data.values.chunks_mut(data_len/n){
				load_crop(&mut self.order, &self.paths, data, &crop);
			}

			self.count += n as u64;
			(vec![data], vec![])

		} else {
			assert_eq!(n, 1, "If cropping isnt specified images must be loaded one at a time. Specify cropping for this supplier, or restrict evaluation batching to 1");

			let data = load_full(&mut self.order, &self.paths);
			self.count += n as u64;

			(vec![data], vec![])
		}

	}
	
	fn epoch_size(&self) -> usize{
		self.epoch
	}

	fn samples_taken(&self) -> u64{
		self.count
	}

	fn reset(&mut self){
		self.order.reset();
		self.count = 0;
	}

	fn once(self) -> Vec<(Vec<NodeData>, Vec<NodeData>)> {
		unimplemented!();
	}
}

impl ImageFolderSupplier{
		
	pub fn new(folder_path: &Path, crop: Option<(u32,u32, Cropping)>) -> ImageFolderSupplier {
		

		print!("Loading paths for {:?} ... ", folder_path);
		io::stdout().flush().ok();
		let dir = folder_path.read_dir().expect(&format!("Could not read folder: {:?}", folder_path));
		let paths = dir.filter_map(|e| e.ok().and_then(|e| if e.path().is_file() {Some(e.path())} else {None}))
			.collect::<Vec<_>>();
		println!("loaded {} paths.", paths.len());


		let n = paths.len();
		ImageFolderSupplier{
			count: 0,
			epoch: n,
			paths: paths,
			order: ShuffleRandomiser::new(n),
			crop: crop,
		}
	}
}





fn load_crop(order: &mut ShuffleRandomiser, paths: &[PathBuf], data: &mut [f32], &(width, height, cropping): &(u32,u32, Cropping) ){

	let mut iter = 0;
	let mut result = None;
	let mut last_path = paths[0].as_path();

	while iter < 100 && result.is_none() {
		let path_index = order.next();
		last_path = paths[path_index].as_path();
		result = image::open(last_path).ok();
		iter += 1;
	}

	let image = result.expect(&format!("100 consecutive attempts at opening images failed. Last path was: {:?}", last_path));

	let (out_width, img_x_off, data_x_off) = range(&cropping, image.dimensions().0, width);
	let (out_height, img_y_off, data_y_off) = range(&cropping, image.dimensions().1, height);

	
	for y in 0..out_height {
		for x in 0..out_width {
			let pixel = image.get_pixel(x + img_x_off, y + img_y_off);
			let channels = pixel.channels();
			let data = &mut data[(x + data_x_off + (y+data_y_off)*width) as usize*CHANNELS..][..CHANNELS];
			for i in 0..CHANNELS {
				data[i] = channels[i] as f32/255.0;
			}
		}
	}
}

pub fn data_to_img(image_node: NodeData) -> DynamicImage {
		assert_eq!(image_node.shape.channels, CHANNELS);
		let data = &image_node.values;
		let width = image_node.shape.spatial_dimensions[0] as u32;
		let height = image_node.shape.spatial_dimensions[1] as u32;

		// let mut imgbuf = image::ImageBuffer::new(width, height);

		let mut img = DynamicImage::new_rgba8(width, height);

		for y in 0..height {
			for x in 0..width {
				let data = &data[(x + y*width) as usize*CHANNELS..][..CHANNELS];
				img.put_pixel(x, y, image::Rgba::from_channels(f32_channel_to_u8(data[0]), f32_channel_to_u8(data[1]), f32_channel_to_u8(data[2]), 255u8));	
			}
		}
		// for (x, y, pixel) in imgbuf.enumerate_pixels_mut() {
		// 		let data = &mut data[(x + y*width) as usize*CHANNELS..][..CHANNELS];
		// 		*pixel = image::Rgba::from_channels((data[0]*255.0).min(255.0) as u8, (data[1]*255.0).min(255.0) as u8, (data[2]*255.0).min(255.0) as u8, 0u8);		
		// }
		img

}

fn f32_channel_to_u8 (val: f32) -> u8{
	(val*255.0).min(255.0).max(0.0).round() as u8
}

pub fn img_to_data(data: &mut[f32], image: &DynamicImage){
		let width = image.dimensions().0;
		let height = image.dimensions().1;
		for y in 0..height {
			for x in 0..width {
				let pixel = image.get_pixel(x, y);
				let channels = pixel.channels();
				let data = &mut data[(x + y*width) as usize*CHANNELS..][..CHANNELS];
				for i in 0..CHANNELS {
					data[i] = channels[i] as f32/255.0;
				}
			}
		}
}

fn open_randomised_img(order: &mut ShuffleRandomiser, paths: &[PathBuf]) -> DynamicImage{
	let mut iter = 0;
	let mut result = None;
	let mut last_path = paths[0].as_path();

	while iter < 100 && result.is_none() {
		let path_index = order.next();
		last_path = paths[path_index].as_path();
		result = image::open(last_path).ok();
		iter += 1;
	}

	let image = result.expect(&format!("100 consecutive attempts at opening images failed. Last path was: {:?}", last_path));
	image
}

fn load_full(order: &mut ShuffleRandomiser, paths: &[PathBuf]) -> NodeData{
	
	let image = open_randomised_img(order, paths);
	let mut node_data = NodeData::new_blank(DataShape::new(CHANNELS, vec![image.dimensions().0 as usize, image.dimensions().1 as usize], 1));

	img_to_data(&mut node_data.values, &image);

	node_data
}



// returns iterated width, img_x_off, and data_x_off
fn range(cropping: &Cropping, image_x: u32, data_x: u32) -> (u32, u32, u32) {
	match cropping {
		&Cropping::Centre => {
			if image_x < data_x {
				(image_x, 0, (data_x - image_x)/2)
			} else {
				(data_x, (image_x - data_x)/2, 0)
			}
		},

		&Cropping::Random => {
			if image_x < data_x {
				(image_x, 0, thread_rng().gen_range(0, data_x - image_x + 1))
			} else {
				(data_x, thread_rng().gen_range(0, image_x - data_x + 1), 0)
			}
		},
	}
}

#[derive(Copy, Clone)]
pub enum Cropping {
	Centre,
	Random,
}


// pub struct ImagenetSupplier{
// 	count: u64,
// 	epoch: usize,
// 	paths: Vec<Vec<PathBuf>>,
// 	orders: Vec<ShuffleRandomiser>,
// 	folder_order: ShuffleRandomiser,
// 	crop: Option<(u32,u32, Cropping)>
// }

// impl Supplier for ImagenetSupplier{

	
// 	fn next_n(&mut self, n: usize) -> (Vec<NodeData>, Vec<NodeData>){
		
// 		if let Some(crop) = self.crop {
// 			let (width, height, _) = crop;
// 			let mut data = NodeData::new_blank(DataShape::new(CHANNELS, vec![width as usize, height as usize], n));
// 			let mut labels = NodeData::new_blank(NodeShape::new_flat(self.paths.len()).to_data_shape(n).unwrap());
// 			let data_len = data.values.len();
// 			let labels_len = labels.values.len();

// 			for (data, label) in data.values.chunks_mut(data_len/n).zip(labels.values.chunks_mut(labels_len/n)){
// 				let label_index = self.folder_order.next();
// 				label[label_index] = 1.0;

// 				load_crop(&mut self.orders[label_index], &self.paths[label_index], data, &crop);
// 			}

// 			self.count += n as u64;
// 			(vec![data], vec![labels])
// 		} else {
// 			assert_eq!(n, 1, "If cropping isnt specified images but be loaded one at a time. Specifiy cropping for this supplier, or restrict evaluation batching to 1");

// 			let mut labels = NodeData::new_blank(NodeShape::new_flat(self.paths.len()).to_data_shape(n).unwrap());
// 			let label_index = self.folder_order.next();
// 			labels.values[label_index] = 1.0;

// 			let data = load_full(&mut self.orders[label_index], &self.paths[label_index]);
// 			self.count += n as u64;

// 			(vec![data], vec![labels])
// 		}

// 	}
	
// 	fn epoch_size(&self) -> usize{
// 		self.epoch
// 	}

// 	fn samples_taken(&self) -> u64{
// 		self.count
// 	}

// 	fn reset(&mut self){
// 		for order in self.orders.iter_mut() {
// 			order.reset();
// 		}
// 		self.count = 0;
// 	}

// 	fn once(self) -> Vec<(Vec<NodeData>, Vec<NodeData>)> {
// 		unimplemented!();
// 	}
// }

// impl ImagenetSupplier{
		
// 	pub fn new(path: &Path, width: u32, height: u32, cropping: Cropping) -> ImagenetSupplier {
		
// 		let paths = FOLDERS.iter().map(|&(folder, names)| {
// 			print!("Loading paths for {}({}) ... ", folder, names);
// 			io::stdout().flush().ok();
// 			let folder_path = path.join(folder);
// 			let dir = folder_path.read_dir().expect(&format!("Could not read folder {:?} : {}", folder_path, names));
// 			let vec = dir.filter_map(
// 				|e|e.ok().and_then(|e| if e.path().is_file() {Some(e.path())} else {None}))
// 			.collect::<Vec<_>>();
// 			println!("loaded {} paths.", vec.len());
// 			vec
// 		}).collect::<Vec<_>>();

// 		let orders = paths.iter().map(|vec| ShuffleRandomiser::new(vec.len())).collect();
// 		let n = paths.len();
// 		ImagenetSupplier{
// 			count: 0,
// 			epoch: paths.iter().fold(0, |acc, vec| acc + vec.len()),
// 			paths: paths,
// 			orders: orders,
// 			folder_order: ShuffleRandomiser::new(n),
// 			crop: Some((width, height, cropping)),
// 		}
// 	}
// }



// const FOLDERS: [(&'static str, &'static str); 1000] = [
// 	("n01440764", "tench, Tincatinca"),
// 	("n01443537", "goldfish, Carassiusauratus"),
// 	("n01484850", "greatwhiteshark, whiteshark, man-eater, man-eatingshark, Carcharodoncarcharias"),
// 	("n01491361", "tigershark, Galeocerdocuvieri"),
// 	("n01494475", "hammerhead, hammerheadshark"),
// 	("n01496331", "electricray, crampfish, numbfish, torpedo"),
// 	("n01498041", "stingray"),
// 	("n01514668", "cock"),
// 	("n01514859", "hen"),
// 	("n01518878", "ostrich, Struthiocamelus"),
// 	("n01530575", "brambling, Fringillamontifringilla"),
// 	("n01531178", "goldfinch, Cardueliscarduelis"),
// 	("n01532829", "housefinch, linnet, Carpodacusmexicanus"),
// 	("n01534433", "junco, snowbird"),
// 	("n01537544", "indigobunting, indigofinch, indigobird, Passerinacyanea"),
// 	("n01558993", "robin, Americanrobin, Turdusmigratorius"),
// 	("n01560419", "bulbul"),
// 	("n01580077", "jay"),
// 	("n01582220", "magpie"),
// 	("n01592084", "chickadee"),
// 	("n01601694", "waterouzel, dipper"),
// 	("n01608432", "kite"),
// 	("n01614925", "baldeagle, Americaneagle, Haliaeetusleucocephalus"),
// 	("n01616318", "vulture"),
// 	("n01622779", "greatgreyowl, greatgrayowl, Strixnebulosa"),
// 	("n01629819", "Europeanfiresalamander, Salamandrasalamandra"),
// 	("n01630670", "commonnewt, Triturusvulgaris"),
// 	("n01631663", "eft"),
// 	("n01632458", "spottedsalamander, Ambystomamaculatum"),
// 	("n01632777", "axolotl, mudpuppy, Ambystomamexicanum"),
// 	("n01641577", "bullfrog, Ranacatesbeiana"),
// 	("n01644373", "treefrog, tree-frog"),
// 	("n01644900", "tailedfrog, belltoad, ribbedtoad, tailedtoad, Ascaphustrui"),
// 	("n01664065", "loggerhead, loggerheadturtle, Carettacaretta"),
// 	("n01665541", "leatherbackturtle, leatherback, leatheryturtle, Dermochelyscoriacea"),
// 	("n01667114", "mudturtle"),
// 	("n01667778", "terrapin"),
// 	("n01669191", "boxturtle, boxtortoise"),
// 	("n01675722", "bandedgecko"),
// 	("n01677366", "commoniguana, iguana, Iguanaiguana"),
// 	("n01682714", "Americanchameleon, anole, Anoliscarolinensis"),
// 	("n01685808", "whiptail, whiptaillizard"),
// 	("n01687978", "agama"),
// 	("n01688243", "frilledlizard, Chlamydosauruskingi"),
// 	("n01689811", "alligatorlizard"),
// 	("n01692333", "Gilamonster, Helodermasuspectum"),
// 	("n01693334", "greenlizard, Lacertaviridis"),
// 	("n01694178", "Africanchameleon, Chamaeleochamaeleon"),
// 	("n01695060", "Komododragon, Komodolizard, dragonlizard, giantlizard, Varanuskomodoensis"),
// 	("n01697457", "Africancrocodile, Nilecrocodile, Crocodylusniloticus"),
// 	("n01698640", "Americanalligator, Alligatormississipiensis"),
// 	("n01704323", "triceratops"),
// 	("n01728572", "thundersnake, wormsnake, Carphophisamoenus"),
// 	("n01728920", "ringnecksnake, ring-neckedsnake, ringsnake"),
// 	("n01729322", "hognosesnake, puffadder, sandviper"),
// 	("n01729977", "greensnake, grasssnake"),
// 	("n01734418", "kingsnake, kingsnake"),
// 	("n01735189", "gartersnake, grasssnake"),
// 	("n01737021", "watersnake"),
// 	("n01739381", "vinesnake"),
// 	("n01740131", "nightsnake, Hypsiglenatorquata"),
// 	("n01742172", "boaconstrictor, Constrictorconstrictor"),
// 	("n01744401", "rockpython, rocksnake, Pythonsebae"),
// 	("n01748264", "Indiancobra, Najanaja"),
// 	("n01749939", "greenmamba"),
// 	("n01751748", "seasnake"),
// 	("n01753488", "hornedviper, cerastes, sandviper, hornedasp, Cerastescornutus"),
// 	("n01755581", "diamondback, diamondbackrattlesnake, Crotalusadamanteus"),
// 	("n01756291", "sidewinder, hornedrattlesnake, Crotaluscerastes"),
// 	("n01768244", "trilobite"),
// 	("n01770081", "harvestman, daddylonglegs, Phalangiumopilio"),
// 	("n01770393", "scorpion"),
// 	("n01773157", "blackandgoldgardenspider, Argiopeaurantia"),
// 	("n01773549", "barnspider, Araneuscavaticus"),
// 	("n01773797", "gardenspider, Araneadiademata"),
// 	("n01774384", "blackwidow, Latrodectusmactans"),
// 	("n01774750", "tarantula"),
// 	("n01775062", "wolfspider, huntingspider"),
// 	("n01776313", "tick"),
// 	("n01784675", "centipede"),
// 	("n01795545", "blackgrouse"),
// 	("n01796340", "ptarmigan"),
// 	("n01797886", "ruffedgrouse, partridge, Bonasaumbellus"),
// 	("n01798484", "prairiechicken, prairiegrouse, prairiefowl"),
// 	("n01806143", "peacock"),
// 	("n01806567", "quail"),
// 	("n01807496", "partridge"),
// 	("n01817953", "Africangrey, Africangray, Psittacuserithacus"),
// 	("n01818515", "macaw"),
// 	("n01819313", "sulphur-crestedcockatoo, Kakatoegalerita, Cacatuagalerita"),
// 	("n01820546", "lorikeet"),
// 	("n01824575", "coucal"),
// 	("n01828970", "beeeater"),
// 	("n01829413", "hornbill"),
// 	("n01833805", "hummingbird"),
// 	("n01843065", "jacamar"),
// 	("n01843383", "toucan"),
// 	("n01847000", "drake"),
// 	("n01855032", "red-breastedmerganser, Mergusserrator"),
// 	("n01855672", "goose"),
// 	("n01860187", "blackswan, Cygnusatratus"),
// 	("n01871265", "tusker"),
// 	("n01872401", "echidna, spinyanteater, anteater"),
// 	("n01873310", "platypus, duckbill, duckbilledplatypus, duck-billedplatypus, Ornithorhynchusanatinus"),
// 	("n01877812", "wallaby, brushkangaroo"),
// 	("n01882714", "koala, koalabear, kangaroobear, nativebear, Phascolarctoscinereus"),
// 	("n01883070", "wombat"),
// 	("n01910747", "jellyfish"),
// 	("n01914609", "seaanemone, anemone"),
// 	("n01917289", "braincoral"),
// 	("n01924916", "flatworm, platyhelminth"),
// 	("n01930112", "nematode, nematodeworm, roundworm"),
// 	("n01943899", "conch"),
// 	("n01944390", "snail"),
// 	("n01945685", "slug"),
// 	("n01950731", "seaslug, nudibranch"),
// 	("n01955084", "chiton, coat-of-mailshell, seacradle, polyplacophore"),
// 	("n01968897", "chamberednautilus, pearlynautilus, nautilus"),
// 	("n01978287", "Dungenesscrab, Cancermagister"),
// 	("n01978455", "rockcrab, Cancerirroratus"),
// 	("n01980166", "fiddlercrab"),
// 	("n01981276", "kingcrab, Alaskacrab, Alaskankingcrab, Alaskakingcrab, Paralithodescamtschatica"),
// 	("n01983481", "Americanlobster, Northernlobster, Mainelobster, Homarusamericanus"),
// 	("n01984695", "spinylobster, langouste, rocklobster, crawfish, crayfish, seacrawfish"),
// 	("n01985128", "crayfish, crawfish, crawdad, crawdaddy"),
// 	("n01986214", "hermitcrab"),
// 	("n01990800", "isopod"),
// 	("n02002556", "whitestork, Ciconiaciconia"),
// 	("n02002724", "blackstork, Ciconianigra"),
// 	("n02006656", "spoonbill"),
// 	("n02007558", "flamingo"),
// 	("n02009229", "littleblueheron, Egrettacaerulea"),
// 	("n02009912", "Americanegret, greatwhiteheron, Egrettaalbus"),
// 	("n02011460", "bittern"),
// 	("n02012849", "crane"),
// 	("n02013706", "limpkin, Aramuspictus"),
// 	("n02017213", "Europeangallinule, Porphyrioporphyrio"),
// 	("n02018207", "Americancoot, marshhen, mudhen, waterhen, Fulicaamericana"),
// 	("n02018795", "bustard"),
// 	("n02025239", "ruddyturnstone, Arenariainterpres"),
// 	("n02027492", "red-backedsandpiper, dunlin, Eroliaalpina"),
// 	("n02028035", "redshank, Tringatotanus"),
// 	("n02033041", "dowitcher"),
// 	("n02037110", "oystercatcher, oystercatcher"),
// 	("n02051845", "pelican"),
// 	("n02056570", "kingpenguin, Aptenodytespatagonica"),
// 	("n02058221", "albatross, mollymawk"),
// 	("n02066245", "greywhale, graywhale, devilfish, Eschrichtiusgibbosus, Eschrichtiusrobustus"),
// 	("n02071294", "killerwhale, killer, orca, grampus, seawolf, Orcinusorca"),
// 	("n02074367", "dugong, Dugongdugon"),
// 	("n02077923", "sealion"),
// 	("n02085620", "Chihuahua"),
// 	("n02085782", "Japanesespaniel"),
// 	("n02085936", "Maltesedog, Malteseterrier, Maltese"),
// 	("n02086079", "Pekinese, Pekingese, Peke"),
// 	("n02086240", "Shih-Tzu"),
// 	("n02086646", "Blenheimspaniel"),
// 	("n02086910", "papillon"),
// 	("n02087046", "toyterrier"),
// 	("n02087394", "Rhodesianridgeback"),
// 	("n02088094", "Afghanhound, Afghan"),
// 	("n02088238", "basset, bassethound"),
// 	("n02088364", "beagle"),
// 	("n02088466", "bloodhound, sleuthhound"),
// 	("n02088632", "bluetick"),
// 	("n02089078", "black-and-tancoonhound"),
// 	("n02089867", "Walkerhound, Walkerfoxhound"),
// 	("n02089973", "Englishfoxhound"),
// 	("n02090379", "redbone"),
// 	("n02090622", "borzoi, Russianwolfhound"),
// 	("n02090721", "Irishwolfhound"),
// 	("n02091032", "Italiangreyhound"),
// 	("n02091134", "whippet"),
// 	("n02091244", "Ibizanhound, IbizanPodenco"),
// 	("n02091467", "Norwegianelkhound, elkhound"),
// 	("n02091635", "otterhound, otterhound"),
// 	("n02091831", "Saluki, gazellehound"),
// 	("n02092002", "Scottishdeerhound, deerhound"),
// 	("n02092339", "Weimaraner"),
// 	("n02093256", "Staffordshirebullterrier, Staffordshirebullterrier"),
// 	("n02093428", "AmericanStaffordshireterrier, Staffordshireterrier, Americanpitbullterrier, pitbullterrier"),
// 	("n02093647", "Bedlingtonterrier"),
// 	("n02093754", "Borderterrier"),
// 	("n02093859", "Kerryblueterrier"),
// 	("n02093991", "Irishterrier"),
// 	("n02094114", "Norfolkterrier"),
// 	("n02094258", "Norwichterrier"),
// 	("n02094433", "Yorkshireterrier"),
// 	("n02095314", "wire-hairedfoxterrier"),
// 	("n02095570", "Lakelandterrier"),
// 	("n02095889", "Sealyhamterrier, Sealyham"),
// 	("n02096051", "Airedale, Airedaleterrier"),
// 	("n02096177", "cairn, cairnterrier"),
// 	("n02096294", "Australianterrier"),
// 	("n02096437", "DandieDinmont, DandieDinmontterrier"),
// 	("n02096585", "Bostonbull, Bostonterrier"),
// 	("n02097047", "miniatureschnauzer"),
// 	("n02097130", "giantschnauzer"),
// 	("n02097209", "standardschnauzer"),
// 	("n02097298", "Scotchterrier, Scottishterrier, Scottie"),
// 	("n02097474", "Tibetanterrier, chrysanthemumdog"),
// 	("n02097658", "silkyterrier, Sydneysilky"),
// 	("n02098105", "soft-coatedwheatenterrier"),
// 	("n02098286", "WestHighlandwhiteterrier"),
// 	("n02098413", "Lhasa, Lhasaapso"),
// 	("n02099267", "flat-coatedretriever"),
// 	("n02099429", "curly-coatedretriever"),
// 	("n02099601", "goldenretriever"),
// 	("n02099712", "Labradorretriever"),
// 	("n02099849", "ChesapeakeBayretriever"),
// 	("n02100236", "Germanshort-hairedpointer"),
// 	("n02100583", "vizsla, Hungarianpointer"),
// 	("n02100735", "Englishsetter"),
// 	("n02100877", "Irishsetter, redsetter"),
// 	("n02101006", "Gordonsetter"),
// 	("n02101388", "Brittanyspaniel"),
// 	("n02101556", "clumber, clumberspaniel"),
// 	("n02102040", "Englishspringer, Englishspringerspaniel"),
// 	("n02102177", "Welshspringerspaniel"),
// 	("n02102318", "cockerspaniel, Englishcockerspaniel, cocker"),
// 	("n02102480", "Sussexspaniel"),
// 	("n02102973", "Irishwaterspaniel"),
// 	("n02104029", "kuvasz"),
// 	("n02104365", "schipperke"),
// 	("n02105056", "groenendael"),
// 	("n02105162", "malinois"),
// 	("n02105251", "briard"),
// 	("n02105412", "kelpie"),
// 	("n02105505", "komondor"),
// 	("n02105641", "OldEnglishsheepdog, bobtail"),
// 	("n02105855", "Shetlandsheepdog, Shetlandsheepdog, Shetland"),
// 	("n02106030", "collie"),
// 	("n02106166", "Bordercollie"),
// 	("n02106382", "BouvierdesFlandres, BouviersdesFlandres"),
// 	("n02106550", "Rottweiler"),
// 	("n02106662", "Germanshepherd, Germanshepherddog, Germanpolicedog, alsatian"),
// 	("n02107142", "Doberman, Dobermanpinscher"),
// 	("n02107312", "miniaturepinscher"),
// 	("n02107574", "GreaterSwissMountaindog"),
// 	("n02107683", "Bernesemountaindog"),
// 	("n02107908", "Appenzeller"),
// 	("n02108000", "EntleBucher"),
// 	("n02108089", "boxer"),
// 	("n02108422", "bullmastiff"),
// 	("n02108551", "Tibetanmastiff"),
// 	("n02108915", "Frenchbulldog"),
// 	("n02109047", "GreatDane"),
// 	("n02109525", "SaintBernard, StBernard"),
// 	("n02109961", "Eskimodog, husky"),
// 	("n02110063", "malamute, malemute, Alaskanmalamute"),
// 	("n02110185", "Siberianhusky"),
// 	("n02110341", "dalmatian, coachdog, carriagedog"),
// 	("n02110627", "affenpinscher, monkeypinscher, monkeydog"),
// 	("n02110806", "basenji"),
// 	("n02110958", "pug, pug-dog"),
// 	("n02111129", "Leonberg"),
// 	("n02111277", "Newfoundland, Newfoundlanddog"),
// 	("n02111500", "GreatPyrenees"),
// 	("n02111889", "Samoyed, Samoyede"),
// 	("n02112018", "Pomeranian"),
// 	("n02112137", "chow, chowchow"),
// 	("n02112350", "keeshond"),
// 	("n02112706", "Brabancongriffon"),
// 	("n02113023", "Pembroke, PembrokeWelshcorgi"),
// 	("n02113186", "Cardigan, CardiganWelshcorgi"),
// 	("n02113624", "toypoodle"),
// 	("n02113712", "miniaturepoodle"),
// 	("n02113799", "standardpoodle"),
// 	("n02113978", "Mexicanhairless"),
// 	("n02114367", "timberwolf, greywolf, graywolf, Canislupus"),
// 	("n02114548", "whitewolf, Arcticwolf, Canislupustundrarum"),
// 	("n02114712", "redwolf, manedwolf, Canisrufus, Canisniger"),
// 	("n02114855", "coyote, prairiewolf, brushwolf, Canislatrans"),
// 	("n02115641", "dingo, warrigal, warragal, Canisdingo"),
// 	("n02115913", "dhole, Cuonalpinus"),
// 	("n02116738", "Africanhuntingdog, hyenadog, Capehuntingdog, Lycaonpictus"),
// 	("n02117135", "hyena, hyaena"),
// 	("n02119022", "redfox, Vulpesvulpes"),
// 	("n02119789", "kitfox, Vulpesmacrotis"),
// 	("n02120079", "Arcticfox, whitefox, Alopexlagopus"),
// 	("n02120505", "greyfox, grayfox, Urocyoncinereoargenteus"),
// 	("n02123045", "tabby, tabbycat"),
// 	("n02123159", "tigercat"),
// 	("n02123394", "Persiancat"),
// 	("n02123597", "Siamesecat, Siamese"),
// 	("n02124075", "Egyptiancat"),
// 	("n02125311", "cougar, puma, catamount, mountainlion, painter, panther, Felisconcolor"),
// 	("n02127052", "lynx, catamount"),
// 	("n02128385", "leopard, Pantherapardus"),
// 	("n02128757", "snowleopard, ounce, Pantherauncia"),
// 	("n02128925", "jaguar, panther, Pantheraonca, Felisonca"),
// 	("n02129165", "lion, kingofbeasts, Pantheraleo"),
// 	("n02129604", "tiger, Pantheratigris"),
// 	("n02130308", "cheetah, chetah, Acinonyxjubatus"),
// 	("n02132136", "brownbear, bruin, Ursusarctos"),
// 	("n02133161", "Americanblackbear, blackbear, Ursusamericanus, Euarctosamericanus"),
// 	("n02134084", "icebear, polarbear, UrsusMaritimus, Thalarctosmaritimus"),
// 	("n02134418", "slothbear, Melursusursinus, Ursusursinus"),
// 	("n02137549", "mongoose"),
// 	("n02138441", "meerkat, mierkat"),
// 	("n02165105", "tigerbeetle"),
// 	("n02165456", "ladybug, ladybeetle, ladybeetle, ladybird, ladybirdbeetle"),
// 	("n02167151", "groundbeetle, carabidbeetle"),
// 	("n02168699", "long-hornedbeetle, longicorn, longicornbeetle"),
// 	("n02169497", "leafbeetle, chrysomelid"),
// 	("n02172182", "dungbeetle"),
// 	("n02174001", "rhinocerosbeetle"),
// 	("n02177972", "weevil"),
// 	("n02190166", "fly"),
// 	("n02206856", "bee"),
// 	("n02219486", "ant, emmet, pismire"),
// 	("n02226429", "grasshopper, hopper"),
// 	("n02229544", "cricket"),
// 	("n02231487", "walkingstick, walkingstick, stickinsect"),
// 	("n02233338", "cockroach, roach"),
// 	("n02236044", "mantis, mantid"),
// 	("n02256656", "cicada, cicala"),
// 	("n02259212", "leafhopper"),
// 	("n02264363", "lacewing, lacewingfly"),
// 	("n02268443", "dragonfly, darningneedle, devilsdarningneedle, sewingneedle, snakefeeder, snakedoctor, mosquitohawk, skeeterhawk"),
// 	("n02268853", "damselfly"),
// 	("n02276258", "admiral"),
// 	("n02277742", "ringlet, ringletbutterfly"),
// 	("n02279972", "monarch, monarchbutterfly, milkweedbutterfly, Danausplexippus"),
// 	("n02280649", "cabbagebutterfly"),
// 	("n02281406", "sulphurbutterfly, sulfurbutterfly"),
// 	("n02281787", "lycaenid, lycaenidbutterfly"),
// 	("n02317335", "starfish, seastar"),
// 	("n02319095", "seaurchin"),
// 	("n02321529", "seacucumber, holothurian"),
// 	("n02325366", "woodrabbit, cottontail, cottontailrabbit"),
// 	("n02326432", "hare"),
// 	("n02328150", "Angora, Angorarabbit"),
// 	("n02342885", "hamster"),
// 	("n02346627", "porcupine, hedgehog"),
// 	("n02356798", "foxsquirrel, easternfoxsquirrel, Sciurusniger"),
// 	("n02361337", "marmot"),
// 	("n02363005", "beaver"),
// 	("n02364673", "guineapig, Caviacobaya"),
// 	("n02389026", "sorrel"),
// 	("n02391049", "zebra"),
// 	("n02395406", "hog, pig, grunter, squealer, Susscrofa"),
// 	("n02396427", "wildboar, boar, Susscrofa"),
// 	("n02397096", "warthog"),
// 	("n02398521", "hippopotamus, hippo, riverhorse, Hippopotamusamphibius"),
// 	("n02403003", "ox"),
// 	("n02408429", "waterbuffalo, waterox, Asiaticbuffalo, Bubalusbubalis"),
// 	("n02410509", "bison"),
// 	("n02412080", "ram, tup"),
// 	("n02415577", "bighorn, bighornsheep, cimarron, RockyMountainbighorn, RockyMountainsheep, Oviscanadensis"),
// 	("n02417914", "ibex, Capraibex"),
// 	("n02422106", "hartebeest"),
// 	("n02422699", "impala, Aepycerosmelampus"),
// 	("n02423022", "gazelle"),
// 	("n02437312", "Arabiancamel, dromedary, Camelusdromedarius"),
// 	("n02437616", "llama"),
// 	("n02441942", "weasel"),
// 	("n02442845", "mink"),
// 	("n02443114", "polecat, fitch, foulmart, foumart, Mustelaputorius"),
// 	("n02443484", "black-footedferret, ferret, Mustelanigripes"),
// 	("n02444819", "otter"),
// 	("n02445715", "skunk, polecat, woodpussy"),
// 	("n02447366", "badger"),
// 	("n02454379", "armadillo"),
// 	("n02457408", "three-toedsloth, ai, Bradypustridactylus"),
// 	("n02480495", "orangutan, orang, orangutang, Pongopygmaeus"),
// 	("n02480855", "gorilla, Gorillagorilla"),
// 	("n02481823", "chimpanzee, chimp, Pantroglodytes"),
// 	("n02483362", "gibbon, Hylobateslar"),
// 	("n02483708", "siamang, Hylobatessyndactylus, Symphalangussyndactylus"),
// 	("n02484975", "guenon, guenonmonkey"),
// 	("n02486261", "patas, hussarmonkey, Erythrocebuspatas"),
// 	("n02486410", "baboon"),
// 	("n02487347", "macaque"),
// 	("n02488291", "langur"),
// 	("n02488702", "colobus, colobusmonkey"),
// 	("n02489166", "proboscismonkey, Nasalislarvatus"),
// 	("n02490219", "marmoset"),
// 	("n02492035", "capuchin, ringtail, Cebuscapucinus"),
// 	("n02492660", "howlermonkey, howler"),
// 	("n02493509", "titi, titimonkey"),
// 	("n02493793", "spidermonkey, Atelesgeoffroyi"),
// 	("n02494079", "squirrelmonkey, Saimirisciureus"),
// 	("n02497673", "Madagascarcat, ring-tailedlemur, Lemurcatta"),
// 	("n02500267", "indri, indris, Indriindri, Indribrevicaudatus"),
// 	("n02504013", "Indianelephant, Elephasmaximus"),
// 	("n02504458", "Africanelephant, Loxodontaafricana"),
// 	("n02509815", "lesserpanda, redpanda, panda, bearcat, catbear, Ailurusfulgens"),
// 	("n02510455", "giantpanda, panda, pandabear, coonbear, Ailuropodamelanoleuca"),
// 	("n02514041", "barracouta, snoek"),
// 	("n02526121", "eel"),
// 	("n02536864", "coho, cohoe, cohosalmon, bluejack, silversalmon, Oncorhynchuskisutch"),
// 	("n02606052", "rockbeauty, Holocanthustricolor"),
// 	("n02607072", "anemonefish"),
// 	("n02640242", "sturgeon"),
// 	("n02641379", "gar, garfish, garpike, billfish, Lepisosteusosseus"),
// 	("n02643566", "lionfish"),
// 	("n02655020", "puffer, pufferfish, blowfish, globefish"),
// 	("n02666196", "abacus"),
// 	("n02667093", "abaya"),
// 	("n02669723", "academicgown, academicrobe, judgesrobe"),
// 	("n02672831", "accordion, pianoaccordion, squeezebox"),
// 	("n02676566", "acousticguitar"),
// 	("n02687172", "aircraftcarrier, carrier, flattop, attackaircraftcarrier"),
// 	("n02690373", "airliner"),
// 	("n02692877", "airship, dirigible"),
// 	("n02699494", "altar"),
// 	("n02701002", "ambulance"),
// 	("n02704792", "amphibian, amphibiousvehicle"),
// 	("n02708093", "analogclock"),
// 	("n02727426", "apiary, beehouse"),
// 	("n02730930", "apron"),
// 	("n02747177", "ashcan, trashcan, garbagecan, wastebin, ashbin, ash-bin, ashbin, dustbin, trashbarrel, trashbin"),
// 	("n02749479", "assaultrifle, assaultgun"),
// 	("n02769748", "backpack, backpack, knapsack, packsack, rucksack, haversack"),
// 	("n02776631", "bakery, bakeshop, bakehouse"),
// 	("n02777292", "balancebeam, beam"),
// 	("n02782093", "balloon"),
// 	("n02783161", "ballpoint, ballpointpen, ballpen, Biro"),
// 	("n02786058", "BandAid"),
// 	("n02787622", "banjo"),
// 	("n02788148", "bannister, banister, balustrade, balusters, handrail"),
// 	("n02790996", "barbell"),
// 	("n02791124", "barberchair"),
// 	("n02791270", "barbershop"),
// 	("n02793495", "barn"),
// 	("n02794156", "barometer"),
// 	("n02795169", "barrel, cask"),
// 	("n02797295", "barrow, gardencart, lawncart, wheelbarrow"),
// 	("n02799071", "baseball"),
// 	("n02802426", "basketball"),
// 	("n02804414", "bassinet"),
// 	("n02804610", "bassoon"),
// 	("n02807133", "bathingcap, swimmingcap"),
// 	("n02808304", "bathtowel"),
// 	("n02808440", "bathtub, bathingtub, bath, tub"),
// 	("n02814533", "beachwagon, stationwagon, wagon, estatecar, beachwaggon, stationwaggon, waggon"),
// 	("n02814860", "beacon, lighthouse, beaconlight, pharos"),
// 	("n02815834", "beaker"),
// 	("n02817516", "bearskin, busby, shako"),
// 	("n02823428", "beerbottle"),
// 	("n02823750", "beerglass"),
// 	("n02825657", "bellcote, bellcot"),
// 	("n02834397", "bib"),
// 	("n02835271", "bicycle-built-for-two, tandembicycle, tandem"),
// 	("n02837789", "bikini, two-piece"),
// 	("n02840245", "binder, ring-binder"),
// 	("n02841315", "binoculars, fieldglasses, operaglasses"),
// 	("n02843684", "birdhouse"),
// 	("n02859443", "boathouse"),
// 	("n02860847", "bobsled, bobsleigh, bob"),
// 	("n02865351", "bolotie, bolo, bolatie, bola"),
// 	("n02869837", "bonnet, pokebonnet"),
// 	("n02870880", "bookcase"),
// 	("n02871525", "bookshop, bookstore, bookstall"),
// 	("n02877765", "bottlecap"),
// 	("n02879718", "bow"),
// 	("n02883205", "bowtie, bow-tie, bowtie"),
// 	("n02892201", "brass, memorialtablet, plaque"),
// 	("n02892767", "brassiere, bra, bandeau"),
// 	("n02894605", "breakwater, groin, groyne, mole, bulwark, seawall, jetty"),
// 	("n02895154", "breastplate, aegis, egis"),
// 	("n02906734", "broom"),
// 	("n02909870", "bucket, pail"),
// 	("n02910353", "buckle"),
// 	("n02916936", "bulletproofvest"),
// 	("n02917067", "bullettrain, bullet"),
// 	("n02927161", "butchershop, meatmarket"),
// 	("n02930766", "cab, hack, taxi, taxicab"),
// 	("n02939185", "caldron, cauldron"),
// 	("n02948072", "candle, taper, waxlight"),
// 	("n02950826", "cannon"),
// 	("n02951358", "canoe"),
// 	("n02951585", "canopener, tinopener"),
// 	("n02963159", "cardigan"),
// 	("n02965783", "carmirror"),
// 	("n02966193", "carousel, carrousel, merry-go-round, roundabout, whirligig"),
// 	("n02966687", "carpenterskit, toolkit"),
// 	("n02971356", "carton"),
// 	("n02974003", "carwheel"),
// 	("n02977058", "cashmachine, cashdispenser, automatedtellermachine, automatictellermachine, automatedteller, automaticteller, ATM"),
// 	("n02978881", "cassette"),
// 	("n02979186", "cassetteplayer"),
// 	("n02980441", "castle"),
// 	("n02981792", "catamaran"),
// 	("n02988304", "CDplayer"),
// 	("n02992211", "cello, violoncello"),
// 	("n02992529", "cellulartelephone, cellularphone, cellphone, cell, mobilephone"),
// 	("n02999410", "chain"),
// 	("n03000134", "chainlinkfence"),
// 	("n03000247", "chainmail, ringmail, mail, chainarmor, chainarmour, ringarmor, ringarmour"),
// 	("n03000684", "chainsaw, chainsaw"),
// 	("n03014705", "chest"),
// 	("n03016953", "chiffonier, commode"),
// 	("n03017168", "chime, bell, gong"),
// 	("n03018349", "chinacabinet, chinacloset"),
// 	("n03026506", "Christmasstocking"),
// 	("n03028079", "church, churchbuilding"),
// 	("n03032252", "cinema, movietheater, movietheatre, moviehouse, picturepalace"),
// 	("n03041632", "cleaver, meatcleaver, chopper"),
// 	("n03042490", "cliffdwelling"),
// 	("n03045698", "cloak"),
// 	("n03047690", "clog, geta, patten, sabot"),
// 	("n03062245", "cocktailshaker"),
// 	("n03063599", "coffeemug"),
// 	("n03063689", "coffeepot"),
// 	("n03065424", "coil, spiral, volute, whorl, helix"),
// 	("n03075370", "combinationlock"),
// 	("n03085013", "computerkeyboard, keypad"),
// 	("n03089624", "confectionery, confectionary, candystore"),
// 	("n03095699", "containership, containership, containervessel"),
// 	("n03100240", "convertible"),
// 	("n03109150", "corkscrew, bottlescrew"),
// 	("n03110669", "cornet, horn, trumpet, trump"),
// 	("n03124043", "cowboyboot"),
// 	("n03124170", "cowboyhat, ten-gallonhat"),
// 	("n03125729", "cradle"),
// 	("n03126707", "crane"),
// 	("n03127747", "crashhelmet"),
// 	("n03127925", "crate"),
// 	("n03131574", "crib, cot"),
// 	("n03133878", "CrockPot"),
// 	("n03134739", "croquetball"),
// 	("n03141823", "crutch"),
// 	("n03146219", "cuirass"),
// 	("n03160309", "dam, dike, dyke"),
// 	("n03179701", "desk"),
// 	("n03180011", "desktopcomputer"),
// 	("n03187595", "dialtelephone, dialphone"),
// 	("n03188531", "diaper, nappy, napkin"),
// 	("n03196217", "digitalclock"),
// 	("n03197337", "digitalwatch"),
// 	("n03201208", "diningtable, board"),
// 	("n03207743", "dishrag, dishcloth"),
// 	("n03207941", "dishwasher, dishwasher, dishwashingmachine"),
// 	("n03208938", "diskbrake, discbrake"),
// 	("n03216828", "dock, dockage, dockingfacility"),
// 	("n03218198", "dogsled, dogsled, dogsleigh"),
// 	("n03220513", "dome"),
// 	("n03223299", "doormat, welcomemat"),
// 	("n03240683", "drillingplatform, offshorerig"),
// 	("n03249569", "drum, membranophone, tympan"),
// 	("n03250847", "drumstick"),
// 	("n03255030", "dumbbell"),
// 	("n03259280", "Dutchoven"),
// 	("n03271574", "electricfan, blower"),
// 	("n03272010", "electricguitar"),
// 	("n03272562", "electriclocomotive"),
// 	("n03290653", "entertainmentcenter"),
// 	("n03291819", "envelope"),
// 	("n03297495", "espressomaker"),
// 	("n03314780", "facepowder"),
// 	("n03325584", "featherboa, boa"),
// 	("n03337140", "file, filecabinet, filingcabinet"),
// 	("n03344393", "fireboat"),
// 	("n03345487", "fireengine, firetruck"),
// 	("n03347037", "firescreen, fireguard"),
// 	("n03355925", "flagpole, flagstaff"),
// 	("n03372029", "flute, transverseflute"),
// 	("n03376595", "foldingchair"),
// 	("n03379051", "footballhelmet"),
// 	("n03384352", "forklift"),
// 	("n03388043", "fountain"),
// 	("n03388183", "fountainpen"),
// 	("n03388549", "four-poster"),
// 	("n03393912", "freightcar"),
// 	("n03394916", "Frenchhorn, horn"),
// 	("n03400231", "fryingpan, frypan, skillet"),
// 	("n03404251", "furcoat"),
// 	("n03417042", "garbagetruck, dustcart"),
// 	("n03424325", "gasmask, respirator, gashelmet"),
// 	("n03425413", "gaspump, gasolinepump, petrolpump, islanddispenser"),
// 	("n03443371", "goblet"),
// 	("n03444034", "go-kart"),
// 	("n03445777", "golfball"),
// 	("n03445924", "golfcart, golfcart"),
// 	("n03447447", "gondola"),
// 	("n03447721", "gong, tam-tam"),
// 	("n03450230", "gown"),
// 	("n03452741", "grandpiano, grand"),
// 	("n03457902", "greenhouse, nursery, glasshouse"),
// 	("n03459775", "grille, radiatorgrille"),
// 	("n03461385", "grocerystore, grocery, foodmarket, market"),
// 	("n03467068", "guillotine"),
// 	("n03476684", "hairslide"),
// 	("n03476991", "hairspray"),
// 	("n03478589", "halftrack"),
// 	("n03481172", "hammer"),
// 	("n03482405", "hamper"),
// 	("n03483316", "handblower, blowdryer, blowdrier, hairdryer, hairdrier"),
// 	("n03485407", "hand-heldcomputer, hand-heldmicrocomputer"),
// 	("n03485794", "handkerchief, hankie, hanky, hankey"),
// 	("n03492542", "harddisc, harddisk, fixeddisk"),
// 	("n03494278", "harmonica, mouthorgan, harp, mouthharp"),
// 	("n03495258", "harp"),
// 	("n03496892", "harvester, reaper"),
// 	("n03498962", "hatchet"),
// 	("n03527444", "holster"),
// 	("n03529860", "hometheater, hometheatre"),
// 	("n03530642", "honeycomb"),
// 	("n03532672", "hook, claw"),
// 	("n03534580", "hoopskirt, crinoline"),
// 	("n03535780", "horizontalbar, highbar"),
// 	("n03538406", "horsecart, horse-cart"),
// 	("n03544143", "hourglass"),
// 	("n03584254", "iPod"),
// 	("n03584829", "iron, smoothingiron"),
// 	("n03590841", "jack-o-lantern"),
// 	("n03594734", "jean, bluejean, denim"),
// 	("n03594945", "jeep, landrover"),
// 	("n03595614", "jersey, T-shirt, teeshirt"),
// 	("n03598930", "jigsawpuzzle"),
// 	("n03599486", "jinrikisha, ricksha, rickshaw"),
// 	("n03602883", "joystick"),
// 	("n03617480", "kimono"),
// 	("n03623198", "kneepad"),
// 	("n03627232", "knot"),
// 	("n03630383", "labcoat, laboratorycoat"),
// 	("n03633091", "ladle"),
// 	("n03637318", "lampshade, lampshade"),
// 	("n03642806", "laptop, laptopcomputer"),
// 	("n03649909", "lawnmower, mower"),
// 	("n03657121", "lenscap, lenscover"),
// 	("n03658185", "letteropener, paperknife, paperknife"),
// 	("n03661043", "library"),
// 	("n03662601", "lifeboat"),
// 	("n03666591", "lighter, light, igniter, ignitor"),
// 	("n03670208", "limousine, limo"),
// 	("n03673027", "liner, oceanliner"),
// 	("n03676483", "lipstick, liprouge"),
// 	("n03680355", "Loafer"),
// 	("n03690938", "lotion"),
// 	("n03691459", "loudspeaker, speaker, speakerunit, loudspeakersystem, speakersystem"),
// 	("n03692522", "loupe, jewelersloupe"),
// 	("n03697007", "lumbermill, sawmill"),
// 	("n03706229", "magneticcompass"),
// 	("n03709823", "mailbag, postbag"),
// 	("n03710193", "mailbox, letterbox"),
// 	("n03710637", "maillot"),
// 	("n03710721", "maillot, tanksuit"),
// 	("n03717622", "manholecover"),
// 	("n03720891", "maraca"),
// 	("n03721384", "marimba, xylophone"),
// 	("n03724870", "mask"),
// 	("n03729826", "matchstick"),
// 	("n03733131", "maypole"),
// 	("n03733281", "maze, labyrinth"),
// 	("n03733805", "measuringcup"),
// 	("n03742115", "medicinechest, medicinecabinet"),
// 	("n03743016", "megalith, megalithicstructure"),
// 	("n03759954", "microphone, mike"),
// 	("n03761084", "microwave, microwaveoven"),
// 	("n03763968", "militaryuniform"),
// 	("n03764736", "milkcan"),
// 	("n03769881", "minibus"),
// 	("n03770439", "miniskirt, mini"),
// 	("n03770679", "minivan"),
// 	("n03773504", "missile"),
// 	("n03775071", "mitten"),
// 	("n03775546", "mixingbowl"),
// 	("n03776460", "mobilehome, manufacturedhome"),
// 	("n03777568", "ModelT"),
// 	("n03777754", "modem"),
// 	("n03781244", "monastery"),
// 	("n03782006", "monitor"),
// 	("n03785016", "moped"),
// 	("n03786901", "mortar"),
// 	("n03787032", "mortarboard"),
// 	("n03788195", "mosque"),
// 	("n03788365", "mosquitonet"),
// 	("n03791053", "motorscooter, scooter"),
// 	("n03792782", "mountainbike, all-terrainbike, off-roader"),
// 	("n03792972", "mountaintent"),
// 	("n03793489", "mouse, computermouse"),
// 	("n03794056", "mousetrap"),
// 	("n03796401", "movingvan"),
// 	("n03803284", "muzzle"),
// 	("n03804744", "nail"),
// 	("n03814639", "neckbrace"),
// 	("n03814906", "necklace"),
// 	("n03825788", "nipple"),
// 	("n03832673", "notebook, notebookcomputer"),
// 	("n03837869", "obelisk"),
// 	("n03838899", "oboe, hautboy, hautbois"),
// 	("n03840681", "ocarina, sweetpotato"),
// 	("n03841143", "odometer, hodometer, mileometer, milometer"),
// 	("n03843555", "oilfilter"),
// 	("n03854065", "organ, pipeorgan"),
// 	("n03857828", "oscilloscope, scope, cathode-rayoscilloscope, CRO"),
// 	("n03866082", "overskirt"),
// 	("n03868242", "oxcart"),
// 	("n03868863", "oxygenmask"),
// 	("n03871628", "packet"),
// 	("n03873416", "paddle, boatpaddle"),
// 	("n03874293", "paddlewheel, paddlewheel"),
// 	("n03874599", "padlock"),
// 	("n03876231", "paintbrush"),
// 	("n03877472", "pajama, pyjama, pjs, jammies"),
// 	("n03877845", "palace"),
// 	("n03884397", "panpipe, pandeanpipe, syrinx"),
// 	("n03887697", "papertowel"),
// 	("n03888257", "parachute, chute"),
// 	("n03888605", "parallelbars, bars"),
// 	("n03891251", "parkbench"),
// 	("n03891332", "parkingmeter"),
// 	("n03895866", "passengercar, coach, carriage"),
// 	("n03899768", "patio, terrace"),
// 	("n03902125", "pay-phone, pay-station"),
// 	("n03903868", "pedestal, plinth, footstall"),
// 	("n03908618", "pencilbox, pencilcase"),
// 	("n03908714", "pencilsharpener"),
// 	("n03916031", "perfume, essence"),
// 	("n03920288", "Petridish"),
// 	("n03924679", "photocopier"),
// 	("n03929660", "pick, plectrum, plectron"),
// 	("n03929855", "pickelhaube"),
// 	("n03930313", "picketfence, paling"),
// 	("n03930630", "pickup, pickuptruck"),
// 	("n03933933", "pier"),
// 	("n03935335", "piggybank, pennybank"),
// 	("n03937543", "pillbottle"),
// 	("n03938244", "pillow"),
// 	("n03942813", "ping-pongball"),
// 	("n03944341", "pinwheel"),
// 	("n03947888", "pirate, pirateship"),
// 	("n03950228", "pitcher, ewer"),
// 	("n03954731", "plane, carpentersplane, woodworkingplane"),
// 	("n03956157", "planetarium"),
// 	("n03958227", "plasticbag"),
// 	("n03961711", "platerack"),
// 	("n03967562", "plow, plough"),
// 	("n03970156", "plunger, plumbershelper"),
// 	("n03976467", "Polaroidcamera, PolaroidLandcamera"),
// 	("n03976657", "pole"),
// 	("n03977966", "policevan, policewagon, paddywagon, patrolwagon, wagon, blackMaria"),
// 	("n03980874", "poncho"),
// 	("n03982430", "pooltable, billiardtable, snookertable"),
// 	("n03983396", "popbottle, sodabottle"),
// 	("n03991062", "pot, flowerpot"),
// 	("n03992509", "potterswheel"),
// 	("n03995372", "powerdrill"),
// 	("n03998194", "prayerrug, prayermat"),
// 	("n04004767", "printer"),
// 	("n04005630", "prison, prisonhouse"),
// 	("n04008634", "projectile, missile"),
// 	("n04009552", "projector"),
// 	("n04019541", "puck, hockeypuck"),
// 	("n04023962", "punchingbag, punchbag, punchingball, punchball"),
// 	("n04026417", "purse"),
// 	("n04033901", "quill, quillpen"),
// 	("n04033995", "quilt, comforter, comfort, puff"),
// 	("n04037443", "racer, racecar, racingcar"),
// 	("n04039381", "racket, racquet"),
// 	("n04040759", "radiator"),
// 	("n04041544", "radio, wireless"),
// 	("n04044716", "radiotelescope, radioreflector"),
// 	("n04049303", "rainbarrel"),
// 	("n04065272", "recreationalvehicle, RV, R.V."),
// 	("n04067472", "reel"),
// 	("n04069434", "reflexcamera"),
// 	("n04070727", "refrigerator, icebox"),
// 	("n04074963", "remotecontrol, remote"),
// 	("n04081281", "restaurant, eatinghouse, eatingplace, eatery"),
// 	("n04086273", "revolver, six-gun, six-shooter"),
// 	("n04090263", "rifle"),
// 	("n04099969", "rockingchair, rocker"),
// 	("n04111531", "rotisserie"),
// 	("n04116512", "rubbereraser, rubber, pencileraser"),
// 	("n04118538", "rugbyball"),
// 	("n04118776", "rule, ruler"),
// 	("n04120489", "runningshoe"),
// 	("n04125021", "safe"),
// 	("n04127249", "safetypin"),
// 	("n04131690", "saltshaker, saltshaker"),
// 	("n04133789", "sandal"),
// 	("n04136333", "sarong"),
// 	("n04141076", "sax, saxophone"),
// 	("n04141327", "scabbard"),
// 	("n04141975", "scale, weighingmachine"),
// 	("n04146614", "schoolbus"),
// 	("n04147183", "schooner"),
// 	("n04149813", "scoreboard"),
// 	("n04152593", "screen, CRTscreen"),
// 	("n04153751", "screw"),
// 	("n04154565", "screwdriver"),
// 	("n04162706", "seatbelt, seatbelt"),
// 	("n04179913", "sewingmachine"),
// 	("n04192698", "shield, buckler"),
// 	("n04200800", "shoeshop, shoe-shop, shoestore"),
// 	("n04201297", "shoji"),
// 	("n04204238", "shoppingbasket"),
// 	("n04204347", "shoppingcart"),
// 	("n04208210", "shovel"),
// 	("n04209133", "showercap"),
// 	("n04209239", "showercurtain"),
// 	("n04228054", "ski"),
// 	("n04229816", "skimask"),
// 	("n04235860", "sleepingbag"),
// 	("n04238763", "sliderule, slipstick"),
// 	("n04239074", "slidingdoor"),
// 	("n04243546", "slot, one-armedbandit"),
// 	("n04251144", "snorkel"),
// 	("n04252077", "snowmobile"),
// 	("n04252225", "snowplow, snowplough"),
// 	("n04254120", "soapdispenser"),
// 	("n04254680", "soccerball"),
// 	("n04254777", "sock"),
// 	("n04258138", "solardish, solarcollector, solarfurnace"),
// 	("n04259630", "sombrero"),
// 	("n04263257", "soupbowl"),
// 	("n04264628", "spacebar"),
// 	("n04265275", "spaceheater"),
// 	("n04266014", "spaceshuttle"),
// 	("n04270147", "spatula"),
// 	("n04273569", "speedboat"),
// 	("n04275548", "spiderweb, spidersweb"),
// 	("n04277352", "spindle"),
// 	("n04285008", "sportscar, sportcar"),
// 	("n04286575", "spotlight, spot"),
// 	("n04296562", "stage"),
// 	("n04310018", "steamlocomotive"),
// 	("n04311004", "steelarchbridge"),
// 	("n04311174", "steeldrum"),
// 	("n04317175", "stethoscope"),
// 	("n04325704", "stole"),
// 	("n04326547", "stonewall"),
// 	("n04328186", "stopwatch, stopwatch"),
// 	("n04330267", "stove"),
// 	("n04332243", "strainer"),
// 	("n04335435", "streetcar, tram, tramcar, trolley, trolleycar"),
// 	("n04336792", "stretcher"),
// 	("n04344873", "studiocouch, daybed"),
// 	("n04346328", "stupa, tope"),
// 	("n04347754", "submarine, pigboat, sub, U-boat"),
// 	("n04350905", "suit, suitofclothes"),
// 	("n04355338", "sundial"),
// 	("n04355933", "sunglass"),
// 	("n04356056", "sunglasses, darkglasses, shades"),
// 	("n04357314", "sunscreen, sunblock, sunblocker"),
// 	("n04366367", "suspensionbridge"),
// 	("n04367480", "swab, swob, mop"),
// 	("n04370456", "sweatshirt"),
// 	("n04371430", "swimmingtrunks, bathingtrunks"),
// 	("n04371774", "swing"),
// 	("n04372370", "switch, electricswitch, electricalswitch"),
// 	("n04376876", "syringe"),
// 	("n04380533", "tablelamp"),
// 	("n04389033", "tank, armytank, armoredcombatvehicle, armouredcombatvehicle"),
// 	("n04392985", "tapeplayer"),
// 	("n04398044", "teapot"),
// 	("n04399382", "teddy, teddybear"),
// 	("n04404412", "television, televisionsystem"),
// 	("n04409515", "tennisball"),
// 	("n04417672", "thatch, thatchedroof"),
// 	("n04418357", "theatercurtain, theatrecurtain"),
// 	("n04423845", "thimble"),
// 	("n04428191", "thresher, thrasher, threshingmachine"),
// 	("n04429376", "throne"),
// 	("n04435653", "tileroof"),
// 	("n04442312", "toaster"),
// 	("n04443257", "tobaccoshop, tobacconistshop, tobacconist"),
// 	("n04447861", "toiletseat"),
// 	("n04456115", "torch"),
// 	("n04458633", "totempole"),
// 	("n04461696", "towtruck, towcar, wrecker"),
// 	("n04462240", "toyshop"),
// 	("n04465501", "tractor"),
// 	("n04467665", "trailertruck, tractortrailer, truckingrig, rig, articulatedlorry, semi"),
// 	("n04476259", "tray"),
// 	("n04479046", "trenchcoat"),
// 	("n04482393", "tricycle, trike, velocipede"),
// 	("n04483307", "trimaran"),
// 	("n04485082", "tripod"),
// 	("n04486054", "triumphalarch"),
// 	("n04487081", "trolleybus, trolleycoach, tracklesstrolley"),
// 	("n04487394", "trombone"),
// 	("n04493381", "tub, vat"),
// 	("n04501370", "turnstile"),
// 	("n04505470", "typewriterkeyboard"),
// 	("n04507155", "umbrella"),
// 	("n04509417", "unicycle, monocycle"),
// 	("n04515003", "upright, uprightpiano"),
// 	("n04517823", "vacuum, vacuumcleaner"),
// 	("n04522168", "vase"),
// 	("n04523525", "vault"),
// 	("n04525038", "velvet"),
// 	("n04525305", "vendingmachine"),
// 	("n04532106", "vestment"),
// 	("n04532670", "viaduct"),
// 	("n04536866", "violin, fiddle"),
// 	("n04540053", "volleyball"),
// 	("n04542943", "waffleiron"),
// 	("n04548280", "wallclock"),
// 	("n04548362", "wallet, billfold, notecase, pocketbook"),
// 	("n04550184", "wardrobe, closet, press"),
// 	("n04552348", "warplane, militaryplane"),
// 	("n04553703", "washbasin, handbasin, washbowl, lavabo, wash-handbasin"),
// 	("n04554684", "washer, automaticwasher, washingmachine"),
// 	("n04557648", "waterbottle"),
// 	("n04560804", "waterjug"),
// 	("n04562935", "watertower"),
// 	("n04579145", "whiskeyjug"),
// 	("n04579432", "whistle"),
// 	("n04584207", "wig"),
// 	("n04589890", "windowscreen"),
// 	("n04590129", "windowshade"),
// 	("n04591157", "Windsortie"),
// 	("n04591713", "winebottle"),
// 	("n04592741", "wing"),
// 	("n04596742", "wok"),
// 	("n04597913", "woodenspoon"),
// 	("n04599235", "wool, woolen, woollen"),
// 	("n04604644", "wormfence, snakefence, snake-railfence, Virginiafence"),
// 	("n04606251", "wreck"),
// 	("n04612504", "yawl"),
// 	("n04613696", "yurt"),
// 	("n06359193", "website, website, internetsite, site"),
// 	("n06596364", "comicbook"),
// 	("n06785654", "crosswordpuzzle, crossword"),
// 	("n06794110", "streetsign"),
// 	("n06874185", "trafficlight, trafficsignal, stoplight"),
// 	("n07248320", "bookjacket, dustcover, dustjacket, dustwrapper"),
// 	("n07565083", "menu"),
// 	("n07579787", "plate"),
// 	("n07583066", "guacamole"),
// 	("n07584110", "consomme"),
// 	("n07590611", "hotpot, hotpot"),
// 	("n07613480", "trifle"),
// 	("n07614500", "icecream, icecream"),
// 	("n07615774", "icelolly, lolly, lollipop, popsicle"),
// 	("n07684084", "Frenchloaf"),
// 	("n07693725", "bagel, beigel"),
// 	("n07695742", "pretzel"),
// 	("n07697313", "cheeseburger"),
// 	("n07697537", "hotdog, hotdog, redhot"),
// 	("n07711569", "mashedpotato"),
// 	("n07714571", "headcabbage"),
// 	("n07714990", "broccoli"),
// 	("n07715103", "cauliflower"),
// 	("n07716358", "zucchini, courgette"),
// 	("n07716906", "spaghettisquash"),
// 	("n07717410", "acornsquash"),
// 	("n07717556", "butternutsquash"),
// 	("n07718472", "cucumber, cuke"),
// 	("n07718747", "artichoke, globeartichoke"),
// 	("n07720875", "bellpepper"),
// 	("n07730033", "cardoon"),
// 	("n07734744", "mushroom"),
// 	("n07742313", "GrannySmith"),
// 	("n07745940", "strawberry"),
// 	("n07747607", "orange"),
// 	("n07749582", "lemon"),
// 	("n07753113", "fig"),
// 	("n07753275", "pineapple, ananas"),
// 	("n07753592", "banana"),
// 	("n07754684", "jackfruit, jak, jack"),
// 	("n07760859", "custardapple"),
// 	("n07768694", "pomegranate"),
// 	("n07802026", "hay"),
// 	("n07831146", "carbonara"),
// 	("n07836838", "chocolatesauce, chocolatesyrup"),
// 	("n07860988", "dough"),
// 	("n07871810", "meatloaf, meatloaf"),
// 	("n07873807", "pizza, pizzapie"),
// 	("n07875152", "potpie"),
// 	("n07880968", "burrito"),
// 	("n07892512", "redwine"),
// 	("n07920052", "espresso"),
// 	("n07930864", "cup"),
// 	("n07932039", "eggnog"),
// 	("n09193705", "alp"),
// 	("n09229709", "bubble"),
// 	("n09246464", "cliff, drop, drop-off"),
// 	("n09256479", "coralreef"),
// 	("n09288635", "geyser"),
// 	("n09332890", "lakeside, lakeshore"),
// 	("n09399592", "promontory, headland, head, foreland"),
// 	("n09421951", "sandbar, sandbar"),
// 	("n09428293", "seashore, coast, seacoast, sea-coast"),
// 	("n09468604", "valley, vale"),
// 	("n09472597", "volcano"),
// 	("n09835506", "ballplayer, baseballplayer"),
// 	("n10148035", "groom, bridegroom"),
// 	("n10565667", "scubadiver"),
// 	("n11879895", "rapeseed"),
// 	("n11939491", "daisy"),
// 	("n12057211", "yellowladysslipper, yellowlady-slipper, Cypripediumcalceolus, Cypripediumparviflorum"),
// 	("n12144580", "corn"),
// 	("n12267677", "acorn"),
// 	("n12620546", "hip, rosehip, rosehip"),
// 	("n12768682", "buckeye, horsechestnut, conker"),
// 	("n12985857", "coralfungus"),
// 	("n12998815", "agaric"),
// 	("n13037406", "gyromitra"),
// 	("n13040303", "stinkhorn, carrionfungus"),
// 	("n13044778", "earthstar"),
// 	("n13052670", "hen-of-the-woods, henofthewoods, Polyporusfrondosus, Grifolafrondosa"),
// 	("n13054560", "bolete"),
// 	("n13133613", "ear, spike, capitulum"),
// 	("n15075141", "toilettissue, toiletpaper, bathroomtissue"),
// ];