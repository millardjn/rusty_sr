# Rusty_SR
![LogoNN](docs/logo_nn.png)![LogoLin](docs/logo_lin.png)![Logo](docs/logo_rs.png)

Deep learning super resolution written in rust.
Use baked in neural networks to upscale your images, or train your own specialised neural network!

For best performance compile using environmental variable `RUSTFLAGS="-C target-cpu=native" ` and release mode `cargo build --release`
or: `cargo rustc --release -- -C target-cpu=native`

Feel free to open an issue to raise any problems or for general discussion.

## Examples
Set14 Cartoon

![CartoonLowRes](docs/cartoon_nn.png)![Cartoon](docs/cartoon_rsa.png)

Set14 Butterfly

![ButterflyLowRes](docs/butterfly_nn.png)
![Butterfly](docs/butterfly_rs.png)

Bank Lobby (test image for [Neural Enhance](https://github.com/alexjc/neural-enhance))
CC-BY-SA @benarent

![BankLowRes](docs/bank_nn.png)
![Bank](docs/bank_rs.png)

## Note
Attemping to upscale images with significant noise or jpeg artifacts is likely to produce poor results. Input and output colorspace are nominally sRGB.

## License
MIT
