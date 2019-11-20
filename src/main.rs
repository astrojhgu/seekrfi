extern crate fitsimg;
extern crate ndarray;
extern crate seekrfi;
use ndarray::{Array2};
use fitsimg::write_img;
use seekrfi::gaussian_filter;
use seekrfi::FlagState;

fn main() {
    let mut mat=Array2::<f64>::zeros((1024,1024));

    mat[(512,512)]=1.0;

    let mut mask=Array2::<FlagState>::default((1024,1024));
    mask[(512,512)]=FlagState::Flagged;

    let mat1=gaussian_filter(&mat, &mask, 16,32,4.0, 4.0);

    write_img("./a.fits".to_string(), &mat1.into_dyn()).unwrap();


    let mut mask=Array2::<FlagState>::default((32,32));
    mask[(16,16)]=FlagState::Flagged;

    let mask=seekrfi::binary_mask_dialation(mask, 3,3,1).map(|&x|{x as i32});

    write_img("./b.fits".to_string(), &mask.into_dyn()).unwrap();
}
