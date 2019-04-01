extern crate fitsimg;
extern crate ndarray;
extern crate seekrfi;
use ndarray::{Array2, ArrayD};
use fitsimg::write_img;
use seekrfi::gaussian_filter;


fn main() {
    let mut mat=Array2::<f64>::zeros((1024,1024));

    mat[(512,512)]=1.0;

    let mut mask=!Array2::<bool>::default((1024,1024));
    //mask[(512,512)]=true;

    let mat1=gaussian_filter(mat, &mask, 16,32,4.0, 4.0);

    write_img("./a.fits".to_string(), &mat1.into_dyn());
}
