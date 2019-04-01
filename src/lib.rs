#[macro_use]
extern crate ndarray;
extern crate num_traits;
use num_traits::{zero, one};
use ndarray::{Array2, Array1};



pub fn gaussian_filter<T>(v:Array2<T>, mask:&Array2<bool>, kernel_x:usize, kernel_y:usize, sigma_x:T, sigma_y:T)->Array2<T>
where T:num_traits::Float+std::fmt::Debug
{
    let two:T=one::<T>()+one::<T>();
    let wd=|n:isize,m:isize|->T{
            (-(T::from(n*n)).unwrap()/(two*sigma_y.powi(2))-(T::from(m*m)).unwrap()/(two*sigma_x.powi(2))).exp()
    };
    let mut vp=Array2::zeros((v.rows()+kernel_y, v.cols()+kernel_x));

    vp.slice_mut(s![kernel_y/2..vp.rows()-kernel_y/2, kernel_x/2..vp.cols()-kernel_x/2]).assign(&v);

    let mut wfp=Array2::zeros((v.rows()+kernel_y, v.cols()+kernel_x));
    wfp.slice_mut(s![kernel_y/2..wfp.rows()-kernel_y/2, kernel_x/2.. wfp.cols()-kernel_x/2]).assign(&mask.map(|&x|{if x {one()} else {zero()}}));
    let kernel_0=(-(kernel_y as isize)/2..kernel_y as isize/2+1).map(|n|wd(n,0)).collect::<Array1<T>>();
    let kernel_1=(-(kernel_x as isize)/2..kernel_x as isize/2+1).map(|m|wd(0,m)).collect::<Array1<T>>();
    println!("{:?}",kernel_1);
    let vh=run_gaussian_filter(&vp, v.rows(), v.cols(), wfp, mask, kernel_0, kernel_1, kernel_x, kernel_y);
    let mut vh=vh.slice(s![kernel_y/2..vh.rows()-kernel_y/2, kernel_x/2..vh.cols()-kernel_x/2]).to_owned();
    //vh.iter_mut().zip(v.iter().zip(mask.iter())).for_each(|(a,(b,&c))|{if !c {*a=*b}});
    vh
}

fn run_gaussian_filter<T>(vp:&Array2<T>, vs0:usize, vs1:usize,
                          wfp:Array2<T>, mask:&Array2<bool>,
                          kernel_0:Array1<T>, kernel_1:Array1<T>, nx:usize, ny:usize)->Array2<T>
where T:num_traits::Float+std::fmt::Debug{
    let mut vh=Array2::<T>::zeros((vp.rows(), vp.cols()));
    let mut vh2=Array2::<T>::zeros((vp.rows(), vp.cols()));
    let nx2=nx/2;
    let ny2=ny/2;
    for i in ny2..vs0+ny2{
        for j in nx2..vs1+nx2{
            if !mask[(i-ny2, j-nx2)]{
                vh[(i,j)]=zero();
            }else{
                let val=(&wfp.slice(s![i-ny2..i+ny2+1, j])* &vp.slice(s![i-ny2..i+ny2+1, j])* &kernel_0).sum();
                vh[(i,j)]=val/(&wfp.slice(s![i-ny2..i+ny2+1, j])* &kernel_0).sum();

            }
        }
    }

    for j in nx2..vs1+nx2{
        for i in ny2..vs0+ny2{
            if !mask[(i-ny2, j-nx2)]{
                vh2[(i,j)]=zero();
            }else{
                let val=(&wfp.slice(s![i, j-nx2..j+nx2+1])*&vh.slice(s![i, j-nx2..j+nx2+1])*&kernel_1).sum();
                vh2[(i,j)]=val/((&wfp.slice(s![i, j-nx2..j+nx2+1])*&kernel_1)).sum();

            }
        }
    }
    vh2
}
