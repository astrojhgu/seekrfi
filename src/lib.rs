#![allow(non_snake_case)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::let_and_return)]
#[macro_use]
extern crate ndarray;
extern crate num_traits;
extern crate imageproc;
extern crate image;
extern crate statistical;
extern crate fitsimg;

use num_traits::{zero, one};
use ndarray::{Array2, Array1, ArrayView2};
use image::ImageBuffer;
use image::Luma;
use statistical::median;
use fitsimg::write_img;
use std::iter::FromIterator;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FlagState{
    Normal,
    Flagged,
}

impl Default for FlagState{
    fn default()->FlagState{
        FlagState::Normal
    }
}

pub fn flag_if_both_flagged(f1:FlagState, f2:FlagState)->FlagState{
    if f1==FlagState::Flagged && f2==FlagState::Flagged{
        FlagState::Flagged
    }else{
        FlagState::Normal
    }
}

pub fn flag_if_either_flagged(f1:FlagState, f2:FlagState)->FlagState {
    if f1==FlagState::Flagged || f2==FlagState::Flagged{
        FlagState::Flagged
    }else{
        FlagState::Normal
    }
}

pub fn flag_if(b:bool)->FlagState {
    if b{
        FlagState::Flagged
    }else{
        FlagState::Normal
    }
}

impl std::ops::BitXor for FlagState{
    type Output=FlagState;

    fn bitxor(self, rhs:FlagState)->FlagState{
        if self==rhs{
            FlagState::Normal
        }else{
            FlagState::Flagged
        }
    }
}

impl std::ops::Not for FlagState{
    type Output=FlagState;

    fn not(self)->FlagState{
        match self{
            FlagState::Flagged=>FlagState::Normal,
            _=>FlagState::Flagged
        }
    }
}

impl std::ops::BitOr for FlagState{
    type Output=FlagState;

    fn bitor(self, rhs:FlagState)->FlagState{
        if self==FlagState::Flagged || rhs==FlagState::Flagged{
            FlagState::Flagged
        }else{
            FlagState::Normal
        }
    }
}

pub fn write_mask(mask:&Array2<FlagState>, fname:&str){
    let m=mask.map(|&x|{
        match x{
            FlagState::Flagged=>1,
            _=>0
        }
    });

    write_img(fname.to_string(), &m.into_dyn()).unwrap();
}

pub fn write_data<T>(img:&Array2<T>, fname:&str)
where T:num_traits::Float+std::fmt::Debug+fitsimg::TypeToImageType+fitsio::images::WriteImage
{
    let img=img.clone().into_dyn();

    write_img(fname.to_string(), &img).unwrap();
}


pub fn gaussian_filter<T>(v:&Array2<T>, mask:&Array2<FlagState>, kernel_x:usize, kernel_y:usize, sigma_x:T, sigma_y:T)->Array2<T>
where T:num_traits::Float+std::fmt::Debug+fitsimg::TypeToImageType+fitsio::images::WriteImage
{
    let two:T=one::<T>()+one::<T>();
    let wd=|n:isize,m:isize|->T{
            (-(T::from(n*n)).unwrap()/(two*sigma_y.powi(2))-(T::from(m*m)).unwrap()/(two*sigma_x.powi(2))).exp()
    };
    let mut vp=Array2::zeros((v.nrows()+kernel_y, v.ncols()+kernel_x));

    vp.slice_mut(s![kernel_y/2..vp.nrows()-kernel_y/2, kernel_x/2..vp.ncols()-kernel_x/2]).assign(&v);

    let mut wfp=Array2::zeros((v.nrows()+kernel_y, v.ncols()+kernel_x));
    wfp.slice_mut(s![kernel_y/2..wfp.nrows()-kernel_y/2, kernel_x/2.. wfp.ncols()-kernel_x/2]).assign(&mask.map(|&x|{
        match x{
            FlagState::Flagged=>zero(),
            FlagState::Normal=>one()
        }
    }));
    let kernel_0=(-(kernel_y as isize)/2..=kernel_y as isize/2).map(|n|wd(n,0)).collect::<Array1<T>>();
    let kernel_1=(-(kernel_x as isize)/2..=kernel_x as isize/2).map(|m|wd(0,m)).collect::<Array1<T>>();
    println!("{:?}",kernel_1);
    let vh=run_gaussian_filter(&vp, v.nrows(), v.ncols(), wfp, mask, kernel_0, kernel_1, kernel_x, kernel_y);
    let vh=vh.slice(s![kernel_y/2..vh.nrows()-kernel_y/2, kernel_x/2..vh.ncols()-kernel_x/2]).to_owned();
    //vh.iter_mut().zip(v.iter().zip(mask.iter())).for_each(|(a,(b,&c))|{if !c {*a=*b}});
    vh
}

fn run_gaussian_filter<T>(vp:&Array2<T>, vs0:usize, vs1:usize,
                          wfp:Array2<T>, mask:&Array2<FlagState>,
                          kernel_0:Array1<T>, kernel_1:Array1<T>, nx:usize, ny:usize)->Array2<T>
where T:num_traits::Float+std::fmt::Debug+fitsimg::TypeToImageType+fitsio::images::WriteImage{
    let mut vh=Array2::<T>::zeros((vp.nrows(), vp.ncols()));
    let mut vh2=Array2::<T>::zeros((vp.nrows(), vp.ncols()));
    let nx2=nx/2;
    let ny2=ny/2;
    for i in ny2..vs0+ny2{
        for j in nx2..vs1+nx2{
            match mask[(i-ny2,j-nx2)]{
                FlagState::Flagged=>{vh[(i,j)]=zero();},
                _=>{
                    let val=(&wfp.slice(s![i-ny2..=i+ny2, j])* &vp.slice(s![i-ny2..=i+ny2, j])* &kernel_0).sum();
                    vh[(i,j)]=val/(&wfp.slice(s![i-ny2..=i+ny2, j])* &kernel_0).sum();
                }
            }
        }
    }

    for j in nx2..vs1+nx2{
        for i in ny2..vs0+ny2{
            match mask[(i-ny2, j-nx2)]{
                FlagState::Flagged=>{
                    vh2[(i,j)]=zero();
                },
                _=>{
                    let val=(&wfp.slice(s![i, j-nx2..=j+nx2])*&vh.slice(s![i, j-nx2..=j+nx2])*&kernel_1).sum();
                    vh2[(i,j)]=val/(&wfp.slice(s![i, j-nx2..=j+nx2])*&kernel_1).sum();
                }
            }
        }
    }
    vh2
}


pub fn binary_mask_dialation(mask:Array2<FlagState>, ss0:usize, ss1:usize, iter:usize)->Array2<FlagState>{
    let width=mask.ncols();
    let height=mask.nrows();
    let mut img=
        ImageBuffer::<Luma<u8>, Vec<u8>>::from_raw(
            width as u32, height as u32,
            mask.into_shape((width*height,)).unwrap().to_vec().into_iter().map(|x|{
                match x{
                    FlagState::Flagged=>1,
                    _=>0
                }
            }).collect::<Vec<_>>()).unwrap();


    for _i in 0..iter{
        img=imageproc::filter::separable_filter(&img, &vec![1_u16; ss1], &vec![1_u16; ss0]);
    }
    //let img=imageproc::filter::box_filter(&img, 5,5);

    Array1::from_iter(img.into_vec().into_iter().map(|pix|{
        if pix==0 {
            FlagState::Normal
        }else{
            FlagState::Flagged
        }
    })).into_shape((height, width)).unwrap()
}

pub fn normalize<T>(mut data:Array2<T>, mask:&Array2<FlagState>)->Array2<T>
where T:num_traits::Float+std::fmt::Debug+fitsimg::TypeToImageType+fitsio::images::WriteImage{
    for i in 0..data.nrows(){
        let r:Vec<_>=data.row(i).iter().zip(mask.row(i)
            .iter())
            .filter(|(&_a,&b)|{b==FlagState::Normal}).map(|(&a,_b)|{a}).collect();
        let med=median(&r);
        data.row_mut(i).iter_mut().for_each(|x|{*x=(*x-med).abs()});
    }
    data
}


pub fn _sumthreshold<T>(data:&ArrayView2<T>, mask:Array2<FlagState>, i:usize, chi:T)->Array2<FlagState>
where T:num_traits::Float+std::fmt::Debug+fitsimg::TypeToImageType+fitsio::images::WriteImage{
    let mut tmp_mask=mask.clone();
    let ds0=data.nrows();
    let ds1=data.ncols();
    for x in 0..ds0{
        let mut sum=zero::<T>();
        let mut cnt=0;

        for ii in 0..i{
            if mask[(x, ii)]==FlagState::Normal{
                sum=sum+data[(x, ii)];
                cnt+=1;
            }
        }

        for y in i..ds1{
            if sum> chi*T::from(cnt).unwrap(){
                for ii2 in 0..i{
                    tmp_mask[(x, y-ii2-1)]=FlagState::Flagged;
                }
                //tmp_mask.slice_mut(s![x, (y-i)..(y-1)]).fill(FlagState::Flagged);
            }

            if mask[(x,y)]==FlagState::Normal{
                sum=sum+data[(x,y)];
                cnt+=1;
            }

            if mask[(x, y-i)]==FlagState::Normal{
                sum=sum-data[(x,y-i)];
                cnt-=1;
            }
        }
        println!("{:?} {}", sum, cnt);
        //panic!();
    }
    tmp_mask
}

pub fn _run_sumthreshold<T>(data:&Array2<T>, init_mask:&Array2<FlagState>,
                            eta:T, M:&[usize], chi_i:&[T],
                            kernel_m:usize, kernel_n:usize, sigma_m:T, sigma_n:T)->Array2<FlagState>
    where T:num_traits::Float+std::fmt::Debug+fitsimg::TypeToImageType+fitsio::images::WriteImage{
    let smoothed_data=gaussian_filter(data, init_mask, kernel_m, kernel_n, sigma_m, sigma_n);
    let res=data-&smoothed_data;
    let mut st_mask=init_mask.clone();
    for (&m, &chi) in M.iter().zip(chi_i.iter()){
        let chi=chi/eta;
        if m==1{
            st_mask.zip_mut_with(&res, |a, &b|{
                if b>chi{
                    *a=FlagState::Flagged;
                }
            });
        }else{
            write_img("res.fits".to_string(), &res.clone().into_dyn()).unwrap();
            write_mask( &st_mask.clone(),"mask0.fits");
            println!("{} {:?}", m, chi);
            st_mask=_sumthreshold(&res.view(), st_mask, m, chi);
            write_mask(&st_mask, &format!("mask_{:?}_{}_1.fits", eta, m));
            panic!();
            st_mask=_sumthreshold(&res.t(), st_mask.t().to_owned(), m, chi).t().to_owned();
            write_mask(&st_mask, &format!("mask_{:?}_{}_2.fits", eta, m));

        }
    }
    st_mask
}


pub fn get_rfi_mask<T>(data1:ArrayView2<T>, mask1:Option<ArrayView2<FlagState>>, chi_1:T, eta_i:&[T], normalize_standing_wave:bool, suppress_dialation:bool,
kernel_m:usize, kernel_n:usize, sigma_m:T, sigma_n:T, di_args:(usize, usize))->Array2<FlagState>
where T:num_traits::Float+std::fmt::Debug+fitsimg::TypeToImageType+fitsio::images::WriteImage{
    let mask=mask1
        .map_or(Array2::<FlagState>::default((data1.nrows(), data1.ncols())), |x|{x.to_owned()});
    let mut data=data1.to_owned();
    if normalize_standing_wave{
        data=normalize(data, &mask);
    }

    write_data(&data, "normalized.fits");
    let p=T::from(1.5).unwrap();
    let M:Vec<_>=(1..8).map(|m| (2_usize.pow(m-1))).collect();
    let chi_i:Vec<_>=M.iter().map(|&m|{chi_1/p.powf(T::from(m).unwrap().log2())}).collect();
    let mut st_mask=mask;
    for &eta in eta_i{
        st_mask=_run_sumthreshold(&data, &st_mask, eta, &M[..], &chi_i, kernel_m, kernel_n, sigma_m, sigma_n);
    }
    write_mask(&st_mask, "st_mask.fits");
    if suppress_dialation{
        st_mask
    }else{
        binary_mask_dialation(
            mask1.map_or(
                st_mask.clone(), |x|{!((&x)^(&st_mask))}), di_args.0, di_args.1, 2)|st_mask
    }
}
