extern crate blas;
extern crate openblas_src;

use std::env;
use std::time::{Instant};
use ndarray::{Zip,Array2,ArrayView2,ArrayViewMut2,s};
use blas::dgemm;

//Computes C+=A*B
fn gemm_base(a : ArrayView2<f64>,b : ArrayView2<f64>,mut c : ArrayViewMut2<f64>) -> () {
    assert_eq!(a.shape()[1],b.shape()[0]);
    assert_eq!(a.shape()[0],c.shape()[0]);
    assert_eq!(b.shape()[1],c.shape()[1]);


    for (mut ci,ai) in c.genrows_mut().into_iter().zip(a.genrows().into_iter()){
        for (bk,aik) in b.genrows().into_iter().zip(ai.iter()){
            Zip::from(&mut ci).and(&bk).apply({
                |x,y|{
                    *x += (*y)*aik;
                }
            });
        }
    }
}

fn gemm_optimized(a : ArrayView2<f64>,b : ArrayView2<f64>,mut c : ArrayViewMut2<f64>) -> () {
    const MAXSIZE : usize = 256;
    assert_eq!(a.shape()[1],b.shape()[0]);
    assert_eq!(a.shape()[0],c.shape()[0]);
    assert_eq!(b.shape()[1],c.shape()[1]);
    let m=a.shape()[0];
    let p=a.shape()[1];
    let n=b.shape()[1];
    if m>MAXSIZE || n>MAXSIZE{
        let m2=m/2;
        let n2=n/2;
        let p2=p/2;
        {
            let (c11,c12,c21,c22) = c.multi_slice_mut(
                (s![0..m2,0..n2],s![0..m2,n2..n],s![m2..m,0..n2],s![m2..m,n2..n]));
            rayon::join(
            ||{
                //Update c11 block
                gemm_optimized(a.slice(s![0..m2,0..p2]),b.slice(s![0..p2,0..n2]),c11);
                gemm_optimized(a.slice(s![m2..m,0..p2]),b.slice(s![0..p2,0..n2]),c21);
            },
            ||{
                //Update c12 block
                gemm_optimized(a.slice(s![0..m2,0..p2]),b.slice(s![0..p2,n2..n]),c12);
                gemm_optimized(a.slice(s![m2..m,0..p2]),b.slice(s![0..p2,n2..n]),c22);
            });
        }
        {
            let (c11,c12,c21,c22) = c.multi_slice_mut(
                (s![0..m2,0..n2],s![0..m2,n2..n],s![m2..m,0..n2],s![m2..m,n2..n]));
            rayon::join(
            ||{
                //Update c21 block
                gemm_optimized(a.slice(s![0..m2,p2..p]),b.slice(s![p2..p,0..n2]),c11);
                gemm_optimized(a.slice(s![m2..m,p2..p]),b.slice(s![p2..p,0..n2]),c21);
            },
            ||{ 
                //Update c22 block
                gemm_optimized(a.slice(s![0..m2,p2..p]),b.slice(s![p2..p,n2..n]),c12);
                gemm_optimized(a.slice(s![m2..m,p2..p]),b.slice(s![p2..p,n2..n]),c22);
            });
        }
    }
    else{
        gemm_base(a,b,c);
    }
}



fn main() {
    let args: Vec<String> = env::args().collect();
    let m=args[1].parse::<usize>().unwrap();
    let n=m as usize;
    let nruns=100;
    let mut c1 = Array2::<f64>::zeros((m,n));
    let mut c2 = Array2::<f64>::zeros((m,n));
    //Just make up some data for a,b.
    let (a,b) = {
        let mut a = Array2::<f64>::zeros((m,n));
        let mut b = Array2::<f64>::zeros((m,n));
        for ((r,c),v) in a.indexed_iter_mut(){
            let rf = r as f64;
            let cf = c as f64;
            *v = (rf+cf).sin()+2.0;
        }        
        for ((r,c),v) in b.indexed_iter_mut(){
            let rf = r as f64;
            let cf = c as f64;
            *v = (rf+cf).cos()+2.0;
        }
        (a,b)
    };



    let times_ref = {
        let mut times_ref = Vec::<f64>::new();
        for _ in 0..nruns{
            let now = Instant::now();
            unsafe{
                dgemm(b'T',b'T',m as i32,m as i32,m as i32,1.0,
                      b.as_slice_memory_order().unwrap(),m as i32,a.as_slice_memory_order().unwrap(),m as i32,1.0,c1.as_slice_memory_order_mut().unwrap(),m as i32);
            }
            times_ref.push(now.elapsed().as_secs_f64());
        }
        times_ref.sort_by(|x,y|x.partial_cmp(y).unwrap());
        times_ref
    };

    let times_opt = {
        let mut times_opt = Vec::<f64>::new();
        for _ in 0..nruns{
            let now = Instant::now();
            gemm_optimized(a.view(),b.view(),c2.view_mut());
            times_opt.push(now.elapsed().as_secs_f64());
        }
        times_opt.sort_by(|x,y|x.partial_cmp(y).unwrap());
        times_opt
    };


    //now check for maximum relative error between c1,c2 make sure we just 
    //computed the same thing.
    let mut max_relerr = 0.0;
    for (&x,&y) in c1.iter().zip(c2.iter()){
        let err = (x-y).abs();
        let minxy = if x.abs()<y.abs() { x.abs() } else { y.abs() };
        let relerr = err/minxy;
        max_relerr = if max_relerr<relerr { relerr } else { max_relerr };
    }
    //Print out the result for inspection
    print!("Maximum relative error: {}\n",max_relerr);
    //Print out timing statistics for reference run
    {
        let min=times_ref[0];
        let max=*times_ref.last().unwrap();
        let avg=times_ref.iter().fold(0.0,|acc,x|acc+x)/(times_ref.len() as f64);
        let std=(times_ref.iter().map(|x|(avg-x)*(avg-x)).fold(0.0,|acc,x|acc+x)/(times_ref.len() as f64)).sqrt();
        print!("Reference:    min = {},   max = {},  avg = {},  std = {}\n",min,max,avg,std);
    }
    //Print out timing statistics for optimized run
    {
        let min=times_opt[0];
        let max=*times_opt.last().unwrap();
        let avg=times_opt.iter().fold(0.0,|acc,x|acc+x)/(times_opt.len() as f64);
        let std=(times_opt.iter().map(|x|(avg-x)*(avg-x)).fold(0.0,|acc,x|acc+x)/(times_opt.len() as f64)).sqrt();
        print!("Optimized:    min = {},   max = {},  avg = {},  std = {}\n",min,max,avg,std);
    }






}
