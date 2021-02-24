use std::env;
use std::time::{Instant};
use ndarray::{Array2,ArrayView2,ArrayViewMut2,s};

//Computes C+=A*B
fn gemm_reference(a : &ArrayView2<f64>,b : &ArrayView2<f64>,c : &mut ArrayViewMut2<f64>) -> () {
    assert_eq!(a.shape()[1],b.shape()[0]);
    assert_eq!(a.shape()[0],c.shape()[0]);
    assert_eq!(b.shape()[1],c.shape()[1]);
    let m=a.shape()[0];
    let k=a.shape()[1];
    let n=b.shape()[1];

    for i in 0..m{
        for j in 0..n{
            //Get i-th row of a
            let ai = a.slice(s![i,..]);
            //Get j-th column of b
            let bj = b.slice(s![..,j]);
            //c(i,j) is dot product of ai,bj
            c[[i,j]] += ai.iter().zip(bj.iter()).map(|(&x,&y)|{x*y}).fold(0.0,|acc,x|{acc+x});
        }
    }
}


fn gemm_optimized(a : &ArrayView2<f64>,b : &ArrayView2<f64>,c : &mut ArrayViewMut2<f64>) -> () {
    gemm_reference(a,b,c);
}


fn main() {
    let args: Vec<String> = env::args().collect();
    let m=args[1].parse::<usize>().unwrap();
    let n=100 as usize;
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
            gemm_reference(&a.view(),&b.view(),&mut c1.view_mut());
            times_ref.push(now.elapsed().as_secs_f64());
        }
        times_ref
    };

    let times_opt = {
        let mut times_opt = Vec::<f64>::new();
        for _ in 0..nruns{
            let now = Instant::now();
            gemm_optimized(&a.view(),&b.view(),&mut c2.view_mut());
            times_opt.push(now.elapsed().as_secs_f64());
        }
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
}
