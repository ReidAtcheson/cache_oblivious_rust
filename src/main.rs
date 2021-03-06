use std::env;
use std::time::{Instant};
use ndarray::{Zip,Array2,ArrayView2,ArrayViewMut2,s};
use rayon::join;

//Computes C+=A*B
fn gemm_base(a : ArrayView2<f64>,b : ArrayView2<f64>,mut c : ArrayViewMut2<f64>) -> () {
    assert_eq!(a.shape()[1],b.shape()[0]);
    assert_eq!(a.shape()[0],c.shape()[0]);
    assert_eq!(b.shape()[1],c.shape()[1]);
    let m=a.shape()[0];
    let p=a.shape()[1];
    let n=b.shape()[1];

    for i in 0..m{
        //Get i-th row of c
        let mut ci = c.slice_mut(s![i,..]);
        for k in 0..p{
            //Get k-th row of b
            let bk = b.slice(s![k,..]);
            let aik=a[[i,k]];
            Zip::from(&mut ci).and(&bk).apply(|x,&y|{*x=*x+y*aik});
        }
    }
}


fn gemm_reference(a : ArrayView2<f64>,b : ArrayView2<f64>,mut c : ArrayViewMut2<f64>) -> () {
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
        //Update c11 block
        gemm_reference(a.slice(s![0..m2,0..p2]),b.slice(s![0..p2,0..n2]),c.slice_mut(s![0..m2,0..n2]));
        gemm_reference(a.slice(s![0..m2,p2..p]),b.slice(s![p2..p,0..n2]),c.slice_mut(s![0..m2,0..n2]));
        //Update c12 block
        gemm_reference(a.slice(s![0..m2,0..p2]),b.slice(s![0..p2,n2..n]),c.slice_mut(s![0..m2,n2..n]));
        gemm_reference(a.slice(s![0..m2,p2..p]),b.slice(s![p2..p,n2..n]),c.slice_mut(s![0..m2,n2..n]));
        //Update c21 block
        gemm_reference(a.slice(s![m2..m,0..p2]),b.slice(s![0..p2,0..n2]),c.slice_mut(s![m2..m,0..n2]));
        gemm_reference(a.slice(s![m2..m,p2..p]),b.slice(s![p2..p,0..n2]),c.slice_mut(s![m2..m,0..n2]));
        //Update c22 block
        gemm_reference(a.slice(s![m2..m,0..p2]),b.slice(s![0..p2,n2..n]),c.slice_mut(s![m2..m,n2..n]));
        gemm_reference(a.slice(s![m2..m,p2..p]),b.slice(s![p2..p,n2..n]),c.slice_mut(s![m2..m,n2..n]));
    }
    else{
        gemm_base(a,b,c);
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
            gemm_reference(a.view(),b.view(),c1.view_mut());
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
