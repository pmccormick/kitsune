#include "Kokkos_DualView.hpp"

#include <iostream>
#include <iomanip>
#include <chrono>
#include <cmath>

typedef Kokkos::DualView<float*, Kokkos::LayoutRight, 
                         Kokkos::DefaultExecutionSpace> 
  FloatDualView;

typedef Kokkos::DualView<int*, Kokkos::LayoutRight, 
                         Kokkos::DefaultExecutionSpace> 
  IntDualView;


void random_matrix(FloatDualView &I, int rows, int cols) {
  srand(7);
  using namespace std;
  auto start_time = chrono::steady_clock::now();  
  for(int i = 0 ; i < rows ; i++) {
    for (int j = 0 ; j < cols ; j++) {
      I.h_view(i * cols + j) = rand()/(float)RAND_MAX ;
    }
  }
  I.modify_host();    
  auto end_time = chrono::steady_clock::now();
  double elapsed_time = chrono::duration<double>(end_time-start_time).count();
  cout << "random matrix creation time " << elapsed_time << "\n";
}

void usage(int argc, char **argv) {
  fprintf(stderr,
        "Usage: %s <rows> <cols> <y1> <y2> <x1> <x2> <lambda> <no. of iter>\n",
        argv[0]);
  fprintf(stderr, "\t<rows>        - number of rows\n");
  fprintf(stderr, "\t<cols>        - number of cols\n");
  fprintf(stderr, "\t<y1> 	       - y1 value of the speckle\n");
  fprintf(stderr, "\t<y2>          - y2 value of the speckle\n");
  fprintf(stderr, "\t<x1>          - x1 value of the speckle\n");
  fprintf(stderr, "\t<x2>          - x2 value of the speckle\n");
  fprintf(stderr, "\t<lambda>      - lambda (0,1)\n");
  fprintf(stderr, "\t<no. of iter> - number of iterations\n");
  exit(1); 
}

int main(int argc, char* argv[])
{
  using namespace std;

  int rows, cols, size_I, size_R, niter;
  float q0sqr, sum, sum2, tmp, meanROI,varROI ;
  int r1, r2, c1, c2;
  float lambda;

  if (argc == 9) {
    rows = atoi(argv[1]); //number of rows in the domain
    cols = atoi(argv[2]); //number of cols in the domain
    r1   = atoi(argv[3]); //y1 position of the speckle
    r2   = atoi(argv[4]); //y2 position of the speckle
    c1   = atoi(argv[5]); //x1 position of the speckle
    c2   = atoi(argv[6]); //x2 position of the speckle
    lambda = atof(argv[7]); //Lambda value
    niter = atoi(argv[8]); //number of iterations
  } else if (argc == 1) {
    // run with default configuration...
    rows = 6400;
    cols = 6400;
    r1 = 0;
    r2 = 127;
    c1 = 0;
    c2 = 127;
    lambda = 0.5;
    niter = 100;
  } else {
    usage(argc, argv);
  }

  if ((rows%16!=0) || (cols%16!=0)){
    fprintf(stderr, "rows and cols must be multiples of 16\n");
    exit(1);
  }

  cout << setprecision(5);
  cout << "\n";
  cout << "---- srad benchmark (kokkos) ----\n"
       << "  Row size    : " << rows << ".\n"
       << "  Column size : " << cols << ".\n" 
       << "  Iterations  : " << niter << ".\n\n";

  Kokkos::initialize(argc, argv); {
    size_I = cols * rows;
    size_R = (r2-r1+1)*(c2-c1+1);

    cout << "  Allocating arrays and building random matrix..."     
         << std::flush;

    FloatDualView I  = FloatDualView("I", size_I);
    FloatDualView J  = FloatDualView("J", size_I);
    FloatDualView c  = FloatDualView("c", size_I);
    IntDualView   iN = IntDualView("iN", rows);
    IntDualView   iS = IntDualView("iS", rows);
    IntDualView   jW = IntDualView("jW", cols);
    IntDualView   jE = IntDualView("jE", cols);
    FloatDualView dN = FloatDualView("dN", size_I);
    FloatDualView dS = FloatDualView("dS", size_I);
    FloatDualView dW = FloatDualView("dW", size_I);
    FloatDualView dE = FloatDualView("dE", size_I);

    random_matrix(I, rows, cols);

    cout << "  Starting benchmark...\n" << std::flush;
    auto start_time = chrono::steady_clock::now();    

    iN.modify_device();
    iS.modify_device();
    Kokkos::parallel_for("rows", rows, KOKKOS_LAMBDA(const int &i) {
	iN.d_view(i) = i-1;
	iS.d_view(i) = i+1;
      });
    Kokkos::fence();

    jW.modify_device();
    jE.modify_device();
    Kokkos::parallel_for("cols", cols, KOKKOS_LAMBDA(const int &j) {
	jW.d_view(j) = j-1;
	jE.d_view(j) = j+1;
      });
    Kokkos::fence();

    iN.sync_host();
    iN.modify_host();    
    iN.h_view(0) = 0;

    iS.sync_host();
    iS.modify_host();    
    iS.h_view(rows-1) = rows-1;

    jW.sync_host();
    jW.modify_host();    
    jW.h_view(0) = 0;
    
    jE.sync_host();
    jE.modify_host();    
    jE.h_view(cols-1) = cols-1;
    jE.modify_host();

    I.sync_device();
    J.modify_device();        
    Kokkos::parallel_for("size_I", size_I, KOKKOS_LAMBDA(const int &k) {
	J.d_view(k) = (float)exp(I.d_view(k));
      });
    Kokkos::fence();    

    double loop1_total_time = 0.0;
    double loop2_total_time = 0.0;
    double loop1_max_time = 0.0, loop1_min_time = 1000.0;
    double loop2_max_time = 0.0, loop2_min_time = 1000.0;
    iN.sync_device();
    iS.sync_device();
    jE.sync_device();
    jW.sync_device();
    for (int iter=0; iter < niter; iter++) {
      sum=0; sum2=0;

      J.sync_host();
      for (int i=r1; i<= r2; i++) {
        for (int j=c1; j<= c2; j++) {
          tmp   = J.h_view(i * cols + j);
          sum  += tmp ;
          sum2 += tmp*tmp;
      	}
      }
      meanROI = sum / size_R;
      varROI  = (sum2 / size_R) - meanROI*meanROI;
      q0sqr   = varROI / (meanROI*meanROI);

      auto loop1_start_time = chrono::steady_clock::now();
      c.modify_device();              
      Kokkos::parallel_for("loop1", rows, KOKKOS_LAMBDA(const int &i) {
	  for (int j = 0; j < cols; j++) {
	    int k = i * cols + j;
	    float Jc = J.d_view(k);
	    // directional derivatives
	    dN.d_view(k) = J.d_view(iN.d_view(i) * cols + j) - Jc;
	    dS.d_view(k) = J.d_view(iS.d_view(i) * cols + j) - Jc;
	    dE.d_view(k) = J.d_view(i * cols + jE.d_view(j)) - Jc;	  
	    dW.d_view(k) = J.d_view(i * cols + jW.d_view(j)) - Jc;
	  
	    float G2 = (dN.d_view(k)*dN.d_view(k) + dS.d_view(k)*dS.d_view(k) +
			dW.d_view(k)*dW.d_view(k) + dE.d_view(k)*dE.d_view(k)) /
	               (Jc*Jc);

	    float L = (dN.d_view(k) + dS.d_view(k) + dW.d_view(k) +
		       dE.d_view(k)) / Jc;

	    float num  = (0.5*G2) - ((1.0/16.0)*(L*L));
	    float den  = 1 + (.25*L);
	    float qsqr = num/(den*den);

	    // diffusion coefficient (equ 33)
	    den = (qsqr-q0sqr) / (q0sqr * (1+q0sqr));
	    c.d_view(k) = 1.0 / (1.0+den);
	    // saturate diffusion coefficient
	    if (c.d_view(k) < 0)
	      c.d_view(k) = 0.0;
	    else if (c.d_view(k) > 1)
	      c.d_view(k) = 1.0;
	  }
	});
      Kokkos::fence();

      auto loop1_end_time = chrono::steady_clock::now();
      double etime = chrono::duration<double>(loop1_end_time-loop1_start_time).count(); 
      loop1_total_time += etime;
      if (etime > loop1_max_time)
	loop1_max_time = etime;
      else if (etime < loop1_min_time)
	loop1_min_time = etime;      

      auto loop2_start_time = chrono::steady_clock::now();
      J.modify_device();      
      Kokkos::parallel_for("loop2", rows, KOKKOS_LAMBDA(const int i) {
	  for (int j = 0; j < cols; j++) {
	    // current index
	    int k = i * cols + j;
	    // diffusion coefficient
	    float cN = c.d_view(k);
	    float cS = c.d_view(iS.d_view(i) * cols + j);
	    float cW = c.d_view(k);
	    float cE = c.d_view(i * cols + jE.d_view(j));

	    // divergence (equ 58)
	    float D = cN * dN.d_view(k) + cS * dS.d_view(k) +
	      cW * dW.d_view(k) + cE * dE.d_view(k);
	    // image update (equ 61)
	    J.d_view(k) = J.d_view(k) + 0.25*lambda*D;
	  }
	});
      Kokkos::fence();

      auto loop2_end_time = chrono::steady_clock::now();
      etime = chrono::duration<double>(loop2_end_time-loop2_start_time).count(); 
      loop2_total_time += etime;
      if (etime > loop2_max_time)
	loop2_max_time = etime;
      else if (etime < loop2_min_time)
	loop2_min_time = etime;      
    }
    auto end_time = chrono::steady_clock::now();
    double elapsed_time = chrono::duration<double>(end_time-start_time).count();
    cout << "  Avg. loop 1 time: " << loop1_total_time / niter << "\n"
	 << "       loop 1 min : " << loop1_min_time << "\n"
	 << "       loop 1 max : " << loop1_max_time << "\n"
	 << "  Avg. loop 2 time: " << loop2_total_time / niter << "\n"
	 << "       loop 2 min : " << loop2_min_time << "\n"
	 << "       loop 2 max : " << loop2_max_time << "\n";
    cout << "  Running time: " << elapsed_time << " seconds.\n"    
         << "----\n\n";

    J.sync_host();
    auto V = J.view_host().data();
    FILE *fp = fopen("srad-kokkos.dat", "wb");
    if (fp != NULL) {
      fwrite((void*)V, sizeof(float), size_I, fp);
      fclose(fp);
    }

  } Kokkos::finalize();

  return 0;
}
