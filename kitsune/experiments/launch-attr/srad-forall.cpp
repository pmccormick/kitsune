#include <iostream>
#include <iomanip>
#include <chrono>
#include <cmath>
#include <stdlib.h>
#include <kitsune.h>

void random_matrix(float *I, unsigned int rows, unsigned int cols) {
  srand(7);
  using namespace std;
  auto start_time = chrono::steady_clock::now();
  for(unsigned int i = 0 ; i < rows ; i++) {
   for (unsigned int j = 0 ; j < cols ; j++) {
      I[i*cols+j] = rand()/(float)RAND_MAX;
    }
  }
  
  auto end_time = chrono::steady_clock::now();
  double elapsed_time = chrono::duration<double>(end_time-start_time).count();
  cout << "random matrix creation time " << elapsed_time << "\n";
}

void usage(int argc, char **argv)
{
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
  float *I, *J, q0sqr, sum, sum2, tmp, meanROI,varROI ;
  float Jc, G2, L, num, den, qsqr;
  int *iN,*iS,*jE,*jW;
  float *dN,*dS,*dW,*dE;
  int r1, r2, c1, c2;
  float cN,cS,cW,cE;
  float *c, D;
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
    // run with a default configuration.
    rows = 6400;
    cols = 6400;
    r1 = 0;
    r2 = 127;
    c1 = 0;
    c2 = 127;
    lambda = 0.5;
    niter = 200;
  } else {
    usage(argc, argv);
  }

  if ((rows%16!=0) || (cols%16!=0)){
    fprintf(stderr, "rows and cols must be multiples of 16\n");
    exit(1);
  }

  cout << setprecision(5);
  cout << "\n";
  cout << "---- srad benchmark (forall) ----\n"
       << "  Row size    : " << rows << ".\n"
       << "  Column size : " << cols << ".\n" 
       << "  Iterations  : " << niter << ".\n\n";
       
  cout << "  Allocating arrays and building random matrix..." 
       << std::flush;

  size_I = cols * rows;
  size_R = (r2-r1+1)*(c2-c1+1);

  I = alloc<float>(size_I);
  J = alloc<float>(size_I);
  c = alloc<float>(size_I);
  iN = alloc<int>(rows);
  iS = alloc<int>(rows);
  jW = alloc<int>(cols);
  jE = alloc<int>(cols);
  dN = alloc<float>(size_I);
  dS = alloc<float>(size_I);
  dW = alloc<float>(size_I);
  dE = alloc<float>(size_I);


  // Right now this initialization hides a lot of other details 
  // (due to the slow performance of rand() on certain systems).
  // So we do this before we start the timer...
  random_matrix(I, rows, cols);
  cout << "  done.\n\n";  
    
  cout << "  Starting benchmark...\n" << std::flush;  
  auto start_time = chrono::steady_clock::now();
  [[kitsune::launch(8)]]
  forall(int i = 0; i < rows; i++) {
    iN[i] = i-1;
    iS[i] = i+1;
  }

  [[kitsune::launch(8)]]
  forall(int j=0; j < cols; j++) {
    jW[j] = j-1;
    jE[j] = j+1;
  }

  iN[0] = 0;
  iS[rows-1] = rows-1;
  jW[0] = 0;
  jE[cols-1] = cols-1;

  [[kitsune::launch(64)]]
  forall(int k = 0;  k < size_I; k++)
    J[k] = (float)exp(I[k]) ;

  double loop1_total_time = 0.0;
  double loop2_total_time = 0.0;  
  double loop1_max_time = 0.0, loop1_min_time = 1000.0;
  double loop2_max_time = 0.0, loop2_min_time = 1000.0;
  
  for (int iter=0; iter < niter; iter++) {
    sum=0; sum2=0;

    for(int i=r1; i <= r2; i++) {
      for(int j = c1; j<=c2; j++) {
        tmp   = J[i * cols + j];
        sum  += tmp ;
        sum2 += tmp*tmp;
      }
    }
    meanROI = sum / size_R;
    varROI  = (sum2 / size_R) - meanROI*meanROI;
    q0sqr   = varROI / (meanROI*meanROI);

    auto loop1_start_time = chrono::steady_clock::now();
    [[kitsune::launch(16)]]
    forall(int i = 0 ; i < rows; i++) {
      
      for(int j = 0; j < cols; j++) {
        int k = i * cols + j;
        float Jc = J[k];
        // directional derivatives
        dN[k] = J[iN[i] * cols + j] - Jc;
        dS[k] = J[iS[i] * cols + j] - Jc;
        dE[k] = J[i * cols + jE[j]] - Jc;
        dW[k] = J[i * cols + jW[j]] - Jc;

        float G2 = (dN[k]*dN[k] + dS[k]*dS[k] +
                    dW[k]*dW[k] + dE[k]*dE[k]) / (Jc*Jc);

        float L = (dN[k] + dS[k] + dW[k] + dE[k]) / Jc;

        float num  = (0.5f*G2) - ((1.0f/16.0f)*(L*L)) ;
        float den  = 1 + (.25f*L);
        float qsqr = num/(den*den);

        // diffusion coefficient (equ 33)
        den = (qsqr-q0sqr) / (q0sqr * (1+q0sqr)) ;
        c[k] = 1.0f / (1.0f+den) ;

        // saturate diffusion coefficient
        if (c[k] < 0)
          c[k] = 0.0;
        else if (c[k] > 1)
          c[k] = 1.0;
      }
    }
    auto loop1_end_time = chrono::steady_clock::now();
    double etime = chrono::duration<double>(loop1_end_time - loop1_start_time).count(); 
    //cout << "\t- loop 1 time: " << etime << "\n";
    loop1_total_time += etime;
    if (etime > loop1_max_time)
      loop1_max_time = etime;
    else if (etime < loop1_min_time)
      loop1_min_time = etime;

    auto loop2_start_time = chrono::steady_clock::now();
    [[kitsune::launch(8)]]
    forall(int i = 0; i < rows; i++) {
      for(int j = 0; j < cols; j++) {
        // current index
        int k = i * cols + j;
        // diffusion coefficient
        float cN = c[k];
        float cS = c[iS[i] * cols + j];
        float cW = c[k];
        float cE = c[i * cols + jE[j]];
        // divergence (equ 58)
        float D = cN * dN[k] + cS * dS[k] + cW * dW[k] + cE * dE[k];
        // image update (equ 61)
        J[k] = J[k] + 0.25*lambda*D;
      }
    }
    auto loop2_end_time = chrono::steady_clock::now();
    etime = chrono::duration<double>(loop2_end_time - loop2_start_time).count(); 
    //cout << "\t- loop 2 time: " << etime << "\n";
    loop2_total_time += etime;
    if (etime > loop2_max_time)
      loop2_max_time = etime;
    else if (etime < loop2_min_time)
      loop2_min_time = etime;
  }
  auto end_time = chrono::steady_clock::now();
  double elapsed_time = chrono::duration<double>(end_time - start_time).count();
  
  cout << "  Avg. loop 1 time: " << loop1_total_time / niter << "\n"
       << "       loop 1 min : " << loop1_min_time << "\n"
       << "       loop 1 max : " << loop1_max_time << "\n"
       << "  Avg. loop 2 time: " << loop2_total_time / niter << "\n"
       << "       loop 2 min : " << loop2_min_time << "\n"
       << "       loop 2 max : " << loop2_max_time << "\n";
  cout << "  Running time: " << elapsed_time << " seconds.\n"
       << "*** " << elapsed_time << ", " << elapsed_time << "\n"
       << "----\n\n";

  FILE *fp = fopen("srad-forall.dat", "wb");
  if (fp != NULL) {
    fwrite((void*)J, sizeof(float), size_I, fp);
    fclose(fp);
  }
  return 0;
}
