/// Copyright 2009, Andrew Corrigan, acorriga@gmu.edu
// This code is from the AIAA-2009-4001 paper

#include <iostream>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <kitsune.h>
#include <cmath>
#include "euler-types.h"

using namespace std;

void dump(float* variables, int nel, int nelr)
{
  {
    ofstream file("density-forall.dat");
    file << nel << " " << nelr << endl;
    for(int i = 0; i < nel; i++)
      file << variables[i + VAR_DENSITY*nelr] << endl;
  }

  {
    ofstream file("momentum-forall.dat");
    file << nel << " " << nelr << endl;
    for(int i = 0; i < nel; i++) {
    	for(int j = 0; j != NDIM; j++)
        file << variables[i + (VAR_MOMENTUM+j)*nelr] << " ";
    	file << endl;
    }
  }

  {
    ofstream file("density_energy-forall.dat");
    file << nel << " " << nelr << endl;
    for(int i = 0; i < nel; i++)
      file << variables[i + VAR_DENSITY_ENERGY*nelr] << endl;
  }

}


void initialize_variables(int nelr,
                          float* variables,
                          float* ff_variable)
{
  forall(int i = 0; i < nelr; i++) {
    for(int j = 0; j < NVAR; j++)
      variables[i + j*nelr] = ff_variable[j];
  }
}


/*
 * Main function
 */
int main(int argc, char** argv)
{
  using namespace std;

  if (argc < 2) {
    cout << "specify data file name" << endl;
    return 0;
  }

  int iterations = 4000;  
  if (argc > 2)
    iterations = atoi(argv[2]);

  const char* data_file_name = argv[1];

  cout << setprecision(5);
  cout << "\n";
  cout << "---- euler3d benchmark (forall) ----\n\n"
       << "  Input file : " << data_file_name << "\n" 
       << "  Iterations : " << iterations << ".\n\n"; 

      
  cout << "  Reading input data, allocating arrays, initializing data, etc..." 
       << std::flush;

  auto total_start_time = chrono::steady_clock::now();

  // these need to be computed the first time in order to compute time step
    float *ff_variable = alloc<float>(NVAR);
  Float3 ff_flux_contribution_momentum_x,
    ff_flux_contribution_momentum_y,
    ff_flux_contribution_momentum_z;
  Float3 ff_flux_contribution_density_energy;

  // set far field conditions
  const float angle_of_attack = float(3.1415926535897931 / 180.0f) *
    float(deg_angle_of_attack);

  ff_variable[VAR_DENSITY] = float(1.4);

  float ff_pressure = float(1.0f);
  float ff_speed_of_sound = sqrtf(GAMMA*ff_pressure /
                                    ff_variable[VAR_DENSITY]);
  float ff_speed = float(ff_mach)*ff_speed_of_sound;

  Float3 ff_velocity;
  ff_velocity.x = ff_speed*float(cos((float)angle_of_attack));
  ff_velocity.y = ff_speed*float(sin((float)angle_of_attack));
  ff_velocity.z = 0.0f;

  ff_variable[VAR_MOMENTUM+0] = ff_variable[VAR_DENSITY] * ff_velocity.x;
  ff_variable[VAR_MOMENTUM+1] = ff_variable[VAR_DENSITY] * ff_velocity.y;
  ff_variable[VAR_MOMENTUM+2] = ff_variable[VAR_DENSITY] * ff_velocity.z;

  ff_variable[VAR_DENSITY_ENERGY] = ff_variable[VAR_DENSITY]*(float(0.5f)*
                                           (ff_speed*ff_speed)) +
    (ff_pressure / float(GAMMA-1.0f));


  Float3 ff_momentum;
  ff_momentum.x = *(ff_variable+VAR_MOMENTUM+0);
  ff_momentum.y = *(ff_variable+VAR_MOMENTUM+1);
  ff_momentum.z = *(ff_variable+VAR_MOMENTUM+2);
  compute_flux_contribution(ff_variable[VAR_DENSITY], ff_momentum,
                            ff_variable[VAR_DENSITY_ENERGY],
                            ff_pressure, ff_velocity,
                            ff_flux_contribution_momentum_x,
                            ff_flux_contribution_momentum_y,
                            ff_flux_contribution_momentum_z,
                            ff_flux_contribution_density_energy);

  int nel;
  int nelr;

  // read in domain geometry
  float* areas;
  int* elements_surrounding_elements;
  float* normals;

  ifstream file(data_file_name);
  file >> nel;
  nelr = block_length*((nel / block_length )+ min(1, nel % block_length));

  areas = alloc<float>(nelr);
  elements_surrounding_elements = alloc<int>(nelr*NNB);
  normals = alloc<float>(NDIM*NNB*nelr);

  // read in data
  for(int i = 0; i < nel; i++) {
    file >> areas[i];
    for(int j = 0; j < NNB; j++) {
      file >> elements_surrounding_elements[i + j*nelr];
      if (elements_surrounding_elements[i+j*nelr] < 0)
        elements_surrounding_elements[i+j*nelr] = -1;
      // it's coming in with Fortran numbering
      elements_surrounding_elements[i + j*nelr]--;

      for(int k = 0; k < NDIM; k++) {
        file >>  normals[i + (j + k*NNB)*nelr];
        normals[i + (j + k*NNB)*nelr] = -normals[i + (j + k*NNB)*nelr];
      }
    }
  }

  // fill in remaining data
  int last = nel-1;
  for(int i = nel; i < nelr; i++) {
    areas[i] = areas[last];
    for(int j = 0; j < NNB; j++) {
      // duplicate the last element
      elements_surrounding_elements[i + j*nelr] =
              elements_surrounding_elements[last + j*nelr];
      for(int k = 0; k < NDIM; k++)
        normals[i + (j + k*NNB)*nelr] = normals[last + (j + k*NNB)*nelr];
    }
  }

  // Create arrays and set initial conditions
  float* variables = alloc<float>(nelr*NVAR);
  cout << "  done.\n\n";

  cout << "  Starting benchmark...\n" << std::flush;
  auto start_time = chrono::steady_clock::now();
  initialize_variables(nelr, variables, ff_variable);
  float* old_variables = alloc<float>(nelr*NVAR);
  float* fluxes = alloc<float>(nelr*NVAR);
  float* step_factors = alloc<float>(nelr);
  double *rk_times = new double[iterations];

  // Begin iterations
  double copy_total = 0.0;
  double sf_total = 0.0;
  double rk_total = 0.0;

  for(int i = 0; i < iterations; i++) {
    auto copy_start = chrono::steady_clock::now();
    cpy(old_variables, variables, nelr*NVAR);
    auto copy_end = chrono::steady_clock::now();
    double time = chrono::duration<double>(copy_end-copy_start).count();
    copy_total += time;

    // for the first iteration we compute the time step
    auto sf_start = chrono::steady_clock::now();
    compute_step_factor(nelr, variables, areas, step_factors);
    auto sf_end = chrono::steady_clock::now();
    time = chrono::duration<double>(sf_end-sf_start).count();    
    sf_total += time;

    auto rk_start = chrono::steady_clock::now();
    for(int j = 0; j < RK; j++) {
      compute_flux(nelr, elements_surrounding_elements, normals, variables,
                   fluxes, ff_variable,
                   ff_flux_contribution_momentum_x,
                   ff_flux_contribution_momentum_y,
                   ff_flux_contribution_momentum_z,
                   ff_flux_contribution_density_energy);
      time_step(j, nelr, old_variables, variables, step_factors, fluxes);
    }
    auto rk_end = chrono::steady_clock::now();
    time = chrono::duration<double>(rk_end-rk_start).count();
    if (i > 0) {
      rk_times[i] = time;
      rk_total += time;
    }
  }
  
  dump(variables, nel, nelr);

  auto end_time = chrono::steady_clock::now();
  double elapsed_time = chrono::duration<double>(end_time-start_time).count();
  double total_time = chrono::duration<double>(end_time-total_start_time).count();
  double rk_mean = rk_total / (iterations-1);
  double sum = 0.0;
  for(int i = 1; i < iterations; i++) {
    double dist = rk_times[i] - rk_mean;
    sum += dist * dist; 
  }
  double rk_std_dev = sqrt(sum / iterations);

  cout << "\n"
       << "      Total time : " << total_time << " seconds.\n"
       << "    Compute time : " << elapsed_time << " seconds.\n"
       << "            copy : " << copy_total << " seconds (average: " << copy_total / iterations << " seconds).\n"
       << "              sf : " << sf_total << " seconds (average: " << sf_total / iterations << " seconds).\n"
       << "              rk : " << rk_total << " seconds (average: " << rk_mean << " seconds / std dev: " << rk_std_dev << ").\n"
       << "*** " << elapsed_time << ", " << elapsed_time << "\n"                
       << "----\n\n";

  dealloc(ff_variable);
  dealloc(areas);
  dealloc(elements_surrounding_elements);
  dealloc(normals);
  dealloc(variables);
  dealloc(old_variables);
  dealloc(fluxes);
  dealloc(step_factors);

  return 0;
}