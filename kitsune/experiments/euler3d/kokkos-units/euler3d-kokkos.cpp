/// Copyright 2009, Andrew Corrigan, acorriga@gmu.edu
// This code is from the AIAA-2009-4001 paper

#include <iostream>
#include <fstream>
#include <chrono>
#include <cmath>

#include "euler-types.h"

using namespace std;


void dump(View<float> &variables, int nel, int nelr)
{
  variables.sync_host();

  {
    std::ofstream file("density-kokkos.dat");
    file << nel << " " << nelr << endl;
    for(int i = 0; i < nel; i++)
      file << variables.h_view(i + VAR_DENSITY*nelr) << endl;
  }

  {
    std::ofstream file("momentum-kokkos.dat");
    file << nel << " " << nelr << endl;
    for(int i = 0; i < nel; i++) {
      for(int j = 0; j != NDIM; j++)
        file << variables.h_view(i + (VAR_MOMENTUM+j)*nelr) << " ";
      file << endl;
    }
  }

  {
    ofstream file("density_energy-kokkos.dat");
    file << nel << " " << nelr << endl;
    for(int i = 0; i < nel; i++)
      file << variables.h_view(i + VAR_DENSITY_ENERGY*nelr) << endl;
  }

}

void initialize_variables(int nelr,
                          View<float> & variables,
                          View<float> &ff_variable)
{
  variables.sync_device();
  ff_variable.sync_device();
  Kokkos::parallel_for("initialize_variables", nelr,
        KOKKOS_LAMBDA(const int &i) {
    for(int j = 0; j < NVAR; j++)
      variables.d_view(i + j*nelr) = ff_variable.d_view(j);
  });
  Kokkos::fence();
  variables.modify_device();
}

/*
 * Main function
 */
int main(int argc, char** argv)
{
  if (argc < 2) {
    cout << "specify data file name" << endl;
    return 1;
  }

  int iterations = ITERATIONS;
  if (argc > 2)
    iterations = atoi(argv[2]);

  const char* data_file_name = argv[1];

  // these need to be computed the first time in order to compute time step


  Kokkos::initialize(argc, argv); {

    View<float> ff_variable("ff_variable", NVAR);
    Float3 ff_flux_contribution_momentum_x,
           ff_flux_contribution_momentum_y,
           ff_flux_contribution_momentum_z;
    Float3 ff_flux_contribution_density_energy;

    // set far field conditions
    const float angle_of_attack =
          float(3.1415926535897931 / 180.0f) * float(deg_angle_of_attack);

    ff_variable.h_view(VAR_DENSITY) = float(1.4);

    float ff_pressure = float(1.0f);
    float ff_speed_of_sound =
          sqrtf(GAMMA * ff_pressure / ff_variable.h_view(VAR_DENSITY));
    float ff_speed = float(ff_mach) * ff_speed_of_sound;

    Float3 ff_velocity;
    ff_velocity.x = ff_speed * float(cos((float)angle_of_attack));
    ff_velocity.y = ff_speed * float(sin((float)angle_of_attack));
    ff_velocity.z = 0.0f;

    ff_variable.h_view(VAR_MOMENTUM + 0) =
        ff_variable.h_view(VAR_DENSITY) * ff_velocity.x;
    ff_variable.h_view(VAR_MOMENTUM + 1) =
        ff_variable.h_view(VAR_DENSITY) * ff_velocity.y;
    ff_variable.h_view(VAR_MOMENTUM + 2) =
        ff_variable.h_view(VAR_DENSITY) * ff_velocity.z;

    ff_variable.h_view(VAR_DENSITY_ENERGY) =
        ff_variable.h_view(VAR_DENSITY) *
            (float(0.5f) * (ff_speed * ff_speed)) +
        (ff_pressure / float(GAMMA - 1.0f));

    ff_variable.modify_host();

    Float3 ff_momentum;
    ff_momentum.x = ff_variable.h_view(VAR_MOMENTUM);
    ff_momentum.y = ff_variable.h_view(VAR_MOMENTUM + 1);
    ff_momentum.z = ff_variable.h_view(VAR_MOMENTUM + 2);
    compute_flux_contribution(ff_variable.h_view(VAR_DENSITY), ff_momentum,
          ff_variable.h_view(VAR_DENSITY_ENERGY), ff_pressure, ff_velocity,
          ff_flux_contribution_momentum_x, ff_flux_contribution_momentum_y,
          ff_flux_contribution_momentum_z, ff_flux_contribution_density_energy);

    int nel;
    int nelr;

    // read in domain geometry
    ifstream file(data_file_name);
    file >> nel;
    nelr = block_length * ((nel / block_length) + min(1, nel % block_length));
    View<float> areas("areas", nelr);
    View<int> elements_surrounding_elements("elements_surrounding_elements",
                                            nelr * NNB);
    View<float> normals("normals", NDIM * NNB * nelr);

    // read in data
    for (int i = 0; i < nel; i++) {
      file >> areas.h_view(i);
      for (int j = 0; j < NNB; j++) {
        file >> elements_surrounding_elements.h_view(i + j * nelr);
        if (elements_surrounding_elements.h_view(i + j * nelr) < 0)
          elements_surrounding_elements.h_view(i + j * nelr) = -1;
        // it's coming in with Fortran numbering
        elements_surrounding_elements.h_view(i + j * nelr)--;

        for (int k = 0; k < NDIM; k++) {
          file >> normals.h_view(i + (j + k * NNB) * nelr);
          normals.h_view(i + (j + k*NNB)*nelr) =
                      -normals.h_view(i + (j + k*NNB)*nelr);
        }
      }
    }
    normals.modify_host();
    elements_surrounding_elements.modify_host();

    int last = nel-1;
    for(int i = nel; i < nelr; i++) {

      areas.h_view(i) = areas.h_view(last);

      for(int j = 0; j < NNB; j++) {
        // duplicate the last element
        elements_surrounding_elements.h_view(i + j*nelr) =
                elements_surrounding_elements.h_view(last + j*nelr);

        for(int k = 0; k < NDIM; k++)
          normals.h_view(i + (j + k*NNB)*nelr) =
                normals.h_view(last + (j + k*NNB)*nelr);
      }
    }
    normals.modify_host();
    areas.modify_host();
    elements_surrounding_elements.modify_host();

    // Create arrays and set initial conditions
    View<float> variables = View<float>("variables", nelr * NVAR);
    cout << iterations << " "; 

    auto start = chrono::steady_clock::now();
    initialize_variables(nelr, variables, ff_variable);

    View<float> old_variables = View<float>("old_variables", nelr*NVAR);
    View<float> fluxes = View<float>("fluxes", nelr*NVAR);
    View<float> step_factors = View<float>("step_factors", nelr);

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
      sf_total += chrono::duration<double>(sf_end-sf_start).count();

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
      rk_total += chrono::duration<double>(rk_end-rk_start).count();
    }

    auto end = chrono::steady_clock::now();
    cout << copy_total << " "
	 << sf_total << " "
	 << rk_total << " "
	 << chrono::duration<double>(end-start).count() << endl;
    dump(variables, nel, nelr);
  } Kokkos::finalize();

  return 0;
}
