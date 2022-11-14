#include <math.h>
#include "euler-types.h"

void compute_flux(int nelr,
                  const int* elements_surrounding_elements,
                  const float* normals,
                  const float* variables,
                  float* fluxes,
                  const float* ff_variable,
                  const Float3 ff_flux_contribution_momentum_x,
                  const Float3 ff_flux_contribution_momentum_y,
                  const Float3 ff_flux_contribution_momentum_z,
                  const Float3 ff_flux_contribution_density_energy) {
  using namespace std;
  const float smoothing_coefficient = 0.2f;

  forall(int blk = 0; blk < nelr/block_length; ++blk) {
    int b_start = blk*block_length;
    int b_end = (blk+1)*block_length > nelr ? nelr : (blk+1)*block_length;

    for(int i = b_start; i < b_end; ++i) {
      float density_i = variables[i + VAR_DENSITY*nelr];
      Float3 momentum_i;
      momentum_i.x = variables[i + (VAR_MOMENTUM+0)*nelr];
      momentum_i.y = variables[i + (VAR_MOMENTUM+1)*nelr];
      momentum_i.z = variables[i + (VAR_MOMENTUM+2)*nelr];

      float density_energy_i = variables[i + VAR_DENSITY_ENERGY*nelr];

      Float3 velocity_i;
      compute_velocity(density_i, momentum_i, velocity_i);
      float speed_sqd_i = compute_speed_sqd(velocity_i);
      float speed_i = sqrtf(speed_sqd_i);
      float pressure_i = compute_pressure(density_i,
                                          density_energy_i,
                                          speed_sqd_i);
      float speed_of_sound_i = compute_speed_of_sound(density_i, pressure_i);
      Float3 flux_contribution_i_momentum_x, flux_contribution_i_momentum_y,
             flux_contribution_i_momentum_z;

      Float3 flux_contribution_i_density_energy;
      compute_flux_contribution(density_i, momentum_i,
                                density_energy_i, pressure_i,
                                velocity_i, flux_contribution_i_momentum_x,
                                flux_contribution_i_momentum_y,
                                flux_contribution_i_momentum_z,
                                flux_contribution_i_density_energy);

      float flux_i_density = 0.0f;
      Float3 flux_i_momentum;
      flux_i_momentum.x = 0.0f;
      flux_i_momentum.y = 0.0f;
      flux_i_momentum.z = 0.0f;
      float flux_i_density_energy = 0.0f;

      Float3 velocity_nb;
      float density_nb, density_energy_nb;
      Float3 momentum_nb;
      Float3 flux_contribution_nb_momentum_x,
             flux_contribution_nb_momentum_y,
             flux_contribution_nb_momentum_z;
      Float3 flux_contribution_nb_density_energy;
      float speed_sqd_nb, speed_of_sound_nb, pressure_nb;

      for(int j = 0; j < NNB; j++) {
        Float3 normal; float normal_len;
        float factor;

        int nb = elements_surrounding_elements[i + j*nelr];
        normal.x = normals[i + (j + 0*NNB)*nelr];
        normal.y = normals[i + (j + 1*NNB)*nelr];
        normal.z = normals[i + (j + 2*NNB)*nelr];
        normal_len = sqrtf(normal.x*normal.x +
                          normal.y*normal.y +
                          normal.z*normal.z);

        if (nb >= 0) { // a legitimate neighbor
          density_nb = variables[nb + VAR_DENSITY*nelr];
          momentum_nb.x = variables[nb + (VAR_MOMENTUM)*nelr];
          momentum_nb.y = variables[nb + (VAR_MOMENTUM+1)*nelr];
          momentum_nb.z = variables[nb + (VAR_MOMENTUM+2)*nelr];
          density_energy_nb = variables[nb + VAR_DENSITY_ENERGY*nelr];
          compute_velocity(density_nb, momentum_nb, velocity_nb);
          speed_sqd_nb = compute_speed_sqd(velocity_nb);
          pressure_nb = compute_pressure(density_nb, density_energy_nb,
                                         speed_sqd_nb);
          speed_of_sound_nb = compute_speed_of_sound(density_nb, pressure_nb);
          compute_flux_contribution(density_nb, momentum_nb, density_energy_nb,
                                    pressure_nb, velocity_nb,
                                    flux_contribution_nb_momentum_x,
                                    flux_contribution_nb_momentum_y,
                                    flux_contribution_nb_momentum_z,
                                    flux_contribution_nb_density_energy);

          // artificial viscosity
          factor = -normal_len*smoothing_coefficient*0.5f *
                        (speed_i + sqrtf(speed_sqd_nb) +
                         speed_of_sound_i + speed_of_sound_nb);
          flux_i_density += factor*(density_i-density_nb);
          flux_i_density_energy += factor*(density_energy_i-density_energy_nb);
          flux_i_momentum.x += factor*(momentum_i.x-momentum_nb.x);
          flux_i_momentum.y += factor*(momentum_i.y-momentum_nb.y);
          flux_i_momentum.z += factor*(momentum_i.z-momentum_nb.z);

          // accumulate cell-centered fluxes
          factor = 0.5f*normal.x;
          flux_i_density += factor*(momentum_nb.x+momentum_i.x);
          flux_i_density_energy += factor*(flux_contribution_nb_density_energy.x
                                    + flux_contribution_i_density_energy.x);
          flux_i_momentum.x += factor*(flux_contribution_nb_momentum_x.x
                                    + flux_contribution_i_momentum_x.x);
          flux_i_momentum.y += factor*(flux_contribution_nb_momentum_y.x
                                    + flux_contribution_i_momentum_y.x);
          flux_i_momentum.z += factor*(flux_contribution_nb_momentum_z.x
                                    + flux_contribution_i_momentum_z.x);

          factor = 0.5f*normal.y;
          flux_i_density += factor*(momentum_nb.y+momentum_i.y);
          flux_i_density_energy += factor*(flux_contribution_nb_density_energy.y
                                    + flux_contribution_i_density_energy.y);
          flux_i_momentum.x += factor*(flux_contribution_nb_momentum_x.y
                                    + flux_contribution_i_momentum_x.y);
          flux_i_momentum.y += factor*(flux_contribution_nb_momentum_y.y
                                    + flux_contribution_i_momentum_y.y);
          flux_i_momentum.z += factor*(flux_contribution_nb_momentum_z.y
                                    + flux_contribution_i_momentum_z.y);

          factor = 0.5f*normal.z;
          flux_i_density += factor*(momentum_nb.z+momentum_i.z);
          flux_i_density_energy += factor*(flux_contribution_nb_density_energy.z
                                    + flux_contribution_i_density_energy.z);
          flux_i_momentum.x += factor*(flux_contribution_nb_momentum_x.z
                                    + flux_contribution_i_momentum_x.z);
          flux_i_momentum.y += factor*(flux_contribution_nb_momentum_y.z
                                    + flux_contribution_i_momentum_y.z);
          flux_i_momentum.z += factor*(flux_contribution_nb_momentum_z.z
                                    + flux_contribution_i_momentum_z.z);
        } else if(nb == -1) { // a wing boundary
          flux_i_momentum.x += normal.x*pressure_i;
          flux_i_momentum.y += normal.y*pressure_i;
          flux_i_momentum.z += normal.z*pressure_i;
        } else if(nb == -2) { // a far field boundary
          factor = 0.5f*normal.x;
          flux_i_density += factor * (ff_variable[VAR_MOMENTUM] + momentum_i.x);
          flux_i_density_energy += factor*(ff_flux_contribution_density_energy.x
                                    + flux_contribution_i_density_energy.x);
          flux_i_momentum.x += factor*(ff_flux_contribution_momentum_x.x
                                    + flux_contribution_i_momentum_x.x);
          flux_i_momentum.y += factor*(ff_flux_contribution_momentum_y.x
                                    + flux_contribution_i_momentum_y.x);
          flux_i_momentum.z += factor*(ff_flux_contribution_momentum_z.x
                                    + flux_contribution_i_momentum_z.x);

          factor = float(0.5f)*normal.y;
          flux_i_density += factor*(ff_variable[VAR_MOMENTUM+1]+momentum_i.y);
          flux_i_density_energy += factor*(ff_flux_contribution_density_energy.y
                                    + flux_contribution_i_density_energy.y);
          flux_i_momentum.x += factor*(ff_flux_contribution_momentum_x.y
                                    + flux_contribution_i_momentum_x.y);
          flux_i_momentum.y += factor*(ff_flux_contribution_momentum_y.y
                                    + flux_contribution_i_momentum_y.y);
          flux_i_momentum.z += factor*(ff_flux_contribution_momentum_z.y
                                    + flux_contribution_i_momentum_z.y);

          factor = float(0.5f)*normal.z;
          flux_i_density += factor*(ff_variable[VAR_MOMENTUM+2]+momentum_i.z);
          flux_i_density_energy += factor*(ff_flux_contribution_density_energy.z
                                    + flux_contribution_i_density_energy.z);
          flux_i_momentum.x += factor*(ff_flux_contribution_momentum_x.z
                                    + flux_contribution_i_momentum_x.z);
          flux_i_momentum.y += factor*(ff_flux_contribution_momentum_y.z
                                    + flux_contribution_i_momentum_y.z);
          flux_i_momentum.z += factor*(ff_flux_contribution_momentum_z.z
                                    + flux_contribution_i_momentum_z.z);
        }
      }

    fluxes[i + VAR_DENSITY*nelr] = flux_i_density;
    fluxes[i + (VAR_MOMENTUM+0)*nelr] = flux_i_momentum.x;
    fluxes[i + (VAR_MOMENTUM+1)*nelr] = flux_i_momentum.y;
    fluxes[i + (VAR_MOMENTUM+2)*nelr] = flux_i_momentum.z;
    fluxes[i + VAR_DENSITY_ENERGY*nelr] = flux_i_density_energy;
    }
  }
}

