#include "euler-types.h"

void compute_step_factor(int nelr,
                        View<float>& __restrict variables,
                        View<float>& areas,
                        View<float>& __restrict step_factors)
{
  variables.sync_device();
  areas.sync_device();
  step_factors.sync_device();
  step_factors.modify_device();

  Kokkos::parallel_for("compute_step_factor", nelr/block_length,
        KOKKOS_LAMBDA(const int &blk) {
    int b_start = blk*block_length;
    int b_end = (blk+1)*block_length > nelr ? nelr : (blk+1)*block_length;

    for(int i = b_start; i < b_end; i++) {
      float density = variables.d_view(i + VAR_DENSITY*nelr);

      Float3 momentum;
      momentum.x = variables.d_view(i + (VAR_MOMENTUM+0)*nelr);
      momentum.y = variables.d_view(i + (VAR_MOMENTUM+1)*nelr);
      momentum.z = variables.d_view(i + (VAR_MOMENTUM+2)*nelr);

      float density_energy = variables.d_view(i + VAR_DENSITY_ENERGY*nelr);
      Float3 velocity;
      compute_velocity(density, momentum, velocity);
      float speed_sqd = compute_speed_sqd(velocity);
      float pressure = compute_pressure(density, density_energy, speed_sqd);
      float speed_of_sound = compute_speed_of_sound(density, pressure);

      // dt = float(0.5f) * sqrt(areas[i]) / (||v|| + c).... but
      // when we do time stepping, this later would need to be divided
      // by the area, so we just do it all at once
      step_factors.d_view(i) = float(0.5f) / (sqrtf(areas.d_view(i)) *
        (sqrtf(speed_sqd) + speed_of_sound));
    }
  });
  Kokkos::fence();
}

