#include "euler-types.h"

void time_step(int j, int nelr,
               View<float>& old_variables,
               View<float>& variables,
               View<float>& step_factors,
               View<float>& fluxes)
{
  old_variables.sync_device();
  variables.sync_device();
  step_factors.sync_device();
  fluxes.sync_device();
  variables.modify_device();
  Kokkos::parallel_for("time_step", nelr/block_length,
    KOKKOS_LAMBDA(const int &blk) {
    int b_start = blk*block_length;
    int b_end = (blk+1)*block_length > nelr ? nelr : (blk+1)*block_length;
    for(int i = b_start; i < b_end; ++i) {
      float factor = step_factors.d_view(i)/float(RK+1-j);
      variables.d_view(i + VAR_DENSITY*nelr) =
                old_variables.d_view(i + VAR_DENSITY*nelr) +
                factor*fluxes.d_view(i + VAR_DENSITY*nelr);
      variables.d_view(i + (VAR_MOMENTUM+0)*nelr) =
                old_variables.d_view(i + (VAR_MOMENTUM+0)*nelr) +
                factor*fluxes.d_view(i + (VAR_MOMENTUM+0)*nelr);
      variables.d_view(i + (VAR_MOMENTUM+1)*nelr) =
                old_variables.d_view(i + (VAR_MOMENTUM+1)*nelr) +
                factor*fluxes.d_view(i + (VAR_MOMENTUM+1)*nelr);
      variables.d_view(i + (VAR_MOMENTUM+2)*nelr) =
                old_variables.d_view(i + (VAR_MOMENTUM+2)*nelr) +
                factor*fluxes.d_view(i + (VAR_MOMENTUM+2)*nelr);
      variables.d_view(i + VAR_DENSITY_ENERGY*nelr) =
                old_variables.d_view(i + VAR_DENSITY_ENERGY*nelr) +
                factor*fluxes.d_view(i + VAR_DENSITY_ENERGY*nelr);
    }
  });
  Kokkos::fence();
}
