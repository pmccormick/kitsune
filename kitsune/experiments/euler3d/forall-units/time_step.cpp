
#include "euler-types.h"

void time_step(int j, int nelr, float* old_variables, float* variables,
               float*  step_factors, float* fluxes)
{
  forall(int blk = 0; blk < nelr/block_length; ++blk) {
    int b_start = blk*block_length;
    int b_end = (blk+1)*block_length > nelr ? nelr : (blk+1)*block_length;
    for(int i = b_start; i < b_end; ++i) {
      float factor = step_factors[i]/float(RK+1-j);
      variables[i + VAR_DENSITY*nelr] = old_variables[i + VAR_DENSITY*nelr]
                  + factor*fluxes[i + VAR_DENSITY*nelr];
      variables[i + (VAR_MOMENTUM+0)*nelr] =
              old_variables[i + (VAR_MOMENTUM+0)*nelr] + factor*fluxes[i
                  + (VAR_MOMENTUM+0)*nelr];
      variables[i + (VAR_MOMENTUM+1)*nelr] = old_variables[i
                  + (VAR_MOMENTUM+1)*nelr] + factor*fluxes[i
                  + (VAR_MOMENTUM+1)*nelr];
      variables[i + (VAR_MOMENTUM+2)*nelr] = old_variables[i
                  + (VAR_MOMENTUM+2)*nelr] + factor*fluxes[i
                  + (VAR_MOMENTUM+2)*nelr];
      variables[i + VAR_DENSITY_ENERGY*nelr] = old_variables[i
                  + VAR_DENSITY_ENERGY*nelr] + factor*fluxes[i
                  + VAR_DENSITY_ENERGY*nelr];
    }
  }
}
