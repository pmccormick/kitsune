#ifndef EULER_TYPES_H_
#define EULER_TYPES_H_
#include <math.h>
#include <kitsune.h>

#define block_length 1
/*
 * Options
 *
 */
#define GAMMA 1.4
#define ITERATIONS 2000
#define NDIM 3
#define NNB 4
#define RK 3	// 3rd order RK
#define ff_mach 1.2
#define deg_angle_of_attack 0.0f

/*
 * not options
 */
#define VAR_DENSITY 0
#define VAR_MOMENTUM  1
#define VAR_DENSITY_ENERGY (VAR_MOMENTUM+NDIM)
#define NVAR (VAR_DENSITY_ENERGY+1)


#ifdef restrict
#define __restrict restrict
#else
#define __restrict
#endif

struct Float3 {
  float x, y, z;
};

template <typename T>
inline __attribute__((always_inline))
void cpy(T* dst, T* src, int N) {
  forall(unsigned int i = 0; i < N; i++)
    dst[i] = src[i];
}

void time_step(int j, int nelr, float* old_variables, float* variables,
		      float*  step_factors, float* fluxes);
void compute_flux(int nelr, const int* elements_surrounding_elements,
		  const float* normals,  const float* variables,
		  float* fluxes, const float* ff_variable,
		  const Float3 ff_flux_contribution_momentum_x,
		  const Float3 ff_flux_contribution_momentum_y,
		  const Float3 ff_flux_contribution_momentum_z,
		  const Float3 ff_flux_contribution_density_energy);
void compute_step_factor(int nelr,
			 const float* __restrict variables,
			 const float* areas,
			 float* __restrict step_factors);

inline __attribute__((always_inline))
void compute_flux_contribution(const float density,
                               const Float3& momentum,
                               const float density_energy,
                               const float pressure,
                               Float3& velocity,
                               Float3& fc_momentum_x,
                               Float3& fc_momentum_y,
                               Float3& fc_momentum_z,
                               Float3& fc_density_energy) {
  fc_momentum_x.x = velocity.x*momentum.x + pressure;
  fc_momentum_x.y = velocity.x*momentum.y;
  fc_momentum_x.z = velocity.x*momentum.z;

  fc_momentum_y.x = fc_momentum_x.y;
  fc_momentum_y.y = velocity.y*momentum.y + pressure;
  fc_momentum_y.z = velocity.y*momentum.z;

  fc_momentum_z.x = fc_momentum_x.z;
  fc_momentum_z.y = fc_momentum_y.z;
  fc_momentum_z.z = velocity.z*momentum.z + pressure;

  float de_p = density_energy+pressure;
  fc_density_energy.x = velocity.x*de_p;
  fc_density_energy.y = velocity.y*de_p;
  fc_density_energy.z = velocity.z*de_p;
}

inline __attribute__((always_inline))
void compute_velocity(float density,
                      const Float3& momentum,
                      Float3& velocity)
{
  velocity.x = momentum.x / density;
  velocity.y = momentum.y / density;
  velocity.z = momentum.z / density;
}

inline __attribute__((always_inline))
float compute_speed_sqd(const Float3 &velocity)
{
  return velocity.x*velocity.x +
         velocity.y*velocity.y +
         velocity.z*velocity.z;
}

inline __attribute__((always_inline))
float compute_pressure(float density,
                       float density_energy,
                       float speed_sqd)
{
  return (float(GAMMA)-float(1.0f))*(density_energy - float(0.5f)*density*speed_sqd);
}

inline __attribute__((always_inline))
float compute_speed_of_sound(float density, float pressure)
{
  return sqrtf(float(GAMMA)*pressure/density);
}

#endif
