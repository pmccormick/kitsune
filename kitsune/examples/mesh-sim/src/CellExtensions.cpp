/**
 * ====================================================================
 * Standard Property Computation Functions
 * ====================================================================
 *
 * This file contains implementations of standard property computation
 * functions for common CFD derived quantities: kinetic energy, vorticity,
 * and divergence.
 *
 * These functions are designed to be registered with the Cell class using
 * the PropertyComputeFunction interface.
 */

#include "Cell.h"
#include <cmath>

/**
 * @brief Register standard property computations for common CFD properties
 */
/**
 * ====================================================================
 * Standard Property Computation Functions
 * ====================================================================
 *
 * This file contains implementations of standard property computation
 * functions for common CFD derived quantities: kinetic energy, vorticity,
 * and divergence.
 *
 * These functions are designed to be registered with the Cell class using
 * the PropertyComputeFunction interface.
 */

#include "Cell.h"
#include "Material.h"
#include <cmath>

/**
 * @brief Register standard property computations for common CFD properties
 */
/**
 * ====================================================================
 * Standard Property Computation Functions
 * ====================================================================
 *
 * This file contains implementations of standard property computation
 * functions for common CFD derived quantities: kinetic energy, vorticity,
 * and divergence.
 */

#include "Cell.h"
#include "Material.h"
#include <cmath>

/**
 * @brief Register standard property computations for common CFD properties
 */
void registerStandardPropertyComputations() {
  // Kinetic Energy Computation
  // Kinetic energy per unit mass (J/kg or m²/s²) = 0.5 * |v|²
  Cell::registerPropertyComputation(
      Cell::PropertyType::KINETIC_ENERGY,
      [](Cell &cell, const std::array<Cell *, 4> *) {
        double vx = cell.getVelocityU();
        double vy = cell.getVelocityV();
        double ke = 0.5 * (vx * vx + vy * vy);
        cell.setKineticEnergy(ke);
      });

  // Vorticity Computation
  // Vorticity (1/s) = ∂v/∂x - ∂u/∂y (curl of velocity field in 2D)
  Cell::registerPropertyComputation(
      Cell::PropertyType::VORTICITY,
      [](Cell &cell, const std::array<Cell *, 4> *neighbors) {
        if (!neighbors)
          return;

        // Neighbor indices: 0=left, 1=right, 2=bottom, 3=top
        Cell *left = (*neighbors)[0];
        Cell *right = (*neighbors)[1];
        Cell *bottom = (*neighbors)[2];
        Cell *top = (*neighbors)[3];

        if (left && right && bottom && top) {
          // Use central difference approximation for derivatives
          // Assuming uniform grid spacing of 1.0 (will be scaled by actual dx,
          // dy in Grid)
          double dv_dx = (right->getVelocityV() - left->getVelocityV()) / 2.0;
          double du_dy = (top->getVelocityU() - bottom->getVelocityU()) / 2.0;

          // Vorticity = ∂v/∂x - ∂u/∂y (curl in 2D)
          double vorticity = dv_dx - du_dy;
          cell.setVorticity(vorticity);
        }
      });

  // Divergence Computation
  // Divergence (1/s) = ∂u/∂x + ∂v/∂y (should be ~0 for incompressible flow)
  Cell::registerPropertyComputation(
      Cell::PropertyType::DIVERGENCE,
      [](Cell &cell, const std::array<Cell *, 4> *neighbors) {
        if (!neighbors)
          return;

        Cell *left = (*neighbors)[0];
        Cell *right = (*neighbors)[1];
        Cell *bottom = (*neighbors)[2];
        Cell *top = (*neighbors)[3];

        if (left && right && bottom && top) {
          // Central difference approximation
          double du_dx = (right->getVelocityU() - left->getVelocityU()) / 2.0;
          double dv_dy = (top->getVelocityV() - bottom->getVelocityV()) / 2.0;

          // Divergence = ∂u/∂x + ∂v/∂y
          double divergence = du_dx + dv_dy;
          cell.setDivergence(divergence);
        }
      });

  // Pressure Correction Computation
  // For SIMPLE/PISO algorithms, often based on divergence
  Cell::registerPropertyComputation(
      Cell::PropertyType::PRESSURE_CORRECTION,
      [](Cell &cell, const std::array<Cell *, 4> *neighbors) {
        if (!neighbors)
          return;

        // First ensure divergence is computed
        Cell::PropertyComputeFunction divFunc =
            Cell::getPropertyComputation(Cell::PropertyType::DIVERGENCE);
        divFunc(cell, neighbors);

        // Simple pressure correction proportional to divergence
        double divergence = cell.getDivergence();
        double density = cell.getDensity();
        double pressureCorrection = -divergence * density;

        cell.setPressureCorrection(pressureCorrection);
      });

  // Shear Stress Computation
  // For simple 2D flow
  // Shear Stress Computation
  // For simple 2D flow
  Cell::registerPropertyComputation(
      Cell::PropertyType::SHEAR_STRESS,
      [](Cell &cell, const std::array<Cell *, 4> *neighbors) {
        if (!neighbors)
          return;

        Cell *left = (*neighbors)[0];
        Cell *right = (*neighbors)[1];
        Cell *bottom = (*neighbors)[2];
        Cell *top = (*neighbors)[3];

        if (left && right && bottom && top) {
          // Compute velocity gradients for strain rate tensor
          double du_dx = (right->getVelocityU() - left->getVelocityU()) / 2.0;
          double du_dy = (top->getVelocityU() - bottom->getVelocityU()) / 2.0;
          double dv_dx = (right->getVelocityV() - left->getVelocityV()) / 2.0;
          double dv_dy = (top->getVelocityV() - bottom->getVelocityV()) / 2.0;

          // Get viscosity from material
          double viscosity = 1.0; // Default if no material
          if (cell.getMaterial()) {
            viscosity = cell.getMaterial()->getProperty(
                Material::MaterialProperty::DYNAMIC_VISCOSITY);
          }

          // For 2D flow, compute the full shear stress tensor
          // τxx = 2μ∂u/∂x
          // τyy = 2μ∂v/∂y
          // τxy = τyx = μ(∂u/∂y + ∂v/∂x)

          // Compute the magnitude of the shear stress
          // |τ| = √(τxx² + 2τxy² + τyy²)
          double tau_xx = 2.0 * viscosity * du_dx;
          double tau_yy = 2.0 * viscosity * dv_dy;
          double tau_xy = viscosity * (du_dy + dv_dx);

          double shearStressMagnitude = std::sqrt(
              tau_xx * tau_xx + 2.0 * tau_xy * tau_xy + tau_yy * tau_yy);

          cell.setShearStress(shearStressMagnitude);
        }
      });

  // Heat Flux Computation
  // For simple 2D heat transfer
  Cell::registerPropertyComputation(
      Cell::PropertyType::HEAT_FLUX_X,
      [](Cell &cell, const std::array<Cell *, 4> *neighbors) {
        if (!neighbors)
          return;

        Cell *left = (*neighbors)[0];
        Cell *right = (*neighbors)[1];

        if (left && right) {
          // Heat flux = -k * dT/dx
          double conductivity = 0.6; // Default if no material (~ air)

          // Get actual thermal conductivity if material is available
          if (cell.getMaterial()) {
            // Use the Material class's proper getter
            conductivity = cell.getMaterial()->getProperty(
                Material::MaterialProperty::THERMAL_CONDUCTIVITY);
          }

          // Compute temperature gradient
          double dT_dx =
              (right->getTemperature() - left->getTemperature()) / 2.0;

          // Compute heat flux (W/m²)
          double heatFlux = -conductivity * dT_dx;
          cell.setHeatFluxX(heatFlux);
        }
      });

  Cell::registerPropertyComputation(
      Cell::PropertyType::HEAT_FLUX_Y,
      [](Cell &cell, const std::array<Cell *, 4> *neighbors) {
        if (!neighbors)
          return;

        Cell *bottom = (*neighbors)[2];
        Cell *top = (*neighbors)[3];

        if (bottom && top) {
          // Heat flux = -k * dT/dy
          double conductivity = 0.6; // Default if no material (~ air)

          // Get actual thermal conductivity if material is available
          if (cell.getMaterial()) {
            // Use the Material class's proper getter
            conductivity = cell.getMaterial()->getProperty(
                Material::MaterialProperty::THERMAL_CONDUCTIVITY);
          }

          // Compute temperature gradient
          double dT_dy =
              (top->getTemperature() - bottom->getTemperature()) / 2.0;

          // Compute heat flux (W/m²)
          double heatFlux = -conductivity * dT_dy;
          cell.setHeatFluxY(heatFlux);
        }
      });
}


/**
 * @brief Compute all derivable properties for a cell
 * @param cell The cell to compute properties for
 * @param neighbors The neighboring cells (left, right, bottom, top)
 */
void computeAllCellProperties(Cell &cell,
                              const std::array<Cell *, 4> &neighbors) {
  // First compute properties that don't depend on other derived properties
  cell.computeDerivedProperties(&neighbors);

  // If there are properties that depend on other derived properties,
  // we could add a second pass here
}
