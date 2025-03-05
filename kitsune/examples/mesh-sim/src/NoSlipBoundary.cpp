/**
 * Implementation of no-slip boundary conditions for CFD simulations.
 *
 * The no-slip boundary condition is the cornerstone of viscous fluid dynamics,
 * expressing the empirical observation that fluid molecules adhere to solid
 * surfaces. This file implements the NoSlipBoundary class that enforces this
 * condition in numerical simulations.
 *
 * Physical basis:
 * The no-slip condition arises from the dominance of viscous forces at the
 * molecular scale near solid boundaries. Fluid molecules interact with the
 * solid surface through intermolecular forces, effectively "sticking" to the
 * surface. This creates a condition where the fluid velocity matches the wall
 * velocity at the boundary interface.
 *
 * Historical context:
 * While intuitive, the no-slip condition was historically debated, with figures
 * like Navier, Stokes, and Couette contributing to its development. Modern
 * experimental techniques have conclusively validated this condition for
 * most engineering applications, though exceptions exist in rarefied flows
 * and certain non-Newtonian fluids.
 *
 * Implementation considerations:
 * 1. Wall treatment: In high Reynolds number flows, resolving the steep
 * velocity gradients near walls requires extremely fine meshes or wall
 * functions
 * 2. Moving boundaries: For moving walls, the fluid velocity at the boundary
 * must match the prescribed wall velocity
 * 3. Thermal conditions: Often paired with thermal boundary conditions (fixed
 *    temperature or heat flux) for heat transfer simulations
 *
 * Common applications:
 * - Flow in pipes and channels
 * - External aerodynamics (vehicles, buildings)
 * - Heat exchangers and thermal systems
 * - Mixing vessels and reactors
 *
 * References:
 * - Day, M.A. (1990) "The no-slip condition of fluid dynamics"
 * - Lauga, E., Brenner, M.P., & Stone, H.A. (2007) "Microfluidics: The no-slip
 * boundary condition"
 * - Pope, S.B. (2000) "Turbulent Flows"
 */
#include <cassert>

#include "NoSlipBoundary.h"
#include "Cell.h"
#include "Grid.h"
#include "Material.h"
/**
 * @brief Constructor for NoSlipBoundary
 *
 * Initializes a no-slip boundary condition with specified wall velocity.
 *
 * @param name Name identifier for this boundary
 * @param wallVelocityX X-component of wall velocity (default = 0 for stationary
 * wall)
 * @param wallVelocityY Y-component of wall velocity (default = 0 for stationary
 * wall)
 */
NoSlipBoundary::NoSlipBoundary(const std::string &name, double wallVelocityX,
                               double wallVelocityY)
    : BoundaryClass(name), m_wallVelocityX(wallVelocityX),
      m_wallVelocityY(wallVelocityY), m_wallTemperature(0.0), m_heatFlux(0.0),
      m_fixTemperature(false), m_fixHeatFlux(false) {}

/**
 * @brief Set the wall velocity for moving boundaries
 *
 * For moving walls (like a sliding lid, rotating cylinder, etc.),
 * this method specifies the wall velocity components.
 *
 * @param vx X-component of wall velocity
 * @param vy Y-component of wall velocity
 */
void NoSlipBoundary::setWallVelocity(double vx, double vy) {
  m_wallVelocityX = vx;
  m_wallVelocityY = vy;
}

/**
 * @brief Set a function for time-dependent wall velocity in X direction
 *
 * For walls with time-varying motion (oscillating walls, accelerating
 * boundaries, etc.), this method allows specification of a function
 * to compute the x-velocity based on position and time.
 *
 * @param func Function taking (x, y, t) and returning wall velocity
 */
void NoSlipBoundary::setWallVelocityXFunction(
    std::function<double(double, double, double)> func) {
  m_velocityXFunc = func;
}

/**
 * @brief Set a function for time-dependent wall velocity in Y direction
 *
 * @param func Function taking (x, y, t) and returning wall velocity
 */
void NoSlipBoundary::setWallVelocityYFunction(
    std::function<double(double, double, double)> func) {
  m_velocityYFunc = func;
}

/**
 * @brief Set a fixed wall temperature (isothermal boundary)
 *
 * Establishes an isothermal boundary condition where the wall
 * temperature is fixed at the specified value.
 *
 * @param temperature Wall temperature value
 */
void NoSlipBoundary::setWallTemperature(double temperature) {
  m_wallTemperature = temperature;
  m_fixTemperature = true;
  m_fixHeatFlux = false; // Temperature and heat flux are mutually exclusive
}

/**
 * @brief Set a function for time-dependent wall temperature
 *
 * @param func Function taking (x, y, t) and returning wall temperature
 */
void NoSlipBoundary::setWallTemperatureFunction(
    std::function<double(double, double, double)> func) {
  m_temperatureFunc = func;
  m_fixTemperature = true;
  m_fixHeatFlux = false; // Temperature and heat flux are mutually exclusive
}

/**
 * @brief Set a fixed heat flux at the wall
 *
 * Establishes a constant heat flux boundary condition.
 * Positive values indicate heat flowing into the fluid.
 *
 * @param heatFlux Heat flux value (positive = into fluid)
 */
void NoSlipBoundary::setHeatFlux(double heatFlux) {
  m_heatFlux = heatFlux;
  m_fixHeatFlux = true;
  m_fixTemperature = false; // Temperature and heat flux are mutually exclusive
}

/**
 * @brief Set a function for time-dependent heat flux
 *
 * @param func Function taking (x, y, t) and returning heat flux
 */
void NoSlipBoundary::setHeatFluxFunction(
    std::function<double(double, double, double)> func) {
  m_heatFluxFunc = func;
  m_fixHeatFlux = true;
  m_fixTemperature = false; // Temperature and heat flux are mutually exclusive
}

/**
 * @brief Apply the no-slip boundary condition
 *
 * This method implements the no-slip condition by setting the velocity
 * of boundary cells to match the wall velocity. It also handles any
 * associated thermal boundary conditions (temperature or heat flux).
 *
 * Implementation notes:
 * 1. For velocity: Direct imposition of the wall velocity (Dirichlet)
 * 2. For pressure: Typically uses a Neumann zero-gradient condition
 * 3. For temperature:
 *    - If fixed temperature: Direct imposition (Dirichlet)
 *    - If fixed heat flux: Uses neighbors to implement the flux condition
 *
 * The method handles both static values and function-based values that
 * may depend on position and time.
 *
 * @param cell The boundary cell to apply the condition to
 * @param neighbors Vector of non-boundary neighboring cells
 * @param x Physical x-coordinate of the cell
 * @param y Physical y-coordinate of the cell
 * @param dt Time step size
 */
void NoSlipBoundary::apply(Cell &cell, double x, double y, double dt,
                           const std::vector<Cell *> *neighbors) {
  assert(neighbors != nullptr && "NoSlipBoundary::apply() requires a neighbor list!");
  // Apply the no-slip velocity condition
  if (m_velocityXFunc) {
    cell.setVelocityU(m_velocityXFunc(x, y, dt));
  } else {
    cell.setVelocityU(m_wallVelocityX);
  }

  if (m_velocityYFunc) {
    cell.setVelocityV(m_velocityYFunc(x, y, dt));
  } else {
    cell.setVelocityV(m_wallVelocityY);
  }

  // For pressure, use Neumann zero-gradient from interior cells
  if (!neighbors->empty()) {
    double avg_p = 0.0;
    for (const auto &neighbor : *neighbors) {
      avg_p += neighbor->getPressure();
    }
    avg_p /= neighbors->size();
    cell.setPressure(avg_p);
  }

  // Apply thermal boundary condition if specified
  if (m_fixTemperature) {
    if (m_temperatureFunc) {
      cell.setTemperature(m_temperatureFunc(x, y, dt));
    } else {
      cell.setTemperature(m_wallTemperature);
    }
  } else if (m_fixHeatFlux && !neighbors->empty()) {
    // For heat flux boundary condition, we need interior cells
    // to approximate the temperature gradient

    // Get the heat flux value (either constant or function-based)
    double flux = m_heatFluxFunc ? m_heatFluxFunc(x, y, dt) : m_heatFlux;

    // Get the first interior cell for a simple 1st-order approximation
    // In a more sophisticated implementation, we would use multiple interior
    // cells for higher-order approximations
    Cell *interiorCell = (*neighbors)[0];

    // Get the material properties needed for heat flux calculation
    double k = 0.0; // Thermal conductivity
    if (cell.getMaterial()) {
      k = cell.getMaterial()->getThermalConductivity();
    } else if (interiorCell->getMaterial()) {
      k = interiorCell->getMaterial()->getThermalConductivity();
    } else {
      // Default to a reasonable value if no material is set
      k = 0.5; // Approximate value for air
    }

    // Get distance to the interior cell
    // In a real implementation, this would use the actual distance
    double dx = cell.getGrid()->getDx();
    double dy = cell.getGrid()->getDy();
    double distance = std::min(dx, dy);

    // Calculate the wall temperature based on the heat flux condition
    // q = -k * (dT/dn) => Twall = Tinterior + (q*distance/k)
    double wallTemp = interiorCell->getTemperature() + (flux * distance / k);
    cell.setTemperature(wallTemp);
  }
}

/**
 * @brief Get the type identifier for this boundary condition
 * @return String "NoSlip" identifying this boundary type
 */
std::string NoSlipBoundary::getType() const { return "NoSlip"; }
