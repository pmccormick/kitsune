#pragma once

#include "BoundaryClass.h"
#include "Cell.h"
#include <functional>
#include <sstream>
#include <string>
#include <vector>

/**
 * @class NoSlipBoundary
 * @brief Implements no-slip boundary conditions for solid walls in CFD
 * simulations
 *
 * No-slip boundary conditions represent the physical behavior of viscous fluids
 * at solid walls, where fluid velocity matches the wall velocity due to viscous
 * effects. This is one of the most fundamental boundary conditions in viscous
 * flow simulations.
 *
 * Mathematically, for a fluid domain Ω with a solid boundary ∂Ω moving at
 * velocity uₘ: u(x) = uₘ  for all x ∈ ∂Ω
 *
 * For stationary walls, this simplifies to:
 *     u(x) = 0  for all x ∈ ∂Ω
 *
 * In fluid dynamics, no-slip conditions are used to simulate:
 * - Stationary solid walls (buildings, obstacles, containment vessels)
 * - Moving solid boundaries (mixers, pumps, turbine blades)
 * - Viscous layers near solid surfaces (boundary layers)
 *
 * This implementation supports:
 * - Zero velocity at stationary walls (classic no-slip)
 * - Prescribed wall velocity for moving boundaries
 * - Optional thermal boundary conditions (fixed temperature or heat flux)
 * - Time-dependent wall velocities and temperatures
 *
 * Theoretical justification:
 * The no-slip condition is empirically validated for most macroscopic flows and
 * theoretically derived from molecular dynamics at the microscopic scale. It
 * breaks down only at extremely low pressures (rarefied gases) or for certain
 * non-Newtonian fluids exhibiting slip behavior.
 *
 * Implementation considerations:
 * - Often combined with wall functions for high-Reynolds turbulent flow
 * - May require special treatment in immersed boundary or cut-cell methods
 * - Critical for accurate boundary layer development and separation prediction
 *
 * References:
 * - Batchelor, G.K. (2000) "An Introduction to Fluid Dynamics"
 * - Schlichting, H. & Gersten, K. (2017) "Boundary-Layer Theory"
 * - White, F.M. (2011) "Fluid Mechanics"
 * - Panton, R.L. (2013) "Incompressible Flow"
 */
class NoSlipBoundary : public BoundaryClass {
private:
  // Wall velocity components (zero for stationary walls)
  double m_wallVelocityX;
  double m_wallVelocityY;

  // Optional thermal condition
  double m_wallTemperature;
  double m_heatFlux;
  bool m_fixTemperature;
  bool m_fixHeatFlux;

  // Function-based boundary values for time-dependent conditions
  std::function<double(double, double, double)> m_velocityXFunc;
  std::function<double(double, double, double)> m_velocityYFunc;
  std::function<double(double, double, double)> m_temperatureFunc;
  std::function<double(double, double, double)> m_heatFluxFunc;

public:
  /**
   * @brief Constructor for NoSlipBoundary
   * @param name Name of the boundary condition
   * @param wallVelocityX X-component of wall velocity (default = 0 for
   * stationary wall)
   * @param wallVelocityY Y-component of wall velocity (default = 0 for
   * stationary wall)
   */
  NoSlipBoundary(const std::string &name, double wallVelocityX = 0.0,
                 double wallVelocityY = 0.0);

  /**
   * @brief Set the wall velocity for moving boundaries
   * @param vx X-component of wall velocity
   * @param vy Y-component of wall velocity
   */
  void setWallVelocity(double vx, double vy);

  /**
   * @brief Set a function for time-dependent wall velocity in X direction
   * @param func Function taking (x, y, t) and returning wall velocity
   */
  void
  setWallVelocityXFunction(std::function<double(double, double, double)> func);

  /**
   * @brief Set a function for time-dependent wall velocity in Y direction
   * @param func Function taking (x, y, t) and returning wall velocity
   */
  void
  setWallVelocityYFunction(std::function<double(double, double, double)> func);

  /**
   * @brief Set a fixed wall temperature (isothermal boundary)
   * @param temperature Wall temperature value
   */
  void setWallTemperature(double temperature);

  /**
   * @brief Set a function for time-dependent wall temperature
   * @param func Function taking (x, y, t) and returning wall temperature
   */
  void setWallTemperatureFunction(
      std::function<double(double, double, double)> func);

  /**
   * @brief Set a fixed heat flux at the wall
   * @param heatFlux Heat flux value (positive = into fluid)
   */
  void setHeatFlux(double heatFlux);

  /**
   * @brief Set a function for time-dependent heat flux
   * @param func Function taking (x, y, t) and returning heat flux
   */
  void setHeatFluxFunction(std::function<double(double, double, double)> func);

  /**
   * @brief Apply the no-slip boundary condition
   * @param cell The cell to apply the boundary condition to
   * @param neighbors Vector of non-boundary neighboring cells
   * @param x Physical x-coordinate of the cell
   * @param y Physical y-coordinate of the cell
   * @param dt Time step size
   */
  void apply(Cell &cell, double x, double y, double dt,
             const std::vector<Cell *> *neighbors = nullptr) override;

  /**
   * @brief Get the type of the boundary condition
   * @return String identifier for the boundary type
   */
  std::string getType() const override;

  /**
   * @brief Serialize the NoSlip boundary condition to a string representation
   * @return String containing serialized boundary data
   */
  std::string serialize() const override {
    // Start with the base class serialization
    std::ostringstream oss;
    oss << BoundaryClass::serialize();

    // Add NoSlip-specific data
    oss << "WALL_VELOCITY_X=" << m_wallVelocityX << "\n";
    oss << "WALL_VELOCITY_Y=" << m_wallVelocityY << "\n";

    // Add thermal conditions
    oss << "WALL_TEMPERATURE=" << m_wallTemperature << "\n";
    oss << "HEAT_FLUX=" << m_heatFlux << "\n";
    oss << "FIX_TEMPERATURE=" << (m_fixTemperature ? 1 : 0) << "\n";
    oss << "FIX_HEAT_FLUX=" << (m_fixHeatFlux ? 1 : 0) << "\n";

    // Note about functions
    oss << "HAS_VELOCITY_X_FUNC=" << (m_velocityXFunc ? 1 : 0) << "\n";
    oss << "HAS_VELOCITY_Y_FUNC=" << (m_velocityYFunc ? 1 : 0) << "\n";
    oss << "HAS_TEMPERATURE_FUNC=" << (m_temperatureFunc ? 1 : 0) << "\n";
    oss << "HAS_HEAT_FLUX_FUNC=" << (m_heatFluxFunc ? 1 : 0) << "\n";

    // Note: We cannot serialize std::function objects directly
    // In a real implementation, would need to use function identifiers or
    // a functional expression language

    return oss.str();
  }

  /**
   * @brief Deserialize NoSlip boundary condition from a string representation
   * @param data String containing serialized boundary data
   * @return True if deserialization was successful
   */
  bool deserialize(const std::string &data) override {
    // First call the base class deserialize method
    if (!BoundaryClass::deserialize(data)) {
      return false;
    }

    // Process the data line by line
    std::istringstream iss(data);
    std::string line;

    while (std::getline(iss, line)) {
      size_t pos = line.find('=');
      if (pos == std::string::npos) {
        continue;
      }

      std::string key = line.substr(0, pos);
      std::string value = line.substr(pos + 1);

      if (key == "WALL_VELOCITY_X") {
        m_wallVelocityX = std::stod(value);
      } else if (key == "WALL_VELOCITY_Y") {
        m_wallVelocityY = std::stod(value);
      } else if (key == "WALL_TEMPERATURE") {
        m_wallTemperature = std::stod(value);
      } else if (key == "HEAT_FLUX") {
        m_heatFlux = std::stod(value);
      } else if (key == "FIX_TEMPERATURE") {
        m_fixTemperature = (std::stoi(value) != 0);
      } else if (key == "FIX_HEAT_FLUX") {
        m_fixHeatFlux = (std::stoi(value) != 0);
      }
      // Note: We cannot deserialize the function objects,
      // these would need to be re-set by the calling code
    }

    return true;
  }
};