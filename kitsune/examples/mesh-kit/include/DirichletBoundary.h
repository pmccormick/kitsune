#pragma once

#include "BoundaryClass.h"
#include "Cell.h"
#include <functional>
#include <sstream>
#include <string>
#include <vector>

/**
 * @class DirichletBoundary
 * @brief Implements Dirichlet (fixed value) boundary conditions
 *
 * Dirichlet boundary conditions (named after Peter Gustav Lejeune Dirichlet,
 * 1805-1859) specify the exact value of variables at the boundary. In
 * mathematical terms, for a PDE defined on domain Ω with boundary ∂Ω, a
 * Dirichlet condition is expressed as: u(x) = g(x) for all x ∈ ∂Ω where u is
 * the variable and g is a known function defining boundary values.
 *
 * In CFD applications, Dirichlet conditions are used to represent:
 * - Inflow boundaries with known velocity profiles
 * - Fixed temperature walls (isothermal)
 * - Fixed pressure outlets
 * - No-slip walls (velocity = 0)
 *
 * This implementation supports:
 * - Constant values for all primary variables (velocity, pressure, temperature)
 * - Function-based values that can vary with position and time
 * - Selective application (can fix some variables while leaving others free)
 *
 * Implementation follows standard CFD discretization practices as described in:
 * - Ferziger, J.H., Perić, M. (2002) "Computational Methods for Fluid Dynamics"
 * - Versteeg, H.K., Malalasekera, W. (2007) "An Introduction to Computational
 * Fluid Dynamics"
 * - Patankar, S.V. (1980) "Numerical Heat Transfer and Fluid Flow"
 *
 * Note: For certain variables (particularly pressure), combining Dirichlet
 * conditions across all boundaries can lead to an over-constrained system. Care
 * should be taken to ensure appropriate combination of boundary condition
 * types.
 */
class DirichletBoundary : public BoundaryClass {
private:
  // Fixed values for boundary variables
  double m_velocityU;
  double m_velocityV;
  double m_pressure;
  double m_temperature;

  // Flags to determine which variables are fixed
  bool m_fixVelocityU;
  bool m_fixVelocityV;
  bool m_fixPressure;
  bool m_fixTemperature;

  // Optional function-based boundary values
  std::function<double(double, double, double)> m_velocityUFunc;
  std::function<double(double, double, double)> m_velocityVFunc;
  std::function<double(double, double, double)> m_pressureFunc;
  std::function<double(double, double, double)> m_temperatureFunc;

public:
  /**
   * @brief Constructor for DirichletBoundary
   * @param name Name of the boundary condition
   */
  DirichletBoundary(const std::string &name);

  /**
   * @brief Set the fixed velocity in X direction
   * @param vx Velocity value to enforce
   */
  void setVelocityU(double vu);

  /**
   * @brief Set the fixed velocity in Y direction
   * @param vy Velocity value to enforce
   */
  void setVelocityV(double vv);

  /**
   * @brief Set the fixed pressure value
   * @param p Pressure value to enforce
   */
  void setPressure(double p);

  /**
   * @brief Set the fixed temperature value
   * @param t Temperature value to enforce
   */
  void setTemperature(double t);

  /**
   * @brief Set a function to compute the X velocity based on position and time
   * @param func Function taking (x, y, t) and returning velocity
   */
  void setVelocityUFunction(std::function<double(double, double, double)> func);

  /**
   * @brief Set a function to compute the Y velocity based on position and time
   * @param func Function taking (x, y, t) and returning velocity
   */
  void setVelocityVFunction(std::function<double(double, double, double)> func);

  /**
   * @brief Set a function to compute the pressure based on position and time
   * @param func Function taking (x, y, t) and returning pressure
   */
  void setPressureFunction(std::function<double(double, double, double)> func);

  /**
   * @brief Set a function to compute the temperature based on position and time
   * @param func Function taking (x, y, t) and returning temperature
   */
  void
  setTemperatureFunction(std::function<double(double, double, double)> func);

  /**
   * @brief Apply the Dirichlet boundary condition
   * @param cell The cell to apply the boundary condition to
   * @param x Physical x-coordinate of the cell
   * @param y Physical y-coordinate of the cell
   * @param dt Time step size
   * @param neighbors Vector of non-boundary neighboring cells
   */
  void apply(Cell &cell, double x, double y, double dt,
             const std::vector<Cell *> *neighbors = nullptr) override;
   
  /**
   * @brief Get the type of the boundary condition
   * @return String identifier for the boundary type
   */
  std::string getType() const override;

  /**
   * @brief Serialize the Dirichlet boundary condition to a string
   * representation
   * @return String containing serialized boundary data
   */
  std::string serialize() const override {
    // Start with the base class serialization
    std::ostringstream oss;
    oss << BoundaryClass::serialize();

    // Add Dirichlet-specific data
    oss << "VELOCITY_U=" << m_velocityU << "\n";
    oss << "VELOCITY_V=" << m_velocityV << "\n";
    oss << "PRESSURE=" << m_pressure << "\n";
    oss << "TEMPERATURE=" << m_temperature << "\n";

    // Add flag information
    oss << "FIX_VELOCITY_U=" << (m_fixVelocityU ? 1 : 0) << "\n";
    oss << "FIX_VELOCITY_V" << (m_fixVelocityV ? 1 : 0) << "\n";
    oss << "FIX_PRESSURE=" << (m_fixPressure ? 1 : 0) << "\n";
    oss << "FIX_TEMPERATURE=" << (m_fixTemperature ? 1 : 0) << "\n";

    // Note about functions
    oss << "HAS_VELOCITY_U_FUNC=" << (m_velocityUFunc ? 1 : 0) << "\n";
    oss << "HAS_VELOCITY_V_FUNC=" << (m_velocityVFunc ? 1 : 0) << "\n";
    oss << "HAS_PRESSURE_FUNC=" << (m_pressureFunc ? 1 : 0) << "\n";
    oss << "HAS_TEMPERATURE_FUNC=" << (m_temperatureFunc ? 1 : 0) << "\n";

    // Note: We cannot serialize std::function objects directly
    // In a real implementation, would need to use function identifiers or
    // a functional expression language

    return oss.str();
  }

  /**
   * @brief Deserialize Dirichlet boundary condition from a string
   * representation
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

      if (key == "VELOCITY_U") {
        m_velocityU = std::stod(value);
      } else if (key == "VELOCITY_V") {
        m_velocityV = std::stod(value);
      } else if (key == "PRESSURE") {
        m_pressure = std::stod(value);
      } else if (key == "TEMPERATURE") {
        m_temperature = std::stod(value);
      } else if (key == "FIX_VELOCITY_U") {
        m_fixVelocityU = (std::stoi(value) != 0);
      } else if (key == "FIX_VELOCITY_V") {
        m_fixVelocityV = (std::stoi(value) != 0);
      } else if (key == "FIX_PRESSURE") {
        m_fixPressure = (std::stoi(value) != 0);
      } else if (key == "FIX_TEMPERATURE") {
        m_fixTemperature = (std::stoi(value) != 0);
      }
      // Note: We cannot deserialize the function objects,
      // these would need to be re-set by the calling code
    }

    return true;
  }
};
