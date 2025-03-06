#pragma once

#include "BoundaryClass.h"
#include <functional>
#include <sstream>
#include <string>

/**
 * @class InflowBoundary
 * @brief Implementation of inflow boundary conditions
 *
 * The inflow boundary condition specifies the values of flow variables at
 * domain inlets where fluid enters. This typically involves setting Dirichlet
 * conditions for velocity components and relevant scalar quantities.
 *
 * This implementation supports both constant and time/space-dependent inflow
 * profiles.
 *
 * Common use cases:
 * - Inlet pipes and channels
 * - Wind tunnel inlets
 * - Jet and fan inflows
 * - Inlet boundary for external flows
 *
 * References:
 * - Versteeg, H. K., & Malalasekera, W. (2007). An Introduction to
 * Computational Fluid Dynamics: The Finite Volume Method. Pearson, Chapter 9.
 * - Tu, J., Yeoh, G. H., & Liu, C. (2018). Computational Fluid Dynamics: A
 * Practical Approach. Butterworth-Heinemann, Chapter 6.
 */
class InflowBoundary : public BoundaryClass {
public:
  // Function types for variable inflow profiles
  using VelocityProfile =
      std::function<double(double, double, double)>; // x, y, t -> value
  using ScalarProfile =
      std::function<double(double, double, double)>; // x, y, t -> value

private:
  // Constant values for each flow variable
  double m_velocityU;
  double m_velocityV;
  double m_pressure;
  double m_temperature;
  double m_density;

  // Variable profile functions
  VelocityProfile m_velocityUProfile;
  VelocityProfile m_velocityVProfile;
  ScalarProfile m_pressureProfile;
  ScalarProfile m_temperatureProfile;
  ScalarProfile m_densityProfile;

  // Flags for which variables use profiles vs. constants
  bool m_useVelocityUProfile;
  bool m_useVelocityVProfile;
  bool m_usePressureProfile;
  bool m_useTemperatureProfile;
  bool m_useDensityProfile;

  // Current simulation time (updated during apply)
  double m_currentTime;

public:
  /**
   * @brief Constructor for constant inflow values
   * @param velocityU Constant u-velocity component at the inflow
   * @param velocityV Constant v-velocity component at the inflow
   * @param pressure Constant pressure at the inflow (often extrapolated)
   * @param temperature Constant temperature at the inflow (if used)
   */
  InflowBoundary(double velocityU = 0.0, double velocityV = 0.0,
                 double pressure = 0.0, double temperature = 0.0);

  /**
   * @brief Apply inflow boundary condition to a cell
   *
   * Sets the specified inflow values for velocity components and other
   * variables, using either constant values or profiles.
   *
   * @param cell The boundary cell to apply the condition to
   * @param x Physical x-coordinate of the cell
   * @param y Physical y-coordinate of the cell
   * @param dt Time step size
   * @param neighbors Vector of non-boundary neighboring cells
   */
  void apply(Cell &cell, double x, double y, double dt,
             const std::vector<Cell *> *neighbors = nullptr) override;

  /**
   * @brief Get the type of the boundary condition
   * @return "Inflow" as the identifier for this boundary type
   */
  std::string getType() const override;

  /**
   * @brief Set a velocity profile function for u-component
   * @param profile Function that takes (x,y,t) and returns velocity value
   */
  void setVelocityUProfile(const VelocityProfile &profile);

  /**
   * @brief Set a velocity profile function for v-component
   * @param profile Function that takes (x,y,t) and returns velocity value
   */
  void setVelocityVProfile(const VelocityProfile &profile);

  /**
   * @brief Set a pressure profile function
   * @param profile Function that takes (x,y,t) and returns pressure value
   */
  void setPressureProfile(const ScalarProfile &profile);

  /**
   * @brief Set a temperature profile function
   * @param profile Function that takes (x,y,t) and returns temperature value
   */
  void setTemperatureProfile(const ScalarProfile &profile);

  /**
   * @brief Set a density profile function
   * @param profile Function that takes (x,y,t) and returns density value
   */
  void setDensityProfile(const ScalarProfile &profile);

  /**
   * @brief Set constant u-velocity value
   * @param velocityU Constant u-velocity value
   */
  void setVelocityU(double velocityU);

  /**
   * @brief Set constant v-velocity value
   * @param velocityV Constant v-velocity value
   */
  void setVelocityV(double velocityV);

  /**
   * @brief Set constant pressure value
   * @param pressure Constant pressure value
   */
  void setPressure(double pressure);

  /**
   * @brief Set constant temperature value
   * @param temperature Constant temperature value
   */
  void setTemperature(double temperature);

  /**
   * @brief Set constant density value
   * @param density Constant density value
   */
  void setDensity(double density);

  /**
   * @brief Update the current simulation time
   * @param time Current simulation time
   */
  void updateTime(double time);

  /**
   * @brief Serialize the inflow boundary condition to a string representation
   * @return String containing serialized boundary data
   */
  std::string serialize() const override {
    // Start with the base class serialization
    std::ostringstream oss;
    oss << BoundaryClass::serialize();

    // Add inflow-specific constant values
    oss << "VELOCITY_U=" << m_velocityU << "\n";
    oss << "VELOCITY_V=" << m_velocityV << "\n";
    oss << "PRESSURE=" << m_pressure << "\n";
    oss << "TEMPERATURE=" << m_temperature << "\n";
    oss << "CURRENT_TIME=" << m_currentTime << "\n";

    // Add flags for which variables use profiles
    oss << "USE_VELOCITY_U_PROFILE=" << (m_useVelocityUProfile ? 1 : 0) << "\n";
    oss << "USE_VELOCITY_V_PROFILE=" << (m_useVelocityVProfile ? 1 : 0) << "\n";
    oss << "USE_PRESSURE_PROFILE=" << (m_usePressureProfile ? 1 : 0) << "\n";
    oss << "USE_TEMPERATURE_PROFILE=" << (m_useTemperatureProfile ? 1 : 0)
        << "\n";
    oss << "USE_DENSITY_PROFILE=" << (m_useDensityProfile ? 1 : 0) << "\n";

    // Note: We cannot serialize std::function objects directly
    // In a real implementation, would need to use function identifiers or
    // a functional expression language

    return oss.str();
  }

  /**
   * @brief Deserialize inflow boundary condition from a string representation
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
      } else if (key == "CURRENT_TIME") {
        m_currentTime = std::stod(value);
      } else if (key == "USE_VELOCITY_U_PROFILE") {
        m_useVelocityUProfile = (std::stoi(value) != 0);
      } else if (key == "USE_VELOCITY_V_PROFILE") {
        m_useVelocityVProfile = (std::stoi(value) != 0);
      } else if (key == "USE_PRESSURE_PROFILE") {
        m_usePressureProfile = (std::stoi(value) != 0);
      } else if (key == "USE_TEMPERATURE_PROFILE") {
        m_useTemperatureProfile = (std::stoi(value) != 0);
      } else if (key == "USE_DENSITY_PROFILE") {
        m_useDensityProfile = (std::stoi(value) != 0);
      }
      // Note: We cannot deserialize the function objects,
      // these would need to be re-set by the calling code
    }

    return true;
  }
};