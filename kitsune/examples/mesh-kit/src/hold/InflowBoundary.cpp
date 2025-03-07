#include "InflowBoundary.h"
#include "Cell.h"

/**
 * Constructor with constant values
 */
InflowBoundary::InflowBoundary(double velocityU, double velocityV,
                               double pressure, double temperature)
    : BoundaryClass("Inflow Boundary"), m_velocityU(velocityU),
      m_velocityV(velocityV), m_pressure(pressure), m_temperature(temperature),
      m_useVelocityUProfile(false), m_useVelocityVProfile(false),
      m_usePressureProfile(false), m_useTemperatureProfile(false),
      m_currentTime(0.0) {

  // Initialize profile functions with defaults (return constant values)
  m_velocityUProfile =
      [this]([[maybe_unused]] double x, [[maybe_unused]] double y,
             [[maybe_unused]] double t) { return m_velocityU; };
  m_velocityVProfile =
      [this]([[maybe_unused]] double x, [[maybe_unused]] double y,
             [[maybe_unused]] double t) { return m_velocityV; };
  m_pressureProfile = [this]([[maybe_unused]] double x,
                             [[maybe_unused]] double y,
                             [[maybe_unused]] double t) { return m_pressure; };
  m_temperatureProfile =
      [this]([[maybe_unused]] double x, [[maybe_unused]] double y,
             [[maybe_unused]] double t) { return m_temperature; };
}

/**
 * Implementation of the inflow boundary condition
 */
void InflowBoundary::apply(Cell &cell, double x, double y, double dt,
                           const std::vector<Cell *> *neighbors) {

  // For inflow, we primarily use Dirichlet conditions (fixed values)
  // but may need neighbor information for some variables (like pressure)

  // Set velocity components based on profiles or constants
  if (m_useVelocityUProfile) {
    cell.setVelocityU(m_velocityUProfile(x, y, m_currentTime));
  } else {
    cell.setVelocityU(m_velocityU);
  }

  if (m_useVelocityVProfile) {
    cell.setVelocityV(m_velocityVProfile(x, y, m_currentTime));
  } else {
    cell.setVelocityV(m_velocityV);
  }

  // For pressure, different approaches exist:
  // 1. Fixed pressure (Dirichlet)
  // 2. Zero-gradient extrapolation from interior (Neumann)
  // 3. Calculated from velocity profile for special cases

  if (m_usePressureProfile) {
    // Use a specified pressure profile
    cell.setPressure(m_pressureProfile(x, y, m_currentTime));
  } else if (m_pressure != 0.0) {
    // Use a constant pressure value
    cell.setPressure(m_pressure);
  } else if (neighbors != nullptr && !neighbors->empty()) {
    // Zero-gradient extrapolation from interior (common for subsonic inflows)
    cell.setPressure((*neighbors)[0]->getPressure());
  }

  // For temperature (if used in the simulation)
  if (cell.hasTemperature()) {
    if (m_useTemperatureProfile) {
      cell.setTemperature(m_temperatureProfile(x, y, m_currentTime));
    } else {
      cell.setTemperature(m_temperature);
    }
  }

  // For any additional scalar quantities
  // In a real implementation, would iterate through all scalar quantities
  // and apply appropriate conditions

  // Increment time (if needed)
  m_currentTime += dt;
}

/**
 * Return the type identifier for this boundary
 */
std::string InflowBoundary::getType() const { return "Inflow"; }

/**
 * Set a profile function for u-velocity
 */
void InflowBoundary::setVelocityUProfile(const VelocityProfile &profile) {
  m_velocityUProfile = profile;
  m_useVelocityUProfile = true;
}

/**
 * Set a profile function for v-velocity
 */
void InflowBoundary::setVelocityVProfile(const VelocityProfile &profile) {
  m_velocityVProfile = profile;
  m_useVelocityVProfile = true;
}

/**
 * Set a profile function for pressure
 */
void InflowBoundary::setPressureProfile(const ScalarProfile &profile) {
  m_pressureProfile = profile;
  m_usePressureProfile = true;
}

/**
 * Set a profile function for temperature
 */
void InflowBoundary::setTemperatureProfile(const ScalarProfile &profile) {
  m_temperatureProfile = profile;
  m_useTemperatureProfile = true;
}

/**
 * Set constant u-velocity
 */
void InflowBoundary::setVelocityU(double velocityU) {
  m_velocityU = velocityU;
  m_useVelocityUProfile = false;
}

/**
 * Set constant v-velocity
 */
void InflowBoundary::setVelocityV(double velocityV) {
  m_velocityV = velocityV;
  m_useVelocityVProfile = false;
}

/**
 * Set constant pressure
 */
void InflowBoundary::setPressure(double pressure) {
  m_pressure = pressure;
  m_usePressureProfile = false;
}

/**
 * Set constant temperature
 */
void InflowBoundary::setTemperature(double temperature) {
  m_temperature = temperature;
  m_useTemperatureProfile = false;
}

/**
 * Set constant density
 */
void InflowBoundary::setDensity(double density) {
  m_density = density;
  m_useDensityProfile = false;
}

/**
 * Set a profile function for density
 */
void InflowBoundary::setDensityProfile(const ScalarProfile &profile) {
  m_densityProfile = profile;
  m_useDensityProfile = true;
}

/**
 * Update current simulation time
 */
void InflowBoundary::updateTime(double time) { m_currentTime = time; }