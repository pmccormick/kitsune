#pragma once

#include "BoundaryClass.h"
#include "Cell.h"
#include <functional>
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
  double m_velocityX;
  double m_velocityY;
  double m_pressure;
  double m_temperature;

  // Flags to determine which variables are fixed
  bool m_fixVelocityX;
  bool m_fixVelocityY;
  bool m_fixPressure;
  bool m_fixTemperature;

  // Optional function-based boundary values
  std::function<double(double, double, double)> m_velocityXFunc;
  std::function<double(double, double, double)> m_velocityYFunc;
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
  void setVelocityX(double vx);

  /**
   * @brief Set the fixed velocity in Y direction
   * @param vy Velocity value to enforce
   */
  void setVelocityY(double vy);

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
  void setVelocityXFunction(std::function<double(double, double, double)> func);

  /**
   * @brief Set a function to compute the Y velocity based on position and time
   * @param func Function taking (x, y, t) and returning velocity
   */
  void setVelocityYFunction(std::function<double(double, double, double)> func);

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
   * @param neighbors Vector of non-boundary neighboring cells
   * @param x Physical x-coordinate of the cell
   * @param y Physical y-coordinate of the cell
   * @param dt Time step size
   */
  void apply(Cell &cell, const std::vector<Cell *> &neighbors, double x,
             double y, double dt) override;

  /**
   * @brief Get the type of the boundary condition
   * @return String identifier for the boundary type
   */
  std::string getType() const override;
};