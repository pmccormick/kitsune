/**
 * Implementation of Dirichlet boundary conditions for CFD simulations.
 *
 * This file contains the concrete implementation of the Dirichlet boundary
 * condition class. Dirichlet conditions directly specify values at boundaries
 * and are among the most straightforward boundary conditions to implement in
 * CFD.
 *
 * Implementation notes:
 * 1. Computational stability: Dirichlet conditions generally provide good
 * stability as they define explicit values rather than derivatives.
 * 2. Physical consistency: Care must be taken to ensure physical consistency
 * when setting Dirichlet values (e.g., ensuring mass conservation).
 * 3. Discretization: For collocated grid arrangements, boundary values are
 * directly applied to boundary cells. For staggered grids, velocity components
 * may need special handling.
 *
 * Common applications of Dirichlet boundaries in CFD:
 * - Velocity inlet conditions (prescribed velocity profile)
 * - Wall temperature conditions (isothermal walls)
 * - Pressure outlet conditions (fixed far-field pressure)
 *
 * References:
 * - Gresho, P. M., & Sani, R. L. (1998). Incompressible Flow and the Finite
 * Element Method.
 * - Wesseling, P. (2001). Principles of Computational Fluid Dynamics.
 * - Anderson, J. D. (1995). Computational Fluid Dynamics: The Basics with
 * Applications.
 */

#include "DirichletBoundary.h"

DirichletBoundary::DirichletBoundary(const std::string &name)
    : BoundaryClass(name), m_velocityX(0.0), m_velocityY(0.0), m_pressure(0.0),
      m_temperature(0.0), m_fixVelocityX(false), m_fixVelocityY(false),
      m_fixPressure(false), m_fixTemperature(false) {}

void DirichletBoundary::setVelocityX(double vx) {
  m_velocityX = vx;
  m_fixVelocityX = true;
}

void DirichletBoundary::setVelocityY(double vy) {
  m_velocityY = vy;
  m_fixVelocityY = true;
}

void DirichletBoundary::setPressure(double p) {
  m_pressure = p;
  m_fixPressure = true;
}

void DirichletBoundary::setTemperature(double t) {
  m_temperature = t;
  m_fixTemperature = true;
}

void DirichletBoundary::setVelocityXFunction(
    std::function<double(double, double, double)> func) {
  m_velocityXFunc = func;
  m_fixVelocityX = true;
}

void DirichletBoundary::setVelocityYFunction(
    std::function<double(double, double, double)> func) {
  m_velocityYFunc = func;
  m_fixVelocityY = true;
}

void DirichletBoundary::setPressureFunction(
    std::function<double(double, double, double)> func) {
  m_pressureFunc = func;
  m_fixPressure = true;
}

void DirichletBoundary::setTemperatureFunction(
    std::function<double(double, double, double)> func) {
  m_temperatureFunc = func;
  m_fixTemperature = true;
}

void DirichletBoundary::apply(Cell &cell, const std::vector<Cell *> &neighbors,
                              double x, double y, double dt) {
  // Apply the fixed values or compute them from functions
  if (m_fixVelocityX) {
    if (m_velocityXFunc) {
      cell.setVelocityX(m_velocityXFunc(x, y, dt));
    } else {
      cell.setVelocityX(m_velocityX);
    }
  }

  if (m_fixVelocityY) {
    if (m_velocityYFunc) {
      cell.setVelocityY(m_velocityYFunc(x, y, dt));
    } else {
      cell.setVelocityY(m_velocityY);
    }
  }

  if (m_fixPressure) {
    if (m_pressureFunc) {
      cell.setPressure(m_pressureFunc(x, y, dt));
    } else {
      cell.setPressure(m_pressure);
    }
  }

  if (m_fixTemperature) {
    if (m_temperatureFunc) {
      cell.setTemperature(m_temperatureFunc(x, y, dt));
    } else {
      cell.setTemperature(m_temperature);
    }
  }
}

std::string DirichletBoundary::getType() const { return "Dirichlet"; }