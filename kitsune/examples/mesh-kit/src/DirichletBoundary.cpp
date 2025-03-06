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
    : BoundaryClass(name), m_velocityU(0.0), m_velocityV(0.0), m_pressure(0.0),
      m_temperature(0.0), m_fixVelocityU(false), m_fixVelocityV(false),
      m_fixPressure(false), m_fixTemperature(false) {}

void DirichletBoundary::setVelocityU(double vu) {
  m_velocityU = vu;
  m_fixVelocityU = true;
}

void DirichletBoundary::setVelocityV(double vv) {
  m_velocityV = vv;
  m_fixVelocityV = true;
}

void DirichletBoundary::setPressure(double p) {
  m_pressure = p;
  m_fixPressure = true;
}

void DirichletBoundary::setTemperature(double t) {
  m_temperature = t;
  m_fixTemperature = true;
}

void DirichletBoundary::setVelocityUFunction(
    std::function<double(double, double, double)> func) {
  m_velocityUFunc = func;
  m_fixVelocityU = true;
}

void DirichletBoundary::setVelocityVFunction(
    std::function<double(double, double, double)> func) {
  m_velocityVFunc = func;
  m_fixVelocityV = true;
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

void DirichletBoundary::apply(Cell &cell, double x, double y, double dt,
                              const std::vector<Cell *> *neighbors) {

  // Dirichlet boundaries does not use neighbors...
  // Cheap shortcut to Silence unused parameter warning.
  (void)neighbors;

  // Apply the fixed values or compute them from functions
  if (m_fixVelocityU) {
    if (m_velocityUFunc) {
      cell.setVelocityU(m_velocityUFunc(x, y, dt));
    } else {
      cell.setVelocityV(m_velocityU);
    }
  }

  if (m_fixVelocityV) {
    if (m_velocityVFunc) {
      cell.setVelocityU(m_velocityVFunc(x, y, dt));
    } else {
      cell.setVelocityV(m_velocityV);
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
