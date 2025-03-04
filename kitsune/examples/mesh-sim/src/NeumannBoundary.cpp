#include "NeumannBoundary.h"
#include "Cell.h" // Include Cell class for implementation

/**
 * Constructor with default zero-gradient
 */
NeumannBoundary::NeumannBoundary(double gradient)
    : BoundaryClass("Neumann Boundary"), m_gradient(gradient) {}

/**
 * Implementation of the Neumann boundary condition
 *
 * This uses a first-order approximation to enforce the specified gradient.
 * For zero-gradient (most common), this means copying values from neighboring
 * cells.
 */
void NeumannBoundary::apply(Cell &cell, const std::vector<Cell *> &neighbors,
                            double x, double y, double dt) {
  // A Neumann boundary needs at least one neighbor to determine the gradient
  if (neighbors.empty()) {
    // Handle error case - not enough neighbors
    // In a real implementation, could throw an exception or log an error
    return;
  }

  // For typical fluid dynamics problems, the Neumann boundary condition
  // requires extrapolation from internal cells

  // Identify the primary interior neighbor (usually the closest one)
  Cell *interiorNeighbor = neighbors[0];

  // Calculate distance between cells
  // Note: In a real implementation, this might use a more sophisticated
  // distance calculation based on cell centers
  double dx = 1.0; // Assume unit distance for simplicity
                   // In production code, would calculate actual distance
                   // between cell centers using x,y coordinates

  // Apply condition to all relevant flow variables
  // For velocity components:
  cell.setVelocityU(interiorNeighbor->getVelocityU() + m_gradient * dx);
  cell.setVelocityV(interiorNeighbor->getVelocityV() + m_gradient * dx);

  // For pressure:
  cell.setPressure(interiorNeighbor->getPressure() + m_gradient * dx);

  // For temperature (if present in the simulation):
  if (cell.hasTemperature() && interiorNeighbor->hasTemperature()) {
    cell.setTemperature(interiorNeighbor->getTemperature() + m_gradient * dx);
  }

  // For any scalar quantities (if present):
  // In a real implementation, you would iterate through all scalar quantities
  // and apply the condition to each one
}

/**
 * Return the type identifier for this boundary
 */
std::string NeumannBoundary::getType() const { return "Neumann"; }

/**
 * Getter for the gradient value
 */
double NeumannBoundary::getGradient() const { return m_gradient; }

/**
 * Setter for the gradient value
 */
void NeumannBoundary::setGradient(double gradient) { m_gradient = gradient; }
