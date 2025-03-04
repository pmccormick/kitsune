#include "SlipBoundary.h"
#include "Cell.h"    // Include Cell class for implementation
#include <algorithm> // For std::transform
#include <cctype>    // For std::toupper
#include <stdexcept> // For std::invalid_argument

/**
 * Constructor with orientation setting
 */
SlipBoundary::SlipBoundary(const std::string &orientation)
    : BoundaryClass("Slip Boundary"), m_orientation(Orientation::AUTO) {

  // Set orientation if provided
  if (orientation != "AUTO") {
    setOrientation(orientation);
  }
}

/**
 * Implementation of the slip boundary condition
 */
void SlipBoundary::apply(Cell &cell, const std::vector<Cell *> &neighbors,
                         double x, double y, double dt) {
  // A slip boundary needs at least one neighbor to determine the values
  if (neighbors.empty()) {
    // Handle error case - not enough neighbors
    return;
  }

  // Determine orientation if set to AUTO
  Orientation orientation = m_orientation;
  if (orientation == Orientation::AUTO) {
    orientation = detectOrientation(neighbors);
  }

  // Get the interior neighbor
  Cell *interiorNeighbor = neighbors[0];

  // Handle slip condition based on the orientation
  switch (orientation) {
  case Orientation::NORTH:
    // North boundary: preserve u (tangential), negate v (normal)
    cell.setVelocityU(
        interiorNeighbor->getVelocityU()); // Preserve tangential component
    cell.setVelocityV(
        -interiorNeighbor->getVelocityV()); // Reflect normal component
    break;

  case Orientation::SOUTH:
    // South boundary: preserve u (tangential), negate v (normal)
    cell.setVelocityU(
        interiorNeighbor->getVelocityU()); // Preserve tangential component
    cell.setVelocityV(
        -interiorNeighbor->getVelocityV()); // Reflect normal component
    break;

  case Orientation::EAST:
    // East boundary: negate u (normal), preserve v (tangential)
    cell.setVelocityU(
        -interiorNeighbor->getVelocityU()); // Reflect normal component
    cell.setVelocityV(
        interiorNeighbor->getVelocityV()); // Preserve tangential component
    break;

  case Orientation::WEST:
    // West boundary: negate u (normal), preserve v (tangential)
    cell.setVelocityU(
        -interiorNeighbor->getVelocityU()); // Reflect normal component
    cell.setVelocityV(
        interiorNeighbor->getVelocityV()); // Preserve tangential component
    break;

  default:
    // Should not happen if orientation is properly set
    // In a real implementation, throw an exception or log an error
    break;
  }

  // For pressure and other scalars, typically use a Neumann condition (zero
  // gradient)
  cell.setPressure(interiorNeighbor->getPressure());

  // For temperature (if present in the simulation):
  if (cell.hasTemperature() && interiorNeighbor->hasTemperature()) {
    cell.setTemperature(interiorNeighbor->getTemperature());
  }

  // For density (crucial for compressible flows)
  if (cell.hasDensity() && interiorNeighbor->hasDensity()) {
    cell.setDensity(interiorNeighbor->getDensity());
  }

  // Similarly for other scalar quantities
}

/**
 * Return the type identifier for this boundary
 */
std::string SlipBoundary::getType() const { return "Slip"; }

/**
 * Detect the orientation based on the pattern of neighbors
 */
SlipBoundary::Orientation
SlipBoundary::detectOrientation(const std::vector<Cell *> &neighbors) const {

  // This is a simplified implementation that assumes a structured grid
  // In a real implementation, you would need to analyze the relative positions
  // of the neighbors to determine the boundary orientation

  // For simplicity, we'll check if there's a clear dominant direction
  // In a more complete implementation, you would check the x,y positions

  // Example implementation (simplified):
  if (neighbors.size() > 0) {
    // Check neighbor positions relative to the boundary cell
    // This requires knowing the grid structure and cell positions
    // For this example, we'll return a default orientation
    return Orientation::NORTH;
  }

  // Default fallback
  return Orientation::NORTH;
}

/**
 * Getter for orientation
 */
SlipBoundary::Orientation SlipBoundary::getOrientation() const {
  return m_orientation;
}

/**
 * Setter for orientation from string
 */
void SlipBoundary::setOrientation(const std::string &orientation) {
  // Convert input to uppercase for case-insensitive comparison
  std::string upperOrientation = orientation;
  std::transform(upperOrientation.begin(), upperOrientation.end(),
                 upperOrientation.begin(), ::toupper);

  if (upperOrientation == "NORTH") {
    m_orientation = Orientation::NORTH;
  } else if (upperOrientation == "SOUTH") {
    m_orientation = Orientation::SOUTH;
  } else if (upperOrientation == "EAST") {
    m_orientation = Orientation::EAST;
  } else if (upperOrientation == "WEST") {
    m_orientation = Orientation::WEST;
  } else if (upperOrientation == "AUTO") {
    m_orientation = Orientation::AUTO;
  } else {
    throw std::invalid_argument("Invalid orientation: " + orientation);
  }
}
