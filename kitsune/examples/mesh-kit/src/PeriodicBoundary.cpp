#include "PeriodicBoundary.h"
#include "Grid.h"

/**
 * Implementation of periodic boundary conditions for CFD simulations.
 *
 * Periodic boundary conditions are an essential tool in CFD for simulating
 * infinite or repeating domains. This file implements the PeriodicBoundary
 * class that works with the Grid class to enforce physical continuity across
 * non-adjacent boundaries.
 *
 * Theoretical background:
 * In periodic boundary conditions, opposite boundaries of the computational
 * domain are connected so that fluid leaving one boundary enters the domain
 * through the opposite boundary. This creates a virtual infinite domain or a
 * seamless repeating pattern.
 *
 * From a mathematical perspective, periodic BCs transform the domain topology
 * from a rectangular domain to a toroidal one (in 2D) or a hyper-torus (in 3D).
 *
 * Implementation considerations:
 * 1. Conservation: Periodic BCs naturally preserve conservation of mass,
 * momentum, and energy
 * 2. Spectral accuracy: For spectral methods, periodic BCs are the most natural
 * boundary conditions and provide spectral accuracy
 * 3. Pressure solver: Special care is needed in pressure solvers as the domain
 * becomes topologically different, often requiring adjustments to the linear
 * system to ensure uniqueness of the solution
 *
 * Common applications:
 * - Channel flows with fully developed conditions
 * - Homogeneous turbulence studies
 * - Flows over repeating geometries (e.g., tube bundles, porous media)
 * - Simulations requiring statistical homogeneity in one or more directions
 *
 * References:
 * - Pope, S.B. (2000) "Turbulent Flows"
 * - Orszag, S.A. (1971) "Numerical Simulation of Incompressible Flows"
 * - Moin, P. & Mahesh, K. (1998) "Direct Numerical Simulation: A Tool in
 * Turbulence Research"
 * - Kim, J., Moin, P., & Moser, R. (1987) "Turbulence statistics in fully
 * developed channel flow"
 */

/**
 * @brief Constructor for PeriodicBoundary
 *
 * Initializes a periodic boundary condition that pairs with another boundary
 * across the domain.
 *
 * @param name Name identifier for this boundary
 * @param pairedBoundaryName Name of the paired boundary on the opposite side
 * @param direction Direction of periodicity ('x' or 'y')
 * @param xOffset Spatial offset in x-direction between paired boundaries
 * @param yOffset Spatial offset in y-direction between paired boundaries
 */
PeriodicBoundary::PeriodicBoundary(const std::string &name,
                                   const std::string &pairedBoundaryName,
                                   char direction, double xOffset,
                                   double yOffset)
    : BoundaryClass(name), m_pairedBoundaryName(pairedBoundaryName),
      m_xOffset(xOffset), m_yOffset(yOffset), m_grid(nullptr),
      m_direction(direction) {}

/**
 * @brief Get the name of the paired boundary
 * @return String identifier for the paired boundary
 */
std::string PeriodicBoundary::getPairedBoundaryName() const {
  return m_pairedBoundaryName;
}

/**
 * @brief Set the grid object that manages this boundary
 * @param grid Pointer to the Grid object
 */
void PeriodicBoundary::setGrid(Grid *grid) { m_grid = grid; }

/**
 * @brief Get the direction of periodicity
 * @return Direction character ('x' or 'y')
 */
char PeriodicBoundary::getDirection() const { return m_direction; }

/**
 * @brief Get the spatial offsets between paired boundaries
 * @return Pair of (xOffset, yOffset) values
 */
std::pair<double, double> PeriodicBoundary::getOffsets() const {
  return {m_xOffset, m_yOffset};
}

/**
 * @brief Apply the periodic boundary condition
 *
 * This method implements the core functionality of periodic boundary
 * conditions. For each boundary cell, it identifies the corresponding interior
 * cell near the paired boundary and copies its values.
 *
 * Implementation note:
 * The actual identification of paired cells is expected to be performed by the
 * Grid class, which provides the appropriate cells in the neighbors list. This
 * design decouples the boundary condition logic from the grid topology
 * management.
 *
 * The method copies all state variables (velocity, pressure, temperature) to
 * ensure complete physical continuity across the periodic boundary.
 *
 * @param cell The boundary cell to apply the condition to
 * @param neighbors Vector of relevant cells for this boundary condition
 * @param x Physical x-coordinate of the cell
 * @param y Physical y-coordinate of the cell
 * @param dt Time step size
 */
void PeriodicBoundary::apply(Cell &cell, [[maybe_unused]] double x,
                             [[maybe_unused]] double y,
                             [[maybe_unused]] double dt,
                             const std::vector<Cell *> *neighbors) {
  // Without properly paired cells from the Grid, this is a no-op
  if (neighbors == nullptr || neighbors->empty() || !m_grid)
    return;

  // For periodic boundary, we need cells from the opposite boundary
  // The Grid class should provide these in the neighbors list
  // We choose one cell that is not a boundary cell (interior cell near the
  // paired boundary)
  Cell *pairedCell = nullptr;
  for (auto *neighbor : *neighbors) {
    if (!neighbor->isBoundary()) {
      pairedCell = neighbor;
      break;
    }
  }

  // If we found a suitable interior cell, copy its values
  if (pairedCell) {
    // Copy all state variables to ensure physical continuity
    cell.setVelocityU(pairedCell->getVelocityU());
    cell.setVelocityV(pairedCell->getVelocityV());
    cell.setPressure(pairedCell->getPressure());
    cell.setTemperature(pairedCell->getTemperature());

    // Also copy material properties if needed
    if (pairedCell->getMaterial()) {
      cell.setMaterial(pairedCell->getMaterial());
    }
  }
}

/**
 * @brief Get the type identifier for this boundary condition
 * @return String "Periodic" identifying this boundary type
 */
std::string PeriodicBoundary::getType() const { return "Periodic"; }
