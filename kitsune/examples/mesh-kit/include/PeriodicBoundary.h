#pragma once

#include "BoundaryClass.h"
#include "Cell.h"
#include <memory>
#include <sstream>
#include <string>
#include <vector>

/**
 * @class PeriodicBoundary
 * @brief Implements periodic boundary conditions for CFD simulations
 *
 * Periodic boundary conditions enforce continuity across non-adjacent
 * boundaries, effectively simulating an infinite domain by connecting opposite
 * edges of the computational domain. Mathematically, for a domain with
 * boundaries ∂Ω1 and ∂Ω2, periodic conditions enforce: u(x∈∂Ω1) = u(x'∈∂Ω2)
 * where x and x' are corresponding points
 *
 * In fluid dynamics, periodic boundaries are used to simulate:
 * - Fully developed flow in channels and pipes
 * - Homogeneous turbulence in idealized domains
 * - Infinitely repeating geometries like array of obstacles or heat exchangers
 * - Flow problems where the inlet and outlet flow statistics are identical
 *
 * This implementation follows a paired boundary approach where:
 * - Each periodic boundary is paired with another boundary at the opposite edge
 * - The boundary values are copied from corresponding interior cells near the
 * paired boundary
 * - Special care is required at corners where two periodic boundaries intersect
 *
 * Key implementation details:
 * - Requires that grid dimensions match at the paired boundaries
 * - Must be applied to complete boundaries (not partial segments)
 * - Grid class handles the actual pairing of cells via neighbor lists
 *
 * Computational Considerations:
 * - Maintains mass, momentum, and energy conservation across domain boundaries
 * - May require special treatment in pressure solvers (domain becomes
 * topologically different)
 * - Allows for significant computational savings by modeling infinite/repeating
 * domains
 *
 * References:
 * - Canuto, C. et al. (2007) "Spectral Methods: Evolution to Complex Geometries
 * and Applications to Fluid Dynamics"
 * - Patankar, S.V. (1980) "Numerical Heat Transfer and Fluid Flow"
 * - Hirsch, C. (2007) "Numerical Computation of Internal and External Flows"
 */
class PeriodicBoundary : public BoundaryClass {
private:
  // Name of the paired boundary at the opposite edge
  std::string m_pairedBoundaryName;

  // Physical distance between periodic boundaries (for handling domains with
  // different sizes)
  double m_xOffset;
  double m_yOffset;

  // Grid object managing the paired cells (set by Grid class)
  Grid *m_grid;

  // Direction of periodicity ('x' or 'y')
  char m_direction;

public:
  /**
   * @brief Constructor for PeriodicBoundary
   * @param name Name of this boundary condition
   * @param pairedBoundaryName Name of the paired boundary at the opposite edge
   * @param direction Direction of periodicity ('x' for left-right or 'y' for
   * top-bottom)
   * @param xOffset Optional x-offset between paired points (for stretched
   * domains)
   * @param yOffset Optional y-offset between paired points (for stretched
   * domains)
   */
  PeriodicBoundary(const std::string &name,
                   const std::string &pairedBoundaryName, char direction = 'x',
                   double xOffset = 0.0, double yOffset = 0.0);

  /**
   * @brief Get the name of the paired boundary
   * @return The paired boundary name
   */
  std::string getPairedBoundaryName() const;

  /**
   * @brief Set the grid object that manages the cell pairings
   * @param grid Pointer to the Grid object
   */
  void setGrid(Grid *grid);

  /**
   * @brief Get the direction of periodicity
   * @return Direction character ('x' or 'y')
   */
  char getDirection() const;

  /**
   * @brief Get the spatial offsets between paired boundaries
   * @return Pair of (xOffset, yOffset) values
   */
  std::pair<double, double> getOffsets() const;

  /**
   * @brief Apply the periodic boundary condition
   *
   * This method applies periodic boundary conditions by copying values from
   * corresponding interior cells near the paired boundary. The actual cell
   * pairing is expected to be managed by the Grid class, which provides the
   * appropriate neighbor cells.
   *
   * @param cell The boundary cell to apply the condition to
   * @param neighbors Vector of non-boundary neighboring cells (includes cells
   * from paired boundary)
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
   * @brief Serialize the periodic boundary condition to a string representation
   * @return String containing serialized boundary data
   */
  std::string serialize() const override {
    // Start with the base class serialization
    std::ostringstream oss;
    oss << BoundaryClass::serialize();

    // Add periodic boundary-specific data
    oss << "PAIRED_BOUNDARY_NAME=" << m_pairedBoundaryName << "\n";
    oss << "DIRECTION=" << m_direction << "\n";
    oss << "X_OFFSET=" << m_xOffset << "\n";
    oss << "Y_OFFSET=" << m_yOffset << "\n";

    // Note: We cannot serialize the Grid pointer
    // That would need to be reset separately after deserialization

    return oss.str();
  }

  /**
   * @brief Deserialize periodic boundary condition from a string representation
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

      if (key == "PAIRED_BOUNDARY_NAME") {
        m_pairedBoundaryName = value;
      } else if (key == "DIRECTION") {
        if (!value.empty()) {
          m_direction = value[0]; // Take first character as direction
        }
      } else if (key == "X_OFFSET") {
        m_xOffset = std::stod(value);
      } else if (key == "Y_OFFSET") {
        m_yOffset = std::stod(value);
      }
    }

    // Note: Grid pointer must be reset separately
    m_grid = nullptr;

    return true;
  }
};
