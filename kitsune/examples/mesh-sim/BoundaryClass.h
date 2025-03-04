
#pragma once

#include <memory>
#include <string>
#include <vector>

// Forward declarations
class Cell;
class Grid;

/**
 * @class BoundaryClass
 * @brief Abstract base class for all boundary conditions in the simulation
 *
 * The BoundaryClass provides a common interface for implementing
 * different types of boundary conditions (e.g. Dirichlet, Neumann,
 * no-slip, etc.) that can be applied to grid cells.
 */
class BoundaryClass {
protected:
  std::string m_name;

public:
  /**
   * @brief Constructor
   * @param name Name of the boundary condition
   */
  BoundaryClass(const std::string &name) : m_name(name) {}

  /**
   * @brief Virtual destructor
   */
  virtual ~BoundaryClass() = default;

  /**
   * @brief Get the name of the boundary condition
   * @return The boundary condition name
   */
  std::string getName() const { return m_name; }

  /**
   * @brief Apply the boundary condition to a cell
   * @param cell The cell to apply the boundary condition to
   * @param neighbors Vector of non-boundary neighboring cells
   * @param x Physical x-coordinate of the cell
   * @param y Physical y-coordinate of the cell
   * @param dt Time step size
   */
  virtual void apply(Cell &cell, const std::vector<Cell *> &neighbors, double x,
                     double y, double dt) = 0;

  /**
   * @brief Get the type of the boundary condition
   * @return String identifier for the boundary type
   */
  virtual std::string getType() const = 0;
};
