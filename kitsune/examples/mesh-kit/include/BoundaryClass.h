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
  virtual void apply(Cell &cell, double x, double y, double dt, 
                     const std::vector<Cell *> *neighbors = nullptr) = 0;

  /**
   * @brief Get the type of the boundary condition
   * @return String identifier for the boundary type
   */
  virtual std::string getType() const = 0;

  /**
   * @brief Serialize the boundary condition to a string representation
   * @return String containing serialized boundary data
   */
  virtual std::string serialize() const {
    // Base implementation just saves the name and type
    return "NAME=" + m_name + "\nTYPE=" + getType() + "\n";
  }

  /**
   * @brief Deserialize boundary condition from a string representation
   * @param data String containing serialized boundary data
   * @return True if deserialization was successful
   */
  virtual bool deserialize(const std::string &data) {
    // Base implementation extracts name
    size_t namePos = data.find("NAME=");
    if (namePos != std::string::npos) {
      size_t valueStart = namePos + 5; // "NAME=".length()
      size_t valueEnd = data.find('\n', valueStart);
      if (valueEnd != std::string::npos) {
        m_name = data.substr(valueStart, valueEnd - valueStart);
        return true;
      }
    }
    return false;
  }
};
