#pragma once

#include <memory>
#include <string>
#include <vector>

// Forward declarations
class Cell;
class Grid;

/**
 * @enum BoundaryType
 * @brief Enumerates all possible boundary condition types in the simulation
 *
 * This enum provides type-safe identification of boundary conditions and
 * improves performance over string comparisons.
 */
enum class BoundaryType {
  UNKNOWN,
  DIRICHLET, // Fixed value boundary
  NEUMANN,   // Fixed gradient boundary
  INFLOW,    // Inflow boundary
  NO_SLIP,   // No-slip wall boundary
  SLIP,      // Slip wall boundary
  PERIODIC,  // Periodic boundary
  OUTFLOW,   // Outflow boundary
  SYMMETRY   // Symmetry boundary
};

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
   * @brief Get the type of the boundary condition as an enum
   * @return Enum value identifying the boundary type
   */
  virtual BoundaryType getTypeEnum() const = 0;

  /**
   * @brief Get the type of the boundary condition as a string
   * @return String identifier for the boundary type
   * @deprecated Use getTypeEnum() for better performance and type safety
   */
  std::string getType() const { return boundaryTypeToString(getTypeEnum()); }

  /**
   * @brief Serialize the boundary condition to a string representation
   * @return String containing serialized boundary data
   */
  virtual std::string serialize() const {
    // Base implementation just saves the name and type
    return "NAME=" + m_name + "\nTYPE=" + boundaryTypeToString(getTypeEnum()) +
           "\n";
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

  /**
   * @brief Convert a BoundaryType enum to its string representation
   * @param type The boundary type enum to convert
   * @return String representation of the boundary type
   */
  static std::string boundaryTypeToString(BoundaryType type) {
    switch (type) {
    case BoundaryType::DIRICHLET:
      return "Dirichlet";
    case BoundaryType::NEUMANN:
      return "Neumann";
    case BoundaryType::INFLOW:
      return "Inflow";
    case BoundaryType::NO_SLIP:
      return "NoSlip";
    case BoundaryType::SLIP:
      return "Slip";
    case BoundaryType::PERIODIC:
      return "Periodic";
    case BoundaryType::OUTFLOW:
      return "Outflow";
    case BoundaryType::SYMMETRY:
      return "Symmetry";
    case BoundaryType::UNKNOWN:
    default:
      return "Unknown";
    }
  }

  /**
   * @brief Convert a string to its corresponding BoundaryType enum
   * @param typeStr The string to convert
   * @return BoundaryType enum corresponding to the string
   */
  static BoundaryType stringToBoundaryType(const std::string &typeStr) {
    if (typeStr == "Dirichlet")
      return BoundaryType::DIRICHLET;
    if (typeStr == "Neumann")
      return BoundaryType::NEUMANN;
    if (typeStr == "Inflow")
      return BoundaryType::INFLOW;
    if (typeStr == "NoSlip")
      return BoundaryType::NO_SLIP;
    if (typeStr == "Slip")
      return BoundaryType::SLIP;
    if (typeStr == "Periodic")
      return BoundaryType::PERIODIC;
    if (typeStr == "Outflow")
      return BoundaryType::OUTFLOW;
    if (typeStr == "Symmetry")
      return BoundaryType::SYMMETRY;
    return BoundaryType::UNKNOWN;
  }
};