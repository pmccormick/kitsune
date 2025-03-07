#pragma once

#include "BoundaryClass.h"
#include <sstream>
#include <string>

/**
 * @class SlipBoundary
 * @brief Implementation of slip (free-slip) boundary conditions
 *
 * The slip boundary condition enforces zero normal velocity component while
 * allowing for tangential velocity (no-friction) at the boundary. This means
 * fluid can move freely along the boundary but cannot penetrate it.
 *
 * Common use cases:
 * - Symmetry planes
 * - Inviscid flow simulations
 * - Frictionless walls in ideal flow
 * - Free surface in some simplified cases
 *
 * References:
 * - Anderson, J. D. (1995). Computational Fluid Dynamics: The Basics with
 * Applications. McGraw-Hill, Chapter 7.
 * - Versteeg, H. K., & Malalasekera, W. (2007). An Introduction to
 * Computational Fluid Dynamics: The Finite Volume Method. Pearson, Chapter 9.
 */
class SlipBoundary : public BoundaryClass {
private:
  enum class Orientation {
    NORTH,
    SOUTH,
    EAST,
    WEST,
    AUTO // Automatically determine orientation from neighbors
  };

  Orientation m_orientation;

public:
  /**
   * @brief Constructor
   * @param orientation Optional manual specification of the boundary
   * orientation (default = AUTO for automatic detection)
   */
  SlipBoundary(const std::string &orientation = "AUTO");

  /**
   * @brief Apply slip boundary condition to a cell
   *
   * Sets the cell velocity components to enforce zero normal velocity while
   * preserving tangential velocity. Other quantities like pressure typically
   * use a Neumann (zero-gradient) condition.
   *
   * @param cell The boundary cell to apply the condition to
   * @param neighbors Vector of non-boundary neighboring cells
   * @param x Physical x-coordinate of the cell
   * @param y Physical y-coordinate of the cell
   * @param dt Time step size
   */
  void apply(Cell &cell, double x, double y, double dt,
             const std::vector<Cell *> *neighbors = nullptr) override;

  /**
   * @brief Get the type of the boundary condition
   * @return "Slip" as the identifier for this boundary type
   */
  std::string getType() const override;

  /**
   * @brief Determine boundary orientation from neighbor pattern
   * @param neighbors Vector of neighboring internal cells
   * @return The detected boundary orientation
   */
  Orientation detectOrientation(const std::vector<Cell *> &neighbors) const;

  /**
   * @brief Get the current orientation of the slip boundary
   * @return Current orientation enum value
   */
  Orientation getOrientation() const;

  /**
   * @brief Set a new boundary orientation
   * @param orientation String representation of the desired orientation
   */
  void setOrientation(const std::string &orientation);

  /**
   * @brief Serialize the slip boundary condition to a string representation
   * @return String containing serialized boundary data
   */
  std::string serialize() const override {
    // Start with the base class serialization
    std::ostringstream oss;
    oss << BoundaryClass::serialize();

    // Add orientation information
    std::string orientationStr;
    switch (m_orientation) {
    case Orientation::NORTH:
      orientationStr = "NORTH";
      break;
    case Orientation::SOUTH:
      orientationStr = "SOUTH";
      break;
    case Orientation::EAST:
      orientationStr = "EAST";
      break;
    case Orientation::WEST:
      orientationStr = "WEST";
      break;
    case Orientation::AUTO:
      orientationStr = "AUTO";
      break;
    }

    oss << "ORIENTATION=" << orientationStr << "\n";

    return oss.str();
  }

  /**
   * @brief Deserialize slip boundary condition from a string representation
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

      if (key == "ORIENTATION") {
        setOrientation(value);
      }
    }

    return true;
  }
};