#pragma once

#include "BoundaryClass.h"
#include <sstream>
#include <string>

/**
 * @class NeumannBoundary
 * @brief Implementation of Neumann boundary conditions
 *
 * The Neumann boundary condition specifies the derivative of the solution
 * at the boundary. For fluid dynamics, this often represents a "zero-gradient"
 * condition where the normal derivative of the flow variables is set to zero,
 * which implies no flux across the boundary in that direction.
 *
 * Common use cases:
 * - Symmetry planes
 * - Fully developed flow regions
 * - Far-field boundaries where flow is parallel to the boundary
 *
 * References:
 * - Ferziger, J. H., & Peric, M. (2002). Computational Methods for Fluid
 * Dynamics. Springer, Chapter 8.
 * - Patankar, S. V. (1980). Numerical Heat Transfer and Fluid Flow.
 *   CRC Press, Chapter 4.3.
 */
class NeumannBoundary : public BoundaryClass {
private:
  double m_gradient; // Specified gradient value at the boundary

public:
  /**
   * @brief Constructor
   * @param gradient The desired gradient value at the boundary (default = 0.0
   * for zero-gradient)
   */
  NeumannBoundary(double gradient = 0.0);

  /**
   * @brief Apply Neumann boundary condition to a cell
   *
   * Sets the cell values based on neighboring cells to enforce the
   * specified gradient at the boundary.
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
   * @return "Neumann" as the identifier for this boundary type
   */
  std::string getType() const override;

  /**
   * @brief Get the specified gradient value
   * @return The gradient value at the boundary
   */
  double getGradient() const;

  /**
   * @brief Set a new gradient value
   * @param gradient The new gradient value to enforce at the boundary
   */
  void setGradient(double gradient);

  /**
   * @brief Serialize the Neumann boundary condition to a string representation
   * @return String containing serialized boundary data
   */
  std::string serialize() const override {
    // Start with the base class serialization
    std::ostringstream oss;
    oss << BoundaryClass::serialize();

    // Add Neumann-specific data
    oss << "GRADIENT=" << m_gradient << "\n";

    return oss.str();
  }

  /**
   * @brief Deserialize Neumann boundary condition from a string representation
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

      if (key == "GRADIENT") {
        m_gradient = std::stod(value);
      }
    }

    return true;
  }
};