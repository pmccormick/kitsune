#pragma once

#include "BoundaryClass.h"

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
  void apply(Cell &cell, const std::vector<Cell *> &neighbors, double x,
             double y, double dt) override;

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
};
