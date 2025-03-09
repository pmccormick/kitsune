/**
 * @file UlpEquivalence.h
 * @brief ULP-based material equivalence for high-precision computational
 * simulations
 *
 * This file implements a Units of Least Precision (ULP) approach to material
 * equivalence checking, which is particularly suited for high-precision
 * simulations where numerical stability and consistency are critical
 * requirements.
 *
 * ULP-based comparison is particularly valuable in computational science
 * because it accounts for the inherent limitations of floating-point
 * representation, providing a mathematically sound way to handle the inevitable
 * approximation errors that accumulate during numerical calculations.
 *
 * References:
 * - "What Every Computer Scientist Should Know About Floating-Point Arithmetic"
 * by Goldberg https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html
 *
 * - "Comparing Floating Point Numbers" by Bruce Dawson
 *   https://randomascii.wordpress.com/2012/02/25/comparing-floating-point-numbers-2012-edition/
 *
 * - "Handbook of Floating-Point Arithmetic" by Muller et al.
 *   Details the IEEE 754 standard and implications for scientific computing
 */

#pragma once

#include "MaterialEquivalence.h"

/**
 * @class UlpEquivalence
 * @brief ULP-based equivalence comparison
 *
 * This class implements floating-point comparisons using Units of Least
 * Precision (ULPs), which provides a more mathematically sound approach to
 * determining if two floating-point values should be considered equal. This is
 * particularly useful for numeric stability in physics-based simulations.
 *
 * Application domains where UlpEquivalence is particularly appropriate:
 *
 * 1. High-precision CFD:
 *    - Direct Numerical Simulation (DNS)
 *    - Large Eddy Simulation (LES)
 *    - Recommended ULP setting: 2-4 ULPs
 *
 * 2. Iterative solvers with strict convergence requirements:
 *    - Finite element analysis with small residual thresholds
 *    - Optimization problems in material design
 *    - Recommended ULP setting: 2-8 ULPs
 *
 * 3. Long-running simulations where error accumulation is a concern:
 *    - Climate models
 *    - N-body astrophysical simulations
 *    - Nuclear reactor modeling
 *    - Recommended ULP setting: 4-16 ULPs (depends on simulation time)
 */
class UlpEquivalence : public MaterialEquivalence {
public:
  /**
   * @brief Constructor
   * @param maxUlps Maximum ULP difference to consider values equivalent
   * (typically 4-10)
   * @param absoluteTolerance Absolute tolerance for near-zero comparisons
   *
   * The maxUlps parameter determines how many representable floating-point
   * values can exist between two numbers while still considering them
   * equivalent.
   *
   * Guidelines for setting maxUlps:
   * - 1-2 ULPs: Extremely strict, mainly for validation tests
   * - 4 ULPs: Good default for high-precision scientific computing
   * - 8-16 ULPs: More forgiving, suitable for iterative methods with
   * accumulated error
   *
   * The absoluteTolerance parameter handles comparisons near zero, where
   * ULP-based comparison alone is insufficient due to the relative nature of
   * ULP spacing.
   */
  UlpEquivalence(int maxUlps = 4, double absoluteTolerance = 1e-10);

  /**
   * @brief Get a human-readable name for this equivalence strategy
   * @return Strategy name
   */
  std::string getName() const override;

  /**
   * @brief Set the maximum ULP difference for equivalence
   * @param maxUlps Maximum ULP difference
   *
   * Smaller values make the comparison more strict, while larger values
   * are more forgiving of numerical differences. Setting this appropriately
   * depends on the numerical methods used in the simulation.
   */
  void setMaxUlps(int maxUlps);

  /**
   * @brief Set property-specific ULP difference
   * @param property The property to configure
   * @param maxUlps Maximum ULP difference for this property
   *
   * Different properties may need different ULP thresholds based on:
   * - Computational history (derived vs. direct properties)
   * - Sensitivity in the physics equations
   * - Expected range and precision requirements
   *
   * Typical settings:
   * - Primary simulation variables (density, temperature): 2-4 ULPs
   * - Derived properties (viscosity, thermal conductivity): 4-8 ULPs
   */
  void setPropertyMaxUlps(Material::MaterialProperty property, int maxUlps);

protected:
  /**
   * @brief Compare two floating-point values using ULP difference
   * @param a First value
   * @param b Second value
   * @param tolerance Not used for ULP comparison
   * @return True if ULP difference is within threshold
   *
   * This method reinterprets the bit patterns of floating-point values as
   * integers, then computes how many representable floating-point values
   * exist between them. This provides a measure of their "distance" that
   * accounts for the non-uniform distribution of floating-point precision.
   */
  bool compareValues(double a, double b, double tolerance) const override;

  /**
   * @brief Compute a consistent hash value based on ULP quantization
   * @param value Value to hash
   * @param tolerance Not used for ULP hashing
   * @return Hash value
   *
   * Creates a hash value that is consistent with ULP-based comparison by
   * quantizing the ULP representation of the value.
   */
  size_t hashValue(double value, double tolerance) const override;

private:
  // Union for reinterpreting doubles as integers for ULP comparison
  union Double_t {
    Double_t(double d) : d(d) {}
    int64_t i;
    double d;
  };

  int m_maxUlps;
  double m_absoluteTolerance;
  int m_propertyMaxUlps[static_cast<size_t>(Material::MaterialProperty::COUNT)];
};
