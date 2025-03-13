/**
 * @file StandardEquivalence.h
 * @brief Standard epsilon-based material equivalence for computational
 * simulations
 *
 * This file implements a standard epsilon-based approach to material
 * equivalence checking, suitable for a wide range of computational science
 * applications. The StandardEquivalence class provides a balance of accuracy
 * and performance for typical engineering and scientific simulations.
 *
 * Epsilon-based comparison is the most widely used approach in scientific
 * software, as it addresses the fundamental issues of floating-point
 * representation while remaining intuitive and computationally efficient.
 *
 * References:
 * - "Numerical Recipes: The Art of Scientific Computing" by Press et al.
 *   Recommends epsilon values of 1e-6 to 1e-8 for most scientific applications
 *
 * - "Engineering Analysis with ANSYS Software" by Nakasone & Yoshimoto
 *   Uses epsilon-based comparisons for material property aggregation
 */

#pragma once

#include "MaterialEquivalence.h"

/**
 * @class StandardEquivalence
 * @brief Standard epsilon-based equivalence comparison
 *
 * This class implements a standard approach to floating-point comparisons
 * using epsilon-based relative and absolute tolerances. It is suitable for
 * most general-purpose material comparison needs.
 *
 * Application domains where StandardEquivalence is particularly appropriate:
 *
 * 1. General CFD simulations:
 *    - Typical fluid density range: 0.5-2000 kg/m³
 *    - Typical viscosity range: 1e-5 to 1e3 Pa·s
 *    - Recommended tolerance: 1e-6 (relative + absolute hybrid)
 *
 * 2. Structural mechanics:
 *    - Young's modulus range: 1e6 to 5e11 Pa
 *    - Density range: 100-20000 kg/m³
 *    - Poisson ratio range: 0.0-0.5
 *    - Recommended tolerance: 1e-6 to 1e-4
 *
 * 3. Heat transfer:
 *    - Thermal conductivity range: 0.01-500 W/(m·K)
 *    - Specific heat range: 100-5000 J/(kg·K)
 *    - Recommended tolerance: 1e-5
 */
class StandardEquivalence : public MaterialEquivalence {
public:
  /**
   * @brief Constructor
   * @param tolerance Default comparison tolerance
   *
   * The default tolerance of 1e-6 is appropriate for most engineering
   * simulations. For reference:
   * - 1e-4: Use for coarse simulations or when performance is critical
   * - 1e-6: Good balance for most applications
   * - 1e-8: Use for high-precision requirements
   */
  StandardEquivalence(double tolerance = 1e-6);

  /**
   * @brief Get a human-readable name for this equivalence strategy
   * @return Strategy name
   */
  std::string getName() const override;

protected:
  /**
   * @brief Compare two floating-point values with tolerance
   * @param a First value
   * @param b Second value
   * @param tolerance Comparison tolerance
   * @return True if values are equivalent within tolerance
   *
   * This implementation uses a hybrid approach:
   * - For small values: Uses absolute comparison (|a-b| <= tolerance)
   * - For larger values: Uses relative comparison (|a-b| <= tolerance *
   * max(|a|,|b|))
   *
   * This handles both near-zero values and values that span multiple orders
   * of magnitude, making it suitable for a wide range of physical properties.
   */
  bool compareValues(double a, double b, double tolerance) const override;

  /**
   * @brief Compute a consistent hash value for a floating-point number
   * @param value Value to hash
   * @param tolerance Tolerance to use for quantization
   * @return Hash value
   *
   * This method quantizes floating-point values to ensure that values within
   * the tolerance range hash to the same bin. This ensures consistency between
   * equivalence comparison and hash-based lookups.
   */
  size_t hashValue(double value, double tolerance) const override;
};
