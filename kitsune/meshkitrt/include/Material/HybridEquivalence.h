/**
 * @file HybridEquivalence.h
 * @brief Hybrid material equivalence for complex multiphysics simulations
 *
 * This file implements a hybrid approach to material equivalence checking,
 * combining multiple comparison strategies to handle the diverse requirements
 * of complex multiphysics simulations. By incorporating absolute tolerance,
 * relative tolerance, and ULP-based comparisons, the HybridEquivalence class
 * provides robust property comparison across different physical regimes.
 *
 * The hybrid approach is particularly valuable in complex computational science
 * applications where no single comparison method is optimal for all the
 * different property ranges and precision requirements encountered.
 *
 * References:
 * - "Verification and Validation in Scientific Computing" by Oberkampf & Roy
 *   Discusses multi-strategy approaches to numerical comparison in V&V
 *
 * - "Multiphysics Modeling: Using COMSOL in a MATLAB Environment" by Pryor
 *   Highlights the challenges of consistent material property handling
 *
 * - "Multiscale Modeling of Materials" by Tadmor & Miller
 *   Examines property variations across different physical scales
 */

#pragma once

#include "MaterialEquivalence.h"

/**
 * @class HybridEquivalence
 * @brief Hybrid approach combining multiple equivalence strategies
 *
 * This class implements a sophisticated approach that combines absolute
 * tolerance, relative tolerance, and ULP-based comparisons to provide robust
 * material equivalence checking across a wide range of values and use cases.
 *
 * Application domains where HybridEquivalence is particularly appropriate:
 *
 * 1. Multiphysics simulations with diverse material regimes:
 *    - FSI (Fluid-Structure Interaction)
 *    - Conjugate heat transfer
 *    - Combustion modeling
 *    - Electromagnetics coupled with thermal/mechanical effects
 *
 * 2. Advanced manufacturing simulations:
 *    - Additive manufacturing (properties spanning solid/liquid/gas phases)
 *    - Material forming processes with extreme property gradients
 *    - Properties can range from 10^-10 to 10^10 in different units
 *
 * 3. Environmental and geophysical modeling:
 *    - Atmospheric-ocean-land coupling
 *    - Subsurface flow with multiphase transport
 *    - Wide range of time and spatial scales
 */
class HybridEquivalence : public MaterialEquivalence {
public:
  /**
   * @brief Constructor
   * @param maxUlps Maximum ULP difference to consider values equivalent
   * @param absoluteTolerance Absolute tolerance for near-zero comparisons
   *
   * The hybrid approach combines:
   * - Absolute tolerance: Handles near-zero values
   * - Relative tolerance: Handles medium-range values
   * - ULP-based comparison: Handles large values and numerical stability
   *
   * Recommended settings for different simulation types:
   * - General multiphysics: maxUlps=4, absoluteTolerance=1e-10
   * - High-precision multiscale: maxUlps=2, absoluteTolerance=1e-12
   * - Large-scale environmental: maxUlps=8, absoluteTolerance=1e-8
   */
  HybridEquivalence(int maxUlps = 4, double absoluteTolerance = 1e-10);

  /**
   * @brief Get a human-readable name for this equivalence strategy
   * @return Strategy name
   */
  std::string getName() const override;

protected:
  /**
   * @brief Compare using multiple strategies (absolute, relative, ULP)
   * @param a First value
   * @param b Second value
   * @param tolerance Relative tolerance component
   * @return True if values are equivalent by any method
   *
   * This method attempts comparison using three different strategies:
   * 1. Absolute difference: |a-b| <= absoluteTolerance
   *    - Optimal for near-zero values
   *
   * 2. Relative difference: |a-b|/max(|a|,|b|) <= relativeTolerance
   *    - Optimal for mid-range values
   *    - Uses tolerance parameter as relativeTolerance
   *
   * 3. ULP-based difference: ULP(a,b) <= maxUlps
   *    - Optimal for large values and numerical stability
   *    - Accounts for floating-point representation
   *
   * Returns true if ANY of these strategies indicates equivalence.
   * This approach handles the widest possible range of values and is
   * extremely robust against floating-point artifacts.
   */
  bool compareValues(double a, double b, double tolerance) const override;

  /**
   * @brief Compute a consistent hash with magnitude-dependent precision
   * @param value Value to hash
   * @param tolerance Relative tolerance component
   * @return Hash value
   *
   * Uses a hybrid hashing approach that selects an appropriate strategy
   * based on the value's magnitude:
   * - Near-zero values: Constant hash (binned to zero)
   * - Small values: Fixed-precision quantization
   * - Large values: Magnitude-relative quantization
   *
   * This ensures hash consistency across the full range of possible values.
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
};
