/**
 * @file QuantizedEquivalence.h
 * @brief Bit-level quantized material equivalence for computational simulations
 *
 * This file implements a bit-level quantization approach to material
 * equivalence checking, which is particularly suited for properties that span
 * many orders of magnitude or for simulations requiring consistent behavior
 * across different physical regimes.
 *
 * Quantized equivalence divides the continuous range of property values into
 * discrete bins, with bin sizes scaled based on property magnitude and
 * customizable bit precision. This approach combines the benefits of relative
 * comparison with deterministic binning.
 *
 * References:
 * - "Discretization Methods for Numerical Analysis in Science and Engineering"
 *   by Roy & Oberkampf - Discusses quantization approaches for multi-scale
 * simulations
 *
 * - "Computational Materials Science: From Ab Initio to Monte Carlo Methods" by
 * Ohno et al. Shows property ranges spanning 10+ orders of magnitude in
 * multi-scale material modeling
 *
 * - "Numerical Methods for Conservation Laws" by LeVeque
 *   Discusses how discretization affects conservation properties in simulations
 */

#pragma once

#include "MaterialEquivalence.h"
#include <unordered_map>

/**
 * @class QuantizedEquivalence
 * @brief Bit-level quantized equivalence comparison
 *
 * This class implements a more sophisticated approach to floating-point
 * comparisons by quantizing values based on their range and a specified
 * bit precision. This creates more stable equivalence comparisons across
 * different numeric ranges.
 *
 * Application domains where QuantizedEquivalence is particularly appropriate:
 *
 * 1. Multi-scale simulations:
 *    - When properties span 10+ orders of magnitude
 *    - Example: Atmospheric models (pressure: 10-10^5 Pa, density: 10^-6-10^3
 * kg/m³)
 *    - Recommended bit precision: 8-10 bits
 *
 * 2. Multiphysics applications:
 *    - When different physics regimes have vastly different property scales
 *    - Example: Fluid-structure-thermal interactions
 *    - Property-specific bit allocations improve cache effectiveness
 *
 * 3. Material science:
 *    - Properties like electrical conductivity: 10^-15 to 10^7 S/m
 *    - Properties like thermal expansion: 10^-8 to 10^-4 K^-1
 *    - Logarithmic quantization essential for reasonable comparison
 */
class QuantizedEquivalence : public MaterialEquivalence {
public:
  /**
   * @brief Constructor
   * @param tolerance Default comparison tolerance
   *
   * While a tolerance is provided for compatibility with the base class,
   * QuantizedEquivalence primarily relies on bit precision and range
   * settings for each property rather than a single tolerance value.
   */
  QuantizedEquivalence(double tolerance = 1e-6);

  /**
   * @brief Get a human-readable name for this equivalence strategy
   * @return Strategy name
   */
  std::string getName() const override;

  /**
   * @brief Set bit precision for a specific property
   * @param property The property to configure
   * @param bits Bit precision (1-16)
   *
   * The bit precision determines how many discrete levels are used to represent
   * the property's range. Higher values provide finer granularity.
   *
   * Recommended settings by property:
   * - Density: 8-10 bits (256-1024 levels)
   * - Viscosity: 8-12 bits (requires more precision due to wide range)
   * - Thermal conductivity: 8 bits (sufficient for most applications)
   * - Specific heat: 6-8 bits (typically narrower range)
   * - Electrical conductivity: 12-16 bits (extremely wide range)
   */
  void setPropertyBits(Material::MaterialProperty property, size_t bits);

  /**
   * @brief Set the expected value range for a property
   * @param property The property to configure
   * @param min Minimum expected value
   * @param max Maximum expected value
   * @param useLogScale Whether to use logarithmic scale
   *
   * Setting appropriate ranges is crucial for effective quantization.
   * For properties spanning many orders of magnitude, logarithmic scaling
   * is strongly recommended.
   *
   * Typical ranges for common properties in scientific simulations:
   * - Density: 0.1-20000 kg/m³ (linear or log)
   * - Viscosity: 10^-6-10^3 Pa·s (log recommended)
   * - Thermal conductivity: 0.01-1000 W/(m·K) (log recommended)
   * - Specific heat: 100-10000 J/(kg·K) (linear usually sufficient)
   * - Thermal expansion: 10^-7-10^-3 K^-1 (log recommended)
   */
  void setPropertyRange(Material::MaterialProperty property, double min,
                        double max, bool useLogScale);

protected:
  /**
   * @brief Compare two floating-point values with bit-level quantization
   * @param a First value
   * @param b Second value
   * @param tolerance Fallback tolerance (not directly used)
   * @return True if values quantize to the same level
   *
   * This method determines equivalence by checking if two values fall into
   * the same quantization bin, based on the configured ranges and bit
   * precision.
   */
  bool compareValues(double a, double b, double tolerance) const override;

  /**
   * @brief Compute a consistent hash value using bit-level quantization
   * @param value Value to hash
   * @param tolerance Fallback tolerance (not directly used)
   * @return Hash value
   *
   * The hash value is directly derived from the quantization bin, ensuring
   * perfect consistency between equivalence checks and hash lookups.
   */
  size_t hashValue(double value, double tolerance) const override;

  /**
   * @brief Quantize a value to a discrete level based on range and bit
   * precision
   * @param value Value to quantize
   * @param min Minimum range value
   * @param max Maximum range value
   * @param useLogScale Whether to use logarithmic scale
   * @param bits Bit precision
   * @return Quantized value
   *
   * This core method maps a continuous value to a discrete bin number.
   * For log-scale properties, the bins are distributed evenly in log-space,
   * providing appropriate precision across the entire range.
   */
  uint64_t quantizeValue(double value, double min, double max, bool useLogScale,
                         size_t bits) const;

  // Property range descriptor
  struct PropertyRange {
    double min;
    double max;
    bool useLogScale;
  };

  // Property configurations
  std::unordered_map<Material::MaterialProperty, size_t> m_bitAllocation;
  std::unordered_map<Material::MaterialProperty, PropertyRange>
      m_propertyRanges;
};
