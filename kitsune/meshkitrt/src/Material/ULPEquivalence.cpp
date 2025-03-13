/**
 * @file UlpEquivalence.cpp
 * @brief Implementation of the UlpEquivalence class
 *
 * This file implements the methods defined in UlpEquivalence.h, providing
 * a Units of Least Precision (ULP) based approach to material equivalence
 * checking. This approach is particularly valuable for high-precision
 * computational science simulations where numerical stability is a critical
 * concern.
 */

#include "ULPEquivalence.h"
#include <cmath>
#include <functional>
#include <limits>

//------------------------------------------------------------------------------
// UlpEquivalence Implementation
//------------------------------------------------------------------------------

UlpEquivalence::UlpEquivalence(int maxUlps, double absoluteTolerance)
    : MaterialEquivalence(absoluteTolerance), m_maxUlps(maxUlps),
      m_absoluteTolerance(absoluteTolerance) {
  // Initialize property-specific ULPs with the global default
  for (size_t i = 0; i < static_cast<size_t>(Material::MaterialProperty::COUNT);
       ++i) {
    m_propertyMaxUlps[i] = maxUlps;
  }

  // Set stricter ULPs for properties where higher precision is typically
  // required in scientific computations
  m_propertyMaxUlps[static_cast<size_t>(Material::MaterialProperty::DENSITY)] =
      std::max(2, maxUlps / 2);
  m_propertyMaxUlps[static_cast<size_t>(
      Material::MaterialProperty::THERMAL_CONDUCTIVITY)] =
      std::max(2, maxUlps / 2);
}

std::string UlpEquivalence::getName() const { return "ULP-Based Equivalence"; }

void UlpEquivalence::setMaxUlps(int maxUlps) {
  m_maxUlps = maxUlps;

  // Update all property ULPs
  for (size_t i = 0; i < static_cast<size_t>(Material::MaterialProperty::COUNT);
       ++i) {
    m_propertyMaxUlps[i] = maxUlps;
  }
}

void UlpEquivalence::setPropertyMaxUlps(Material::MaterialProperty property,
                                        int maxUlps) {
  size_t index = static_cast<size_t>(property);
  if (index < static_cast<size_t>(Material::MaterialProperty::COUNT)) {
    m_propertyMaxUlps[index] = maxUlps;
  }
}

bool UlpEquivalence::compareValues(double a, double b, double tolerance) const {
  // First, check for NaN - never equal to anything (including itself)
  if (std::isnan(a) || std::isnan(b)) {
    return false;
  }

  // Check if the numbers are really close -- needed when comparing numbers near
  // zero Absolute tolerance handles near-zero values where ULP comparison is
  // less meaningful
  double diff = std::fabs(a - b);
  if (diff <= m_absoluteTolerance) {
    return true;
  }

  // Special case for infinity - equal only if same sign
  if (std::isinf(a) || std::isinf(b)) {
    return a == b;
  }

  // Convert to integer representations using the union trick
  // This allows direct access to the bit pattern of the double
  Double_t uA(a);
  Double_t uB(b);

  // Different signs means they are far apart in the number line
  // Need special handling to avoid overflow in ULP calculation
  if ((uA.i < 0) != (uB.i < 0)) {
    return false;
  }

  // For negative numbers, the ordering in ULP space is inverted
  // So we need to negate the integers for proper comparison
  if (uA.i < 0) {
    uA.i = INT64_MIN - uA.i;
    uB.i = INT64_MIN - uB.i;
  }

  // Find the difference in ULPs - how many representable floating-point
  // values exist between a and b
  int64_t ulpsDiff = std::abs(uA.i - uB.i);

  // Compare with the configured ULP threshold
  return ulpsDiff <= m_maxUlps;
}

size_t UlpEquivalence::hashValue(double value, double tolerance) const {
  // Handle special cases
  if (std::isnan(value)) {
    return std::hash<double>{}(std::numeric_limits<double>::quiet_NaN());
  }

  if (std::isinf(value)) {
    return std::hash<double>{}(value);
  }

  // For near-zero values, round to zero
  if (std::fabs(value) <= m_absoluteTolerance) {
    return 0;
  }

  // Convert to integer representation to work with ULPs
  Double_t u(value);

  // For key consistency, use a much larger quantization factor
  // This ensures that values within maxUlps of each other hash to the same
  // bucket
  int64_t quantizationFactor = m_maxUlps * 10;

  // For negative numbers, invert the order for consistent hashing behavior
  if (u.i < 0) {
    u.i = INT64_MIN - u.i;
  }

  // Quantize ULPs by rounding to nearest multiple of quantization factor
  int64_t roundedUlps = (u.i / quantizationFactor) * quantizationFactor;

  // Return the hash of the rounded ULPs
  return static_cast<size_t>(roundedUlps);
}
