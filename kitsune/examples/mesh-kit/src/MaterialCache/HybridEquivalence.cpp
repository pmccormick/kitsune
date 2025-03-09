/**
 * @file HybridEquivalence.cpp
 * @brief Implementation of the HybridEquivalence class
 *
 * This file implements the methods defined in HybridEquivalence.h, providing
 * a combined approach to material equivalence checking that integrates multiple
 * comparison strategies. The hybrid approach is particularly valuable for complex
 * multiphysics simulations with diverse material property ranges and precision needs.
 */

#include "HybridEquivalence.h"
#include <cmath>
#include <functional>
#include <limits>

//------------------------------------------------------------------------------
// HybridEquivalence Implementation
//------------------------------------------------------------------------------

HybridEquivalence::HybridEquivalence(int maxUlps, double absoluteTolerance)
    : MaterialEquivalence(absoluteTolerance), m_maxUlps(maxUlps),
      m_absoluteTolerance(absoluteTolerance) {
  // Initialize with configured values
  // The base class tolerance will be used for relative comparison
}

std::string HybridEquivalence::getName() const {
  return "Hybrid ULP-Quantized Equivalence";
}

bool HybridEquivalence::compareValues(double a, double b, double tolerance) const {
  // First, check for NaN - never equal to anything (including itself)
  if (std::isnan(a) || std::isnan(b)) {
    return false;
  }
  
  // Method 1: Absolute difference comparison
  // This is best for near-zero values where relative and ULP methods can be unstable
  double diff = std::fabs(a - b);
  if (diff <= m_absoluteTolerance) {
    return true;
  }
  
  // Method 2: Relative difference comparison
  // This works well for mid-range values and is intuitive for most properties
  double maxRelDiff = tolerance; // Use the base class tolerance as relative tolerance
  double relDiff = 0.0;
  
  // Calculate relative difference safely, handling possible division by zero
  if (std::fabs(b) > std::fabs(a)) {
    relDiff = std::fabs(b) > 0.0 ? std::fabs((b - a) / b) : 0.0;
  } else {
    relDiff = std::fabs(a) > 0.0 ? std::fabs((b - a) / a) : 0.0;
  }
  
  // If relative difference is small enough, consider them equal
  if (relDiff <= maxRelDiff) {
    return true;
  }
  
  // Special case for infinity - equal only if same sign
  if (std::isinf(a) || std::isinf(b)) {
    return a == b;
  }
  
  // Method 3: ULP-based comparison
  // This works well for edge cases and numeric stability concerns
  
  // Convert to integer representations
  Double_t uA(a);
  Double_t uB(b);
  
  // Different signs means they are far apart
  if ((uA.i < 0) != (uB.i < 0)) {
    // Special case: +0 and -0 are considered equal
    if (a == 0.0 && b == 0.0) {
      return true;
    }
    return false;
  }
  
  // For negative numbers, invert the ordering for proper ULP calculation
  if (uA.i < 0) {
    uA.i = INT64_MIN - uA.i;
    uB.i = INT64_MIN - uB.i;
  }
  
  // Find the difference in ULPs
  int64_t ulpsDiff = std::abs(uA.i - uB.i);
  
  // Compare with maximum allowed ULP difference
  return ulpsDiff <= m_maxUlps;
}

size_t HybridEquivalence::hashValue(double value, double tolerance) const {
  // Handle special cases
  if (std::isnan(value)) {
    return std::hash<double>{}(std::numeric_limits<double>::quiet_NaN());
  }
  
  if (std::isinf(value)) {
    return std::hash<double>{}(value);
  }
  
  // Strategy 1: For near-zero values, round to zero
  if (std::fabs(value) <= m_absoluteTolerance) {
    return 0;
  }
  
  // Strategy 2: For small to medium values, use fixed-precision quantization
  double magnitude = std::fabs(value);
  double roundFactor = 1000.0; // 0.1% precision
  
  if (magnitude < 1.0) {
    // For values < 1, use fixed precision
    double roundedValue = std::round(value * roundFactor) / roundFactor;
    return std::hash<double>{}(roundedValue);
  } 
  
  // Strategy 3: For large values, use magnitude-based precision
  // This scales the quantization based on the magnitude of the value
  // similar to a relative tolerance approach
  double scaleFactor = std::pow(10.0, std::floor(std::log10(magnitude)));
  double roundedValue = std::round(value / (scaleFactor * 0.001)) * (scaleFactor * 0.001);
  return std::hash<double>{}(roundedValue);
}


