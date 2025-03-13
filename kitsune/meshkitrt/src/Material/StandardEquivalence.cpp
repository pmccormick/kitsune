/**
 * @file StandardEquivalence.cpp
 * @brief Implementation of the StandardEquivalence class
 *
 * This file implements the methods defined in StandardEquivalence.h, providing
 * a standard epsilon-based approach to material equivalence checking. The
 * implementation uses a combination of absolute and relative tolerance checks
 * that work well for most general-purpose scientific and engineering
 * simulations.
 */

#include "StandardEquivalence.h"
#include <cmath>
#include <functional>

//------------------------------------------------------------------------------
// StandardEquivalence Implementation
//------------------------------------------------------------------------------

StandardEquivalence::StandardEquivalence(double tolerance)
    : MaterialEquivalence(tolerance) {
  // Constructor uses base class initialization
}

std::string StandardEquivalence::getName() const {
  return "Standard Epsilon-Based Equivalence";
}

bool StandardEquivalence::compareValues(double a, double b,
                                        double tolerance) const {
  // Use the base class implementation for standard epsilon comparison
  // This provides both absolute and relative tolerance handling
  return MaterialEquivalence::compareValues(a, b, tolerance);
}

size_t StandardEquivalence::hashValue(double value, double tolerance) const {
  // Use the base class implementation for standard quantized hashing
  return MaterialEquivalence::hashValue(value, tolerance);
}
