/**
 * @file units.h
 * @brief Main include file for the enhanced SI units library
 * 
 * This file includes all components of the units library, allowing users
 * to import everything with a single include.
 */

#ifndef UNITS_H
#define UNITS_H

// Include all components of the units library
#include "Units/core.h"
#include "Units/constants.h"
#include "Units/si_units.h"
#include "Units/prefixes.h"
#include "Units/derived_units.h"
#include "Units/formatting.h"
#include "Units/validation.h"
#include "Units/conversions.h"
#include "Units/bounds.h"

// Create a compatibility layer for code using the "Units" namespace
namespace Units {
  // Re-export the convert function with the same signature
  [[nodiscard]] inline double convert(double value, const std::string& fromUnit, const std::string& toUnit) {
    return units::convert(value, fromUnit, toUnit);
  }

  // Add other functions from the old Units API as needed
  namespace constants = units::constants;
}

#endif // UNITS_H

