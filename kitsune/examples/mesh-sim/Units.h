#pragma once

#include <cmath>
#include <functional>
#include <stdexcept>
#include <string>
#include <unordered_map>

/**
 * @namespace Units
 * @brief Provides utilities for handling physical units and conversions
 *
 * This namespace contains functions and constants related to unit conversions
 * and standardization. All internal computations in the simulation use SI
 * units:
 * - Temperature: Kelvin (K)
 * - Pressure: Pascal (Pa)
 * - Length: meter (m)
 * - Velocity: meter per second (m/s)
 * - Density: kilogram per cubic meter (kg/m³)
 * - Viscosity: Pascal-second (Pa·s)
 * - Time: second (s)
 * - Energy: Joule (J)
 * - Power: Watt (W)
 * - Thermal conductivity: Watt per meter-Kelvin (W/(m·K))
 * - Specific heat: Joule per kilogram-Kelvin (J/(kg·K))
 */
namespace Units {

//==============================================================================
// Physical constants in SI units
//==============================================================================
constexpr double STANDARD_GRAVITY = 9.80665;       // m/s²
constexpr double STANDARD_ATM_PRESSURE = 101325.0; // Pa
constexpr double STANDARD_TEMPERATURE = 293.15;    // K (20°C)
constexpr double ABSOLUTE_ZERO = 0.0;              // K
constexpr double WATER_FREEZING_POINT = 273.15;    // K (0°C)
constexpr double WATER_BOILING_POINT = 373.15;     // K (100°C)
constexpr double GAS_CONSTANT = 8.31446;           // J/(mol·K)

//==============================================================================
// Temperature conversions
//==============================================================================

/**
 * @brief Convert Celsius to Kelvin
 * @param celsius Temperature in degrees Celsius
 * @return Temperature in Kelvin
 */
inline double celsiusToKelvin(double celsius) { return celsius + 273.15; }

/**
 * @brief Convert Kelvin to Celsius
 * @param kelvin Temperature in Kelvin
 * @return Temperature in degrees Celsius
 */
inline double kelvinToCelsius(double kelvin) { return kelvin - 273.15; }

/**
 * @brief Convert Fahrenheit to Kelvin
 * @param fahrenheit Temperature in degrees Fahrenheit
 * @return Temperature in Kelvin
 */
inline double fahrenheitToKelvin(double fahrenheit) {
  return (fahrenheit - 32.0) * 5.0 / 9.0 + 273.15;
}

/**
 * @brief Convert Kelvin to Fahrenheit
 * @param kelvin Temperature in Kelvin
 * @return Temperature in degrees Fahrenheit
 */
inline double kelvinToFahrenheit(double kelvin) {
  return (kelvin - 273.15) * 9.0 / 5.0 + 32.0;
}

/**
 * @brief Convert Rankine to Kelvin
 * @param rankine Temperature in Rankine
 * @return Temperature in Kelvin
 */
inline double rankineToKelvin(double rankine) { return rankine * 5.0 / 9.0; }

/**
 * @brief Convert Kelvin to Rankine
 * @param kelvin Temperature in Kelvin
 * @return Temperature in Rankine
 */
inline double kelvinToRankine(double kelvin) { return kelvin * 9.0 / 5.0; }

//==============================================================================
// Pressure conversions
//==============================================================================

/**
 * @brief Convert atmospheres to Pascal
 * @param atm Pressure in atmospheres
 * @return Pressure in Pascal
 */
inline double atmToPascal(double atm) { return atm * STANDARD_ATM_PRESSURE; }

/**
 * @brief Convert Pascal to atmospheres
 * @param pascal Pressure in Pascal
 * @return Pressure in atmospheres
 */
inline double pascalToAtm(double pascal) {
  return pascal / STANDARD_ATM_PRESSURE;
}

/**
 * @brief Convert bar to Pascal
 * @param bar Pressure in bar
 * @return Pressure in Pascal
 */
inline double barToPascal(double bar) { return bar * 1.0e5; }

/**
 * @brief Convert Pascal to bar
 * @param pascal Pressure in Pascal
 * @return Pressure in bar
 */
inline double pascalToBar(double pascal) { return pascal / 1.0e5; }

/**
 * @brief Convert torr (mmHg) to Pascal
 * @param torr Pressure in torr
 * @return Pressure in Pascal
 */
inline double torrToPascal(double torr) { return torr * 133.322; }

/**
 * @brief Convert Pascal to torr (mmHg)
 * @param pascal Pressure in Pascal
 * @return Pressure in torr
 */
inline double pascalToTorr(double pascal) { return pascal / 133.322; }

/**
 * @brief Convert pounds per square inch (psi) to Pascal
 * @param psi Pressure in pounds per square inch
 * @return Pressure in Pascal
 */
inline double psiToPascal(double psi) { return psi * 6894.76; }

/**
 * @brief Convert Pascal to pounds per square inch (psi)
 * @param pascal Pressure in Pascal
 * @return Pressure in pounds per square inch
 */
inline double pascalToPsi(double pascal) { return pascal / 6894.76; }

//==============================================================================
// Length conversions
//==============================================================================

/**
 * @brief Convert inches to meters
 * @param inch Length in inches
 * @return Length in meters
 */
inline double inchToMeter(double inch) { return inch * 0.0254; }

/**
 * @brief Convert meters to inches
 * @param meter Length in meters
 * @return Length in inches
 */
inline double meterToInch(double meter) { return meter / 0.0254; }

/**
 * @brief Convert feet to meters
 * @param foot Length in feet
 * @return Length in meters
 */
inline double footToMeter(double foot) { return foot * 0.3048; }

/**
 * @brief Convert meters to feet
 * @param meter Length in meters
 * @return Length in feet
 */
inline double meterToFoot(double meter) { return meter / 0.3048; }

/**
 * @brief Convert miles to meters
 * @param mile Length in miles
 * @return Length in meters
 */
inline double mileToMeter(double mile) { return mile * 1609.34; }

/**
 * @brief Convert meters to miles
 * @param meter Length in meters
 * @return Length in miles
 */
inline double meterToMile(double meter) { return meter / 1609.34; }

//==============================================================================
// Velocity conversions
//==============================================================================

/**
 * @brief Convert miles per hour to meters per second
 * @param mph Velocity in miles per hour
 * @return Velocity in meters per second
 */
inline double mphToMps(double mph) { return mph * 0.44704; }

/**
 * @brief Convert meters per second to miles per hour
 * @param mps Velocity in meters per second
 * @return Velocity in miles per hour
 */
inline double mpsToMph(double mps) { return mps / 0.44704; }

/**
 * @brief Convert knots to meters per second
 * @param knot Velocity in knots
 * @return Velocity in meters per second
 */
inline double knotToMps(double knot) { return knot * 0.51444; }

/**
 * @brief Convert meters per second to knots
 * @param mps Velocity in meters per second
 * @return Velocity in knots
 */
inline double mpsToKnot(double mps) { return mps / 0.51444; }

//==============================================================================
// Density conversions
//==============================================================================

/**
 * @brief Convert pounds per cubic foot to kilograms per cubic meter
 * @param lbft3 Density in pounds per cubic foot
 * @return Density in kilograms per cubic meter
 */
inline double lbft3ToKgm3(double lbft3) { return lbft3 * 16.0185; }

/**
 * @brief Convert kilograms per cubic meter to pounds per cubic foot
 * @param kgm3 Density in kilograms per cubic meter
 * @return Density in pounds per cubic foot
 */
inline double kgm3ToLbft3(double kgm3) { return kgm3 / 16.0185; }

//==============================================================================
// Viscosity conversions
//==============================================================================

/**
 * @brief Convert poise to Pascal-second
 * @param poise Viscosity in poise
 * @return Viscosity in Pascal-second
 */
inline double poiseToPass(double poise) { return poise * 0.1; }

/**
 * @brief Convert Pascal-second to poise
 * @param pass Viscosity in Pascal-second
 * @return Viscosity in poise
 */
inline double passToPoise(double pass) { return pass / 0.1; }

/**
 * @brief Convert centipoise to Pascal-second
 * @param cp Viscosity in centipoise
 * @return Viscosity in Pascal-second
 */
inline double cpToPass(double cp) { return cp * 0.001; }

/**
 * @brief Convert Pascal-second to centipoise
 * @param pass Viscosity in Pascal-second
 * @return Viscosity in centipoise
 */
inline double passToCp(double pass) { return pass / 0.001; }

//==============================================================================
// Thermal conductivity conversions
//==============================================================================

/**
 * @brief Convert BTU/(hr·ft·°F) to W/(m·K)
 * @param btu Thermal conductivity in BTU/(hr·ft·°F)
 * @return Thermal conductivity in W/(m·K)
 */
inline double btuToWmk(double btu) { return btu * 1.73073; }

/**
 * @brief Convert W/(m·K) to BTU/(hr·ft·°F)
 * @param wmk Thermal conductivity in W/(m·K)
 * @return Thermal conductivity in BTU/(hr·ft·°F)
 */
inline double wmkToBtu(double wmk) { return wmk / 1.73073; }

//==============================================================================
// Specific heat conversions
//==============================================================================

/**
 * @brief Convert BTU/(lb·°F) to J/(kg·K)
 * @param btu Specific heat in BTU/(lb·°F)
 * @return Specific heat in J/(kg·K)
 */
inline double btuToJkgk(double btu) { return btu * 4186.8; }

/**
 * @brief Convert J/(kg·K) to BTU/(lb·°F)
 * @param jkgk Specific heat in J/(kg·K)
 * @return Specific heat in BTU/(lb·°F)
 */
inline double jkgkToBtu(double jkgk) { return jkgk / 4186.8; }

//==============================================================================
// Generic unit conversion with unit string identification
//==============================================================================

/**
 * @enum UnitType
 * @brief Identifies the physical quantity type for unit conversion
 */
enum class UnitType {
  TEMPERATURE,
  PRESSURE,
  LENGTH,
  VELOCITY,
  DENSITY,
  VISCOSITY,
  THERMAL_CONDUCTIVITY,
  SPECIFIC_HEAT,
  UNKNOWN
};

/**
 * @struct UnitConversion
 * @brief Provides functions to convert between different units
 */
struct UnitConversion {
  UnitType type;
  std::string siUnit;
  std::function<double(double)> toSI;
  std::function<double(double)> fromSI;
};

/**
 * @brief Get the unit conversion structure for a given unit string
 * @param unitStr The unit string to look up
 * @return The unit conversion structure
 * @throws std::invalid_argument if unit is not recognized
 */
inline UnitConversion getUnitConversion(const std::string &unitStr) {
  static const std::unordered_map<std::string, UnitConversion> conversions = {
      // Temperature
      {"K",
       {UnitType::TEMPERATURE, "K", [](double v) { return v; },
        [](double v) { return v; }}},
      {"C",
       {UnitType::TEMPERATURE, "K", [](double v) { return celsiusToKelvin(v); },
        [](double v) { return kelvinToCelsius(v); }}},
      {"F",
       {UnitType::TEMPERATURE, "K",
        [](double v) { return fahrenheitToKelvin(v); },
        [](double v) { return kelvinToFahrenheit(v); }}},
      {"R",
       {UnitType::TEMPERATURE, "K", [](double v) { return rankineToKelvin(v); },
        [](double v) { return kelvinToRankine(v); }}},

      // Pressure
      {"Pa",
       {UnitType::PRESSURE, "Pa", [](double v) { return v; },
        [](double v) { return v; }}},
      {"atm",
       {UnitType::PRESSURE, "Pa", [](double v) { return atmToPascal(v); },
        [](double v) { return pascalToAtm(v); }}},
      {"bar",
       {UnitType::PRESSURE, "Pa", [](double v) { return barToPascal(v); },
        [](double v) { return pascalToBar(v); }}},
      {"torr",
       {UnitType::PRESSURE, "Pa", [](double v) { return torrToPascal(v); },
        [](double v) { return pascalToTorr(v); }}},
      {"psi",
       {UnitType::PRESSURE, "Pa", [](double v) { return psiToPascal(v); },
        [](double v) { return pascalToPsi(v); }}},

      // Length
      {"m",
       {UnitType::LENGTH, "m", [](double v) { return v; },
        [](double v) { return v; }}},
      {"in",
       {UnitType::LENGTH, "m", [](double v) { return inchToMeter(v); },
        [](double v) { return meterToInch(v); }}},
      {"ft",
       {UnitType::LENGTH, "m", [](double v) { return footToMeter(v); },
        [](double v) { return meterToFoot(v); }}},
      {"mi",
       {UnitType::LENGTH, "m", [](double v) { return mileToMeter(v); },
        [](double v) { return meterToMile(v); }}},

      // Velocity
      {"m/s",
       {UnitType::VELOCITY, "m/s", [](double v) { return v; },
        [](double v) { return v; }}},
      {"mph",
       {UnitType::VELOCITY, "m/s", [](double v) { return mphToMps(v); },
        [](double v) { return mpsToMph(v); }}},
      {"knot",
       {UnitType::VELOCITY, "m/s", [](double v) { return knotToMps(v); },
        [](double v) { return mpsToKnot(v); }}},

      // Density
      {"kg/m³",
       {UnitType::DENSITY, "kg/m³", [](double v) { return v; },
        [](double v) { return v; }}},
      {"lb/ft³",
       {UnitType::DENSITY, "kg/m³", [](double v) { return lbft3ToKgm3(v); },
        [](double v) { return kgm3ToLbft3(v); }}},

      // Viscosity
      {"Pa·s",
       {UnitType::VISCOSITY, "Pa·s", [](double v) { return v; },
        [](double v) { return v; }}},
      {"poise",
       {UnitType::VISCOSITY, "Pa·s", [](double v) { return poiseToPass(v); },
        [](double v) { return passToPoise(v); }}},
      {"cP",
       {UnitType::VISCOSITY, "Pa·s", [](double v) { return cpToPass(v); },
        [](double v) { return passToCp(v); }}},

      // Thermal conductivity
      {"W/(m·K)",
       {UnitType::THERMAL_CONDUCTIVITY, "W/(m·K)", [](double v) { return v; },
        [](double v) { return v; }}},
      {"BTU/(hr·ft·°F)",
       {UnitType::THERMAL_CONDUCTIVITY, "W/(m·K)",
        [](double v) { return btuToWmk(v); },
        [](double v) { return wmkToBtu(v); }}},

      // Specific heat
      {"J/(kg·K)",
       {UnitType::SPECIFIC_HEAT, "J/(kg·K)", [](double v) { return v; },
        [](double v) { return v; }}},
      {"BTU/(lb·°F)",
       {UnitType::SPECIFIC_HEAT, "J/(kg·K)",
        [](double v) { return btuToJkgk(v); },
        [](double v) { return jkgkToBtu(v); }}}};

  auto it = conversions.find(unitStr);
  if (it != conversions.end()) {
    return it->second;
  }

  throw std::invalid_argument("Unrecognized unit: " + unitStr);
}

/**
 * @brief Convert a value from one unit to another
 * @param value Value to convert
 * @param fromUnit Source unit string
 * @param toUnit Target unit string
 * @return Converted value
 * @throws std::invalid_argument if units are not recognized or incompatible
 */
inline double convert(double value, const std::string &fromUnit,
                      const std::string &toUnit) {
  if (fromUnit == toUnit) {
    return value;
  }

  UnitConversion fromConv = getUnitConversion(fromUnit);
  UnitConversion toConv = getUnitConversion(toUnit);

  if (fromConv.type != toConv.type) {
    throw std::invalid_argument(
        "Cannot convert between incompatible unit types: " + fromUnit + " to " +
        toUnit);
  }

  // Convert to SI, then to target unit
  double siValue = fromConv.toSI(value);
  return toConv.fromSI(siValue);
}

//==============================================================================
// Unit validation functions
//==============================================================================

/**
 * @brief Check if a temperature value is physically valid
 * @param value The temperature value in Kelvin
 * @return True if the temperature is valid (above absolute zero)
 */
inline bool isValidTemperature(double value) { return value >= ABSOLUTE_ZERO; }

/**
 * @brief Check if a pressure value is physically valid
 * @param value The pressure value in Pascal
 * @return True if the pressure is valid (positive)
 */
inline bool isValidPressure(double value) { return value >= 0.0; }

/**
 * @brief Check if a density value is physically valid
 * @param value The density value in kg/m³
 * @return True if the density is valid (positive)
 */
inline bool isValidDensity(double value) { return value > 0.0; }

/**
 * @brief Check if a viscosity value is physically valid
 * @param value The viscosity value in Pa·s
 * @return True if the viscosity is valid (non-negative)
 */
inline bool isValidViscosity(double value) { return value >= 0.0; }

/**
 * @brief Enforce a valid temperature by clamping to valid range
 * @param value The temperature value in Kelvin
 * @return Clamped temperature value
 */
inline double enforceValidTemperature(double value) {
  return std::max(ABSOLUTE_ZERO, value);
}

/**
 * @brief Enforce a valid pressure by clamping to valid range
 * @param value The pressure value in Pascal
 * @return Clamped pressure value
 */
inline double enforceValidPressure(double value) {
  return std::max(0.0, value);
}

/**
 * @brief Enforce a valid density by clamping to valid range
 * @param value The density value in kg/m³
 * @return Clamped density value
 */
inline double enforceValidDensity(double value) {
  constexpr double EPSILON = 1e-6;
  return std::max(EPSILON, value);
}

/**
 * @brief Enforce a valid viscosity by clamping to valid range
 * @param value The viscosity value in Pa·s
 * @return Clamped viscosity value
 */
inline double enforceValidViscosity(double value) {
  return std::max(0.0, value);
}

} // namespace Units
