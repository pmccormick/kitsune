#include "Units.h"

namespace Units {
/**
 * @brief Get the unit conversion structure for a given unit string.
 * @param unitStr The unit string to look up.
 * @return The unit conversion structure.
 * @throws std::invalid_argument if unit is not recognized.
 */
UnitConversion getUnitConversion(const std::string &unitStr) {
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
} // namespace Units