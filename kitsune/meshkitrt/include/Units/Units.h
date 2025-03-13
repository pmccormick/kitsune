/**
 * @file Units.h
 * @brief Lightweight SI units library for compile-time checking with backward compatibility
 */

#ifndef UNITS_H
#define UNITS_H

#include <cmath>
#include <type_traits>
#include <ostream>
#include <ratio>
#include <string>
#include <concepts>
#include <stdexcept>

// Define Constants directly in Units.h instead of using forward declarations
namespace units {
  namespace constants {
    // Define key constants here (no extern, full definition)
    constexpr double ABSOLUTE_ZERO = 0.0;                  // K
    constexpr double STANDARD_GRAVITY = 9.80665;           // m/s²
    constexpr double STANDARD_ATM_PRESSURE = 101325.0;     // Pa
    constexpr double STANDARD_TEMPERATURE = 293.15;        // K (20°C)
    constexpr double WATER_FREEZING_POINT = 273.15;        // K (0°C)
    constexpr double WATER_BOILING_POINT = 373.15;         // K (100°C)
    constexpr double GAS_CONSTANT = 8.31446261815324;      // J/(mol⋅K)
  }
}

namespace units {

  /**
   * @brief Unit dimension tags for compile-time checking
   */
  struct dimensionless_dimension {};  // New: dimensionless dimension type
  struct length_dimension {};
  struct time_dimension {};
  struct mass_dimension {};
  struct temperature_dimension {};
  struct angle_dimension {};
  struct area_dimension : length_dimension {}; // L²
  struct volume_dimension : length_dimension {}; // L³

  // Derived dimensions - fixed inheritance hierarchy to avoid ambiguity
  struct velocity_dimension : length_dimension, time_dimension {}; // L/T
  struct acceleration_dimension : length_dimension, time_dimension {}; // L/T²
  struct force_dimension : mass_dimension, length_dimension, time_dimension {}; // M·L/T²
  struct pressure_dimension : mass_dimension, length_dimension, time_dimension {}; // M/(L·T²)
  struct energy_dimension : mass_dimension, length_dimension, time_dimension {}; // M·L²/T²

  // Fix ambiguous inheritance by not inheriting time_dimension again
  struct power_dimension : mass_dimension, length_dimension, time_dimension {}; // M·L²/T³

  struct density_dimension : mass_dimension, volume_dimension {}; // M/L³
  struct frequency_dimension : time_dimension {}; // 1/T

  /**
   * @brief Base template for all unit types
   *
   * @tparam Dimension Type tag for unit dimension
   * @tparam Ratio Compile-time ratio for unit conversion
   */
  template <typename Dimension, typename Ratio = std::ratio<1>>
  class Unit {
  public:
    // Type aliases for concepts and conversion
    using dimension_type = Dimension;
    using ratio_type = Ratio;

    // Default constructor
    constexpr Unit() noexcept : m_value(0.0) {}

    // Value constructor
    explicit constexpr Unit(double value) noexcept : m_value(value) {}

    // Get raw value
    [[nodiscard]] constexpr double value() const noexcept { return m_value; }

    // Arithmetic operations
    [[nodiscard]] constexpr Unit operator+(const Unit& other) const noexcept {
      return Unit(m_value + other.m_value);
    }

    [[nodiscard]] constexpr Unit operator-(const Unit& other) const noexcept {
      return Unit(m_value - other.m_value);
    }

    [[nodiscard]] constexpr Unit operator-() const noexcept {
      return Unit(-m_value);
    }

    [[nodiscard]] constexpr Unit operator*(double scalar) const noexcept {
      return Unit(m_value * scalar);
    }

    [[nodiscard]] constexpr Unit operator/(double scalar) const noexcept {
      return Unit(m_value / scalar);
    }

    // Comparison operators
    [[nodiscard]] constexpr bool operator==(const Unit& other) const noexcept {
      return m_value == other.m_value;
    }

    [[nodiscard]] constexpr bool operator!=(const Unit& other) const noexcept {
      return m_value != other.m_value;
    }

    [[nodiscard]] constexpr bool operator<(const Unit& other) const noexcept {
      return m_value < other.m_value;
    }

    [[nodiscard]] constexpr bool operator<=(const Unit& other) const noexcept {
      return m_value <= other.m_value;
    }

    [[nodiscard]] constexpr bool operator>(const Unit& other) const noexcept {
      return m_value > other.m_value;
    }

    [[nodiscard]] constexpr bool operator>=(const Unit& other) const noexcept {
      return m_value >= other.m_value;
    }

    // Assignment operators
    constexpr Unit& operator+=(const Unit& other) noexcept {
      m_value += other.m_value;
      return *this;
    }

    constexpr Unit& operator-=(const Unit& other) noexcept {
      m_value -= other.m_value;
      return *this;
    }

    constexpr Unit& operator*=(double scalar) noexcept {
      m_value *= scalar;
      return *this;
    }

    constexpr Unit& operator/=(double scalar) noexcept {
      m_value /= scalar;
      return *this;
    }

  private:
    double m_value;
  };

  // Define a dimensionless unit type
  using dimensionless = Unit<dimensionless_dimension>;

  // Provide a conversion from double to dimensionless
  inline dimensionless make_dimensionless(double value) {
    return dimensionless(value);
  }

  /**
   * @brief Concept to check if a type is a unit
   */
  template <typename T>
  concept UnitType = requires(T t) {
    typename T::dimension_type;
    typename T::ratio_type;
    { t.value() } -> std::convertible_to<double>;
  };

  /**
   * @brief Concept to check if two unit types have the same dimension
   */
  template <typename U1, typename U2>
  concept SameDimension = UnitType<U1> && UnitType<U2> &&
    std::is_same_v<typename U1::dimension_type, typename U2::dimension_type>;

  // Allow scalar * unit
  template <UnitType U>
  [[nodiscard]] constexpr U operator*(double scalar, const U& unit) noexcept {
    return unit * scalar;
  }

  /**
   * @brief Helper for unit conversion
   */
  template <UnitType ToUnit, UnitType FromUnit>
  requires SameDimension<ToUnit, FromUnit>
  [[nodiscard]] constexpr ToUnit unit_cast(const FromUnit& from) noexcept {
    // Calculate conversion ratio and return converted value
    constexpr double ratio = static_cast<double>(FromUnit::ratio_type::num) *
      static_cast<double>(ToUnit::ratio_type::den) /
      (static_cast<double>(FromUnit::ratio_type::den) *
       static_cast<double>(ToUnit::ratio_type::num));

    return ToUnit(from.value() * ratio);
  }

  // SI Unit definitions

  // Length units
  using meter = Unit<length_dimension>;
  using millimeter = Unit<length_dimension, std::ratio<1, 1000>>;
  using centimeter = Unit<length_dimension, std::ratio<1, 100>>;
  using kilometer = Unit<length_dimension, std::ratio<1000>>;
  using inch = Unit<length_dimension, std::ratio<254, 10000>>; // 0.0254 meters
  using foot = Unit<length_dimension, std::ratio<3048, 10000>>; // 0.3048 meters
  using mile = Unit<length_dimension, std::ratio<1609344, 1000>>; // 1.609344 kilometers

  // Area units
  using square_meter = Unit<area_dimension>;
  using square_centimeter = Unit<area_dimension, std::ratio<1, 10000>>;
  using square_kilometer = Unit<area_dimension, std::ratio<1000000>>;
  using square_inch = Unit<area_dimension, std::ratio<254*254, 100000000>>;
  using square_foot = Unit<area_dimension, std::ratio<3048*3048, 100000000>>;
  using hectare = Unit<area_dimension, std::ratio<10000>>;

  // Volume units
  using cubic_meter = Unit<volume_dimension>;
  using cubic_centimeter = Unit<volume_dimension, std::ratio<1, 1000000>>;
  using liter = Unit<volume_dimension, std::ratio<1, 1000>>;
  using milliliter = Unit<volume_dimension, std::ratio<1, 1000000>>;

  // Time units
  using second = Unit<time_dimension>;
  using minute = Unit<time_dimension, std::ratio<60>>;
  using hour = Unit<time_dimension, std::ratio<3600>>;
  using day = Unit<time_dimension, std::ratio<86400>>;
  using millisecond = Unit<time_dimension, std::ratio<1, 1000>>;
  using microsecond = Unit<time_dimension, std::ratio<1, 1000000>>;

  // Frequency units
  using hertz = Unit<frequency_dimension>;
  using kilohertz = Unit<frequency_dimension, std::ratio<1000>>;
  using megahertz = Unit<frequency_dimension, std::ratio<1000000>>;

  // Mass units
  using kilogram = Unit<mass_dimension>;
  using gram = Unit<mass_dimension, std::ratio<1, 1000>>;
  using tonne = Unit<mass_dimension, std::ratio<1000>>;

  // Temperature units
  using kelvin = Unit<temperature_dimension>;

  // Angle units
  using radian = Unit<angle_dimension>;
  using degree = Unit<angle_dimension, std::ratio<1745329252, 100000000000>>; // π/180

  // Derived units
  using meters_per_second = Unit<velocity_dimension>;
  using kilometers_per_hour = Unit<velocity_dimension, std::ratio<1000, 3600>>;
  using meters_per_second_squared = Unit<acceleration_dimension>;
  using newton = Unit<force_dimension>;
  using pascal = Unit<pressure_dimension>;
  using kilopascal = Unit<pressure_dimension, std::ratio<1000>>;
  using bar = Unit<pressure_dimension, std::ratio<100000>>;
  using joule = Unit<energy_dimension>;
  using watt = Unit<power_dimension>;
  using kilogram_per_cubic_meter = Unit<density_dimension>;

  // Division of units
  template <UnitType U1, UnitType U2>
  [[nodiscard]] constexpr auto operator/(const U1& lhs, const U2& rhs) noexcept {
    return lhs.value() / rhs.value();
  }

  // Special case for length/time = velocity
  [[nodiscard]] constexpr meters_per_second operator/(const meter& length, const second& time) noexcept {
    return meters_per_second(length.value() / time.value());
  }

  /**
   * @brief Stream output for units with optional Unicode support
   */
  template <UnitType U>
  std::ostream& operator<<(std::ostream& os, const U& unit) {
    os << unit.value();

    // Only add unit symbol if not dimensionless
    if constexpr (!std::is_same_v<typename U::dimension_type, dimensionless_dimension>) {
      os << " ";

      if constexpr (std::is_same_v<typename U::dimension_type, length_dimension>) {
	os << "m";
      }
      else if constexpr (std::is_same_v<typename U::dimension_type, time_dimension>) {
	os << "s";
      }
      else if constexpr (std::is_same_v<typename U::dimension_type, mass_dimension>) {
	os << "kg";
      }
      else if constexpr (std::is_same_v<typename U::dimension_type, temperature_dimension>) {
	os << "K";
      }
      else if constexpr (std::is_same_v<typename U::dimension_type, angle_dimension>) {
	  #ifdef UNITS_USE_UNICODE
	os << "°";
	  #else
	os << "rad";
	  #endif
      }
      else if constexpr (std::is_same_v<typename U::dimension_type, area_dimension>) {
	  #ifdef UNITS_USE_UNICODE
	os << "m²";
	  #else
	os << "m^2";
	  #endif
      }
      else if constexpr (std::is_same_v<typename U::dimension_type, volume_dimension>) {
	  #ifdef UNITS_USE_UNICODE
	os << "m³";
	  #else
	os << "m^3";
	  #endif
      }
      else if constexpr (std::is_same_v<typename U::dimension_type, velocity_dimension>) {
	os << "m/s";
      }
      else if constexpr (std::is_same_v<typename U::dimension_type, acceleration_dimension>) {
	  #ifdef UNITS_USE_UNICODE
	os << "m/s²";
	  #else
	os << "m/s^2";
	  #endif
      }
      else if constexpr (std::is_same_v<typename U::dimension_type, force_dimension>) {
	os << "N";
      }
      else if constexpr (std::is_same_v<typename U::dimension_type, pressure_dimension>) {
	os << "Pa";
      }
      else if constexpr (std::is_same_v<typename U::dimension_type, energy_dimension>) {
	os << "J";
      }
      else if constexpr (std::is_same_v<typename U::dimension_type, power_dimension>) {
	os << "W";
      }
      else if constexpr (std::is_same_v<typename U::dimension_type, density_dimension>) {
	  #ifdef UNITS_USE_UNICODE
	os << "kg/m³";
	  #else
	os << "kg/m^3";
	  #endif
      }
      else if constexpr (std::is_same_v<typename U::dimension_type, frequency_dimension>) {
	os << "Hz";
      }
    }

    return os;
  }

  // SI prefixes for convenience
  constexpr double kilo = 1e3;
  constexpr double mega = 1e6;
  constexpr double giga = 1e9;
  constexpr double milli = 1e-3;
  constexpr double micro = 1e-6;
  constexpr double nano = 1e-9;

  // Temperature conversion helpers (special case due to offsets)
  [[nodiscard]] constexpr kelvin celsius_to_kelvin(double celsius) noexcept {
    return kelvin(celsius + 273.15);
  }

  [[nodiscard]] constexpr double kelvin_to_celsius(const kelvin& k) noexcept {
    return k.value() - 273.15;
  }

  [[nodiscard]] constexpr kelvin fahrenheit_to_kelvin(double fahrenheit) noexcept {
    return kelvin((fahrenheit - 32.0) * 5.0 / 9.0 + 273.15);
  }

  [[nodiscard]] constexpr double kelvin_to_fahrenheit(const kelvin& k) noexcept {
    return (k.value() - 273.15) * 9.0 / 5.0 + 32.0;
  }

  // Unit validation functions
  [[nodiscard]] constexpr bool is_valid_temperature(const kelvin& k) noexcept {
    return k.value() >= constants::ABSOLUTE_ZERO;
  }

  [[nodiscard]] constexpr bool is_valid_pressure(const pascal& p) noexcept {
    return p.value() >= 0.0;
  }

  [[nodiscard]] constexpr kelvin enforce_valid_temperature(const kelvin& k) noexcept {
    return kelvin(k.value() < constants::ABSOLUTE_ZERO ? constants::ABSOLUTE_ZERO : k.value());
  }

  [[nodiscard]] constexpr pascal enforce_valid_pressure(const pascal& p) noexcept {
    return pascal(p.value() < 0.0 ? 0.0 : p.value());
  }

  //==============================================================================
  // String-based unit conversion interface (for backward compatibility)
  //==============================================================================

  /**
   * @brief Convert a value from one unit string to another with runtime checking
   */
  [[nodiscard]] inline double convert(double value, const std::string& fromUnit, const std::string& toUnit) {
    if (fromUnit == toUnit) {
      return value;
    }

    // Temperature conversions (special case due to offsets)
    if (fromUnit == "K" && toUnit == "C") return kelvin_to_celsius(kelvin(value));
    if (fromUnit == "C" && toUnit == "K") return celsius_to_kelvin(value).value();
    if (fromUnit == "F" && toUnit == "K") return fahrenheit_to_kelvin(value).value();
    if (fromUnit == "K" && toUnit == "F") return kelvin_to_fahrenheit(kelvin(value));
    if (fromUnit == "C" && toUnit == "F") return kelvin_to_fahrenheit(celsius_to_kelvin(value));
    if (fromUnit == "F" && toUnit == "C") return kelvin_to_celsius(fahrenheit_to_kelvin(value));

    // Length conversions
    if (fromUnit == "m" && toUnit == "cm") return value * 100.0;
    if (fromUnit == "cm" && toUnit == "m") return value * 0.01;
    if (fromUnit == "m" && toUnit == "km") return value * 0.001;
    if (fromUnit == "km" && toUnit == "m") return value * 1000.0;
    if (fromUnit == "m" && toUnit == "in") return value * 39.3701;
    if (fromUnit == "in" && toUnit == "m") return value * 0.0254;
    if (fromUnit == "m" && toUnit == "ft") return value * 3.28084;
    if (fromUnit == "ft" && toUnit == "m") return value * 0.3048;
    if (fromUnit == "m" && toUnit == "mi") return value * 0.000621371;
    if (fromUnit == "mi" && toUnit == "m") return value * 1609.34;

    // Pressure conversions
    if (fromUnit == "Pa" && toUnit == "bar") return value * 1.0e-5;
    if (fromUnit == "bar" && toUnit == "Pa") return value * 1.0e5;
    if (fromUnit == "Pa" && toUnit == "atm") return value / constants::STANDARD_ATM_PRESSURE;
    if (fromUnit == "atm" && toUnit == "Pa") return value * constants::STANDARD_ATM_PRESSURE;
    if (fromUnit == "Pa" && toUnit == "psi") return value * 0.000145038;
    if (fromUnit == "psi" && toUnit == "Pa") return value * 6894.76;
    if (fromUnit == "Pa" && toUnit == "torr") return value * 0.00750062;
    if (fromUnit == "torr" && toUnit == "Pa") return value * 133.322;

    // Thermal conductivity
    if (fromUnit == "W/(m·K)" && toUnit == "BTU/(hr·ft·°F)") return value * 0.5779;
    if (fromUnit == "BTU/(hr·ft·°F)" && toUnit == "W/(m·K)") return value * 1.73073;

    // Specific heat
    if (fromUnit == "J/(kg·K)" && toUnit == "BTU/(lb·°F)") return value * 0.000238846;
    if (fromUnit == "BTU/(lb·°F)" && toUnit == "J/(kg·K)") return value * 4186.8;

    // Density
    if (fromUnit == "kg/m³" && toUnit == "g/cm³") return value * 0.001;
    if (fromUnit == "g/cm³" && toUnit == "kg/m³") return value * 1000.0;
    if (fromUnit == "kg/m³" && toUnit == "lb/ft³") return value * 0.0624279606;
    if (fromUnit == "lb/ft³" && toUnit == "kg/m³") return value * 16.0185;

    // Viscosity
    if (fromUnit == "Pa·s" && toUnit == "cP") return value * 1000.0;
    if (fromUnit == "cP" && toUnit == "Pa·s") return value * 0.001;
    if (fromUnit == "Pa·s" && toUnit == "lb·s/ft²") return value * 0.0208854;
    if (fromUnit == "lb·s/ft²" && toUnit == "Pa·s") return value * 47.8803;

    // If no conversion was found
    throw std::invalid_argument("Cannot convert between units: " + fromUnit + " to " + toUnit);
  }

  // Compile-time verification examples
  static_assert(unit_cast<centimeter>(meter(1.0)).value() == 100.0, "Unit conversion failed");
  static_assert(unit_cast<kilometer>(meter(1000.0)).value() == 1.0, "Unit conversion failed");
  static_assert(celsius_to_kelvin(0.0).value() == 273.15, "Temperature conversion failed");

} // namespace units

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
