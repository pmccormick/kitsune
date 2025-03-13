/**
 * @file UnitsConstants.h
 * @brief Physical constants and conversion factors for use with Units.h
 *
 * This file provides a comprehensive set of physical constants and conversion
 * factors for scientific and engineering calculations. All values are defined
 * as constexpr to enable compile-time computations.
 */

#ifndef UNITS_CONSTANTS_H
#define UNITS_CONSTANTS_H

namespace units {
  namespace constants {

    //==============================================================================
    // Physical constants in SI units
    //==============================================================================

    // Fundamental constants
    constexpr double PI = 3.14159265358979323846;          // Mathematical constant π
    constexpr double E = 2.71828182845904523536;           // Mathematical constant e
    constexpr double AVOGADRO = 6.02214076e23;             // Avogadro's number [1/mol]
    constexpr double BOLTZMANN = 1.380649e-23;             // Boltzmann constant [J/K]
    constexpr double PLANCK = 6.62607015e-34;              // Planck constant [J⋅s]
    constexpr double SPEED_OF_LIGHT = 299792458.0;         // Speed of light in vacuum [m/s]
    constexpr double ELECTRON_CHARGE = 1.602176634e-19;    // Elementary charge [C]
    constexpr double VACUUM_PERMITTIVITY = 8.8541878128e-12; // Vacuum permittivity [F/m]
    constexpr double VACUUM_PERMEABILITY = 1.25663706212e-6; // Vacuum permeability [H/m]

    // Gravitational constants
    constexpr double STANDARD_GRAVITY = 9.80665;           // Standard gravity [m/s²]
    constexpr double GRAVITATIONAL_CONSTANT = 6.67430e-11; // Gravitational constant [m³/(kg⋅s²)]
    constexpr double EARTH_MASS = 5.9722e24;               // Mass of Earth [kg]
    constexpr double EARTH_RADIUS = 6.3781e6;              // Mean radius of Earth [m]

    // Thermodynamic constants
    constexpr double STANDARD_TEMPERATURE = 293.15;        // Standard temperature [K] (20°C)
    constexpr double ABSOLUTE_ZERO = 0.0;                  // Absolute zero [K]
    constexpr double WATER_FREEZING_POINT = 273.15;        // Water freezing point [K] (0°C)
    constexpr double WATER_BOILING_POINT = 373.15;         // Water boiling point [K] (100°C)
    constexpr double GAS_CONSTANT = 8.31446261815324;      // Universal gas constant [J/(mol⋅K)]
    constexpr double STANDARD_ATM_PRESSURE = 101325.0;     // Standard atmospheric pressure [Pa]

    // Material properties at standard conditions
    constexpr double WATER_DENSITY = 997.0;                // Density of water [kg/m³]
    constexpr double AIR_DENSITY = 1.225;                  // Density of air [kg/m³]
    constexpr double WATER_SPECIFIC_HEAT = 4186.0;         // Specific heat of water [J/(kg⋅K)]
    constexpr double AIR_SPECIFIC_HEAT = 1005.0;           // Specific heat of air [J/(kg⋅K)]
    constexpr double WATER_DYNAMIC_VISCOSITY = 8.90e-4;    // Dynamic viscosity of water [Pa⋅s]
    constexpr double AIR_DYNAMIC_VISCOSITY = 1.81e-5;      // Dynamic viscosity of air [Pa⋅s]
    constexpr double WATER_THERMAL_CONDUCTIVITY = 0.6;     // Thermal conductivity of water [W/(m⋅K)]
    constexpr double AIR_THERMAL_CONDUCTIVITY = 0.026;     // Thermal conductivity of air [W/(m⋅K)]

    // Electromagnetic constants
    constexpr double VACUUM_IMPEDANCE = 376.730313668;     // Impedance of vacuum [Ω]
    constexpr double ELECTRON_MASS = 9.1093837015e-31;     // Electron mass [kg]
    constexpr double PROTON_MASS = 1.67262192369e-27;      // Proton mass [kg]
    constexpr double NEUTRON_MASS = 1.67492749804e-27;     // Neutron mass [kg]
    constexpr double FINE_STRUCTURE = 7.2973525693e-3;     // Fine-structure constant [-]

    // Nuclear constants
    constexpr double ATOMIC_MASS_UNIT = 1.66053906660e-27; // Atomic mass unit [kg]
    constexpr double NUCLEAR_MAGNETON = 5.0507837461e-27;  // Nuclear magneton [J/T]
    constexpr double BOHR_MAGNETON = 9.2740100783e-24;     // Bohr magneton [J/T]

    // Conversion factors (dimensionless)
    constexpr double DEG_TO_RAD = PI / 180.0;              // Degrees to radians conversion
    constexpr double RAD_TO_DEG = 180.0 / PI;              // Radians to degrees conversion
    constexpr double INCH_TO_METER = 0.0254;               // Inches to meters conversion
    constexpr double FOOT_TO_METER = 0.3048;               // Feet to meters conversion
    constexpr double MILE_TO_METER = 1609.344;             // Miles to meters conversion
    constexpr double POUND_TO_KG = 0.45359237;             // Pounds to kilograms conversion
    constexpr double PSI_TO_PA = 6894.76;                  // PSI to Pascals conversion
    constexpr double BTU_TO_JOULE = 1055.05585262;         // BTU to Joules conversion
    constexpr double HP_TO_WATT = 745.699872;              // Horsepower to Watts conversion
    constexpr double GAL_TO_M3 = 0.003785411784;           // US gallon to cubic meters conversion

    // Typical engineering values
    constexpr double STEEL_YOUNGS_MODULUS = 200e9;         // Young's modulus of steel [Pa]
    constexpr double ALUMINUM_YOUNGS_MODULUS = 69e9;       // Young's modulus of aluminum [Pa]
    constexpr double CONCRETE_YOUNGS_MODULUS = 30e9;       // Young's modulus of concrete [Pa]
    constexpr double STEEL_DENSITY = 7850.0;               // Density of steel [kg/m³]
    constexpr double ALUMINUM_DENSITY = 2700.0;            // Density of aluminum [kg/m³]
    constexpr double CONCRETE_DENSITY = 2400.0;            // Density of concrete [kg/m³]
    constexpr double STEEL_POISSON_RATIO = 0.3;            // Poisson's ratio of steel [-]
    constexpr double ALUMINUM_POISSON_RATIO = 0.33;        // Poisson's ratio of aluminum [-]
    constexpr double CONCRETE_POISSON_RATIO = 0.2;         // Poisson's ratio of concrete [-]

    // Tolerance values for numerical computations
    constexpr double EPSILON = 1.0e-10;                    // Small value for floating-point comparisons
    constexpr double TOLERANCE_FACTOR = 1.0e-6;            // Relative tolerance for iterative methods

  } // namespace constants
} // namespace units

#endif // UNITS_CONSTANTS_H
