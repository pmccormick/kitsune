#include <gtest/gtest.h>
#include "units/core.h"
#include "units/si_units.h"
#include "units/derived_units.h"
#include "units/conversions.h"
#include "units/constants.h"
#include <stdexcept>

using namespace units;

TEST(UnitsStringConversion, LengthUnitConversions) {
    // Test string-based conversions for length units
    
    // Meters to centimeters
    EXPECT_DOUBLE_EQ(100.0, convert(1.0, "m", "cm"));
    
    // Centimeters to meters
    EXPECT_DOUBLE_EQ(0.01, convert(1.0, "cm", "m"));
    
    // Meters to kilometers
    EXPECT_DOUBLE_EQ(0.001, convert(1.0, "m", "km"));
    
    // Kilometers to meters
    EXPECT_DOUBLE_EQ(1000.0, convert(1.0, "km", "m"));
    
    // Meters to millimeters
    EXPECT_DOUBLE_EQ(1000.0, convert(1.0, "m", "mm"));
    
    // Millimeters to meters
    EXPECT_DOUBLE_EQ(0.001, convert(1.0, "mm", "m"));
    
    // Meters to inches
    EXPECT_NEAR(39.3701, convert(1.0, "m", "in"), 1e-4);
    
    // Inches to meters
    EXPECT_NEAR(0.0254, convert(1.0, "in", "m"), 1e-6);
    
    // Meters to feet
    EXPECT_NEAR(3.28084, convert(1.0, "m", "ft"), 1e-5);
    
    // Feet to meters
    EXPECT_NEAR(0.3048, convert(1.0, "ft", "m"), 1e-6);
    
    // Meters to miles
    EXPECT_NEAR(0.000621371, convert(1.0, "m", "mi"), 1e-9);
    
    // Miles to meters
    EXPECT_NEAR(1609.34, convert(1.0, "mi", "m"), 1e-2);
}

TEST(UnitsStringConversion, AreaUnitConversions) {
    // Test string-based conversions for area units
    
    // Square meters to square centimeters
    EXPECT_DOUBLE_EQ(10000.0, convert(1.0, "m²", "cm²"));
    
    // Square centimeters to square meters
    EXPECT_DOUBLE_EQ(0.0001, convert(1.0, "cm²", "m²"));
    
    // Square meters to square kilometers
    EXPECT_DOUBLE_EQ(0.000001, convert(1.0, "m²", "km²"));
    
    // Square kilometers to square meters
    EXPECT_DOUBLE_EQ(1000000.0, convert(1.0, "km²", "m²"));
    
    // Square meters to hectares
    EXPECT_DOUBLE_EQ(0.0001, convert(1.0, "m²", "ha"));
    
    // Hectares to square meters
    EXPECT_DOUBLE_EQ(10000.0, convert(1.0, "ha", "m²"));
    
    // Square meters to acres
    EXPECT_NEAR(0.000247105, convert(1.0, "m²", "acre"), 1e-9);
    
    // Acres to square meters
    EXPECT_NEAR(4046.86, convert(1.0, "acre", "m²"), 1e-2);
    
    // Square meters to square feet
    EXPECT_NEAR(10.7639, convert(1.0, "m²", "ft²"), 1e-4);
    
    // Square feet to square meters
    EXPECT_NEAR(0.092903, convert(1.0, "ft²", "m²"), 1e-6);
}

TEST(UnitsStringConversion, VolumeUnitConversions) {
    // Test string-based conversions for volume units
    
    // Cubic meters to liters
    EXPECT_DOUBLE_EQ(1000.0, convert(1.0, "m³", "L"));
    
    // Liters to cubic meters
    EXPECT_DOUBLE_EQ(0.001, convert(1.0, "L", "m³"));
    
    // Liters to milliliters
    EXPECT_DOUBLE_EQ(1000.0, convert(1.0, "L", "mL"));
    
    // Milliliters to liters
    EXPECT_DOUBLE_EQ(0.001, convert(1.0, "mL", "L"));
    
    // Liters to US gallons
    EXPECT_NEAR(0.264172, convert(1.0, "L", "gal"), 1e-6);
    
    // US gallons to liters
    EXPECT_NEAR(3.78541, convert(1.0, "gal", "L"), 1e-5);
    
    // Cubic meters to cubic feet
    EXPECT_NEAR(35.3147, convert(1.0, "m³", "ft³"), 1e-4);
    
    // Cubic feet to cubic meters
    EXPECT_NEAR(0.0283168, convert(1.0, "ft³", "m³"), 1e-6);
}

TEST(UnitsStringConversion, MassUnitConversions) {
    // Test string-based conversions for mass units
    
    // Kilograms to grams
    EXPECT_DOUBLE_EQ(1000.0, convert(1.0, "kg", "g"));
    
    // Grams to kilograms
    EXPECT_DOUBLE_EQ(0.001, convert(1.0, "g", "kg"));
    
    // Kilograms to pounds
    EXPECT_NEAR(2.20462, convert(1.0, "kg", "lb"), 1e-5);
    
    // Pounds to kilograms
    EXPECT_NEAR(0.453592, convert(1.0, "lb", "kg"), 1e-6);
    
    // Grams to ounces
    EXPECT_NEAR(0.035274, convert(1.0, "g", "oz"), 1e-6);
    
    // Ounces to grams
    EXPECT_NEAR(28.3495, convert(1.0, "oz", "g"), 1e-4);
    
    // Kilograms to tonnes
    EXPECT_DOUBLE_EQ(0.001, convert(1.0, "kg", "t"));
    
    // Tonnes to kilograms
    EXPECT_DOUBLE_EQ(1000.0, convert(1.0, "t", "kg"));
}

TEST(UnitsStringConversion, TimeUnitConversions) {
    // Test string-based conversions for time units
    
    // Seconds to minutes
    EXPECT_DOUBLE_EQ(1.0/60.0, convert(1.0, "s", "min"));
    
    // Minutes to seconds
    EXPECT_DOUBLE_EQ(60.0, convert(1.0, "min", "s"));
    
    // Seconds to hours
    EXPECT_DOUBLE_EQ(1.0/3600.0, convert(1.0, "s", "h"));
    
    // Hours to seconds
    EXPECT_DOUBLE_EQ(3600.0, convert(1.0, "h", "s"));
    
    // Seconds to milliseconds
    EXPECT_DOUBLE_EQ(1000.0, convert(1.0, "s", "ms"));
    
    // Milliseconds to seconds
    EXPECT_DOUBLE_EQ(0.001, convert(1.0, "ms", "s"));
    
    // Seconds to microseconds
    EXPECT_DOUBLE_EQ(1000000.0, convert(1.0, "s", "μs"));
    
    // Microseconds to seconds
    EXPECT_DOUBLE_EQ(0.000001, convert(1.0, "μs", "s"));
    
    // Minutes to hours
    EXPECT_DOUBLE_EQ(1.0/60.0, convert(1.0, "min", "h"));
    
    // Hours to minutes
    EXPECT_DOUBLE_EQ(60.0, convert(1.0, "h", "min"));
    
    // Hours to days
    EXPECT_DOUBLE_EQ(1.0/24.0, convert(1.0, "h", "day"));
    
    // Days to hours
    EXPECT_DOUBLE_EQ(24.0, convert(1.0, "day", "h"));
}

TEST(UnitsStringConversion, TemperatureUnitConversions) {
    // Test string-based conversions for temperature units
    
    // Kelvin to Celsius
    EXPECT_DOUBLE_EQ(0.0, convert(273.15, "K", "C"));
    EXPECT_DOUBLE_EQ(100.0, convert(373.15, "K", "C"));
    EXPECT_DOUBLE_EQ(-273.15, convert(0.0, "K", "C"));
    
    // Celsius to Kelvin
    EXPECT_DOUBLE_EQ(273.15, convert(0.0, "C", "K"));
    EXPECT_DOUBLE_EQ(373.15, convert(100.0, "C", "K"));
    EXPECT_DOUBLE_EQ(0.0, convert(-273.15, "C", "K"));
    
    // Kelvin to Fahrenheit
    EXPECT_NEAR(32.0, convert(273.15, "K", "F"), 1e-12);
    EXPECT_NEAR(212.0, convert(373.15, "K", "F"), 1e-12);
    EXPECT_NEAR(-459.67, convert(0.0, "K", "F"), 1e-2);
    
    // Fahrenheit to Kelvin
    EXPECT_NEAR(273.15, convert(32.0, "F", "K"), 1e-12);
    EXPECT_NEAR(373.15, convert(212.0, "F", "K"), 1e-12);
    EXPECT_NEAR(0.0, convert(-459.67, "F", "K"), 1e-12);
    
    // Celsius to Fahrenheit
    EXPECT_NEAR(32.0, convert(0.0, "C", "F"), 1e-12);
    EXPECT_NEAR(212.0, convert(100.0, "C", "F"), 1e-12);
    EXPECT_NEAR(-40.0, convert(-40.0, "C", "F"), 1e-12);
    
    // Fahrenheit to Celsius
    EXPECT_NEAR(0.0, convert(32.0, "F", "C"), 1e-12);
    EXPECT_NEAR(100.0, convert(212.0, "F", "C"), 1e-12);
    EXPECT_NEAR(-40.0, convert(-40.0, "F", "C"), 1e-12);
}

TEST(UnitsStringConversion, PressureUnitConversions) {
    // Test string-based conversions for pressure units
    
    // Pascal to kilopascal
    EXPECT_DOUBLE_EQ(0.001, convert(1.0, "Pa", "kPa"));
    
    // Kilopascal to pascal
    EXPECT_DOUBLE_EQ(1000.0, convert(1.0, "kPa", "Pa"));
    
    // Pascal to bar
    EXPECT_DOUBLE_EQ(1.0e-5, convert(1.0, "Pa", "bar"));
    
    // Bar to pascal
    EXPECT_DOUBLE_EQ(1.0e5, convert(1.0, "bar", "Pa"));
    
    // Pascal to atmosphere
    EXPECT_DOUBLE_EQ(1.0/constants::STANDARD_ATM_PRESSURE, convert(1.0, "Pa", "atm"));
    
    // Atmosphere to pascal
    EXPECT_DOUBLE_EQ(constants::STANDARD_ATM_PRESSURE, convert(1.0, "atm", "Pa"));
    
    // Pascal to PSI
    EXPECT_NEAR(0.000145038, convert(1.0, "Pa", "psi"), 1e-9);
    
    // PSI to pascal
    EXPECT_NEAR(6894.76, convert(1.0, "psi", "Pa"), 1e-2);
}

TEST(UnitsStringConversion, VelocityUnitConversions) {
    // Test string-based conversions for velocity units
    
    // Meters per second to kilometers per hour
    EXPECT_DOUBLE_EQ(3.6, convert(1.0, "m/s", "km/h"));
    
    // Kilometers per hour to meters per second
    EXPECT_DOUBLE_EQ(1.0/3.6, convert(1.0, "km/h", "m/s"));
    
    // Meters per second to miles per hour
    EXPECT_NEAR(2.23694, convert(1.0, "m/s", "mph"), 1e-5);
    
    // Miles per hour to meters per second
    EXPECT_NEAR(0.44704, convert(1.0, "mph", "m/s"), 1e-5);
    
    // Kilometers per hour to miles per hour
    EXPECT_NEAR(0.621371, convert(1.0, "km/h", "mph"), 1e-6);
    
    // Miles per hour to kilometers per hour
    EXPECT_NEAR(1.60934, convert(1.0, "mph", "km/h"), 1e-5);
}

TEST(UnitsStringConversion, EnergyUnitConversions) {
    // Test string-based conversions for energy units
    
    // Joule to kilojoule
    EXPECT_DOUBLE_EQ(0.001, convert(1.0, "J", "kJ"));
    
    // Kilojoule to joule
    EXPECT_DOUBLE_EQ(1000.0, convert(1.0, "kJ", "J"));
    
    // Joule to calorie
    EXPECT_NEAR(0.239006, convert(1.0, "J", "cal"), 1e-6);
    
    // Calorie to joule
    EXPECT_NEAR(4.184, convert(1.0, "cal", "J"), 1e-3);
    
    // Joule to kilowatt-hour
    EXPECT_DOUBLE_EQ(1.0/3600000.0, convert(1.0, "J", "kWh"));
    
    // Kilowatt-hour to joule
    EXPECT_DOUBLE_EQ(3600000.0, convert(1.0, "kWh", "J"));
}

TEST(UnitsStringConversion, PowerUnitConversions) {
    // Test string-based conversions for power units
    
    // Watt to kilowatt
    EXPECT_DOUBLE_EQ(0.001, convert(1.0, "W", "kW"));
    
    // Kilowatt to watt
    EXPECT_DOUBLE_EQ(1000.0, convert(1.0, "kW", "W"));
    
    // Watt to horsepower
    EXPECT_NEAR(0.00134102, convert(1.0, "W", "hp"), 1e-8);
    
    // Horsepower to watt
    EXPECT_NEAR(745.7, convert(1.0, "hp", "W"), 0.1);
}

TEST(UnitsStringConversion, ErrorHandling) {
    // Test error handling for invalid or incompatible unit conversions
    
    // Same source and destination units
    EXPECT_DOUBLE_EQ(42.0, convert(42.0, "m", "m"));
    EXPECT_DOUBLE_EQ(98.6, convert(98.6, "F", "F"));
    
    // Invalid unit names
    EXPECT_THROW(convert(1.0, "invalid", "m"), std::invalid_argument);
    EXPECT_THROW(convert(1.0, "m", "invalid"), std::invalid_argument);
    
    // Incompatible unit types
    EXPECT_THROW(convert(1.0, "m", "s"), std::invalid_argument);
    EXPECT_THROW(convert(1.0, "kg", "W"), std::invalid_argument);
    EXPECT_THROW(convert(1.0, "C", "m"), std::invalid_argument);
}

TEST(UnitsStringConversion, LegacyCompatibility) {
    // Test compatibility with the original Units::convert function
    
    // Use the Units namespace function
    EXPECT_DOUBLE_EQ(100.0, Units::convert(1.0, "m", "cm"));
    EXPECT_DOUBLE_EQ(1000.0, Units::convert(1.0, "kg", "g"));
    EXPECT_DOUBLE_EQ(0.001, Units::convert(1.0, "km", "m"));
    
    // Temperature conversions
    EXPECT_DOUBLE_EQ(0.0, Units::convert(273.15, "K", "C"));
    EXPECT_NEAR(32.0, Units::convert(0.0, "C", "F"), 1e-12);
    
    // Error handling
    EXPECT_THROW(Units::convert(1.0, "invalid", "m"), std::invalid_argument);
    
    // Make sure the Units::convert and units::convert functions behave identically
    EXPECT_DOUBLE_EQ(Units::convert(1.0, "m", "cm"), units::convert(1.0, "m", "cm"));
    EXPECT_DOUBLE_EQ(Units::convert(100.0, "C", "F"), units::convert(100.0, "C", "F"));
}

TEST(UnitsStringConversion, DataStorageConversions) {
    // Test string-based conversions for data storage units
    
    // Byte to kilobyte
    EXPECT_DOUBLE_EQ(1.0/1024.0, convert(1.0, "B", "KB"));
    
    // Kilobyte to byte
    EXPECT_DOUBLE_EQ(1024.0, convert(1.0, "KB", "B"));
    
    // Kilobyte to megabyte
    EXPECT_DOUBLE_EQ(1.0/1024.0, convert(1.0, "KB", "MB"));
    
    // Megabyte to kilobyte
    EXPECT_DOUBLE_EQ(1024.0, convert(1.0, "MB", "KB"));
    
    // Megabyte to gigabyte
    EXPECT_DOUBLE_EQ(1.0/1024.0, convert(1.0, "MB", "GB"));
    
    // Gigabyte to megabyte
    EXPECT_DOUBLE_EQ(1024.0, convert(1.0, "GB", "MB"));
    
    // Byte to bit
    EXPECT_DOUBLE_EQ(8.0, convert(1.0, "B", "bit"));
    
    // Bit to byte
    EXPECT_DOUBLE_EQ(1.0/8.0, convert(1.0, "bit", "B"));
}
