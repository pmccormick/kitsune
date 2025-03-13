#include <gtest/gtest.h>
#include "units/core.h"
#include "units/si_units.h"
#include "units/derived_units.h"
#include "units/validation.h"
#include "units/constants.h"

using namespace units;

TEST(UnitsValidation, TemperatureConversions) {
    // Test Celsius to Kelvin conversion
    double freezingInC = 0.0;
    kelvin freezingInK = celsius_to_kelvin(freezingInC);
    EXPECT_DOUBLE_EQ(273.15, freezingInK.value());
    
    double boilingInC = 100.0;
    kelvin boilingInK = celsius_to_kelvin(boilingInC);
    EXPECT_DOUBLE_EQ(373.15, boilingInK.value());
    
    double roomTempInC = 20.0;
    kelvin roomTempInK = celsius_to_kelvin(roomTempInC);
    EXPECT_DOUBLE_EQ(293.15, roomTempInK.value());
    
    // Test Kelvin to Celsius conversion
    kelvin absZeroK(0.0);
    double absZeroC = kelvin_to_celsius(absZeroK);
    EXPECT_DOUBLE_EQ(-273.15, absZeroC);
    
    kelvin freezingK(273.15);
    double freezingC = kelvin_to_celsius(freezingK);
    EXPECT_DOUBLE_EQ(0.0, freezingC);
    
    kelvin boilingK(373.15);
    double boilingC = kelvin_to_celsius(boilingK);
    EXPECT_DOUBLE_EQ(100.0, boilingC);
    
    // Test Fahrenheit to Kelvin conversion
    double freezingInF = 32.0;
    kelvin freezingKfromF = fahrenheit_to_kelvin(freezingInF);
    EXPECT_NEAR(273.15, freezingKfromF.value(), 1e-12);
    
    double boilingInF = 212.0;
    kelvin boilingKfromF = fahrenheit_to_kelvin(boilingInF);
    EXPECT_NEAR(373.15, boilingKfromF.value(), 1e-12);
    
    double roomTempInF = 68.0;
    kelvin roomTempKfromF = fahrenheit_to_kelvin(roomTempInF);
    EXPECT_NEAR(293.15, roomTempKfromF.value(), 1e-2);
    
    // Test Kelvin to Fahrenheit conversion
    kelvin absZeroKtoF(0.0);
    double absZeroF = kelvin_to_fahrenheit(absZeroKtoF);
    EXPECT_NEAR(-459.67, absZeroF, 1e-2);
    
    kelvin freezingKtoF(273.15);
    double freezingF = kelvin_to_fahrenheit(freezingKtoF);
    EXPECT_NEAR(32.0, freezingF, 1e-12);
    
    kelvin boilingKtoF(373.15);
    double boilingF = kelvin_to_fahrenheit(boilingKtoF);
    EXPECT_NEAR(212.0, boilingF, 1e-12);
}

TEST(UnitsValidation, TemperatureRoundTrip) {
    // Test round-trip conversions for temperature
    
    // Celsius -> Kelvin -> Celsius
    double originalC = 25.0;
    kelvin intermediate = celsius_to_kelvin(originalC);
    double backToC = kelvin_to_celsius(intermediate);
    EXPECT_DOUBLE_EQ(originalC, backToC);
    
    // Kelvin -> Celsius -> Kelvin
    kelvin originalK(300.0);
    double intermediateC = kelvin_to_celsius(originalK);
    kelvin backToK = celsius_to_kelvin(intermediateC);
    EXPECT_DOUBLE_EQ(originalK.value(), backToK.value());
    
    // Fahrenheit -> Kelvin -> Fahrenheit
    double originalF = 98.6;  // Body temperature
    kelvin intermediateK = fahrenheit_to_kelvin(originalF);
    double backToF = kelvin_to_fahrenheit(intermediateK);
    EXPECT_NEAR(originalF, backToF, 1e-12);
    
    // Celsius -> Fahrenheit -> Celsius (via Kelvin)
    double anotherC = 37.0;  // Body temperature
    kelvin toK = celsius_to_kelvin(anotherC);
    double toF = kelvin_to_fahrenheit(toK);
    kelvin backToK2 = fahrenheit_to_kelvin(toF);
    double backToC2 = kelvin_to_celsius(backToK2);
    EXPECT_NEAR(anotherC, backToC2, 1e-12);
}

TEST(UnitsValidation, ValidTemperature) {
    // Test is_valid_temperature function
    
    // Valid temperatures (at or above absolute zero)
    kelvin absZero(constants::ABSOLUTE_ZERO);  // 0 K
    EXPECT_TRUE(is_valid_temperature(absZero));
    
    kelvin roomTemp(293.15);  // ~20°C
    EXPECT_TRUE(is_valid_temperature(roomTemp));
    
    kelvin veryCold(0.001);  // Just above absolute zero
    EXPECT_TRUE(is_valid_temperature(veryCold));
    
    // Invalid temperatures (below absolute zero)
    kelvin belowAbsZero(-0.001);
    EXPECT_FALSE(is_valid_temperature(belowAbsZero));
    
    kelvin veryNegative(-100.0);
    EXPECT_FALSE(is_valid_temperature(veryNegative));
}

TEST(UnitsValidation, ValidPressure) {
    // Test is_valid_pressure function
    
    // Valid pressures (zero or positive)
    pascal zeroPressure(0.0);
    EXPECT_TRUE(is_valid_pressure(zeroPressure));
    
    pascal atmPressure(constants::STANDARD_ATM_PRESSURE);  // ~101325 Pa
    EXPECT_TRUE(is_valid_pressure(atmPressure));
    
    pascal veryHighPressure(1e9);  // 1 GPa (extreme pressure)
    EXPECT_TRUE(is_valid_pressure(veryHighPressure));
    
    // Invalid pressures (negative)
    pascal negativePressure(-1.0);
    EXPECT_FALSE(is_valid_pressure(negativePressure));
    
    pascal veryNegativePressure(-1e5);
    EXPECT_FALSE(is_valid_pressure(veryNegativePressure));
}

TEST(UnitsValidation, ValidDensity) {
    // Test is_valid_density function
    
    // Valid densities (zero or positive)
    kilogram_per_cubic_meter zeroDensity(0.0);
    EXPECT_TRUE(is_valid_density(zeroDensity));
    
    kilogram_per_cubic_meter airDensity(1.225);  // Air at sea level
    EXPECT_TRUE(is_valid_density(airDensity));
    
    kilogram_per_cubic_meter waterDensity(1000.0);  // Water
    EXPECT_TRUE(is_valid_density(waterDensity));
    
    kilogram_per_cubic_meter highDensity(19300.0);  // Gold
    EXPECT_TRUE(is_valid_density(highDensity));
    
    // Invalid densities (negative)
    kilogram_per_cubic_meter negativeDensity(-1.0);
    EXPECT_FALSE(is_valid_density(negativeDensity));
    
    kilogram_per_cubic_meter veryNegativeDensity(-1000.0);
    EXPECT_FALSE(is_valid_density(veryNegativeDensity));
}

TEST(UnitsValidation, ValidMass) {
    // Test is_valid_mass function
    
    // Valid masses (zero or positive)
    kilogram zeroMass(0.0);
    EXPECT_TRUE(is_valid_mass(zeroMass));
    
    kilogram smallMass(0.001);  // 1 gram
    EXPECT_TRUE(is_valid_mass(smallMass));
    
    kilogram personMass(75.0);  // 75 kg person
    EXPECT_TRUE(is_valid_mass(personMass));
    
    kilogram largeMass(1000.0);  // 1 tonne
    EXPECT_TRUE(is_valid_mass(largeMass));
    
    // Invalid masses (negative)
    kilogram negativeMass(-0.1);
    EXPECT_FALSE(is_valid_mass(negativeMass));
    
    kilogram veryNegativeMass(-1000.0);
    EXPECT_FALSE(is_valid_mass(veryNegativeMass));
}

TEST(UnitsValidation, EnforceValidTemperature) {
    // Test enforce_valid_temperature function
    
    // Valid temperatures should remain unchanged
    kelvin validTemp(300.0);
    auto enforcedValidTemp = enforce_valid_temperature(validTemp);
    EXPECT_DOUBLE_EQ(validTemp.value(), enforcedValidTemp.value());
    
    kelvin exactAbsZero(constants::ABSOLUTE_ZERO);  // 0 K
    auto enforcedExactAbsZero = enforce_valid_temperature(exactAbsZero);
    EXPECT_DOUBLE_EQ(exactAbsZero.value(), enforcedExactAbsZero.value());
    
    // Invalid temperatures should be clamped to absolute zero
    kelvin slightlyInvalid(-0.001);
    auto enforcedSlightlyInvalid = enforce_valid_temperature(slightlyInvalid);
    EXPECT_DOUBLE_EQ(constants::ABSOLUTE_ZERO, enforcedSlightlyInvalid.value());
    
    kelvin veryInvalid(-1000.0);
    auto enforcedVeryInvalid = enforce_valid_temperature(veryInvalid);
    EXPECT_DOUBLE_EQ(constants::ABSOLUTE_ZERO, enforcedVeryInvalid.value());
}

TEST(UnitsValidation, EnforceValidPressure) {
    // Test enforce_valid_pressure function
    
    // Valid pressures should remain unchanged
    pascal validPressure(101325.0);
    auto enforcedValidPressure = enforce_valid_pressure(validPressure);
    EXPECT_DOUBLE_EQ(validPressure.value(), enforcedValidPressure.value());
    
    pascal zeroPressure(0.0);
    auto enforcedZeroPressure = enforce_valid_pressure(zeroPressure);
    EXPECT_DOUBLE_EQ(zeroPressure.value(), enforcedZeroPressure.value());
    
    // Invalid pressures should be clamped to zero
    pascal slightlyInvalid(-0.001);
    auto enforcedSlightlyInvalid = enforce_valid_pressure(slightlyInvalid);
    EXPECT_DOUBLE_EQ(0.0, enforcedSlightlyInvalid.value());
    
    pascal veryInvalid(-101325.0);
    auto enforcedVeryInvalid = enforce_valid_pressure(veryInvalid);
    EXPECT_DOUBLE_EQ(0.0, enforcedVeryInvalid.value());
}

TEST(UnitsValidation, EnforceValidDensity) {
    // Test enforce_valid_density function
    
    // Valid densities should remain unchanged
    kilogram_per_cubic_meter validDensity(1000.0);
    auto enforcedValidDensity = enforce_valid_density(validDensity);
    EXPECT_DOUBLE_EQ(validDensity.value(), enforcedValidDensity.value());
    
    kilogram_per_cubic_meter zeroDensity(0.0);
    auto enforcedZeroDensity = enforce_valid_density(zeroDensity);
    EXPECT_DOUBLE_EQ(zeroDensity.value(), enforcedZeroDensity.value());
    
    // Invalid densities should be clamped to zero
    kilogram_per_cubic_meter slightlyInvalid(-0.001);
    auto enforcedSlightlyInvalid = enforce_valid_density(slightlyInvalid);
    EXPECT_DOUBLE_EQ(0.0, enforcedSlightlyInvalid.value());
    
    kilogram_per_cubic_meter veryInvalid(-1000.0);
    auto enforcedVeryInvalid = enforce_valid_density(veryInvalid);
    EXPECT_DOUBLE_EQ(0.0, enforcedVeryInvalid.value());
}

TEST(UnitsValidation, EnforceValidMass) {
    // Test enforce_valid_mass function
    
    // Valid masses should remain unchanged
    kilogram validMass(75.0);
    auto enforcedValidMass = enforce_valid_mass(validMass);
    EXPECT_DOUBLE_EQ(validMass.value(), enforcedValidMass.value());
    
    kilogram zeroMass(0.0);
    auto enforcedZeroMass = enforce_valid_mass(zeroMass);
    EXPECT_DOUBLE_EQ(zeroMass.value(), enforcedZeroMass.value());
    
    // Invalid masses should be clamped to zero
    kilogram slightlyInvalid(-0.001);
    auto enforcedSlightlyInvalid = enforce_valid_mass(slightlyInvalid);
    EXPECT_DOUBLE_EQ(0.0, enforcedSlightlyInvalid.value());
    
    kilogram veryInvalid(-1000.0);
    auto enforcedVeryInvalid = enforce_valid_mass(veryInvalid);
    EXPECT_DOUBLE_EQ(0.0, enforcedVeryInvalid.value());
}

TEST(UnitsValidation, ValidityCheckEdgeCases) {
    // Test edge cases for validity check functions
    
    // Temperatures
    kelvin exactAbsZero(0.0);
    EXPECT_TRUE(is_valid_temperature(exactAbsZero));
    
    kelvin justAboveAbsZero(std::numeric_limits<double>::min());  // Smallest positive double
    EXPECT_TRUE(is_valid_temperature(justAboveAbsZero));
    
    kelvin justBelowAbsZero(-std::numeric_limits<double>::min());  // Smallest negative double
    EXPECT_FALSE(is_valid_temperature(justBelowAbsZero));
    
    // Pressures
    pascal exactZeroPressure(0.0);
    EXPECT_TRUE(is_valid_pressure(exactZeroPressure));
    
    pascal justAboveZeroPressure(std::numeric_limits<double>::min());
    EXPECT_TRUE(is_valid_pressure(justAboveZeroPressure));
    
    pascal justBelowZeroPressure(-std::numeric_limits<double>::min());
    EXPECT_FALSE(is_valid_pressure(justBelowZeroPressure));
    
    // Densities
    kilogram_per_cubic_meter exactZeroDensity(0.0);
    EXPECT_TRUE(is_valid_density(exactZeroDensity));
    
    kilogram_per_cubic_meter justAboveZeroDensity(std::numeric_limits<double>::min());
    EXPECT_TRUE(is_valid_density(justAboveZeroDensity));
    
    kilogram_per_cubic_meter justBelowZeroDensity(-std::numeric_limits<double>::min());
    EXPECT_FALSE(is_valid_density(justBelowZeroDensity));
    
    // Masses
    kilogram exactZeroMass(0.0);
    EXPECT_TRUE(is_valid_mass(exactZeroMass));
    
    kilogram justAboveZeroMass(std::numeric_limits<double>::min());
    EXPECT_TRUE(is_valid_mass(justAboveZeroMass));
    
    kilogram justBelowZeroMass(-std::numeric_limits<double>::min());
    EXPECT_FALSE(is_valid_mass(justBelowZeroMass));
}
