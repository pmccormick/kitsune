#include <gtest/gtest.h>
#include "Units/core.h"
#include "Units/si_units.h"
#include "Units/derived_units.h"
#include "Units/constants.h"
#include <cmath>

using namespace units;

TEST(UnitsDerived, VelocityUnits) {
    // Base unit: meters_per_second
    meters_per_second mps(1.0);
    
    // Test conversions
    EXPECT_DOUBLE_EQ(3.6, mps.as<kilometers_per_hour>().value());  // 1 m/s = 3.6 km/h
    EXPECT_NEAR(2.23694, mps.as<miles_per_hour>().value(), 1e-5);  // 1 m/s ≈ 2.237 mph
    
    // Different starting values
    kilometers_per_hour kph(36.0);  // 36 km/h = 10 m/s
    EXPECT_DOUBLE_EQ(10.0, kph.as<meters_per_second>().value());
    
    miles_per_hour mph(60.0);  // 60 mph ≈ 26.8224 m/s
    EXPECT_NEAR(26.8224, mph.as<meters_per_second>().value(), 1e-4);
    EXPECT_NEAR(96.5606, mph.as<kilometers_per_hour>().value(), 1e-4);  // 60 mph ≈ 96.56 km/h
    
    // Division of length by time
    meter distance(100.0);
    second time(10.0);
    auto resultVelocity = distance / time;  // 100m / 10s = 10 m/s
    
    EXPECT_TRUE((std::is_same_v<decltype(resultVelocity), meters_per_second>));
    EXPECT_DOUBLE_EQ(10.0, resultVelocity.value());
}

TEST(UnitsDerived, AccelerationUnits) {
    // Base unit: meters_per_second_squared
    meters_per_second_squared mps2(1.0);
    
    // Standard gravity
    meters_per_second_squared g(constants::STANDARD_GRAVITY);  // 9.80665 m/s²
    EXPECT_DOUBLE_EQ(9.80665, g.value());
    
    // Division of velocity by time
    meters_per_second velocity(20.0);
    second time(5.0);
    
    // This should convert velocity to meters_per_second_squared by dividing by time
    // Since we're using custom division operators, need to handle this separately
    meters_per_second_squared resultAcceleration(velocity.value() / time.value());
    
    EXPECT_DOUBLE_EQ(4.0, resultAcceleration.value());
}

TEST(UnitsDerived, ForceUnits) {
    // Base unit: newton
    newton n(1.0);
    
    // Test conversions
    EXPECT_NEAR(0.224809, n.as<pound_force>().value(), 1e-6);  // 1 N ≈ 0.2248 lbf
    
    // Different starting values
    pound_force lbf(1.0);
    EXPECT_NEAR(4.44822, lbf.as<newton>().value(), 1e-5);  // 1 lbf ≈ 4.4482 N
    
    // Multiplication of mass by acceleration (F = ma)
    kilogram mass(2.0);
    meters_per_second_squared accel(5.0);
    auto resultForce = mass * accel;  // 2 kg * 5 m/s² = 10 N
    
    EXPECT_TRUE((std::is_same_v<decltype(resultForce), newton>));
    EXPECT_DOUBLE_EQ(10.0, resultForce.value());
    
    // Weight calculation (with standard gravity)
    kilogram weight(75.0);  // 75 kg person
    meters_per_second_squared gravity(constants::STANDARD_GRAVITY);
    auto weightForce = weight * gravity;  // 75 kg * 9.80665 m/s² ≈ 735.5 N
    
    EXPECT_NEAR(735.5, weightForce.value(), 0.1);
}

TEST(UnitsDerived, PressureUnits) {
    // Base unit: pascal
    pascal pa(1.0);
    
    // Test conversions
    EXPECT_DOUBLE_EQ(0.001, pa.as<kilopascal>().value());  // 1 Pa = 0.001 kPa
    EXPECT_DOUBLE_EQ(1e-5, pa.as<bar>().value());  // 1 Pa = 1e-5 bar
    EXPECT_NEAR(0.000145038, pa.as<psi>().value(), 1e-9);  // 1 Pa ≈ 0.000145 psi
    
    // Different starting values
    bar bar_val(1.0);
    EXPECT_DOUBLE_EQ(100000.0, bar_val.as<pascal>().value());  // 1 bar = 100,000 Pa
    
    psi psi_val(14.5038);  // ≈ 1 atmosphere
    EXPECT_NEAR(100000.0, psi_val.as<pascal>().value(), 1.0);  // ≈ 100,000 Pa
    
    // Standard atmospheric pressure
    pascal atm(constants::STANDARD_ATM_PRESSURE);  // 101325 Pa
    EXPECT_DOUBLE_EQ(101325.0, atm.value());
    EXPECT_NEAR(1.01325, atm.as<bar>().value(), 1e-5);  // ≈ 1.01325 bar
    EXPECT_NEAR(14.6959, atm.as<psi>().value(), 1e-4);  // ≈ 14.696 psi
}

TEST(UnitsDerivedFixes, EnergyUnits) {
    // Base unit: joule
    joule j(1.0);
    
    // Test conversions
    EXPECT_DOUBLE_EQ(0.001, j.as<kilojoule>().value());  // 1 J = 0.001 kJ
    
    // The original test expectation was:
    // EXPECT_DOUBLE_EQ(2.77778e-7, j.as<kilowatt_hour>().value());
    // But due to float precision issues, it would be better to:
    
    // Option 1: Use EXPECT_NEAR with appropriate epsilon
    EXPECT_NEAR(2.77778e-7, j.as<kilowatt_hour>().value(), 1e-12);
    
    // Option 2: Use the exact mathematical value (preferred)
    EXPECT_DOUBLE_EQ(1.0/3600000.0, j.as<kilowatt_hour>().value());
    
    EXPECT_NEAR(0.239006, j.as<calorie>().value(), 1e-6);  // 1 J ≈ 0.239 cal
    
    // Different starting values
    kilojoule kj(1.0);
    EXPECT_DOUBLE_EQ(1000.0, kj.as<joule>().value());  // 1 kJ = 1000 J
    
    kilowatt_hour kwh(1.0);
    EXPECT_DOUBLE_EQ(3600000.0, kwh.as<joule>().value());  // 1 kWh = 3,600,000 J
    EXPECT_DOUBLE_EQ(3600.0, kwh.as<kilojoule>().value());  // 1 kWh = 3,600 kJ
    
    calorie cal(1.0);
    EXPECT_NEAR(4.184, cal.as<joule>().value(), 1e-3);  // 1 cal ≈ 4.184 J
}


TEST(UnitsDerived, PowerUnits) {
    // Base unit: watt
    watt w(1.0);
    
    // Test conversions
    EXPECT_DOUBLE_EQ(0.001, w.as<kilowatt>().value());  // 1 W = 0.001 kW
    EXPECT_NEAR(0.00134102, w.as<horsepower>().value(), 1e-8);  // 1 W ≈ 0.00134 hp
    
    // Different starting values
    kilowatt kw(1.0);
    EXPECT_DOUBLE_EQ(1000.0, kw.as<watt>().value());  // 1 kW = 1000 W
    
    horsepower hp(1.0);
    EXPECT_NEAR(745.7, hp.as<watt>().value(), 0.1);  // 1 hp ≈ 745.7 W
    EXPECT_NEAR(0.7457, hp.as<kilowatt>().value(), 1e-4);  // 1 hp ≈ 0.7457 kW
    
    // Division of energy by time (P = E/t)
    joule energy(120.0);
    second time(10.0);
    auto resultPower = energy / time;  // 120 J / 10 s = 12 W
    
    EXPECT_TRUE((std::is_same_v<decltype(resultPower), watt>));
    EXPECT_DOUBLE_EQ(12.0, resultPower.value());
}

TEST(UnitsDerived, DensityUnits) {
    // Base unit: kilogram_per_cubic_meter
    kilogram_per_cubic_meter kgm3(1.0);
    
    // Test conversions
    EXPECT_DOUBLE_EQ(0.001, kgm3.as<gram_per_cubic_centimeter>().value());  // 1 kg/m³ = 0.001 g/cm³
    
    // Different starting values
    gram_per_cubic_centimeter gcm3(1.0);  // Water density ≈ 1 g/cm³
    EXPECT_DOUBLE_EQ(1000.0, gcm3.as<kilogram_per_cubic_meter>().value());  // 1 g/cm³ = 1000 kg/m³
    
    // Density of some common materials
    kilogram_per_cubic_meter airDensity(1.225);  // Air at sea level
    EXPECT_NEAR(0.001225, airDensity.as<gram_per_cubic_centimeter>().value(), 1e-6);
    
    gram_per_cubic_centimeter goldDensity(19.3);  // Gold
    EXPECT_DOUBLE_EQ(19300.0, goldDensity.as<kilogram_per_cubic_meter>().value());
}

TEST(UnitsDerived, AreaCalculations) {
    // Test area calculations from length units
    meter length(4.0);
    meter width(5.0);
    
    // Direct multiplication
    auto area = length * width;
    EXPECT_TRUE((std::is_same_v<decltype(area), square_meter>));
    EXPECT_DOUBLE_EQ(20.0, area.value());
    
    // Using square function
    auto squaredLength = square(length);
    EXPECT_TRUE((std::is_same_v<decltype(squaredLength), square_meter>));
    EXPECT_DOUBLE_EQ(16.0, squaredLength.value());
    
    // Mixed units
    centimeter smallLength(200.0);  // 2 meters
    auto mixedArea = length * smallLength.as<meter>();
    EXPECT_DOUBLE_EQ(8.0, mixedArea.value());
}

TEST(UnitsDerived, VolumeCalculations) {
    // Test volume calculations from length and area units
    meter length(3.0);
    meter width(4.0);
    meter height(5.0);
    
    // We can't directly do length * width * height in the current implementation
    // because (length * width) is an area, and we need special handling for 
    // area * length which was implemented
    
    // Instead, we test area * length
    square_meter area = length * width;  // 12 m²
    cubic_meter volume = area * height;  // 12 m² * 5 m = 60 m³
    
    EXPECT_TRUE((std::is_same_v<decltype(volume), cubic_meter>));
    EXPECT_DOUBLE_EQ(60.0, volume.value());
    
    // Using cube function
    auto cubedLength = cube(length);
    EXPECT_TRUE((std::is_same_v<decltype(cubedLength), cubic_meter>));
    EXPECT_DOUBLE_EQ(27.0, cubedLength.value());
    
    // Also test length * area (commutative)
    auto volume2 = height * area;
    EXPECT_TRUE((std::is_same_v<decltype(volume2), cubic_meter>));
    EXPECT_DOUBLE_EQ(60.0, volume2.value());
}

TEST(UnitsDerived, SquareRootOperations) {
    // Test square root of area units
    square_meter area(9.0);
    
    // Using sqrt function
    auto length = sqrt(area);
    EXPECT_TRUE((std::is_same_v<decltype(length), meter>));
    EXPECT_DOUBLE_EQ(3.0, length.value());
    
    // Using sqrt with different area units
    square_kilometer largeArea(1.0);  // 1 km² = 1,000,000 m²
    auto lengthFromLargeArea = sqrt(largeArea);
    EXPECT_DOUBLE_EQ(1000.0, lengthFromLargeArea.value());
    
    // Non-perfect squares
    square_meter irregularArea(2.0);
    auto irregularLength = sqrt(irregularArea);
    EXPECT_NEAR(std::sqrt(2.0), irregularLength.value(), 1e-10);
}

TEST(UnitsDerived, CubeRootOperations) {
    // Test cube root of volume units
    cubic_meter volume(27.0);
    
    // Using cbrt function
    auto length = cbrt(volume);
    EXPECT_TRUE((std::is_same_v<decltype(length), meter>));
    EXPECT_DOUBLE_EQ(3.0, length.value());
    
    // Using cbrt with different volume units
    // There is no cubic_kilometer defined, so we'll convert cubic_meter to a larger value
    cubic_meter largeVolume(1000000000.0);  // 1 billion m³ = 1 km³
    auto lengthFromLargeVolume = cbrt(largeVolume);
    EXPECT_DOUBLE_EQ(1000.0, lengthFromLargeVolume.value());
    
    // Non-perfect cubes
    cubic_meter irregularVolume(2.0);
    auto irregularLength = cbrt(irregularVolume);
    EXPECT_NEAR(std::cbrt(2.0), irregularLength.value(), 1e-10);
}

TEST(UnitsDerived, FlowRateUnits) {
    // Base unit: cubic_meter_per_second
    cubic_meter_per_second cms(1.0);
    
    // Test conversions
    EXPECT_DOUBLE_EQ(60000.0, cms.as<liter_per_minute>().value());  // 1 m³/s = 60,000 L/min
    EXPECT_NEAR(2118.88, cms.as<cubic_foot_per_minute>().value(), 0.01);  // 1 m³/s ≈ 2118.9 CFM
    
    // Different starting values
    liter_per_minute lpm(60.0);  // 60 L/min = 0.001 m³/s
    EXPECT_DOUBLE_EQ(0.001, lpm.as<cubic_meter_per_second>().value());
    
    // Flow rate derived from volume and time
    cubic_meter volume(5.0);
    minute time(10.0);
    
    // Convert to base units and calculate manually
    double flow_value = volume.value() / time.as<second>().value();  // 5 m³ / 600 s = 0.00833... m³/s
    cubic_meter_per_second resultFlowRate(flow_value);
    
    EXPECT_NEAR(0.00833333, resultFlowRate.value(), 1e-8);
    
    // Convert to liter per minute and check
    liter_per_minute resultInLPM = resultFlowRate.as<liter_per_minute>();
    EXPECT_NEAR(500.0, resultInLPM.value(), 1e-5);  // 0.00833... m³/s ≈ 500 L/min
}

TEST(UnitsDerived, ChainedConversions) {
    // Test chained conversions across different derived units
    
    // Energy to power to force...
    kilowatt_hour energy(1.0);  // 1 kWh
    // 1 kWh = 3,600,000 J
    joule energyInJoules = energy.as<joule>();
    EXPECT_DOUBLE_EQ(3600000.0, energyInJoules.value());
    
    // If applied over 1 hour (3600 s)
    hour time(1.0);
    
    // Calculate power manually
    double power_value = energyInJoules.value() / time.as<second>().value();  // 3,600,000 J / 3600 s = 1000 W
    watt power(power_value);
    
    EXPECT_DOUBLE_EQ(1000.0, power.value());
    EXPECT_DOUBLE_EQ(1.0, power.as<kilowatt>().value());
    
    // If this power moves an object at 2 m/s
    meters_per_second velocity(2.0);
    
    // Manual calculation of force
    double force_value = power.value() / velocity.value();  // 1000 W / 2 m/s = 500 N
    newton force(force_value);
    
    EXPECT_DOUBLE_EQ(500.0, force.value());
    
    // Calculate mass manually
    double mass_value = force.value() / 9.80665;  // 500 N / 9.80665 m/s² ≈ 51 kg
    kilogram mass(mass_value);
    
    EXPECT_NEAR(51.0, mass.value(), 0.1);
}

// Comprehensive verification test for all previously failing unit conversions
TEST(UnitsFixes, ComprehensiveVerification) {
    // AREA UNITS
    square_meter sqm(1.0);
    EXPECT_NEAR(0.000247105, sqm.as<acre>().value(), 1e-9);

    acre ac(1.0);
    EXPECT_NEAR(4046.86, ac.as<square_meter>().value(), 0.01);
    EXPECT_NEAR(0.404686, ac.as<hectare>().value(), 1e-6);

    // VOLUME UNITS
    cubic_meter cubm(1.0);
    EXPECT_NEAR(264.172, cubm.as<gallon_us>().value(), 1e-3);
    EXPECT_NEAR(219.969, cubm.as<gallon_uk>().value(), 1e-3);
    EXPECT_NEAR(35.3147, cubm.as<cubic_foot>().value(), 1e-4);
    EXPECT_NEAR(33814.0, cubm.as<fluid_ounce_us>().value(), 0.5);

    gallon_us gal(1.0);
    EXPECT_NEAR(3.78541, gal.as<liter>().value(), 1e-5);

    cubic_foot cf(1.0);
    EXPECT_NEAR(28.3169, cf.as<liter>().value(), 1e-4);

    // PRESSURE UNITS
    pascal pa(1.0);
    EXPECT_NEAR(0.000145038, pa.as<psi>().value(), 1e-9);

    psi psi_val(14.5038);
    EXPECT_NEAR(100000.0, psi_val.as<pascal>().value(), 1.0);

    pascal atm(constants::STANDARD_ATM_PRESSURE);
    EXPECT_NEAR(14.6959, atm.as<psi>().value(), 1e-4);

    // FLOW RATE UNITS
    cubic_meter_per_second cms(1.0);
    EXPECT_NEAR(2118.88, cms.as<cubic_foot_per_minute>().value(), 0.01);

    liter_per_minute lpm(60.0);
    EXPECT_DOUBLE_EQ(0.001, lpm.as<cubic_meter_per_second>().value());
}
