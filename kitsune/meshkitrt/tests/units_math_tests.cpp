#include <gtest/gtest.h>
#include "units/core.h"
#include "units/si_units.h"
#include "units/derived_units.h"
#include <cmath>

using namespace units;

TEST(UnitsMath, AbsFunction) {
    // Test the abs function with various unit types
    
    // Length units
    meter positiveLength(5.0);
    meter negativeLength(-5.0);
    
    auto absPositive = abs(positiveLength);
    auto absNegative = abs(negativeLength);
    
    EXPECT_DOUBLE_EQ(5.0, absPositive.value());
    EXPECT_DOUBLE_EQ(5.0, absNegative.value());
    EXPECT_TRUE((std::is_same_v<decltype(absPositive), meter>));
    
    // Time units
    second positiveTime(10.0);
    second negativeTime(-10.0);
    
    auto absPositiveTime = abs(positiveTime);
    auto absNegativeTime = abs(negativeTime);
    
    EXPECT_DOUBLE_EQ(10.0, absPositiveTime.value());
    EXPECT_DOUBLE_EQ(10.0, absNegativeTime.value());
    EXPECT_TRUE((std::is_same_v<decltype(absPositiveTime), second>));
    
    // Derived units
    newton positiveForce(20.0);
    newton negativeForce(-20.0);
    
    auto absPositiveForce = abs(positiveForce);
    auto absNegativeForce = abs(negativeForce);
    
    EXPECT_DOUBLE_EQ(20.0, absPositiveForce.value());
    EXPECT_DOUBLE_EQ(20.0, absNegativeForce.value());
    EXPECT_TRUE((std::is_same_v<decltype(absPositiveForce), newton>));
}

TEST(UnitsMath, MinFunction) {
    // Test the min function with various unit types
    
    // Length units
    meter length1(10.0);
    meter length2(5.0);
    
    auto minLength = min(length1, length2);
    EXPECT_DOUBLE_EQ(5.0, minLength.value());
    EXPECT_TRUE((std::is_same_v<decltype(minLength), meter>));
    
    // Time units
    hour time1(2.0);
    hour time2(1.5);
    
    auto minTime = min(time1, time2);
    EXPECT_DOUBLE_EQ(1.5, minTime.value());
    EXPECT_TRUE((std::is_same_v<decltype(minTime), hour>));
    
    // Derived units
    pascal pressure1(101325.0);  // 1 atm
    pascal pressure2(100000.0);  // 1 bar
    
    auto minPressure = min(pressure1, pressure2);
    EXPECT_DOUBLE_EQ(100000.0, minPressure.value());
    EXPECT_TRUE((std::is_same_v<decltype(minPressure), pascal>));
    
    // With negative values
    meter negLength(-8.0);
    auto minWithNegative = min(length1, negLength);
    EXPECT_DOUBLE_EQ(-8.0, minWithNegative.value());
}

TEST(UnitsMath, MaxFunction) {
    // Test the max function with various unit types
    
    // Length units
    meter length1(10.0);
    meter length2(5.0);
    
    auto maxLength = max(length1, length2);
    EXPECT_DOUBLE_EQ(10.0, maxLength.value());
    EXPECT_TRUE((std::is_same_v<decltype(maxLength), meter>));
    
    // Time units
    hour time1(2.0);
    hour time2(1.5);
    
    auto maxTime = max(time1, time2);
    EXPECT_DOUBLE_EQ(2.0, maxTime.value());
    EXPECT_TRUE((std::is_same_v<decltype(maxTime), hour>));
    
    // Derived units
    pascal pressure1(101325.0);  // 1 atm
    pascal pressure2(100000.0);  // 1 bar
    
    auto maxPressure = max(pressure1, pressure2);
    EXPECT_DOUBLE_EQ(101325.0, maxPressure.value());
    EXPECT_TRUE((std::is_same_v<decltype(maxPressure), pascal>));
    
    // With negative values
    meter negLength(-8.0);
    auto maxWithNegative = max(length1, negLength);
    EXPECT_DOUBLE_EQ(10.0, maxWithNegative.value());
}

TEST(UnitsMath, RoundFunction) {
    // Test the round function with various unit types
    
    // Length units
    meter lengthUp(10.7);
    meter lengthDown(10.2);
    
    auto roundedUp = round(lengthUp);
    auto roundedDown = round(lengthDown);
    
    EXPECT_DOUBLE_EQ(11.0, roundedUp.value());
    EXPECT_DOUBLE_EQ(10.0, roundedDown.value());
    EXPECT_TRUE((std::is_same_v<decltype(roundedUp), meter>));
    
    // Time units with fractional values
    second timeUp(2.8);
    second timeDown(2.3);
    
    auto roundedTimeUp = round(timeUp);
    auto roundedTimeDown = round(timeDown);
    
    EXPECT_DOUBLE_EQ(3.0, roundedTimeUp.value());
    EXPECT_DOUBLE_EQ(2.0, roundedTimeDown.value());
    EXPECT_TRUE((std::is_same_v<decltype(roundedTimeUp), second>));
    
    // Derived units
    newton forceUp(15.6);
    newton forceDown(15.4);
    
    auto roundedForceUp = round(forceUp);
    auto roundedForceDown = round(forceDown);
    
    EXPECT_DOUBLE_EQ(16.0, roundedForceUp.value());
    EXPECT_DOUBLE_EQ(15.0, roundedForceDown.value());
    EXPECT_TRUE((std::is_same_v<decltype(roundedForceUp), newton>));
    
    // Edge cases
    meter exactHalf(10.5);
    auto roundedHalf = round(exactHalf);
    EXPECT_DOUBLE_EQ(11.0, roundedHalf.value());  // Rounds away from zero
    
    meter negativeHalf(-10.5);
    auto roundedNegativeHalf = round(negativeHalf);
    EXPECT_DOUBLE_EQ(-11.0, roundedNegativeHalf.value());  // Rounds away from zero
}

TEST(UnitsMath, CeilFunction) {
    // Test the ceil function with various unit types
    
    // Length units
    meter lengthJustUnder(10.1);
    meter lengthExact(10.0);
    
    auto ceiledJustUnder = ceil(lengthJustUnder);
    auto ceiledExact = ceil(lengthExact);
    
    EXPECT_DOUBLE_EQ(11.0, ceiledJustUnder.value());
    EXPECT_DOUBLE_EQ(10.0, ceiledExact.value());
    EXPECT_TRUE((std::is_same_v<decltype(ceiledJustUnder), meter>));
    
    // Time units with fractional values
    second timeJustUnder(2.9999);
    second timeExact(3.0);
    
    auto ceiledTimeJustUnder = ceil(timeJustUnder);
    auto ceiledTimeExact = ceil(timeExact);
    
    EXPECT_DOUBLE_EQ(3.0, ceiledTimeJustUnder.value());
    EXPECT_DOUBLE_EQ(3.0, ceiledTimeExact.value());
    EXPECT_TRUE((std::is_same_v<decltype(ceiledTimeJustUnder), second>));
    
    // Negative values (ceil rounds towards positive infinity)
    meter negativeLength(-10.9);
    auto ceiledNegative = ceil(negativeLength);
    EXPECT_DOUBLE_EQ(-10.0, ceiledNegative.value());
}

TEST(UnitsMath, FloorFunction) {
    // Test the floor function with various unit types
    
    // Length units
    meter lengthJustOver(10.9);
    meter lengthExact(10.0);
    
    auto flooredJustOver = floor(lengthJustOver);
    auto flooredExact = floor(lengthExact);
    
    EXPECT_DOUBLE_EQ(10.0, flooredJustOver.value());
    EXPECT_DOUBLE_EQ(10.0, flooredExact.value());
    EXPECT_TRUE((std::is_same_v<decltype(flooredJustOver), meter>));
    
    // Time units with fractional values
    second timeJustOver(2.0001);
    second timeExact(2.0);
    
    auto flooredTimeJustOver = floor(timeJustOver);
    auto flooredTimeExact = floor(timeExact);
    
    EXPECT_DOUBLE_EQ(2.0, flooredTimeJustOver.value());
    EXPECT_DOUBLE_EQ(2.0, flooredTimeExact.value());
    EXPECT_TRUE((std::is_same_v<decltype(flooredTimeJustOver), second>));
    
    // Negative values (floor rounds towards negative infinity)
    meter negativeLength(-10.1);
    auto flooredNegative = floor(negativeLength);
    EXPECT_DOUBLE_EQ(-11.0, flooredNegative.value());
}

TEST(UnitsMath, TruncFunction) {
    // Test the trunc function with various unit types
    
    // Length units
    meter lengthPos(10.9);
    meter lengthNeg(-10.9);
    
    auto truncatedPos = trunc(lengthPos);
    auto truncatedNeg = trunc(lengthNeg);
    
    EXPECT_DOUBLE_EQ(10.0, truncatedPos.value());
    EXPECT_DOUBLE_EQ(-10.0, truncatedNeg.value());  // Truncation removes decimal part (towards zero)
    EXPECT_TRUE((std::is_same_v<decltype(truncatedPos), meter>));
    
    // Time units with fractional values
    second timePos(2.7);
    second timeNeg(-2.7);
    
    auto truncatedTimePos = trunc(timePos);
    auto truncatedTimeNeg = trunc(timeNeg);
    
    EXPECT_DOUBLE_EQ(2.0, truncatedTimePos.value());
    EXPECT_DOUBLE_EQ(-2.0, truncatedTimeNeg.value());
    EXPECT_TRUE((std::is_same_v<decltype(truncatedTimePos), second>));
}

TEST(UnitsMath, SquareAndCubeRootOperations) {
    // Test more complex square and cube root operations
    
    // Square roots of areas with different unit types
    square_centimeter small_area(100.0);  // 100 cm² = 0.01 m²
    auto length_from_small = sqrt(small_area);
    EXPECT_DOUBLE_EQ(10.0, length_from_small.as<centimeter>().value());  // 10 cm
    
    square_kilometer large_area(4.0);  // 4 km² = 4,000,000 m²
    auto length_from_large = sqrt(large_area);
    EXPECT_DOUBLE_EQ(2000.0, length_from_large.value());  // 2000 m = 2 km
    
    // Cube roots of volumes with different unit types
    cubic_centimeter small_volume(1000.0);  // 1000 cm³ = 0.001 m³
    auto length_from_small_vol = cbrt(small_volume);
    EXPECT_DOUBLE_EQ(10.0, length_from_small_vol.as<centimeter>().value());  // 10 cm
    
    cubic_kilometer large_volume(8.0);  // 8 km³ = 8,000,000,000 m³
    auto length_from_large_vol = cbrt(large_volume);
    EXPECT_DOUBLE_EQ(2000.0, length_from_large_vol.value());  // 2000 m = 2 km
}

TEST(UnitsMath, SquareAndCubeFunctions) {
    // Test square and cube functions for creating areas and volumes
    
    // Square of lengths
    centimeter small_length(10.0);  // 10 cm = 0.1 m
    auto small_area = square(small_length.as<meter>());
    EXPECT_DOUBLE_EQ(0.01, small_area.value());  // 0.01 m²
    
    kilometer large_length(2.0);  // 2 km = 2000 m
    auto large_area = square(large_length.as<meter>());
    EXPECT_DOUBLE_EQ(4000000.0, large_area.value());  // 4,000,000 m²
    
    // Cube of lengths
    centimeter small_length_cube(10.0);  // 10 cm = 0.1 m
    auto small_volume = cube(small_length_cube.as<meter>());
    EXPECT_DOUBLE_EQ(0.001, small_volume.value());  // 0.001 m³
    
    kilometer large_length_cube(2.0);  // 2 km = 2000 m
    auto large_volume = cube(large_length_cube.as<meter>());
    EXPECT_DOUBLE_EQ(8000000000.0, large_volume.value());  // 8,000,000,000 m³
}

TEST(UnitsMath, ScientificCalculations) {
    // Test more complex scientific calculations using the units library
    
    // Calculate kinetic energy: E = 0.5 * m * v²
    kilogram mass(75.0);  // 75 kg
    meters_per_second velocity(10.0);  // 10 m/s
    
    auto velocitySquared = velocity.value() * velocity.value();  // Manual calculation
    auto kineticEnergy = 0.5 * mass.value() * velocitySquared;  // 0.5 * 75 * 10² = 3750 J
    
    EXPECT_DOUBLE_EQ(3750.0, kineticEnergy);
    
    // Calculate potential energy: E = m * g * h
    meter height(10.0);  // 10 m
    meters_per_second_squared gravity(constants::STANDARD_GRAVITY);  // 9.80665 m/s²
    
    auto force = mass * gravity;  // Weight force
    auto potentialEnergy = force * height;  // m * g * h
    EXPECT_NEAR(7355.0, potentialEnergy.value(), 1.0);  // ≈ 7355 J
    
    // Calculate pressure from force and area: P = F / A
    newton weightForce(1000.0);  // 1000 N
    square_meter contactArea(0.1);  // 0.1 m²
    
    auto pressureValue = weightForce.value() / contactArea.value();  // 1000 N / 0.1 m² = 10000 Pa
    EXPECT_DOUBLE_EQ(10000.0, pressureValue);
}

TEST(UnitsMath, ChainedUnitOperations) {
    // Test more complex chained operations with units
    
    // Calculate stopping distance: d = v² / (2 * μ * g)
    // where v is velocity, μ is friction coefficient, g is gravity
    
    meters_per_second velocity(20.0);  // 20 m/s
    dimensionless frictionCoeff(0.7);  // Typical dry road
    meters_per_second_squared gravity(constants::STANDARD_GRAVITY);  // 9.80665 m/s²
    
    auto velocitySquared = velocity.value() * velocity.value();  // v²
    auto denominator = 2.0 * frictionCoeff.value() * gravity.value();  // 2 * μ * g
    auto stoppingDistance = velocitySquared / denominator;  // v² / (2 * μ * g)
    
    EXPECT_NEAR(29.14, stoppingDistance, 0.01);  // ≈ 29.14 meters
}

TEST(UnitsMath, TrigonometricOperations) {
    // While the units library doesn't directly support trig functions with units,
    // we can still test manual calculations with appropriate unit handling
    
    // Calculate height from distance and angle: h = d * tan(θ)
    meter distance(100.0);  // 100 m
    double angle = 30.0 * constants::PI / 180.0;  // 30 degrees in radians
    
    auto height = distance.value() * std::tan(angle);  // 100 * tan(30°) ≈ 57.74 m
    EXPECT_NEAR(57.74, height, 0.01);
    
    // Calculate projectile range: R = (v² * sin(2θ)) / g
    meters_per_second initialVelocity(50.0);  // 50 m/s
    double launchAngle = 45.0 * constants::PI / 180.0;  // 45 degrees in radians
    meters_per_second_squared gravity(constants::STANDARD_GRAVITY);  // 9.80665 m/s²
    
    auto v2 = initialVelocity.value() * initialVelocity.value();
    auto sin2Theta = std::sin(2 * launchAngle);  // sin(2*45°) = sin(90°) = 1
    auto range = v2 * sin2Theta / gravity.value();  // (50² * 1) / 9.80665 ≈ 255 m
    
    EXPECT_NEAR(255.0, range, 1.0);
}
