#include <gtest/gtest.h>
#include "units/core.h"

using namespace units;

// Define some test unit types with different ratios
using base_length = Unit<length_dimension, std::ratio<1>>;        // base unit (1)
using kilo_length = Unit<length_dimension, std::ratio<1000>>;     // 1000x base
using milli_length = Unit<length_dimension, std::ratio<1, 1000>>; // 1/1000 of base
using micro_length = Unit<length_dimension, std::ratio<1, 1000000>>; // 1/1000000 of base

// Another dimension for testing
using base_mass = Unit<mass_dimension, std::ratio<1>>;
using kilo_mass = Unit<mass_dimension, std::ratio<1000>>;

TEST(UnitsCoreConversion, UnitCastBasic) {
    base_length meter(1.0);
    
    // Convert to larger unit (smaller value)
    auto kilometers = unit_cast<kilo_length>(meter);
    EXPECT_DOUBLE_EQ(0.001, kilometers.value());
    
    // Convert to smaller unit (larger value)
    auto millimeters = unit_cast<milli_length>(meter);
    EXPECT_DOUBLE_EQ(1000.0, millimeters.value());
    
    // Convert to even smaller unit
    auto micrometers = unit_cast<micro_length>(meter);
    EXPECT_DOUBLE_EQ(1000000.0, micrometers.value());
    
    // Convert from one non-base unit to another
    kilo_length km(1.0);  // 1 km
    auto mm = unit_cast<milli_length>(km);  // Convert to mm
    EXPECT_DOUBLE_EQ(1000000.0, mm.value());  // 1 km = 1,000,000 mm
}

TEST(UnitsCoreConversion, AsMethodBasic) {
    base_length meter(5.0);
    
    // Convert to larger unit (smaller value)
    auto kilometers = meter.as<kilo_length>();
    EXPECT_DOUBLE_EQ(0.005, kilometers.value());
    
    // Convert to smaller unit (larger value)
    auto millimeters = meter.as<milli_length>();
    EXPECT_DOUBLE_EQ(5000.0, millimeters.value());
    
    // Convert to even smaller unit
    auto micrometers = meter.as<micro_length>();
    EXPECT_DOUBLE_EQ(5000000.0, micrometers.value());
    
    // Convert from one non-base unit to another
    milli_length mm(5000.0);  // 5000 mm
    auto km = mm.as<kilo_length>();  // Convert to km
    EXPECT_DOUBLE_EQ(0.005, km.value());  // 5000 mm = 0.005 km
}

TEST(UnitsCoreConversion, RoundTripConversion) {
    // Test round-trip conversion (should get back the same value)
    base_length original(42.0);
    
    // base -> kilo -> base
    auto kilo = unit_cast<kilo_length>(original);
    auto back_to_base = unit_cast<base_length>(kilo);
    EXPECT_DOUBLE_EQ(original.value(), back_to_base.value());
    
    // base -> milli -> micro -> milli -> base
    auto milli = unit_cast<milli_length>(original);
    auto micro = unit_cast<micro_length>(milli);
    auto back_to_milli = unit_cast<milli_length>(micro);
    auto back_to_base2 = unit_cast<base_length>(back_to_milli);
    EXPECT_DOUBLE_EQ(original.value(), back_to_base2.value());
}

TEST(UnitsCoreConversion, NonIntegerRatios) {
    // Unit with a non-integer ratio (e.g., inches to meters: 0.0254)
    using inch = Unit<length_dimension, std::ratio<254, 10000>>; // 0.0254 meters
    
    base_length meter(1.0);
    
    // Convert meter to inches (1m = 39.3701 inches)
    auto inches = unit_cast<inch>(meter);
    EXPECT_NEAR(39.3701, inches.value(), 1e-4);
    
    // Convert inches to meters (100 inches = 2.54m)
    inch hundredInches(100.0);
    auto meters = unit_cast<base_length>(hundredInches);
    EXPECT_NEAR(2.54, meters.value(), 1e-6);
}

TEST(UnitsCoreConversion, CompoundRatios) {
    // Create compound ratios (e.g., std::ratio<2, 3> - represents 2/3)
    using two_thirds_length = Unit<length_dimension, std::ratio<2, 3>>;
    using three_quarters_length = Unit<length_dimension, std::ratio<3, 4>>;
    
    base_length meter(12.0);
    
    // Convert to fraction of a meter
    auto twoThirds = unit_cast<two_thirds_length>(meter);
    EXPECT_DOUBLE_EQ(18.0, twoThirds.value());  // 12m * 3/2 = 18 two_thirds_units
    
    // Convert between fractions
    auto threeQuarters = unit_cast<three_quarters_length>(twoThirds);
    // 18 two_thirds_units * (2/3) / (3/4) = 18 * (2/3) * (4/3) = 18 * 8/9 = 16
    EXPECT_DOUBLE_EQ(16.0, threeQuarters.value());
}

TEST(UnitsCoreConversion, ZeroValues) {
    // Zero values should convert to zero in any compatible unit
    base_length zeroLength;  // 0.0
    
    EXPECT_DOUBLE_EQ(0.0, unit_cast<kilo_length>(zeroLength).value());
    EXPECT_DOUBLE_EQ(0.0, unit_cast<milli_length>(zeroLength).value());
    EXPECT_DOUBLE_EQ(0.0, unit_cast<micro_length>(zeroLength).value());
    
    // Also with the as() method
    EXPECT_DOUBLE_EQ(0.0, zeroLength.as<kilo_length>().value());
}

TEST(UnitsCoreConversion, NegativeValues) {
    // Negative values should maintain their sign when converted
    base_length negativeMeter(-5.0);
    
    EXPECT_DOUBLE_EQ(-0.005, unit_cast<kilo_length>(negativeMeter).value());
    EXPECT_DOUBLE_EQ(-5000.0, unit_cast<milli_length>(negativeMeter).value());
    
    // Also with the as() method
    EXPECT_DOUBLE_EQ(-0.005, negativeMeter.as<kilo_length>().value());
    EXPECT_DOUBLE_EQ(-5000.0, negativeMeter.as<milli_length>().value());
}

TEST(UnitsCoreConversion, LargeAndSmallValues) {
    // Test with very large values
    base_length veryLarge(1e12);  // 1 trillion base units
    
    auto kiloLarge = unit_cast<kilo_length>(veryLarge);
    EXPECT_DOUBLE_EQ(1e9, kiloLarge.value());  // 1 trillion base = 1 billion kilo
    
    // Test with very small values
    base_length verySmall(1e-12);  // 1 trillionth base units
    
    auto microSmall = unit_cast<micro_length>(verySmall);
    EXPECT_DOUBLE_EQ(1e-6, microSmall.value());  // 1 trillionth base = 1 millionth micro
}

TEST(UnitsCoreConversion, SameUnitNoChange) {
    // Converting to the same unit type should return the same value
    base_length original(42.0);
    
    auto same = unit_cast<base_length>(original);
    EXPECT_DOUBLE_EQ(original.value(), same.value());
    
    // Also with the as() method
    auto sameToo = original.as<base_length>();
    EXPECT_DOUBLE_EQ(original.value(), sameToo.value());
}

TEST(UnitsCoreConversion, DifferentDimensionsCompilationError) {
    // This test verifies that we can't convert between different dimensions.
    // Since this is a compile-time check, we can't directly test it in runtime.
    // Instead, we're testing that the same-dimension conversions do work.
    
    base_length length(10.0);
    base_mass mass(10.0);
    
    // These should compile fine
    auto kiloLength = unit_cast<kilo_length>(length);
    auto kiloMass = unit_cast<kilo_mass>(mass);
    
    EXPECT_DOUBLE_EQ(0.01, kiloLength.value());
    EXPECT_DOUBLE_EQ(0.01, kiloMass.value());
    
    // The following would not compile, which is what we want:
    // auto invalidConversion = unit_cast<kilo_mass>(length);  // Different dimensions
}

TEST(UnitsCoreConversion, UnitTypeConcept) {
    // Test that our UnitType concept works correctly
    
    // These should satisfy UnitType
    EXPECT_TRUE((UnitType<base_length>));
    EXPECT_TRUE((UnitType<kilo_mass>));
    EXPECT_TRUE((UnitType<dimensionless>));
    
    // These should not satisfy UnitType (but we can't directly test in runtime)
    // EXPECT_FALSE((UnitType<int>));
    // EXPECT_FALSE((UnitType<double>));
    // EXPECT_FALSE((UnitType<std::string>));
    
    // Test that we can create a function that uses UnitType concept
    auto getUnitValue = [](UnitType auto unit) { return unit.value(); };
    
    EXPECT_DOUBLE_EQ(42.0, getUnitValue(base_length(42.0)));
    EXPECT_DOUBLE_EQ(3.14, getUnitValue(dimensionless(3.14)));
}

TEST(UnitsCoreConversion, SameDimensionConcept) {
    // Test that our SameDimension concept works correctly
    
    // These should satisfy SameDimension
    EXPECT_TRUE((SameDimension<base_length, kilo_length>));
    EXPECT_TRUE((SameDimension<base_mass, kilo_mass>));
    EXPECT_TRUE((SameDimension<dimensionless, dimensionless>));
    
    // These should not satisfy SameDimension (but we can't directly test in runtime)
    // EXPECT_FALSE((SameDimension<base_length, base_mass>));
    // EXPECT_FALSE((SameDimension<kilo_length, dimensionless>));
    
    // Test that we can create a function that uses SameDimension concept
    auto addUnits = [](SameDimension<base_length> auto unit1, SameDimension<base_length> auto unit2) {
        return unit_cast<base_length>(unit1).value() + unit_cast<base_length>(unit2).value();
    };
    
    EXPECT_DOUBLE_EQ(2.0, addUnits(base_length(1.0), base_length(1.0)));
    EXPECT_DOUBLE_EQ(1001.0, addUnits(base_length(1.0), milli_length(1000.0)));
    EXPECT_DOUBLE_EQ(1.001, addUnits(kilo_length(0.001), base_length(1.0)));
}
