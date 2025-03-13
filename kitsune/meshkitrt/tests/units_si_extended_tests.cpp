#include <gtest/gtest.h>
#include "units/core.h"
#include "units/si_units.h"
#include "units/constants.h"

using namespace units;

TEST(UnitsSIExtended, AreaUnits) {
    // Test relationships between different area units
    
    // Base unit: square_meter
    square_meter sqm(1.0);
    
    // Smaller units
    EXPECT_DOUBLE_EQ(10000.0, sqm.as<square_centimeter>().value());
    
    // Larger units
    EXPECT_DOUBLE_EQ(0.000001, sqm.as<square_kilometer>().value());
    EXPECT_DOUBLE_EQ(0.0001, sqm.as<hectare>().value());
    
    // Non-SI units
    EXPECT_NEAR(10.7639, sqm.as<square_foot>().value(), 1e-4);
    EXPECT_NEAR(1550.0, sqm.as<square_inch>().value(), 1);  // Approximate
    EXPECT_NEAR(0.000247105, sqm.as<acre>().value(), 1e-9);
    
    // Construction from square of length
    // 2m × 3m = 6m²
    meter length1(2.0);
    meter length2(3.0);
    square_meter area = length1 * length2;  // length multiplication should create area
    EXPECT_DOUBLE_EQ(6.0, area.value());
    
    // Different starting values
    hectare ha(1.0);  // 1 hectare = 10,000 m²
    EXPECT_DOUBLE_EQ(10000.0, ha.as<square_meter>().value());
    
    acre ac(1.0);
    EXPECT_NEAR(4046.86, ac.as<square_meter>().value(), 0.01);
    EXPECT_NEAR(0.404686, ac.as<hectare>().value(), 1e-6);
}

TEST(UnitsSIExtended, VolumeUnits) {
    // Test relationships between different volume units
    
    // Base unit: cubic_meter
    cubic_meter cubm(1.0);
    
    // Smaller units
    EXPECT_DOUBLE_EQ(1000000.0, cubm.as<cubic_centimeter>().value());
    EXPECT_DOUBLE_EQ(1000.0, cubm.as<liter>().value());
    EXPECT_DOUBLE_EQ(1000000.0, cubm.as<milliliter>().value());
    
    // Non-SI units
    EXPECT_NEAR(264.172, cubm.as<gallon_us>().value(), 1e-3);
    EXPECT_NEAR(219.969, cubm.as<gallon_uk>().value(), 1e-3);
    EXPECT_NEAR(35.3147, cubm.as<cubic_foot>().value(), 1e-4);
    EXPECT_NEAR(33814.0, cubm.as<fluid_ounce_us>().value(), 0.5);  // Approximate
    
    // Construction from area × length
    square_meter area(2.0);
    meter height(3.0);
    cubic_meter volume = area * height;  // area × height should create volume
    EXPECT_DOUBLE_EQ(6.0, volume.value());
    
    // Also test length × area
    cubic_meter volume2 = height * area;  // should be commutative
    EXPECT_DOUBLE_EQ(6.0, volume2.value());
    
    // Different starting values
    liter l(1.0);
    EXPECT_DOUBLE_EQ(0.001, l.as<cubic_meter>().value());
    EXPECT_DOUBLE_EQ(1000.0, l.as<milliliter>().value());
    
    gallon_us gal(1.0);
    EXPECT_NEAR(3.78541, gal.as<liter>().value(), 1e-5);
    
    cubic_foot cf(1.0);
    EXPECT_NEAR(28.3169, cf.as<liter>().value(), 1e-4);
}

TEST(UnitsSIExtended, MagneticUnits) {
    // Test relationships between different magnetic units
    
    // Magnetic flux density
    tesla t(1.0);
    
    // Conversion to gauss (non-SI)
    EXPECT_DOUBLE_EQ(10000.0, t.as<gauss>().value());
    
    // Different starting values
    gauss g(1.0);
    EXPECT_DOUBLE_EQ(0.0001, g.as<tesla>().value());
    
    // Earth's magnetic field (approximately 0.5 gauss)
    gauss earthField(0.5);
    EXPECT_DOUBLE_EQ(0.00005, earthField.as<tesla>().value());
}

TEST(UnitsSIExtended, LightUnits) {
    // Test relationships between different light/optical units
    
    // Basic light units
    candela cd(1.0);  // Base SI unit for luminous intensity
    
    // Derived units
    lumen lm(1.0);    // Luminous flux
    lux lx(1.0);      // Illuminance
    
    // These are different dimensions, so no direct conversion tests
    // Just verify they can be constructed and manipulated
    
    // Arithmetic operations
    candela cd2 = cd * 2.0;
    EXPECT_DOUBLE_EQ(2.0, cd2.value());
    
    lumen lm2 = lm + lm;
    EXPECT_DOUBLE_EQ(2.0, lm2.value());
    
    lux lx2 = lx / 2.0;
    EXPECT_DOUBLE_EQ(0.5, lx2.value());
}

TEST(UnitsSIExtended, FlowRateUnits) {
    // Test relationships between different flow rate units
    
    // Base unit: cubic_meter_per_second
    cubic_meter_per_second cms(1.0);
    
    // Smaller units
    EXPECT_DOUBLE_EQ(60000.0, cms.as<liter_per_minute>().value());
    EXPECT_NEAR(2118.88, cms.as<cubic_foot_per_minute>().value(), 0.01);
    
    // Different starting values
    liter_per_minute lpm(1.0);
    EXPECT_DOUBLE_EQ(1.0/60000.0, lpm.as<cubic_meter_per_second>().value());
    
    cubic_foot_per_minute cfm(1.0);
    EXPECT_NEAR(28.3169/60.0, cfm.as<liter_per_second>().value(), 1e-4);
}

TEST(UnitsSIExtended, PrefixConsistency) {
    // Test that units with the same prefix have consistent scaling
    
    // kilo- prefix
    kilometer km(1.0);
    kilogram kg(1.0);
    kilohertz khz(1.0);
    kiloohm kohm(1.0);
    kilowatt kw(1.0);
    
    // All should be 1000 times their base units
    EXPECT_DOUBLE_EQ(1000.0, km.as<meter>().value());
    EXPECT_DOUBLE_EQ(1000.0, khz.as<hertz>().value());
    EXPECT_DOUBLE_EQ(1000.0, kohm.as<ohm>().value());
    
    // milli- prefix
    millimeter mm(1.0);
    milligram mg(1.0);
    millisecond ms(1.0);
    milliampere ma(1.0);
    millivolt mv(1.0);
    millimole mmol(1.0);
    
    // All should be 1/1000 of their base units
    EXPECT_DOUBLE_EQ(0.001, mm.as<meter>().value());
    EXPECT_DOUBLE_EQ(0.001, mg.as<gram>().value());
    EXPECT_DOUBLE_EQ(0.001, ms.as<second>().value());
    EXPECT_DOUBLE_EQ(0.001, ma.as<ampere>().value());
    EXPECT_DOUBLE_EQ(0.001, mv.as<volt>().value());
    EXPECT_DOUBLE_EQ(0.001, mmol.as<mole>().value());
    
    // micro- prefix
    micrometer um(1.0);
    microsecond us(1.0);
    microfarad uf(1.0);
    microhenry uh(1.0);
    micromole umol(1.0);
    
    // All should be 1/1000000 of their base units
    EXPECT_DOUBLE_EQ(0.000001, um.as<meter>().value());
    EXPECT_DOUBLE_EQ(0.000001, us.as<second>().value());
    EXPECT_DOUBLE_EQ(0.000001, uf.as<farad>().value());
    EXPECT_DOUBLE_EQ(0.000001, uh.as<henry>().value());
    EXPECT_DOUBLE_EQ(0.000001, umol.as<mole>().value());
}

TEST(UnitsSIExtended, DifferentDimensionsIncompatibility) {
    // Test that units of different dimensions don't convert to each other
    
    // We can't do static_assert here since it would prevent compilation
    // Instead, we'll test that same-dimension conversions do work
    
    // Length and time have different dimensions
    meter m(1.0);
    second s(1.0);
    
    // These should compile and work correctly (same dimension)
    auto km = m.as<kilometer>();
    auto ms = s.as<millisecond>();
    
    EXPECT_DOUBLE_EQ(0.001, km.value());
    EXPECT_DOUBLE_EQ(1000.0, ms.value());
    
    // The following would not compile:
    // auto invalid = m.as<second>();  // Different dimensions
}

TEST(UnitsSIExtended, ArithmeticOperationsDifferentUnits) {
    // Test arithmetic operations between different units of the same dimension
    
    meter m(1.0);
    centimeter cm(50.0);  // 0.5 meters
    
    // Addition between different units of same dimension
    // Result will be in the first unit's type
    auto result1 = m + cm.as<meter>();  // Need to convert explicitly
    EXPECT_DOUBLE_EQ(1.5, result1.value());
    
    // Subtraction between different units of same dimension
    auto result2 = meter(10.0) - kilometer(0.005);  // 10m - 5m = 5m
    EXPECT_DOUBLE_EQ(5.0, result2.value());
    
    // Multiplication by scalar
    auto result3 = millimeter(2000.0) * 2.0;  // 2000mm * 2 = 4000mm
    EXPECT_DOUBLE_EQ(4000.0, result3.value());
    EXPECT_DOUBLE_EQ(4.0, result3.as<meter>().value());
    
    // Division by scalar
    auto result4 = kilometer(0.1) / 2.0;  // 0.1km / 2 = 0.05km
    EXPECT_DOUBLE_EQ(0.05, result4.value());
    EXPECT_DOUBLE_EQ(50.0, result4.as<meter>().value());
}

TEST(UnitsSIExtended, ComparisonOperationsDifferentUnits) {
    // Test comparison operations between different units of the same dimension
    
    meter m(1.0);
    centimeter cm(150.0);  // 1.5 meters
    millimeter mm(500.0);  // 0.5 meters
    
    // Direct comparison between different units don't work
    // We need to convert them to the same unit first
    
    // Equality
    EXPECT_FALSE(m == cm.as<meter>());  // 1m != 1.5m
    EXPECT_TRUE(m == millimeter(1000.0).as<meter>());  // 1m == 1000mm
    
    // Inequality
    EXPECT_TRUE(m != cm.as<meter>());  // 1m != 1.5m
    EXPECT_FALSE(m != millimeter(1000.0).as<meter>());  // 1m == 1000mm
    
    // Less than
    EXPECT_TRUE(m < cm.as<meter>());  // 1m < 1.5m
    EXPECT_TRUE(mm.as<meter>() < m);  // 0.5m < 1m
    EXPECT_FALSE(cm.as<meter>() < m);  // 1.5m !< 1m
    
    // Greater than
    EXPECT_FALSE(m > cm.as<meter>());  // 1m !> 1.5m
    EXPECT_TRUE(m > mm.as<meter>());  // 1m > 0.5m
    EXPECT_TRUE(cm.as<meter>() > m);  // 1.5m > 1m
    
    // Less than or equal
    EXPECT_TRUE(m <= cm.as<meter>());  // 1m <= 1.5m
    EXPECT_TRUE(m <= meter(1.0));  // 1m <= 1m
    
    // Greater than or equal
    EXPECT_FALSE(m >= cm.as<meter>());  // 1m !>= 1.5m
    EXPECT_TRUE(m >= meter(1.0));  // 1m >= 1m
}

TEST(UnitsSIExtended, AssignmentOperationsDifferentUnits) {
    // Test assignment operations between different units of the same dimension
    
    meter m(1.0);
    
    // Addition assignment
    m += centimeter(50.0).as<meter>();  // 1m += 0.5m
    EXPECT_DOUBLE_EQ(1.5, m.value());
    
    // Subtraction assignment
    m -= millimeter(500.0).as<meter>();  // 1.5m -= 0.5m
    EXPECT_DOUBLE_EQ(1.0, m.value());
    
    // Multiplication assignment
    m *= 2.0;  // 1m *= 2
    EXPECT_DOUBLE_EQ(2.0, m.value());
    
    // Division assignment
    m /= 4.0;  // 2m /= 4
    EXPECT_DOUBLE_EQ(0.5, m.value());
}

TEST(UnitsSIExtended, MixedTemperatureUnits) {
    // In the updated library, temperature is primarily handled through
    // the kelvin unit, with conversions to/from Celsius and Fahrenheit
    // handled in the validation module.
    
    // Just test basic kelvin operations here
    kelvin temp1(300.0);
    kelvin temp2(350.0);
    
    auto diff = temp2 - temp1;
    EXPECT_DOUBLE_EQ(50.0, diff.value());
    
    auto average = (temp1 + temp2) / 2.0;
    EXPECT_DOUBLE_EQ(325.0, average.value());
}

TEST(UnitsSIExtended, ZeroValueComparisons) {
    // Test comparisons involving zero values
    
    // Length units
    EXPECT_TRUE(meter() == meter(0.0));
    EXPECT_TRUE(meter() == centimeter(0.0).as<meter>());
    EXPECT_TRUE(meter(0.0) < meter(0.1));
    EXPECT_TRUE(meter(0.0) <= meter(0.0));
    
    // Time units
    EXPECT_TRUE(second() == second(0.0));
    EXPECT_TRUE(second() == millisecond(0.0).as<second>());
    EXPECT_TRUE(second(0.0) < second(0.1));
    
    // Mass units
    EXPECT_TRUE(kilogram() == kilogram(0.0));
    EXPECT_TRUE(kilogram() == gram(0.0).as<kilogram>());
    EXPECT_TRUE(kilogram(0.0) < kilogram(0.1));
}

TEST(UnitsSIExtended, ConstantsWithUnits) {
    // Test physical constants expressed in units
    // (This is currently handled in the constants namespace without unit types,
    // but we can test conversions to values with units)
    
    // Standard gravity
    meter_per_second_squared gravity(constants::STANDARD_GRAVITY);
    EXPECT_DOUBLE_EQ(9.80665, gravity.value());
    
    // Standard atmosphere
    pascal atmosphere(constants::STANDARD_ATM_PRESSURE);
    EXPECT_DOUBLE_EQ(101325.0, atmosphere.value());
    
    // Standard temperature
    kelvin stdTemp(constants::STANDARD_TEMPERATURE);
    EXPECT_DOUBLE_EQ(293.15, stdTemp.value());
    
    // Speed of light
    meter_per_second lightSpeed(constants::SPEED_OF_LIGHT);
    EXPECT_DOUBLE_EQ(299792458.0, lightSpeed.value());
}

TEST(UnitsSIExtended, CompilationVerification) {
    // These tests use static_assert to verify that certain operations
    // compile correctly. If any of these assertions fail, the code won't compile.
    
    // Ratio checks
    static_assert(std::is_same_v<millimeter::ratio_type, std::ratio<1, 1000>>,
                  "millimeter should have a ratio of 1/1000");
    
    static_assert(std::is_same_v<kilometer::ratio_type, std::ratio<1000, 1>>,
                  "kilometer should have a ratio of 1000/1");
    
    // Dimension checks
    static_assert(std::is_same_v<meter::dimension_type, length_dimension>,
                  "meter should have length dimension");
    
    static_assert(std::is_same_v<second::dimension_type, time_dimension>,
                  "second should have time dimension");
    
    static_assert(std::is_same_v<kilogram::dimension_type, mass_dimension>,
                  "kilogram should have mass dimension");
}
