#include <gtest/gtest.h>
#include "units/core.h"
#include "units/si_units.h"
#include "units/constants.h"

using namespace units;

TEST(UnitsSIBasic, LengthUnits) {
    // Test relationships between different length units
    
    // Base unit: meter
    meter m(1.0);
    
    // Smaller units
    EXPECT_DOUBLE_EQ(1000.0, m.as<millimeter>().value());
    EXPECT_DOUBLE_EQ(100.0, m.as<centimeter>().value());
    EXPECT_DOUBLE_EQ(1000000.0, m.as<micrometer>().value());
    EXPECT_DOUBLE_EQ(1000000000.0, m.as<nanometer>().value());
    
    // Larger units
    EXPECT_DOUBLE_EQ(0.001, m.as<kilometer>().value());
    
    // Non-SI units
    EXPECT_NEAR(39.3701, m.as<inch>().value(), 1e-4);
    EXPECT_NEAR(3.28084, m.as<foot>().value(), 1e-5);
    EXPECT_NEAR(1.09361, m.as<yard>().value(), 1e-5);
    EXPECT_NEAR(0.000621371, m.as<mile>().value(), 1e-9);
    EXPECT_NEAR(0.000539957, m.as<nautical_mile>().value(), 1e-9);
    
    // Round-trip conversions
    meter original(42.0);
    
    // meter -> cm -> meter
    auto cm = original.as<centimeter>();
    auto backToM = cm.as<meter>();
    EXPECT_DOUBLE_EQ(original.value(), backToM.value());
    
    // meter -> inch -> foot -> meter
    auto inch_val = original.as<inch>();
    auto foot_val = inch_val.as<foot>();
    auto backToM2 = foot_val.as<meter>();
    EXPECT_NEAR(original.value(), backToM2.value(), 1e-10);
}

TEST(UnitsSIBasic, TimeUnits) {
    // Test relationships between different time units
    
    // Base unit: second
    second s(1.0);
    
    // Smaller units
    EXPECT_DOUBLE_EQ(1000.0, s.as<millisecond>().value());
    EXPECT_DOUBLE_EQ(1000000.0, s.as<microsecond>().value());
    EXPECT_DOUBLE_EQ(1000000000.0, s.as<nanosecond>().value());
    
    // Larger units
    EXPECT_DOUBLE_EQ(1.0/60.0, s.as<minute>().value());
    EXPECT_DOUBLE_EQ(1.0/3600.0, s.as<hour>().value());
    EXPECT_DOUBLE_EQ(1.0/86400.0, s.as<day>().value());
    EXPECT_DOUBLE_EQ(1.0/604800.0, s.as<week>().value());
    EXPECT_DOUBLE_EQ(1.0/31536000.0, s.as<year>().value());
    
    // Different starting values
    minute min(1.0);
    EXPECT_DOUBLE_EQ(60.0, min.as<second>().value());
    EXPECT_DOUBLE_EQ(1.0/60.0, min.as<hour>().value());
    
    hour hr(24.0);
    EXPECT_DOUBLE_EQ(24.0, hr.as<day>().value());
    EXPECT_DOUBLE_EQ(1440.0, hr.as<minute>().value());
    EXPECT_DOUBLE_EQ(86400.0, hr.as<second>().value());
}

TEST(UnitsSIBasic, MassUnits) {
    // Test relationships between different mass units
    
    // Base unit: kilogram
    kilogram kg(1.0);
    
    // Smaller units
    EXPECT_DOUBLE_EQ(1000.0, kg.as<gram>().value());
    EXPECT_DOUBLE_EQ(1000000.0, kg.as<milligram>().value());
    
    // Larger units
    EXPECT_DOUBLE_EQ(0.001, kg.as<tonne>().value());
    
    // Non-SI units
    EXPECT_NEAR(2.20462, kg.as<pound>().value(), 1e-5);
    EXPECT_NEAR(35.274, kg.as<ounce>().value(), 1e-3);
    
    // Different starting values
    gram g(1000.0);
    EXPECT_DOUBLE_EQ(1.0, g.as<kilogram>().value());
    
    pound lb(1.0);
    EXPECT_NEAR(0.45359, lb.as<kilogram>().value(), 1e-5);
    EXPECT_NEAR(16.0, lb.as<ounce>().value(), 1e-10);
}

TEST(UnitsSIBasic, TemperatureUnits) {
    // For temperature, we're only testing the base unit kelvin
    // Temperature conversion is special and tested in validation tests
    
    // Base unit: kelvin
    kelvin k1(273.15);  // Freezing point of water
    kelvin k2(373.15);  // Boiling point of water
    
    // Test absolute zero
    kelvin absoluteZero(constants::ABSOLUTE_ZERO);
    EXPECT_DOUBLE_EQ(0.0, absoluteZero.value());
    
    // Test standard constants
    EXPECT_DOUBLE_EQ(273.15, constants::WATER_FREEZING_POINT);
    EXPECT_DOUBLE_EQ(373.15, constants::WATER_BOILING_POINT);
    EXPECT_DOUBLE_EQ(293.15, constants::STANDARD_TEMPERATURE);
    
    // Test arithmetic with kelvin
    kelvin sum = k1 + k2;
    EXPECT_DOUBLE_EQ(646.3, sum.value());
    
    kelvin diff = k2 - k1;
    EXPECT_DOUBLE_EQ(100.0, diff.value());
    
    kelvin scaled = k1 * 2.0;
    EXPECT_DOUBLE_EQ(546.3, scaled.value());
}

TEST(UnitsSIBasic, AngleUnits) {
    // Test relationships between different angle units
    
    // Base unit: radian
    radian rad(1.0);
    
    // Conversion to degrees
    EXPECT_NEAR(57.2958, rad.as<degree>().value(), 1e-4);
    
    // Full circle
    radian fullCircle(2.0 * constants::PI);
    EXPECT_NEAR(360.0, fullCircle.as<degree>().value(), 1e-10);
    
    // Right angle
    radian rightAngle(constants::PI / 2.0);
    EXPECT_NEAR(90.0, rightAngle.as<degree>().value(), 1e-10);
    
    // Degree to radian
    degree deg(45.0);
    EXPECT_NEAR(constants::PI / 4.0, deg.as<radian>().value(), 1e-10);
}

TEST(UnitsSIBasic, FrequencyUnits) {
    // Test relationships between different frequency units
    
    // Base unit: hertz
    hertz hz(1.0);
    
    // Larger units
    EXPECT_DOUBLE_EQ(0.001, hz.as<kilohertz>().value());
    EXPECT_DOUBLE_EQ(0.000001, hz.as<megahertz>().value());
    EXPECT_DOUBLE_EQ(0.000000001, hz.as<gigahertz>().value());
    
    // Different starting values
    kilohertz khz(1.0);
    EXPECT_DOUBLE_EQ(1000.0, khz.as<hertz>().value());
    
    megahertz mhz(1.0);
    EXPECT_DOUBLE_EQ(1000.0, mhz.as<kilohertz>().value());
    EXPECT_DOUBLE_EQ(1000000.0, mhz.as<hertz>().value());
    
    gigahertz ghz(2.4);  // WiFi frequency
    EXPECT_DOUBLE_EQ(2400.0, ghz.as<megahertz>().value());
    EXPECT_DOUBLE_EQ(2400000.0, ghz.as<kilohertz>().value());
}

TEST(UnitsSIBasic, ElectricalCurrentUnits) {
    // Test relationships between different electrical current units
    
    // Base unit: ampere
    ampere a(1.0);
    
    // Smaller units
    EXPECT_DOUBLE_EQ(1000.0, a.as<milliampere>().value());
    
    // Different starting values
    milliampere ma(500.0);
    EXPECT_DOUBLE_EQ(0.5, ma.as<ampere>().value());
}

TEST(UnitsSIBasic, ElectricalPotentialUnits) {
    // Test relationships between different electrical potential units
    
    // Base unit: volt
    volt v(1.0);
    
    // Smaller units
    EXPECT_DOUBLE_EQ(1000.0, v.as<millivolt>().value());
    
    // Different starting values
    millivolt mv(1500.0);
    EXPECT_DOUBLE_EQ(1.5, mv.as<volt>().value());
}

TEST(UnitsSIBasic, ElectricalResistanceUnits) {
    // Test relationships between different electrical resistance units
    
    // Base unit: ohm
    ohm o(1.0);
    
    // Larger units
    EXPECT_DOUBLE_EQ(0.001, o.as<kiloohm>().value());
    EXPECT_DOUBLE_EQ(0.000001, o.as<megaohm>().value());
    
    // Different starting values
    kiloohm ko(4.7);  // Common resistor value
    EXPECT_DOUBLE_EQ(4700.0, ko.as<ohm>().value());
    EXPECT_DOUBLE_EQ(0.0047, ko.as<megaohm>().value());
    
    megaohm mo(1.0);
    EXPECT_DOUBLE_EQ(1000.0, mo.as<kiloohm>().value());
    EXPECT_DOUBLE_EQ(1000000.0, mo.as<ohm>().value());
}

TEST(UnitsSIBasic, ElectricalCapacitanceUnits) {
    // Test relationships between different electrical capacitance units
    
    // Base unit: farad
    farad f(1.0);
    
    // Smaller units
    EXPECT_DOUBLE_EQ(1000000.0, f.as<microfarad>().value());
    EXPECT_DOUBLE_EQ(1000000000.0, f.as<nanofarad>().value());
    EXPECT_DOUBLE_EQ(1000000000000.0, f.as<picofarad>().value());
    
    // Different starting values
    microfarad uf(47.0);  // Common capacitor value
    EXPECT_DOUBLE_EQ(0.000047, uf.as<farad>().value());
    EXPECT_DOUBLE_EQ(47000.0, uf.as<nanofarad>().value());
    EXPECT_DOUBLE_EQ(47000000.0, uf.as<picofarad>().value());
}

TEST(UnitsSIBasic, ElectricalInductanceUnits) {
    // Test relationships between different electrical inductance units
    
    // Base unit: henry
    henry h(1.0);
    
    // Smaller units
    EXPECT_DOUBLE_EQ(1000.0, h.as<millihenry>().value());
    EXPECT_DOUBLE_EQ(1000000.0, h.as<microhenry>().value());
    
    // Different starting values
    millihenry mh(100.0);  // Common inductor value
    EXPECT_DOUBLE_EQ(0.1, mh.as<henry>().value());
    EXPECT_DOUBLE_EQ(100000.0, mh.as<microhenry>().value());
}

TEST(UnitsSIBasic, SubstanceAmountUnits) {
    // Test relationships between different substance amount units
    
    // Base unit: mole
    mole mol(1.0);
    
    // Smaller units
    EXPECT_DOUBLE_EQ(1000.0, mol.as<millimole>().value());
    EXPECT_DOUBLE_EQ(1000000.0, mol.as<micromole>().value());
    
    // Different starting values
    millimole mmol(500.0);
    EXPECT_DOUBLE_EQ(0.5, mmol.as<mole>().value());
    EXPECT_DOUBLE_EQ(500000.0, mmol.as<micromole>().value());
}

TEST(UnitsSIBasic, DataUnits) {
    // Test relationships between different data units
    
    // Base unit: byte
    byte b(1.0);
    
    // Larger units (using binary prefixes)
    EXPECT_DOUBLE_EQ(1.0/1024.0, b.as<kilobyte>().value());
    EXPECT_DOUBLE_EQ(1.0/1048576.0, b.as<megabyte>().value());
    EXPECT_DOUBLE_EQ(1.0/1073741824.0, b.as<gigabyte>().value());
    EXPECT_DOUBLE_EQ(1.0/1099511627776.0, b.as<terabyte>().value());
    
    // Bit conversion
    EXPECT_DOUBLE_EQ(8.0, b.as<bit>().value());
    
    // Different starting values
    kilobyte kb(1.0);
    EXPECT_DOUBLE_EQ(1024.0, kb.as<byte>().value());
    EXPECT_DOUBLE_EQ(8192.0, kb.as<bit>().value());
    
    megabyte mb(1.0);
    EXPECT_DOUBLE_EQ(1024.0, mb.as<kilobyte>().value());
    EXPECT_DOUBLE_EQ(1048576.0, mb.as<byte>().value());
}
