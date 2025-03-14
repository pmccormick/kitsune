#include <gtest/gtest.h>
#include <sstream>
#include "Units/core.h"
#include "Units/formatting.h"
#include "Units/si_units.h"
#include "Units/derived_units.h"

using namespace units;

class UnitsFormattingTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Save the original display style to restore it later
        originalStyle = getUnitDisplayStyle();
    }
    
    void TearDown() override {
        // Restore the original display style
        setUnitDisplayStyle(originalStyle);
    }
    
    UnitDisplayStyle originalStyle;
};

TEST_F(UnitsFormattingTest, FormatUnitSymbolPlain) {
    // Set to PLAIN style
    setUnitDisplayStyle(UnitDisplayStyle::PLAIN);
    
    // Test basic units
    EXPECT_EQ("m", (formatUnitSymbol<meter>()));
    EXPECT_EQ("s", (formatUnitSymbol<second>()));
    EXPECT_EQ("kg", (formatUnitSymbol<kilogram>()));
    EXPECT_EQ("K", (formatUnitSymbol<kelvin>()));
    
    // Test derived units with special formatting in PLAIN style
    EXPECT_EQ("m^2", (formatUnitSymbol<square_meter>()));
    EXPECT_EQ("m^3", (formatUnitSymbol<cubic_meter>()));
    EXPECT_EQ("m/s^2", (formatUnitSymbol<meters_per_second_squared>()));
    EXPECT_EQ("kg/m^3", (formatUnitSymbol<kilogram_per_cubic_meter>()));
    
    // Test angle units
    EXPECT_EQ("rad", (formatUnitSymbol<radian>()));
    
    // Test electrical units
    EXPECT_EQ("ohm", (formatUnitSymbol<ohm>()));
}

TEST_F(UnitsFormattingTest, FormatUnitSymbolUnicode) {
    // Set to UNICODE style
    setUnitDisplayStyle(UnitDisplayStyle::UNICODE);
    
    // Test basic units (same as PLAIN)
    EXPECT_EQ("m", (formatUnitSymbol<meter>()));
    EXPECT_EQ("s", (formatUnitSymbol<second>()));
    EXPECT_EQ("kg", (formatUnitSymbol<kilogram>()));
    EXPECT_EQ("K", (formatUnitSymbol<kelvin>()));
    
    // Test derived units with special formatting in UNICODE style
    EXPECT_EQ("m²", (formatUnitSymbol<square_meter>()));
    EXPECT_EQ("m³", (formatUnitSymbol<cubic_meter>()));
    EXPECT_EQ("m/s²", (formatUnitSymbol<meters_per_second_squared>()));
    EXPECT_EQ("kg/m³", (formatUnitSymbol<kilogram_per_cubic_meter>()));
    
    // Test angle units
    EXPECT_EQ("°", (formatUnitSymbol<radian>()));
    
    // Test electrical units
    EXPECT_EQ("Ω", (formatUnitSymbol<ohm>()));
}

TEST_F(UnitsFormattingTest, FormatUnitSymbolHTML) {
    // Set to HTML style
    setUnitDisplayStyle(UnitDisplayStyle::HTML);
    
    // Test basic units (same as PLAIN)
    EXPECT_EQ("m", (formatUnitSymbol<meter>()));
    EXPECT_EQ("s", (formatUnitSymbol<second>()));
    EXPECT_EQ("kg", (formatUnitSymbol<kilogram>()));
    EXPECT_EQ("K", (formatUnitSymbol<kelvin>()));
    
    // Test derived units with special formatting in HTML style
    EXPECT_EQ("m&sup2;", (formatUnitSymbol<square_meter>()));
    EXPECT_EQ("m&sup3;", (formatUnitSymbol<cubic_meter>()));
    EXPECT_EQ("m/s&sup2;", (formatUnitSymbol<meters_per_second_squared>()));
    EXPECT_EQ("kg/m&sup3;", (formatUnitSymbol<kilogram_per_cubic_meter>()));
    
    // Test angle units
    EXPECT_EQ("&deg;", (formatUnitSymbol<radian>()));
    
    // Test electrical units
    EXPECT_EQ("&Omega;", (formatUnitSymbol<ohm>()));
}

TEST_F(UnitsFormattingTest, FormatDimensionlessUnit) {
    // Dimensionless units should return empty string in all styles
    
    // PLAIN
    setUnitDisplayStyle(UnitDisplayStyle::PLAIN);
    EXPECT_EQ("", (formatUnitSymbol<dimensionless>()));
    
    // UNICODE
    setUnitDisplayStyle(UnitDisplayStyle::UNICODE);
    EXPECT_EQ("", (formatUnitSymbol<dimensionless>()));
    
    // HTML
    setUnitDisplayStyle(UnitDisplayStyle::HTML);
    EXPECT_EQ("", (formatUnitSymbol<dimensionless>()));
}

TEST_F(UnitsFormattingTest, ToStringBasic) {
    // Test toString function with different units and styles
    
    // Length - PLAIN style
    setUnitDisplayStyle(UnitDisplayStyle::PLAIN);
    EXPECT_EQ("10 m", toString(meter(10.0)));
    
    // Mass - UNICODE style
    setUnitDisplayStyle(UnitDisplayStyle::UNICODE);
    EXPECT_EQ("5 kg", toString(kilogram(5.0)));
    
    // Area - HTML style 
    setUnitDisplayStyle(UnitDisplayStyle::HTML);
    EXPECT_EQ("7.5 m&sup2;", toString(square_meter(7.5)));
    
    // Dimensionless - should not append unit symbol
    EXPECT_EQ("0.75", toString(dimensionless(0.75)));
}

TEST_F(UnitsFormattingTest, ToStringWithPrecision) {
    // Test toString with various numeric precision cases
    
    // Integer value
    EXPECT_EQ("10 s", toString(second(10.0)));
    
    // Fractional value
    EXPECT_EQ("3.14 m", toString(meter(3.14)));
    
    // Very small value (should use scientific notation)
    EXPECT_TRUE(toString(meter(1e-10)).find("e-") != std::string::npos);
    
    // Very large value (should use scientific notation)
    EXPECT_TRUE(toString(meter(1e+15)).find("e+") != std::string::npos);
}

TEST_F(UnitsFormattingTest, ToStringExplicitStyle) {
    // Test toString with explicit style parameter (overriding global setting)
    
    // Set global style to PLAIN
    setUnitDisplayStyle(UnitDisplayStyle::PLAIN);
    
    // Use explicit UNICODE style
    EXPECT_EQ("10 m²", toString(square_meter(10.0), UnitDisplayStyle::UNICODE));
    
    // Use explicit HTML style
    EXPECT_EQ("5 m&sup3;", toString(cubic_meter(5.0), UnitDisplayStyle::HTML));
    
    // Use explicit PLAIN style (same as global)
    EXPECT_EQ("2.5 m^2", toString(square_meter(2.5), UnitDisplayStyle::PLAIN));
}

TEST_F(UnitsFormattingTest, StreamOutput) {
    // Test operator<< for output streams
    
    std::ostringstream oss;
    
    // Length
    oss.str("");
    oss << meter(10.0);
    EXPECT_EQ("10 m", oss.str());
    
    // Time
    oss.str("");
    oss << second(5.5);
    EXPECT_EQ("5.5 s", oss.str());
    
    // Area with UNICODE style
    setUnitDisplayStyle(UnitDisplayStyle::UNICODE);
    oss.str("");
    oss << square_meter(2.5);
    EXPECT_EQ("2.5 m²", oss.str());
    
    // Dimensionless
    oss.str("");
    oss << dimensionless(3.14);
    EXPECT_EQ("3.14", oss.str());
}

TEST_F(UnitsFormattingTest, StreamMultipleUnits) {
    // Test streaming multiple units in sequence
    
    std::ostringstream oss;
    
    // Test with spaces between units
    oss << meter(5.0) << " " << second(2.0) << " " << kilogram(10.0);
    EXPECT_EQ("5 m 2 s 10 kg", oss.str());
}

TEST_F(UnitsFormattingTest, DisplayStyleChange) {
    // Test changing global display style 
    
    // Start with PLAIN
    setUnitDisplayStyle(UnitDisplayStyle::PLAIN);
    
    std::ostringstream oss;
    oss << square_meter(9.0);
    EXPECT_EQ("9 m^2", oss.str());
    
    // Change to UNICODE and test again
    setUnitDisplayStyle(UnitDisplayStyle::UNICODE);
    oss.str("");
    oss << square_meter(9.0);
    EXPECT_EQ("9 m²", oss.str());
    
    // Change to HTML and test again
    setUnitDisplayStyle(UnitDisplayStyle::HTML);
    oss.str("");
    oss << square_meter(9.0);
    EXPECT_EQ("9 m&sup2;", oss.str());
    
    // Get the current style
    EXPECT_EQ(UnitDisplayStyle::HTML, getUnitDisplayStyle());
}

TEST_F(UnitsFormattingTest, NegativeValues) {
    // Test formatting of negative values
    
    std::ostringstream oss;
    
    // Negative length
    oss << meter(-5.0);
    EXPECT_EQ("-5 m", oss.str());
    
    // Negative temperature
    oss.str("");
    oss << kelvin(-10.0);
    EXPECT_EQ("-10 K", oss.str());
}

TEST_F(UnitsFormattingTest, ZeroValues) {
    // Test formatting of zero values
    
    std::ostringstream oss;
    
    // Zero length
    oss << meter(0.0);
    EXPECT_EQ("0 m", oss.str());
    
    // Default-constructed unit (zero value)
    oss.str("");
    oss << meter();
    EXPECT_EQ("0 m", oss.str());
}

TEST_F(UnitsFormattingTest, ComplexDerivedUnits) {
    // Test formatting of more complex derived units
    
    // Test UNICODE mode for complex derived units
    setUnitDisplayStyle(UnitDisplayStyle::UNICODE);
    std::ostringstream oss;
    
    // Newton (force)
    oss << newton(15.0);
    EXPECT_EQ("15 N", oss.str());
    
    // Pascal (pressure)
    oss.str("");
    oss << pascal(101325.0);
    EXPECT_EQ("101325 Pa", oss.str());
    
    // Joule (energy)
    oss.str("");
    oss << joule(1000.0);
    EXPECT_EQ("1000 J", oss.str());
    
    // Watt (power)
    oss.str("");
    oss << watt(750.0);
    EXPECT_EQ("750 W", oss.str());
}
