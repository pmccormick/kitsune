#include <iostream>
#include <cassert>
#include <cmath>
#include <string>
#include <vector>
#include <stdexcept>

// Include necessary headers
#include "Units.h"
#include "Cell.h"

// Helper function to check if two doubles are approximately equal
bool approxEqual(double a, double b, double epsilon = 1e-6) {
    return std::fabs(a - b) < epsilon;
}

// Test output helper
#define RUN_TEST(test) \
    do { \
        std::cout << "Running test: " << #test << "... "; \
        bool result = test(); \
        std::cout << (result ? "PASSED" : "FAILED") << std::endl; \
        testsPassed += result; \
        testsTotal++; \
    } while(0)

/**
 * Test suite specifically focused on Units functionality in the Cell class
 */
class CellUnitsTestSuite {
public:
    CellUnitsTestSuite() : testsPassed(0), testsTotal(0) {}
    
    // Run all tests
    bool runAllTests() {
        RUN_TEST(testTemperatureUnitConversions);
        RUN_TEST(testPressureUnitConversions);
        RUN_TEST(testDensityUnitConversions);
        RUN_TEST(testVelocityUnitConversions);
        RUN_TEST(testVertexVelocityUnitConversions);
        RUN_TEST(testRoundTripConversions);
        RUN_TEST(testInvalidUnitHandling);
        RUN_TEST(testUnitBoundaryValues);
        RUN_TEST(testValidationWithUnits);
        
        std::cout << "\nUnit Tests Results: " << testsPassed << " of " << testsTotal << " tests passed." << std::endl;
        return testsPassed == testsTotal;
    }
    
private:
    int testsPassed;
    int testsTotal;
    
    // Test temperature unit conversions
    bool testTemperatureUnitConversions() {
        Cell cell;
        bool pass = true;
        
        // Test basic temperature unit conversions
        cell.setTemperature(273.15); // 0°C / 32°F
        
        // Verify Kelvin to Celsius
        pass &= approxEqual(cell.getTemperatureWithUnits("C"), 0.0);
        
        // Verify Kelvin to Fahrenheit
        pass &= approxEqual(cell.getTemperatureWithUnits("F"), 32.0);
        
        // Verify Kelvin to Rankine
        pass &= approxEqual(cell.getTemperatureWithUnits("R"), 491.67, 0.01);
        
        // Test setting temperature in different units
        cell.setTemperatureWithUnits(100.0, "C");
        pass &= approxEqual(cell.getTemperature(), 373.15);
        
        cell.setTemperatureWithUnits(68.0, "F");
        pass &= approxEqual(cell.getTemperature(), 293.15, 0.01);
        
        return pass;
    }
    
    // Test pressure unit conversions
    bool testPressureUnitConversions() {
        Cell cell;
        bool pass = true;
        
        // Test basic pressure unit conversions
        cell.setPressure(101325.0); // 1 atm / 1.01325 bar / 14.6959 psi
        
        // Verify Pascal to atm
        pass &= approxEqual(cell.getPressureWithUnits("atm"), 1.0);
        
        // Verify Pascal to bar
        pass &= approxEqual(cell.getPressureWithUnits("bar"), 1.01325);
        
        // Verify Pascal to psi
        pass &= approxEqual(cell.getPressureWithUnits("psi"), 14.6959, 0.0001);
        
        // Verify Pascal to torr
        pass &= approxEqual(cell.getPressureWithUnits("torr"), 760.0, 0.1);
        
        // Test setting pressure in different units
        cell.setPressureWithUnits(2.0, "atm");
        pass &= approxEqual(cell.getPressure(), 202650.0);
        
        cell.setPressureWithUnits(10.0, "psi");
        pass &= approxEqual(cell.getPressure(), 68947.6, 0.1);
        
        return pass;
    }
    
    // Test density unit conversions
    bool testDensityUnitConversions() {
        Cell cell;
        bool pass = true;
        
        // Test basic density unit conversions
        cell.setDensity(1000.0); // 1000 kg/m³ ≈ 62.4 lb/ft³
        
        // Verify kg/m³ to lb/ft³
        pass &= approxEqual(cell.getDensityWithUnits("lb/ft³"), 62.428, 0.001);
        
        // Test setting density in different units
        cell.setDensityWithUnits(50.0, "lb/ft³");
        pass &= approxEqual(cell.getDensity(), 800.925, 0.1);
        
        return pass;
    }
    
    // Test velocity unit conversions
    bool testVelocityUnitConversions() {
        Cell cell;
        bool pass = true;
        
        // Test basic velocity unit conversions
        cell.setVelocityX(1.0);     // 1 m/s ≈ 2.237 mph ≈ 1.944 knot
        cell.setVelocityY(0.0);
        
        // Verify m/s to mph
        pass &= approxEqual(cell.getVelocityXWithUnits("mph"), 2.237, 0.001);
        
        // Verify m/s to knot
        pass &= approxEqual(cell.getVelocityXWithUnits("knot"), 1.944, 0.001);
        
        // Test setting velocity in different units
        cell.setVelocityWithUnits(100.0, 50.0, "mph");
        pass &= approxEqual(cell.getVelocityX(), 44.704, 0.001);
        pass &= approxEqual(cell.getVelocityY(), 22.352, 0.001);
        
        cell.setVelocityWithUnits(10.0, 5.0, "knot");
        pass &= approxEqual(cell.getVelocityX(), 5.144, 0.001);
        pass &= approxEqual(cell.getVelocityY(), 2.572, 0.001);
        
        return pass;
    }
    
    // Test vertex velocity unit conversions
    bool testVertexVelocityUnitConversions() {
        Cell cell;
        bool pass = true;
        
        // Test vertex velocity unit conversions
        cell.setVertexVelocity(Cell::VertexPosition::NORTHWEST, 10.0, 5.0);
        
        // Verify m/s to mph for vertex
        auto [vx_mph, vy_mph] = cell.getVertexVelocityWithUnits(Cell::VertexPosition::NORTHWEST, "mph");
        pass &= approxEqual(vx_mph, 22.37, 0.01);
        pass &= approxEqual(vy_mph, 11.185, 0.01);
        
        // Verify m/s to knot for vertex
        auto [vx_knot, vy_knot] = cell.getVertexVelocityWithUnits(Cell::VertexPosition::NORTHWEST, "knot");
        pass &= approxEqual(vx_knot, 19.44, 0.01);
        pass &= approxEqual(vy_knot, 9.72, 0.01);
        
        // Test setting vertex velocity with units
        cell.setVertexVelocityWithUnits(Cell::VertexPosition::NORTHEAST, 50.0, 25.0, "mph");
        auto [vx, vy] = cell.getVertexVelocity(Cell::VertexPosition::NORTHEAST);
        pass &= approxEqual(vx, 22.352, 0.001);
        pass &= approxEqual(vy, 11.176, 0.001);
        
        return pass;
    }
    
    // Test round-trip conversions for all unit types
    bool testRoundTripConversions() {
        Cell cell;
        bool pass = true;
        
        // Temperature round-trip
        for (const auto& unit : {"K", "C", "F", "R"}) {
            double initial = 300.0; // Use Kelvin as base
            cell.setTemperature(initial);
            double converted = cell.getTemperatureWithUnits(unit);
            cell.setTemperatureWithUnits(converted, unit);
            pass &= approxEqual(cell.getTemperature(), initial, 0.01);
        }
        
        // Pressure round-trip
        for (const auto& unit : {"Pa", "atm", "bar", "psi", "torr"}) {
            double initial = 101325.0; // Use Pascal as base
            cell.setPressure(initial);
            double converted = cell.getPressureWithUnits(unit);
            cell.setPressureWithUnits(converted, unit);
            pass &= approxEqual(cell.getPressure(), initial, 0.1);
        }
        
        // Density round-trip
        for (const auto& unit : {"kg/m³", "lb/ft³"}) {
            double initial = 1000.0; // Use kg/m³ as base
            cell.setDensity(initial);
            double converted = cell.getDensityWithUnits(unit);
            cell.setDensityWithUnits(converted, unit);
            pass &= approxEqual(cell.getDensity(), initial, 0.1);
        }
        
        // Velocity round-trip
        for (const auto& unit : {"m/s", "mph", "knot"}) {
            double initial_x = 10.0;
            double initial_y = 5.0;
            cell.setVelocityX(initial_x);
            cell.setVelocityY(initial_y);
            double converted_x = cell.getVelocityXWithUnits(unit);
            double converted_y = cell.getVelocityYWithUnits(unit);
            cell.setVelocityWithUnits(converted_x, converted_y, unit);
            pass &= approxEqual(cell.getVelocityX(), initial_x, 0.01);
            pass &= approxEqual(cell.getVelocityY(), initial_y, 0.01);
        }
        
        return pass;
    }
    
    // Test handling of invalid units
    bool testInvalidUnitHandling() {
        Cell cell;
        bool pass = true;
        
        // Test invalid unit handling
        try {
            cell.setTemperatureWithUnits(20.0, "invalid_unit");
            pass = false; // Should throw an exception
        } catch (const std::invalid_argument&) {
            // Expected exception
        }
        
        try {
            double temp = cell.getTemperatureWithUnits("invalid_unit");
            (void)temp; // Suppress unused variable warning
            pass = false; // Should throw an exception
        } catch (const std::invalid_argument&) {
            // Expected exception
        }
        
        try {
            // Try to convert between incompatible units
            cell.setTemperature(300.0);
            double pressure = cell.getTemperatureWithUnits("Pa");
            (void)pressure; // Suppress unused variable warning
            pass = false; // Should throw an exception
        } catch (const std::invalid_argument&) {
            // Expected exception
        }
        
        return pass;
    }
    
    // Test boundary values for units
    bool testUnitBoundaryValues() {
        Cell cell;
        bool pass = true;

	// Test basic pressure unit conversions
        cell.setPressure(101325.0); // 1 atm / 1.01325 bar / 14.6959 psi
        pass &= approxEqual(cell.getPressureWithUnits("atm"), 1.0);
        pass &= approxEqual(cell.getPressureWithUnits("bar"), 1.01325);
        pass &= approxEqual(cell.getPressureWithUnits("psi"), 14.6959, 0.0001);
        pass &= approxEqual(cell.getPressureWithUnits("torr"), 760.0, 0.1);

        // Test very high and low temperature values
        cell.setTemperatureWithUnits(5000.0, "C"); // Very high temperature
        pass &= cell.getTemperature() > 5000.0; // Should be higher than 5000K
        assert(pass == true);						
        
        cell.setTemperatureWithUnits(-300.0, "C"); // Below absolute zero
        pass &= cell.getTemperature() == 0.0; // Should be clamped to absolute zero
        assert(pass == true);						
        
        // Test very high and low pressure values
	
        cell.setPressureWithUnits(101325.0, "atm"); // Very high pressure
        pass &= cell.getPressure() > 101325000.0; // Should be higher than 1000 atm in Pa
        assert(pass == true);						
        
        cell.setPressureWithUnits(-10.0, "atm"); // Negative pressure
        pass &= cell.getPressure() == 0.0; // Should be clamped to zero
        assert(pass == true);						
        
        // Test very high and low density values
        cell.setDensityWithUnits(2000.0, "kg/m³"); // Very high density
        pass &= cell.getDensity() > 1990.0; // Should be approximately 2000 kg/m³
        assert(pass == true);						
        
        cell.setDensityWithUnits(0.0, "kg/m³"); // Zero density
        pass &= cell.getDensity() > 0.0; // Should be clamped to small positive value
        assert(pass == true);						
        
        return pass;
    }
    
    // Test validation with units
    bool testValidationWithUnits() {
        Cell cell;
        bool pass = true;
        
        // Test temperature validation
        cell.setTemperatureWithUnits(-300.0, "F"); // Very low temp in Fahrenheit
        pass &= cell.getTemperature() >= 0.0; // Should be clamped to absolute zero
        
        cell.setTemperatureWithUnits(-500.0, "C"); // Very low temp in Celsius
        pass &= cell.getTemperature() >= 0.0; // Should be clamped to absolute zero
        
        // Test pressure validation
        cell.setPressureWithUnits(-5.0, "bar"); // Negative pressure in bar
        pass &= cell.getPressure() >= 0.0; // Should be clamped to zero
        
        cell.setPressureWithUnits(-20.0, "psi"); // Negative pressure in psi
        pass &= cell.getPressure() >= 0.0; // Should be clamped to zero
        
        // Test density validation
        cell.setDensityWithUnits(-10.0, "lb/ft³"); // Negative density in lb/ft³
        pass &= cell.getDensity() > 0.0; // Should be clamped to positive value
        
        cell.setDensityWithUnits(0.0, "kg/m³"); // Zero density
        pass &= cell.getDensity() > 0.0; // Should be clamped to positive value
        
        return pass;
    }
};

// Main function to run the unit tests
int main() {
    std::cout << "===== Cell Units Test Suite =====" << std::endl;
    
    CellUnitsTestSuite testSuite;
    bool allPassed = testSuite.runAllTests();
    
    if (allPassed) {
        std::cout << "\nAll unit tests passed successfully!" << std::endl;
        return 0;
    } else {
        std::cout << "\nSome unit tests failed. See above for details." << std::endl;
        return 1;
    }
}


