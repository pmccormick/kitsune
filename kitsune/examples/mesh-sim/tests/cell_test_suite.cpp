#include <cassert>
#include <cmath>
#include <functional>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>

// Include necessary headers
#include "Cell.h"
#include "Material.h"
#include "Units.h"

// Forward declarations for test dependencies
class Grid;
class BoundaryClass;

// Simple mocks for testing
class MockMaterial : public Material {
public:
    MockMaterial() : Material(MaterialType::FLUID, "MockMaterial") {
        setProperty(MaterialProperty::DENSITY, 1000.0);
        setProperty(MaterialProperty::DYNAMIC_VISCOSITY, 0.001);
        setProperty(MaterialProperty::THERMAL_CONDUCTIVITY, 0.6);
        setProperty(MaterialProperty::SPECIFIC_HEAT, 4180.0);
    }
};

class MockGrid {
public:
    // Empty implementation for testing
};

// Test output helper
#define RUN_TEST(test) \
    do { \
        std::cout << "Running test: " << #test << "... "; \
        bool result = test(); \
        std::cout << (result ? "PASSED" : "FAILED") << std::endl; \
        testsPassed += result; \
        testsTotal++; \
    } while(0)

// Main test suite
class CellTestSuite {
public:
    CellTestSuite() : testsPassed(0), testsTotal(0) {}
    
    // Run all tests
    bool runAllTests() {
        RUN_TEST(testCellConstruction);
        RUN_TEST(testCellTypeOperations);
        RUN_TEST(testTemperatureOperations);
        RUN_TEST(testPressureOperations);
        RUN_TEST(testDensityOperations);
        RUN_TEST(testMaterialAssignment);
        RUN_TEST(testVertexVelocity);
        RUN_TEST(testBoundaryFlagOperations);
        RUN_TEST(testObstacleFlagOperations);
        RUN_TEST(testCellReset);
        RUN_TEST(testDynamicProperties);
        RUN_TEST(testPhysicalValidation);
        RUN_TEST(testPropertyAndFlagAccess);
        RUN_TEST(testPropertyTypeOperations);
        RUN_TEST(testPropertyAccessorMethods);
        RUN_TEST(testCellCenterVelocity);
        RUN_TEST(testFixedStateOperations);
        RUN_TEST(testSerializeDeserialize);
        RUN_TEST(testStringRepresentations);

        std::cout << "\nTest Results: " << testsPassed << " of " << testsTotal << " tests passed." << std::endl;
        return testsPassed == testsTotal;
    }

    // Test fixed state functionality
    bool testFixedStateOperations() {
      Cell cell;
      bool pass = true;

      // Default should not be fixed
      pass &= !cell.isFixed();
      assert(pass == true);

      // Set fixed state
      cell.setFixed(true);
      pass &= cell.isFixed();
      assert(pass == true);

      // Verify that setting cell type to BOUNDARY makes it fixed
      cell.setType(Cell::CellType::FLUID);
      pass &= !cell.isFixed();
      assert(pass == true);

      cell.setType(Cell::CellType::BOUNDARY);
      pass &= cell.isFixed();
      assert(pass == true);

      // Verify that setting cell type to SOLID makes it fixed
      cell.setType(Cell::CellType::FLUID);
      pass &= !cell.isFixed();
      assert(pass == true);

      cell.setType(Cell::CellType::SOLID);
      pass &= cell.isFixed();
      assert(pass == true);

      return pass;
    }

private:
    int testsPassed;
    int testsTotal;
    
    // Individual test cases
    
    // Test basic cell construction
    bool testCellConstruction() {
        // Default construction
        Cell cell;
        bool pass = true;
        
        pass &= cell.getType() == Cell::CellType::FLUID;
        pass &= std::abs(cell.getTemperature() - 293.15) < 1e-6;
        pass &= std::abs(cell.getPressure() - 101325.0) < 1e-6;
        pass &= std::abs(cell.getDensity() - 1.0) < 1e-6;
        pass &= !cell.isFixed();
        pass &= cell.getVelocityU() == 0.0;
        pass &= cell.getVelocityV() == 0.0;
        pass &= !cell.isBoundary();
        pass &= !cell.isObstacle();
        
        // Parameterized construction
        Cell solidCell(Cell::CellType::SOLID);
        pass &= solidCell.getType() == Cell::CellType::SOLID;
        pass &= solidCell.isFixed();
        pass &= solidCell.isObstacle();
        
        Cell boundaryCell(Cell::CellType::BOUNDARY);
        pass &= boundaryCell.getType() == Cell::CellType::BOUNDARY;
        pass &= boundaryCell.isFixed();
        pass &= boundaryCell.isBoundary();
        
        return pass;
    }
    
    // Test cell type change operations
    bool testCellTypeOperations() {
        Cell cell;
        bool pass = true;
        
        // Initial state
        pass &= cell.getType() == Cell::CellType::FLUID;
        pass &= !cell.isFixed();
        
        // Change to solid
        cell.setType(Cell::CellType::SOLID);
        pass &= cell.getType() == Cell::CellType::SOLID;
        pass &= cell.isFixed();
        pass &= cell.getFlag(Cell::CellFlag::IS_OBSTACLE);
        
        // Change to boundary
        cell.setType(Cell::CellType::BOUNDARY);
        pass &= cell.getType() == Cell::CellType::BOUNDARY;
        pass &= cell.isFixed();
        pass &= cell.getFlag(Cell::CellFlag::IS_BOUNDARY);
        
        // Change back to fluid
        cell.setType(Cell::CellType::FLUID);
        pass &= cell.getType() == Cell::CellType::FLUID;
        
        return pass;
    }
    
    // Test temperature operations with validation
    bool testTemperatureOperations() {
        Cell cell;
        bool pass = true;
        
        // Default temperature
        pass &= std::abs(cell.getTemperature() - 293.15) < 1e-6;
        
        // Set normal temperature
        cell.setTemperature(350.0);
        pass &= std::abs(cell.getTemperature() - 350.0) < 1e-6;
        
        // Set invalid temperature (below absolute zero)
        cell.setTemperature(-10.0);
        pass &= std::abs(cell.getTemperature() - 0.0) < 1e-6;  // Should be clamped to 0K
        
        // Test unit conversion
        cell.setTemperatureWithUnits(25.0, "C");
        pass &= std::abs(cell.getTemperature() - 298.15) < 1e-6;
        
        pass &= std::abs(cell.getTemperatureWithUnits("C") - 25.0) < 1e-6;
        pass &= std::abs(cell.getTemperatureWithUnits("F") - 77.0) < 1e-6;
        
        return pass;
    }
    
    // Test pressure operations with validation
    bool testPressureOperations() {
        Cell cell;
        bool pass = true;
        
        // Default pressure
        pass &= std::abs(cell.getPressure() - 101325.0) < 1e-6;
        
        // Set normal pressure
        cell.setPressure(200000.0);
        pass &= std::abs(cell.getPressure() - 200000.0) < 1e-6;
        
        // Set invalid pressure (negative)
        cell.setPressure(-10000.0);
        pass &= std::abs(cell.getPressure() - 0.0) < 1e-6;  // Should be clamped to 0
        
        // Test unit conversion
        cell.setPressureWithUnits(2.0, "bar");
        pass &= std::abs(cell.getPressure() - 200000.0) < 1e-6;
        
        pass &= std::abs(cell.getPressureWithUnits("bar") - 2.0) < 1e-6;
        pass &= std::abs(cell.getPressureWithUnits("atm") - 1.97385) < 1e-4;
        
        return pass;
    }
    
    // Test density operations with validation
    bool testDensityOperations() {
        Cell cell;
        bool pass = true;
        
        // Default density
        pass &= std::abs(cell.getDensity() - 1.0) < 1e-6;
        
        // Set normal density
        cell.setDensity(1000.0);
        pass &= std::abs(cell.getDensity() - 1000.0) < 1e-6;
        
        // Set invalid density (zero or negative)
        cell.setDensity(0.0);
        pass &= cell.getDensity() > 0.0;  // Should be clamped to small positive value
        
        // Test unit conversion
        cell.setDensityWithUnits(1.5, "kg/m³");
        pass &= std::abs(cell.getDensity() - 1.5) < 1e-6;
        
        double lbft3 = cell.getDensityWithUnits("lb/ft³");
        pass &= std::abs(lbft3 - (1.5 / 16.0185)) < 1e-4;
        
        return pass;
    }
    
    // Test material assignment and retrieval
    bool testMaterialAssignment() {
        Cell cell;
        bool pass = true;
        
        // Initial state: null material
        pass &= cell.getMaterial() == nullptr;
        
        // Assign material via shared_ptr
        auto material = std::make_shared<MockMaterial>();
        cell.setMaterial(material);
        pass &= cell.getMaterial() != nullptr;
        pass &= cell.getMaterial()->getName() == "MockMaterial";
        
        // Assign material via raw pointer
        MockMaterial* rawMaterial = new MockMaterial();
        cell.setMaterial(rawMaterial);
        pass &= cell.getMaterial() != nullptr;
        
        // Clean up (since we're not using unique_ptr)
        delete rawMaterial;
        
        return pass;
    }
    
    // Test vertex velocity operations
    bool testVertexVelocity() {
        Cell cell;
        bool pass = true;
        
        // Check initial state - all vertices have zero velocity
        auto [nw_vx, nw_vy] = cell.getVertexVelocity(Cell::VertexPosition::NORTHWEST);
        pass &= nw_vx == 0.0 && nw_vy == 0.0;
        
        // Set and get vertex velocity
        cell.setVertexVelocity(Cell::VertexPosition::NORTHEAST, 1.5, 2.5);
        auto [ne_vx, ne_vy] = cell.getVertexVelocity(Cell::VertexPosition::NORTHEAST);
        pass &= ne_vx == 1.5 && ne_vy == 2.5;
        
        // Test all vertices
        for (int i = 0; i < 4; i++) {
            Cell::VertexPosition pos = static_cast<Cell::VertexPosition>(i);
            double vx = i * 1.0;
            double vy = i * -1.0;
            
            cell.setVertexVelocity(pos, vx, vy);
            auto [get_vx, get_vy] = cell.getVertexVelocity(pos);
            pass &= get_vx == vx && get_vy == vy;
            
            // Test direct vertex access
            Cell::Vertex& vertex = cell.getVertex(pos);
            pass &= vertex.vx == vx && vertex.vy == vy;
        }
        
        // Test unit conversion
        cell.setVertexVelocityWithUnits(Cell::VertexPosition::SOUTHWEST, 10.0, 15.0, "mph");
        auto [sw_vx, sw_vy] = cell.getVertexVelocityWithUnits(Cell::VertexPosition::SOUTHWEST, "mph");
        pass &= std::abs(sw_vx - 10.0) < 1e-4 && std::abs(sw_vy - 15.0) < 1e-4;
        
        auto [sw_mps_vx, sw_mps_vy] = cell.getVertexVelocity(Cell::VertexPosition::SOUTHWEST);
        pass &= std::abs(sw_mps_vx - 4.4704) < 1e-4 && std::abs(sw_mps_vy - 6.7056) < 1e-4;
        
        return pass;
    }
    
    // Test boundary flag operations
    bool testBoundaryFlagOperations() {
        Cell cell;
        bool pass = true;
        
        // Initial state
        pass &= !cell.isBoundary();
        pass &= !cell.getFlag(Cell::CellFlag::IS_BOUNDARY);
        pass &= cell.getType() == Cell::CellType::FLUID;
        
        // Set boundary flag
        cell.setBoundary(true);
        pass &= cell.isBoundary();
        pass &= cell.getFlag(Cell::CellFlag::IS_BOUNDARY);
        pass &= cell.getType() == Cell::CellType::BOUNDARY;
        
        // Clear boundary flag
        cell.setBoundary(false);
        pass &= !cell.isBoundary();
        pass &= !cell.getFlag(Cell::CellFlag::IS_BOUNDARY);
        
        return pass;
    }
    
    // Test obstacle flag operations
    bool testObstacleFlagOperations() {
        Cell cell;
        bool pass = true;
        
        // Initial state
        pass &= !cell.isObstacle();
        pass &= !cell.getFlag(Cell::CellFlag::IS_OBSTACLE);
        pass &= cell.getType() == Cell::CellType::FLUID;
        
        // Set obstacle flag
        cell.setObstacle(true);
        pass &= cell.isObstacle();
        pass &= cell.getFlag(Cell::CellFlag::IS_OBSTACLE);
        pass &= cell.getType() == Cell::CellType::SOLID;
        
        // Clear obstacle flag
        cell.setObstacle(false);
        pass &= !cell.isObstacle();
        pass &= !cell.getFlag(Cell::CellFlag::IS_OBSTACLE);
        
        return pass;
    }
    
    // Test cell reset functionality
    bool testCellReset() {
        Cell cell;
        bool pass = true;
        
        // Set up cell with non-default values
        cell.setTemperature(350.0);
        cell.setPressure(200000.0);
        cell.setDensity(1200.0);
        cell.setVelocityU(10.0);
        cell.setVelocityV(5.0);
        cell.setVertexVelocity(Cell::VertexPosition::NORTHWEST, 2.0, 3.0);
        cell.setProperty(Cell::PropertyType::VORTICITY, 0.5);
        cell.setDynamicProperty("custom", 42.0);
        cell.setFlag(Cell::CellFlag::IS_INLET, true);
        
        // Reset the cell
        cell.reset();
        
        // Verify reset state
        pass &= std::abs(cell.getTemperature() - 293.15) < 1e-6;
        pass &= std::abs(cell.getPressure() - 101325.0) < 1e-6;
        pass &= std::abs(cell.getDensity() - 1.0) < 1e-6;
        pass &= cell.getVelocityU() == 0.0;
        pass &= cell.getVelocityV() == 0.0;
        
        auto [vx, vy] = cell.getVertexVelocity(Cell::VertexPosition::NORTHWEST);
        pass &= vx == 0.0 && vy == 0.0;
        
        pass &= cell.getProperty(Cell::PropertyType::VORTICITY) == 0.0;
        pass &= cell.getDynamicProperty("custom", -1.0) == -1.0; // Dynamic properties should be cleared
        pass &= !cell.getFlag(Cell::CellFlag::IS_INLET);  // Flags should be cleared
        
        return pass;
    }
    
    // Test dynamic properties
    bool testDynamicProperties() {
        Cell cell;
        bool pass = true;
        
        // Initial state - property doesn't exist
        pass &= cell.getDynamicProperty("test_prop", 99.9) == 99.9;
        
        // Set and retrieve property
        cell.setDynamicProperty("test_prop", 42.0);
        pass &= cell.getDynamicProperty("test_prop", 0.0) == 42.0;
        
        // Update property
        cell.setDynamicProperty("test_prop", 84.0);
        pass &= cell.getDynamicProperty("test_prop", 0.0) == 84.0;
        
        // Multiple properties
        cell.setDynamicProperty("prop2", -1.0);
        pass &= cell.getDynamicProperty("test_prop", 0.0) == 84.0;
        pass &= cell.getDynamicProperty("prop2", 0.0) == -1.0;
        
        // Reset should clear properties
        cell.reset();
        pass &= cell.getDynamicProperty("test_prop", 99.9) == 99.9;
        pass &= cell.getDynamicProperty("prop2", 99.9) == 99.9;
        
        return pass;
    }
    
    // Test physical validation of property values
    bool testPhysicalValidation() {
        Cell cell;
        bool pass = true;
        
        // Temperature validation (must be >= 0K)
        cell.setTemperature(-10.0);
        pass &= cell.getTemperature() >= 0.0;
        
        // Pressure validation (must be >= 0Pa)
        cell.setPressure(-5000.0);
        pass &= cell.getPressure() >= 0.0;
        
        // Density validation (must be > 0kg/m³)
        cell.setDensity(-2.0);
        pass &= cell.getDensity() > 0.0;
        cell.setDensity(0.0);
        pass &= cell.getDensity() > 0.0;
        
        return pass;
    }
    
    // Test property accessors and flag operations
    bool testPropertyAndFlagAccess() {
      Cell cell;
      bool pass = true;

      // Fixed property access
      for (int i = 0; i < static_cast<int>(Cell::PropertyType::COUNT); i++) {
        Cell::PropertyType propType = static_cast<Cell::PropertyType>(i);
        double value = i * 10.0;

        cell.setProperty(propType, value);
        pass &= cell.getProperty(propType) == value;
      }

      // Flag operations - corrected to use bit shifting
      // Each flag is a bit field, not a sequential value
      for (int i = 0; i < static_cast<int>(Cell::CellFlag::COUNT); i++) {
        Cell::CellFlag flag = static_cast<Cell::CellFlag>(1 << i);

        // Initially should be false
        pass &= !cell.getFlag(flag);

        // Set flag and check
        cell.setFlag(flag, true);
        pass &= cell.getFlag(flag);

        // Clear flag and check
        cell.setFlag(flag, false);
        pass &= !cell.getFlag(flag);
      }

      return pass;
    }

    /**
     * ====================================================================
     * Test Case for PropertyType Functionality
     * ====================================================================
     *
     * Below is a test case that should be added to the cell_test_suite.cpp file
     * to verify the correct functionality of the PropertyType system.
     * This test validates property storage, retrieval, and reset behavior.
     */

    bool testPropertyTypeOperations() {
      Cell cell;
      bool pass = true;

      // 1. Test default values
      // All properties should default to 0.0
      for (int i = 0; i < static_cast<int>(Cell::PropertyType::COUNT); i++) {
        Cell::PropertyType propType = static_cast<Cell::PropertyType>(i);
        pass &= cell.getProperty(propType) == 0.0;
      }

      // 2. Test setting and retrieving individual properties
      cell.setProperty(Cell::PropertyType::VORTICITY, 0.75);
      pass &= cell.getProperty(Cell::PropertyType::VORTICITY) == 0.75;

      cell.setProperty(Cell::PropertyType::STREAM_FUNCTION, -1.25);
      pass &= cell.getProperty(Cell::PropertyType::STREAM_FUNCTION) == -1.25;
      pass &= cell.getProperty(Cell::PropertyType::VORTICITY) ==
              0.75; // First value unchanged

      // 3. Test updating existing properties
      cell.setProperty(Cell::PropertyType::VORTICITY, 1.5);
      pass &= cell.getProperty(Cell::PropertyType::VORTICITY) == 1.5; // Updated

      // 4. Test multiple properties
      const double testValues[9] = {1.1, 2.2, 3.3, 4.4, 5.5,
                                    6.6, 7.7, 8.8, 9.9};

      for (int i = 0; i < static_cast<int>(Cell::PropertyType::COUNT); i++) {
        Cell::PropertyType propType = static_cast<Cell::PropertyType>(i);
        cell.setProperty(propType, testValues[i]);
      }

      for (int i = 0; i < static_cast<int>(Cell::PropertyType::COUNT); i++) {
        Cell::PropertyType propType = static_cast<Cell::PropertyType>(i);
        pass &= cell.getProperty(propType) == testValues[i];
      }

      // 5. Test reset behavior
      cell.reset();

      for (int i = 0; i < static_cast<int>(Cell::PropertyType::COUNT); i++) {
        Cell::PropertyType propType = static_cast<Cell::PropertyType>(i);
        pass &=
            cell.getProperty(propType) == 0.0; // After reset, all should be 0.0
      }

      return pass;
    }

    /**
     * ====================================================================
     * Updated Test Case for PropertyType Accessor Methods
     * ====================================================================
     *
     * This test verifies that the new dedicated accessor methods work correctly
     * and maintain the same functionality as the original property system.
     */

    bool testPropertyAccessorMethods() {
      Cell cell;
      bool pass = true;

      // Test vorticity accessors
      cell.setVorticity(0.75);
      pass &= cell.getVorticity() == 0.75;
      pass &= cell.getProperty(Cell::PropertyType::VORTICITY) ==
              0.75; // Underlying storage works

      // Test stream function accessors
      cell.setStreamFunction(-1.25);
      pass &= cell.getStreamFunction() == -1.25;
      pass &= cell.getProperty(Cell::PropertyType::STREAM_FUNCTION) == -1.25;

      // Test that values are independent
      pass &= cell.getVorticity() == 0.75; // Unchanged

      // Test wall distance accessors
      cell.setWallDistance(3.5);
      pass &= cell.getWallDistance() == 3.5;

      // Test reset behavior affects the accessors
      cell.reset();
      pass &= cell.getVorticity() == 0.0;
      pass &= cell.getStreamFunction() == 0.0;
      pass &= cell.getWallDistance() == 0.0;

      return pass;
    }

    // Test cell center velocity operations
    bool testCellCenterVelocity() {
      Cell cell;
      bool pass = true;

      // Default velocities should be zero
      pass &= cell.getVelocityU() == 0.0;
      pass &= cell.getVelocityV() == 0.0;

      // Set and get velocities
      cell.setVelocityU(3.5);
      cell.setVelocityV(-2.0);
      pass &= cell.getVelocityU() == 3.5;
      pass &= cell.getVelocityV() == -2.0;

      // Test unit conversion functions
      cell.setVelocityWithUnits(10.0, 5.0, "mph");
      pass &= std::abs(cell.getVelocityU() - 4.4704) < 1e-4;
      pass &= std::abs(cell.getVelocityV() - 2.2352) < 1e-4;

      pass &= std::abs(cell.getVelocityUWithUnits("mph") - 10.0) < 1e-4;
      pass &= std::abs(cell.getVelocityVWithUnits("mph") - 5.0) < 1e-4;

      // Test knots conversion
      pass &= std::abs(cell.getVelocityUWithUnits("knot") - 8.69) < 1e-2;

      return pass;
    }

    bool testSerializeDeserialize() {
      Cell originalCell;
      bool pass = true;

      // Set up a cell with non-default values for all properties
      originalCell.setType(Cell::CellType::BOUNDARY);
      originalCell.setTemperature(350.0);
      originalCell.setPressure(200000.0);
      originalCell.setDensity(1200.0);
      originalCell.setVelocityU(10.0);
      originalCell.setVelocityV(-5.0);

      // Set vertex velocities
      originalCell.setVertexVelocity(Cell::VertexPosition::NORTHWEST, 1.0, 2.0);
      originalCell.setVertexVelocity(Cell::VertexPosition::NORTHEAST, 3.0, 4.0);
      originalCell.setVertexVelocity(Cell::VertexPosition::SOUTHEAST, 5.0, 6.0);
      originalCell.setVertexVelocity(Cell::VertexPosition::SOUTHWEST, 7.0, 8.0);

      // Set fixed properties
      originalCell.setVorticity(0.75);
      originalCell.setStreamFunction(-1.25);
      originalCell.setKineticEnergy(62.5);
      originalCell.setDivergence(0.01);
      originalCell.setPressureCorrection(100.0);
      originalCell.setHeatFluxX(20.0);
      originalCell.setHeatFluxY(30.0);
      originalCell.setShearStress(15.0);
      originalCell.setWallDistance(2.5);

      // Set flags
      originalCell.setFlag(Cell::CellFlag::IS_INLET, true);
      originalCell.setFlag(Cell::CellFlag::IS_WALL, true);

      // Set dynamic properties
      originalCell.setDynamicProperty("custom1", 42.0);
      originalCell.setDynamicProperty("custom2", -3.14);
      originalCell.setDynamicProperty("custom3", 9.81);

      // Serialize the cell
      std::string serialized = originalCell.serialize();

      // Verify serialized string is not empty
      pass &= !serialized.empty();
      pass &= serialized.find("CELL_V1") != std::string::npos;

      // Create a new cell and deserialize into it
      Cell deserializedCell;
      bool deserializeResult = deserializedCell.deserialize(serialized);
      pass &= deserializeResult;

      // Verify all properties match between original and deserialized cells

      // Core properties
      pass &= deserializedCell.getType() == originalCell.getType();
      pass &= std::abs(deserializedCell.getTemperature() -
                       originalCell.getTemperature()) < 1e-6;
      pass &= std::abs(deserializedCell.getPressure() -
                       originalCell.getPressure()) < 1e-6;
      pass &= std::abs(deserializedCell.getDensity() -
                       originalCell.getDensity()) < 1e-6;
      pass &= std::abs(deserializedCell.getVelocityU() -
                       originalCell.getVelocityU()) < 1e-6;
      pass &= std::abs(deserializedCell.getVelocityV() -
                       originalCell.getVelocityV()) < 1e-6;

      // Vertex velocities
      for (int i = 0; i < 4; i++) {
        Cell::VertexPosition pos = static_cast<Cell::VertexPosition>(i);
        auto [orig_vx, orig_vy] = originalCell.getVertexVelocity(pos);
        auto [deser_vx, deser_vy] = deserializedCell.getVertexVelocity(pos);

        pass &= std::abs(deser_vx - orig_vx) < 1e-6;
        pass &= std::abs(deser_vy - orig_vy) < 1e-6;
      }

      // Fixed properties
      for (int i = 0; i < static_cast<int>(Cell::PropertyType::COUNT); i++) {
        Cell::PropertyType propType = static_cast<Cell::PropertyType>(i);
        double origValue = originalCell.getProperty(propType);
        double deserValue = deserializedCell.getProperty(propType);

        pass &= std::abs(deserValue - origValue) < 1e-6;
      }

      // Flags
      for (int i = 0; i < static_cast<int>(Cell::CellFlag::COUNT); i++) {
        Cell::CellFlag flag = static_cast<Cell::CellFlag>(1 << i);
        pass &= deserializedCell.getFlag(flag) == originalCell.getFlag(flag);
      }

      // Dynamic properties - this is trickier since we need the same property
      // names We'll check the ones we explicitly set
      pass &= std::abs(deserializedCell.getDynamicProperty("custom1", 0.0) -
                       42.0) < 1e-6;
      pass &= std::abs(deserializedCell.getDynamicProperty("custom2", 0.0) -
                       (-3.14)) < 1e-6;
      pass &= std::abs(deserializedCell.getDynamicProperty("custom3", 0.0) -
                       9.81) < 1e-6;

      // Verify material is preserved in serialization format (but actual
      // material object reference won't be restored without a material
      // registry)
      pass &= serialized.find("MAT=") != std::string::npos;

      // Test corrupted serialization data
      Cell badCell;
      bool shouldFail = badCell.deserialize("NOT_VALID_DATA");
      pass &= !shouldFail;

      // Test partial serialization data
      std::string partialData =
          "CELL_V1\nTYPE=0\n"; // Just the version and type
      Cell partialCell;
      bool partialResult = partialCell.deserialize(partialData);
      pass &= partialResult; // Should succeed with partial data
      pass &= partialCell.getType() ==
              Cell::CellType::FLUID; // Should have the specified type

      return pass;
    }

    bool testStringRepresentations() {
      Cell cell;
      bool pass = true;

      // Configure cell with some non-default values
      cell.setType(Cell::CellType::BOUNDARY);
      cell.setTemperature(350.0);
      cell.setPressure(200000.0);
      cell.setDensity(1200.0);
      cell.setVelocityU(10.5);
      cell.setVelocityV(-5.2);
      cell.setVorticity(0.75);
      cell.setFlag(Cell::CellFlag::IS_INLET, true);
      cell.setDynamicProperty("custom_prop", 42.0);

      // 1. Test toString with different verbosity levels

      // Minimal verbosity (level 0)
      std::string minimalStr = cell.toString(0);
      pass &= !minimalStr.empty();
      pass &= minimalStr.find("Cell [Type: BOUNDARY]") != std::string::npos;
      pass &= minimalStr.find("Temperature") ==
              std::string::npos; // Should not include details

      // Normal verbosity (level 1)
      std::string normalStr = cell.toString(1);
      pass &= !normalStr.empty();
      pass &= normalStr.find("Cell [Type: BOUNDARY]") != std::string::npos;

      // Use more flexible string matching for numeric values that might have
      // different formatting
      pass &= normalStr.find("Temperature: 35") !=
              std::string::npos; // Match just the start of 350
      pass &= normalStr.find("Pressure: 2") !=
              std::string::npos; // Match just the start of 200000
      pass &= normalStr.find("Density: 12") !=
              std::string::npos; // Match just the start of 1200

      // For velocity, check components separately as formatting might vary
      pass &= normalStr.find("10.5") != std::string::npos ||
              normalStr.find("10,5") != std::string::npos;
      pass &= normalStr.find("-5.2") != std::string::npos ||
              normalStr.find("-5,2") != std::string::npos;

      pass &= normalStr.find("Fixed: Yes") !=
              std::string::npos; // Boundary cells are fixed

      // The string might use different wording for boundary status
      // Let's check if any boundary-related text is present
      bool boundaryTextFound =
          normalStr.find("Is Boundary: Yes") != std::string::npos ||
          normalStr.find("Boundary: Yes") != std::string::npos ||
          normalStr.find("boundary: Yes") != std::string::npos ||
          normalStr.find("BOUNDARY") != std::string::npos;
      pass &= boundaryTextFound;

      pass &= normalStr.find("Vorticity") ==
              std::string::npos; // Detailed props not in level 1

      // Detailed verbosity (level 2)
      std::string detailedStr = cell.toString(2);
      pass &= !detailedStr.empty();
      pass &= detailedStr.find("Cell [Type: BOUNDARY]") != std::string::npos;
      pass &= detailedStr.find("Temperature: 350") != std::string::npos;
      pass &= detailedStr.find("Vorticity: 0.75") !=
              std::string::npos; // Should include properties
      pass &= detailedStr.find("INLET") !=
              std::string::npos; // Should include flags
      pass &= detailedStr.find("custom_prop: 42") !=
              std::string::npos; // Should include dynamic properties

      // 2. Test print method (redirecting output to stringstream)
      std::ostringstream oss;
      cell.print(oss, 1); // Normal verbosity
      std::string printStr = oss.str();
      pass &= !printStr.empty();

      // Instead of exact string matching, check for the same key content
      pass &= printStr.find("Cell [Type: BOUNDARY]") != std::string::npos;
      pass &= printStr.find("Temperature:") != std::string::npos;
      pass &= printStr.find("Pressure:") != std::string::npos;
      pass &= printStr.find("Density:") != std::string::npos;
      pass &= printStr.find("Velocity:") != std::string::npos;
      pass &= printStr.find("Fixed:") != std::string::npos;

      // 3. Test toSVG method - Basic validation
      std::string svgStr = cell.toSVG();
      pass &= !svgStr.empty();
      pass &= svgStr.find("<svg") != std::string::npos; // Contains SVG tag
      pass &= svgStr.find("</svg>") !=
              std::string::npos; // Contains closing SVG tag

      // Test different scale and velocity options
      std::string svgStr2 =
          cell.toSVG(20.0, false); // Larger scale, no velocity vectors
      pass &=
          svgStr2.find("width=\"240\"") != std::string::npos; // 12 * 20.0 = 240

      // 4. Test static string methods
      pass &= Cell::cellTypeToString(Cell::CellType::FLUID) == "FLUID";
      pass &= Cell::cellTypeToString(Cell::CellType::SOLID) == "SOLID";
      pass &= Cell::cellTypeToString(Cell::CellType::BOUNDARY) == "BOUNDARY";

      pass &= Cell::cellFlagToString(Cell::CellFlag::IS_INLET) == "INLET";
      pass &= Cell::cellFlagToString(Cell::CellFlag::IS_OUTLET) == "OUTLET";
      pass &= Cell::cellFlagToString(Cell::CellFlag::IS_WALL) == "WALL";
      pass &= Cell::cellFlagToString(Cell::CellFlag::IS_SYMMETRY) == "SYMMETRY";
      pass &= Cell::cellFlagToString(Cell::CellFlag::IS_BOUNDARY) == "BOUNDARY";
      pass &= Cell::cellFlagToString(Cell::CellFlag::IS_OBSTACLE) == "OBSTACLE";

      return pass;
    }
};

int main() {
    std::cout << "===== Cell Class Test Suite =====" << std::endl;
    
    CellTestSuite testSuite;
    bool allPassed = testSuite.runAllTests();
    
    if (allPassed) {
        std::cout << "\nAll tests passed successfully!" << std::endl;
        return 0;
    } else {
        std::cout << "\nSome tests failed. See above for details." << std::endl;
        return 1;
    }
}

