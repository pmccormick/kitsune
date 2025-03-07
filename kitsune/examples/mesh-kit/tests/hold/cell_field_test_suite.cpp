/**
 * @file cell_field_test_suite.cpp
 * @brief Tests for the Cell class with field-based storage
 *
 * This suite tests that the Cell class correctly accesses and modifies
 * data stored in Fields owned by the Mesh.
 */

#include "cell_test_fixture.h"
#include <cassert>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

// Simple test framework
#define TEST(name) void name(CellTestFixture &fixture)
#define ASSERT(condition)                                                      \
  if (!(condition)) {                                                          \
    std::cerr << "Assertion failed: " << #condition << " at " << __FILE__      \
              << ":" << __LINE__ << std::endl;                                 \
    assert(condition);                                                         \
  }
#define ASSERT_EQ(a, b) ASSERT((a) == (b))
#define ASSERT_DOUBLE_EQ(a, b) ASSERT(std::abs((a) - (b)) < 1e-10)
#define ASSERT_NE(a, b) ASSERT((a) != (b))
#define ASSERT_TRUE(a) ASSERT(a)
#define ASSERT_FALSE(a) ASSERT(!(a))

// Test that Cell reads from fields correctly
TEST(TestCellReadsFromFields) {
  // Get the central cell for testing
  Cell &cell = fixture.getCentralCell();
  Mesh &mesh = fixture.getMesh();

  // Test temperature reading
  double expectedTemp = mesh.getTemperatureField()(1, 1);
  ASSERT_DOUBLE_EQ(cell.getTemperature(), expectedTemp);

  // Test pressure reading
  double expectedPressure = mesh.getPressureField()(1, 1);
  ASSERT_DOUBLE_EQ(cell.getPressure(), expectedPressure);

  // Test density reading
  double expectedDensity = mesh.getDensityField()(1, 1);
  ASSERT_DOUBLE_EQ(cell.getDensity(), expectedDensity);

  // Test velocity reading
  double expectedVelocityU = mesh.getVelocityUField()(1, 1);
  double expectedVelocityV = mesh.getVelocityVField()(1, 1);
  ASSERT_DOUBLE_EQ(cell.getVelocityU(), expectedVelocityU);
  ASSERT_DOUBLE_EQ(cell.getVelocityV(), expectedVelocityV);

  // Test fixed property reading
  for (size_t p = 0; p < static_cast<size_t>(Cell::PropertyType::COUNT); ++p) {
    double expectedValue = mesh.getPropertiesField()(1, 1, p);
    ASSERT_DOUBLE_EQ(cell.getProperty(static_cast<Cell::PropertyType>(p)),
                     expectedValue);
  }

  // Test vertex velocity reading
  for (size_t v = 0; v < 4; ++v) {
    auto expectedVertexVelocity =
        std::make_pair(mesh.getVertexVelocityXField()(1, 1, v),
                       mesh.getVertexVelocityYField()(1, 1, v));

    auto actualVertexVelocity =
        cell.getVertexVelocity(static_cast<Cell::VertexPosition>(v));
    ASSERT_DOUBLE_EQ(actualVertexVelocity.first, expectedVertexVelocity.first);
    ASSERT_DOUBLE_EQ(actualVertexVelocity.second,
                     expectedVertexVelocity.second);
  }

  // Test material reading
  ASSERT_EQ(cell.getMaterial(), fixture.getDefaultMaterial());

  // Test type flags
  ASSERT_EQ(cell.getType(), Cell::CellType::FLUID);
  ASSERT_FALSE(cell.isBoundary());
  ASSERT_FALSE(cell.isFixed());
  ASSERT_FALSE(cell.isObstacle());
}

// Test that Cell writes to fields correctly
TEST(TestCellWritesToFields) {
  // Get the central cell for testing
  Cell &cell = fixture.getCentralCell();
  Mesh &mesh = fixture.getMesh();

  // Test temperature writing
  double newTemp = 350.0;
  cell.setTemperature(newTemp);
  ASSERT_DOUBLE_EQ(mesh.getTemperatureField()(1, 1), newTemp);

  // Test pressure writing
  double newPressure = 98000.0;
  cell.setPressure(newPressure);
  ASSERT_DOUBLE_EQ(mesh.getPressureField()(1, 1), newPressure);

  // Test density writing
  double newDensity = 1.5;
  cell.setDensity(newDensity);
  ASSERT_DOUBLE_EQ(mesh.getDensityField()(1, 1), newDensity);

  // Test velocity writing
  double newVelocityU = 2.5;
  double newVelocityV = -1.3;
  cell.setVelocityU(newVelocityU);
  cell.setVelocityV(newVelocityV);
  ASSERT_DOUBLE_EQ(mesh.getVelocityUField()(1, 1), newVelocityU);
  ASSERT_DOUBLE_EQ(mesh.getVelocityVField()(1, 1), newVelocityV);

  // Test fixed property writing
  for (size_t p = 0; p < static_cast<size_t>(Cell::PropertyType::COUNT); ++p) {
    double newValue = 100.0 + p;
    cell.setProperty(static_cast<Cell::PropertyType>(p), newValue);
    ASSERT_DOUBLE_EQ(mesh.getPropertiesField()(1, 1, p), newValue);
  }

  // Test vertex velocity writing
  for (size_t v = 0; v < 4; ++v) {
    double newVx = 0.5 + 0.1 * v;
    double newVy = 0.7 + 0.1 * v;
    cell.setVertexVelocity(static_cast<Cell::VertexPosition>(v), newVx, newVy);
    ASSERT_DOUBLE_EQ(mesh.getVertexVelocityXField()(1, 1, v), newVx);
    ASSERT_DOUBLE_EQ(mesh.getVertexVelocityYField()(1, 1, v), newVy);
  }

  // Test type flag writing
  cell.setType(Cell::CellType::BOUNDARY);
  ASSERT_EQ(mesh.getCellTypeField()(1, 1), Cell::CellType::BOUNDARY);
  ASSERT_TRUE(mesh.getBoundaryFlagField()(1, 1));

  cell.setType(Cell::CellType::SOLID);
  ASSERT_EQ(mesh.getCellTypeField()(1, 1), Cell::CellType::SOLID);
  ASSERT_TRUE(mesh.getObstacleFlagField()(1, 1));

  cell.setType(Cell::CellType::FLUID);
  ASSERT_EQ(mesh.getCellTypeField()(1, 1), Cell::CellType::FLUID);
  ASSERT_FALSE(mesh.getBoundaryFlagField()(1, 1));
  ASSERT_FALSE(mesh.getObstacleFlagField()(1, 1));
}

// Test unit conversions still work correctly
TEST(TestCellUnitsConversion) {
  Cell &cell = fixture.getCentralCell();

  // Temperature conversion
  double tempK = 300.0;
  cell.setTemperature(tempK);
  ASSERT_DOUBLE_EQ(cell.getTemperature(), tempK);
  ASSERT_DOUBLE_EQ(cell.getTemperatureWithUnits("C"), tempK - 273.15);

  double tempC = 25.0;
  cell.setTemperatureWithUnits(tempC, "C");
  ASSERT_DOUBLE_EQ(cell.getTemperature(), tempC + 273.15);

  // Pressure conversion
  double pressurePa = 101325.0;
  cell.setPressure(pressurePa);
  ASSERT_DOUBLE_EQ(cell.getPressure(), pressurePa);
  ASSERT_DOUBLE_EQ(cell.getPressureWithUnits("bar"), pressurePa / 1.0e5);

  double pressureBar = 2.0;
  cell.setPressureWithUnits(pressureBar, "bar");
  ASSERT_DOUBLE_EQ(cell.getPressure(), pressureBar * 1.0e5);

  // Density conversion
  double densityKgm3 = 1.2;
  cell.setDensity(densityKgm3);
  ASSERT_DOUBLE_EQ(cell.getDensity(), densityKgm3);
  ASSERT_DOUBLE_EQ(cell.getDensityWithUnits("g/cm³"), densityKgm3 / 1000.0);

  // Velocity conversion
  cell.setVelocityWithUnits(10.0, 5.0, "mph");
  ASSERT_DOUBLE_EQ(cell.getVelocityUWithUnits("mph"), 10.0);
  ASSERT_DOUBLE_EQ(cell.getVelocityVWithUnits("mph"), 5.0);
}

// Test material operations
TEST(TestCellMaterialOperations) {
  Cell &cell = fixture.getCentralCell();

  // Initially has the default material
  ASSERT_EQ(cell.getMaterial(), fixture.getDefaultMaterial());

  // Create a new material
  auto newMaterial = std::make_shared<Material>(Material::MaterialType::FLUID,
                                                "NewTestMaterial");
  newMaterial->setProperty(Material::MaterialProperty::DENSITY, 2.0);

  // Set the new material
  cell.setMaterial(newMaterial);
  ASSERT_EQ(cell.getMaterial(), newMaterial);
  ASSERT_NE(cell.getMaterial(), fixture.getDefaultMaterial());

  // Test material mixing
  double mixFraction = 0.3;
  cell.mixMaterial(fixture.getDefaultMaterial(), mixFraction);

  // The mixed material should be different from both original materials
  ASSERT_NE(cell.getMaterial(), newMaterial);
  ASSERT_NE(cell.getMaterial(), fixture.getDefaultMaterial());

  // But should have a mixed density
  double expectedDensity = 2.0 * (1.0 - mixFraction) + 1.0 * mixFraction;
  double actualDensity =
      cell.getMaterial()->getProperty(Material::MaterialProperty::DENSITY);
  ASSERT_DOUBLE_EQ(actualDensity, expectedDensity);
}

// Test boundary cell types
TEST(TestBoundaryCellTypes) {
  // Get a boundary cell
  Cell &boundaryCell = fixture.getWestBoundaryCell();

  // Verify it has boundary type
  ASSERT_EQ(boundaryCell.getType(), Cell::CellType::BOUNDARY);
  ASSERT_TRUE(boundaryCell.isBoundary());
  ASSERT_TRUE(boundaryCell.isFixed());
  ASSERT_FALSE(boundaryCell.isObstacle());

  // Create and set a boundary condition
  auto dirichletBoundary = std::make_shared<DirichletBoundary>("TestBoundary");
  boundaryCell.setBoundaryCondition(dirichletBoundary);

  // Verify boundary condition is set
  ASSERT_EQ(boundaryCell.getBoundaryCondition(), dirichletBoundary);
}

// Test cell diagnostic output still works
TEST(TestCellDiagnosticOutput) {
  Cell &cell = fixture.getCentralCell();

  // Set some known values
  cell.setTemperature(300.0);
  cell.setPressure(101325.0);
  cell.setVelocityU(1.0);
  cell.setVelocityV(2.0);

  // Generate and check string representation
  std::string cellStr = cell.toString();

  // Basic checking that the string contains expected information
  ASSERT_TRUE(cellStr.find("FLUID") != std::string::npos);
  ASSERT_TRUE(cellStr.find("300") != std::string::npos);
  ASSERT_TRUE(cellStr.find("101325") != std::string::npos);
}

// Test cell serialization/deserialization
TEST(TestCellSerialization) {
  Cell &cell = fixture.getCentralCell();

  // Set some known values
  cell.setTemperature(325.0);
  cell.setPressure(98000.0);
  cell.setDensity(1.2);
  cell.setVelocityU(0.5);
  cell.setVelocityV(0.8);

  // Serialize the cell
  std::string serialized = cell.serialize();

  // Reset the cell to different values
  cell.setTemperature(200.0);
  cell.setPressure(100000.0);

  // Deserialize and verify values are restored
  bool success = cell.deserialize(serialized);
  ASSERT_TRUE(success);

  ASSERT_DOUBLE_EQ(cell.getTemperature(), 325.0);
  ASSERT_DOUBLE_EQ(cell.getPressure(), 98000.0);
  ASSERT_DOUBLE_EQ(cell.getDensity(), 1.2);
  ASSERT_DOUBLE_EQ(cell.getVelocityU(), 0.5);
  ASSERT_DOUBLE_EQ(cell.getVelocityV(), 0.8);
}

// Test accessing neighboring cells through the mesh
TEST(TestCellNeighborAccess) {
  Cell &centralCell = fixture.getCentralCell();
  Mesh &mesh = fixture.getMesh();

  // Get neighbors directly from mesh
  auto neighbors = mesh.getNeighbors(1, 1);

  // Verify the neighbors match our expectations
  ASSERT_EQ(neighbors[0], &fixture.getNorthBoundaryCell()); // North
  ASSERT_EQ(neighbors[1], &fixture.getEastBoundaryCell());  // East
  ASSERT_EQ(neighbors[2], &fixture.getSouthBoundaryCell()); // South
  ASSERT_EQ(neighbors[3], &fixture.getWestBoundaryCell());  // West

  // Check that properties can be accessed from neighbors
  ASSERT_EQ(neighbors[0]->getType(), Cell::CellType::BOUNDARY);
  ASSERT_EQ(neighbors[1]->getType(), Cell::CellType::BOUNDARY);
  ASSERT_EQ(neighbors[2]->getType(), Cell::CellType::BOUNDARY);
  ASSERT_EQ(neighbors[3]->getType(), Cell::CellType::BOUNDARY);
}

// Test dynamic property access (which is not field-based)
TEST(TestDynamicProperties) {
  Cell &cell = fixture.getCentralCell();

  // Dynamic properties are stored directly in the Cell object
  std::string propName = "TestProperty";
  double propValue = 42.0;

  // Set and get a dynamic property
  cell.setDynamicProperty(propName, propValue);
  ASSERT_DOUBLE_EQ(cell.getDynamicProperty(propName), propValue);

  // Default value for nonexistent property
  ASSERT_DOUBLE_EQ(cell.getDynamicProperty("NonexistentProperty", 99.0), 99.0);
}

// Main function to run all tests
int main() {
  try {
    std::cout << "Running Cell-Field integration test suite..." << std::endl;

    // Create the test fixture
    CellTestFixture fixture;

    // Run all tests
    TestCellReadsFromFields(fixture);
    TestCellWritesToFields(fixture);
    TestCellUnitsConversion(fixture);
    TestCellMaterialOperations(fixture);
    TestBoundaryCellTypes(fixture);
    TestCellDiagnosticOutput(fixture);
    TestCellSerialization(fixture);
    TestCellNeighborAccess(fixture);
    TestDynamicProperties(fixture);

    std::cout << "All tests passed!" << std::endl;
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "Exception caught: " << e.what() << std::endl;
    return 1;
  } catch (...) {
    std::cerr << "Unknown exception caught!" << std::endl;
    return 1;
  }
}