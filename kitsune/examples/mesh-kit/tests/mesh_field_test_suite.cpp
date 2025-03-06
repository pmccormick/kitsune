/**
 * @file mesh_field_test_suite.cpp
 * @brief Tests for Mesh class using Field-based storage
 *
 * This test suite focuses on the Mesh class and its management of Fields
 * for storing Cell data.
 */

#include "BoundaryFactory.h"
#include "Cell.h"
#include "DirichletBoundary.h"
#include "Field.h"
#include "InflowBoundary.h"
#include "Material.h"
#include "Mesh.h"
#include "NeumannBoundary.h"
#include "SlipBoundary.h"

#include <cassert>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>

// Simple test framework
#define TEST(name) void name(std::shared_ptr<Mesh> mesh)
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

// Utility function to create a test mesh
std::shared_ptr<Mesh> createTestMesh() {
  // Create a 5x4 mesh
  auto mesh = std::make_shared<Mesh>(5, 4, 1.0, 1.0);

  // Create a default material
  auto defaultMaterial =
      std::make_shared<Material>(Material::MaterialType::FLUID, "TestMaterial");
  defaultMaterial->setProperty(Material::MaterialProperty::DENSITY, 1.0);
  defaultMaterial->setProperty(Material::MaterialProperty::DYNAMIC_VISCOSITY,
                               0.01);
  defaultMaterial->setProperty(Material::MaterialProperty::THERMAL_CONDUCTIVITY,
                               0.5);
  defaultMaterial->setProperty(Material::MaterialProperty::SPECIFIC_HEAT,
                               1000.0);

  // Initialize the mesh with the material
  mesh->initialize(defaultMaterial);

  return mesh;
}

// Test mesh construction and dimensions
TEST(TestMeshConstruction) {
  // Verify dimensions
  size_t nx = 5, ny = 4;
  ASSERT_EQ(mesh->getNx(), nx);
  ASSERT_EQ(mesh->getNy(), ny);

  // Verify cell type initialization (boundaries and internal cells)
  // Boundaries should be set automatically
  for (size_t j = 0; j < ny; ++j) {
    for (size_t i = 0; i < nx; ++i) {
      if (i == 0 || i == nx - 1 || j == 0 || j == ny - 1) {
        // Boundary cells
        ASSERT_EQ(mesh->getCellTypeField()(i, j), Cell::CellType::BOUNDARY);
        ASSERT_TRUE(mesh->getBoundaryFlagField()(i, j));
      } else {
        // Internal cells
        ASSERT_EQ(mesh->getCellTypeField()(i, j), Cell::CellType::FLUID);
        ASSERT_FALSE(mesh->getBoundaryFlagField()(i, j));
      }
    }
  }

  // Verify field sizes match the mesh dimensions
  ASSERT_EQ(mesh->getTemperatureField().nx(), nx);
  ASSERT_EQ(mesh->getTemperatureField().ny(), ny);
  ASSERT_EQ(mesh->getPressureField().nx(), nx);
  ASSERT_EQ(mesh->getPressureField().ny(), ny);

  // Test physical coordinates match expected values
  double dx = 1.0 / (nx - 1);
  double dy = 1.0 / (ny - 1);

  // Check some sample points
  ASSERT_DOUBLE_EQ(mesh->physicalX(0), 0.0);
  ASSERT_DOUBLE_EQ(mesh->physicalX(nx - 1), 1.0);
  ASSERT_DOUBLE_EQ(mesh->physicalY(0), 0.0);
  ASSERT_DOUBLE_EQ(mesh->physicalY(ny - 1), 1.0);

  // Check midpoints
  ASSERT_DOUBLE_EQ(mesh->physicalX(nx / 2), 0.5);
  ASSERT_DOUBLE_EQ(mesh->physicalY(ny / 2), 0.5);
}

// Test setting and getting cell types
TEST(TestCellTypeOperations) {
  // Set some cells as obstacles
  mesh->setCellAsObstacle(1, 1, nullptr);
  mesh->setCellAsObstacle(2, 2, nullptr);

  // Verify the cells are marked correctly
  ASSERT_EQ(mesh->getCellTypeField()(1, 1), Cell::CellType::SOLID);
  ASSERT_TRUE(mesh->getObstacleFlagField()(1, 1));
  ASSERT_EQ(mesh->getCellTypeField()(2, 2), Cell::CellType::SOLID);
  ASSERT_TRUE(mesh->getObstacleFlagField()(2, 2));

  // Check accessing through the cell interface works too
  Cell &cell1 = mesh->getCell(1, 1);
  Cell &cell2 = mesh->getCell(2, 2);
  ASSERT_EQ(cell1.getType(), Cell::CellType::SOLID);
  ASSERT_TRUE(cell1.isObstacle());
  ASSERT_EQ(cell2.getType(), Cell::CellType::SOLID);
  ASSERT_TRUE(cell2.isObstacle());

  // Convert obstacle back to fluid
  mesh->setCellAsFluid(1, 1, nullptr);

  // Verify the cell is now fluid
  ASSERT_EQ(mesh->getCellTypeField()(1, 1), Cell::CellType::FLUID);
  ASSERT_FALSE(mesh->getObstacleFlagField()(1, 1));
  ASSERT_EQ(cell1.getType(), Cell::CellType::FLUID);
  ASSERT_FALSE(cell1.isObstacle());
}

// Test boundary condition operations
TEST(TestBoundaryConditions) {
  // Create some boundary conditions
  auto inflowBC = std::make_shared<InflowBoundary>(1.0, 0.0);
  auto outflowBC = std::make_shared<NeumannBoundary>();

  // Set on specific boundaries
  mesh->setBoundaryCondition(Mesh::BoundaryLocation::LEFT, inflowBC);
  mesh->setBoundaryCondition(Mesh::BoundaryLocation::RIGHT, outflowBC);

  // Verify the boundary conditions are set correctly
  for (size_t j = 0; j < mesh->getNy(); ++j) {
    // Left boundary
    Cell &leftCell = mesh->getCell(0, j);
    ASSERT_EQ(leftCell.getBoundaryCondition(), inflowBC);

    // Right boundary
    Cell &rightCell = mesh->getCell(mesh->getNx() - 1, j);
    ASSERT_EQ(rightCell.getBoundaryCondition(), outflowBC);
  }

  // Set a boundary condition in a custom region
  auto slipBC = std::make_shared<SlipBoundary>();
  mesh->setBoundaryConditionRegion(1, 3, 0, 0, slipBC);

  // Verify the custom region
  for (size_t i = 1; i <= 3; ++i) {
    Cell &bottomCell = mesh->getCell(i, 0);
    ASSERT_EQ(bottomCell.getBoundaryCondition(), slipBC);
  }
}

// Test material handling
TEST(TestMaterialHandling) {
  // Create a different material
  auto newMaterial =
      std::make_shared<Material>(Material::MaterialType::FLUID, "NewMaterial");
  newMaterial->setProperty(Material::MaterialProperty::DENSITY, 2.0);

  // Set the material for some cells
  mesh->setCellMaterial(1, 1, newMaterial);
  mesh->setCellMaterial(2, 2, newMaterial);

  // Verify the material is set correctly through cell access
  Cell &cell1 = mesh->getCell(1, 1);
  Cell &cell2 = mesh->getCell(2, 2);
  ASSERT_EQ(cell1.getMaterial(), newMaterial);
  ASSERT_EQ(cell2.getMaterial(), newMaterial);

  // Create a mixture and set it
  auto material1 = cell1.getMaterial();
  auto material2 =
      std::make_shared<Material>(Material::MaterialType::FLUID, "Material2");
  material2->setProperty(Material::MaterialProperty::DENSITY, 3.0);

  auto mixedMaterial = material1->createMixture(material2, 0.5);
  mesh->setCellMaterial(3, 3, mixedMaterial);

  // Verify the mixed material is set correctly
  Cell &cell3 = mesh->getCell(3, 3);
  ASSERT_EQ(cell3.getMaterial(), mixedMaterial);

  // Check the density reflects the mixture (should be between 2.0 and 3.0)
  double mixedDensity =
      cell3.getMaterial()->getProperty(Material::MaterialProperty::DENSITY);
  ASSERT_TRUE(mixedDensity > 2.0 && mixedDensity < 3.0);
}

// Test field data extraction
TEST(TestFieldDataExtraction) {
  // Set some predictable values in the fields
  for (size_t j = 0; j < mesh->getNy(); ++j) {
    for (size_t i = 0; i < mesh->getNx(); ++i) {
      mesh->getTemperatureField()(i, j) = 300.0 + i + j;
      mesh->getPressureField()(i, j) = 101325.0 + 100.0 * (i + j);
      mesh->getVelocityUField()(i, j) = 0.1 * i;
      mesh->getVelocityVField()(i, j) = 0.1 * j;
    }
  }

  // Extract temperature field
  std::vector<double> temperatureField = mesh->getTemperatureField();

  // Verify extracted field has the correct values
  for (size_t j = 0; j < mesh->getNy(); ++j) {
    for (size_t i = 0; i < mesh->getNx(); ++i) {
      size_t idx = i + j * mesh->getNx();
      ASSERT_DOUBLE_EQ(temperatureField[idx], 300.0 + i + j);
    }
  }

  // Extract velocity field components
  std::vector<double> velocityU, velocityV;
  mesh->getVelocityField(velocityU, velocityV);

  // Verify velocity components
  for (size_t j = 0; j < mesh->getNy(); ++j) {
    for (size_t i = 0; i < mesh->getNx(); ++i) {
      size_t idx = i + j * mesh->getNx();
      ASSERT_DOUBLE_EQ(velocityU[idx], 0.1 * i);
      ASSERT_DOUBLE_EQ(velocityV[idx], 0.1 * j);
    }
  }
}

// Test cell neighborhood and connectivity
TEST(TestCellNeighborhood) {
  // Set up an interior cell to test neighborhood
  size_t i = 2, j = 2;
  Cell &cell = mesh->getCell(i, j);

  // Get its neighbors
  auto neighbors = mesh->getNeighbors(i, j);

  // Verify neighbor types - should be FLUID since they're interior cells
  ASSERT_TRUE(neighbors[0] != nullptr); // North
  ASSERT_EQ(neighbors[0]->getType(), Cell::CellType::FLUID);

  ASSERT_TRUE(neighbors[1] != nullptr); // East
  ASSERT_EQ(neighbors[1]->getType(), Cell::CellType::FLUID);

  ASSERT_TRUE(neighbors[2] != nullptr); // South
  ASSERT_EQ(neighbors[2]->getType(), Cell::CellType::FLUID);

  ASSERT_TRUE(neighbors[3] != nullptr); // West
  ASSERT_EQ(neighbors[3]->getType(), Cell::CellType::FLUID);

  // Check a corner cell (0,0) which should have some null neighbors
  auto cornerNeighbors = mesh->getNeighbors(0, 0);

  // South and West should be null
  ASSERT_TRUE(cornerNeighbors[2] == nullptr); // South
  ASSERT_TRUE(cornerNeighbors[3] == nullptr); // West

  // North and East should exist and be boundaries
  ASSERT_TRUE(cornerNeighbors[0] != nullptr); // North
  ASSERT_TRUE(cornerNeighbors[1] != nullptr); // East
}

// Test cell cache consistency
TEST(TestCellCacheConsistency) {
  // Get references to the same cell twice
  Cell &cell1 = mesh->getCell(2, 2);
  Cell &cell2 = mesh->getCell(2, 2);

  // They should be the same object
  ASSERT_TRUE(&cell1 == &cell2);

  // Changes to one should affect the other
  cell1.setTemperature(350.0);
  ASSERT_DOUBLE_EQ(cell2.getTemperature(), 350.0);

  // Both should access the same field data
  ASSERT_DOUBLE_EQ(mesh->getTemperatureField()(2, 2), 350.0);
}

// Test unit conversion operations
TEST(TestUnitConversions) {
  // Set values in SI units
  mesh->getTemperatureField()(2, 2) = 300.0; // K
  mesh->getPressureField()(2, 2) = 101325.0; // Pa

  // Get cell to test unit conversions
  Cell &cell = mesh->getCell(2, 2);

  // Check temperature conversions
  double tempC = cell.getTemperatureWithUnits("C");
  ASSERT_DOUBLE_EQ(tempC, 300.0 - 273.15);

  double tempF = cell.getTemperatureWithUnits("F");
  ASSERT_DOUBLE_EQ(tempF, (300.0 - 273.15) * 9.0 / 5.0 + 32.0);

  // Check pressure conversions
  double pressureBar = cell.getPressureWithUnits("bar");
  ASSERT_DOUBLE_EQ(pressureBar, 101325.0 / 100000.0);

  double pressureAtm = cell.getPressureWithUnits("atm");
  ASSERT_DOUBLE_EQ(pressureAtm, 101325.0 / 101325.0);

  // Set using unit conversions
  cell.setTemperatureWithUnits(25.0, "C");
  ASSERT_DOUBLE_EQ(mesh->getTemperatureField()(2, 2), 25.0 + 273.15);

  cell.setPressureWithUnits(2.0, "bar");
  ASSERT_DOUBLE_EQ(mesh->getPressureField()(2, 2), 2.0 * 100000.0);
}

// Main function to run all tests
int main() {
  try {
    std::cout << "Running Mesh-Field integration test suite..." << std::endl;

    // Create test mesh
    auto mesh = createTestMesh();

    // Run all tests
    TestMeshConstruction(mesh);
    TestCellTypeOperations(mesh);
    TestBoundaryConditions(mesh);
    TestMaterialHandling(mesh);
    TestFieldDataExtraction(mesh);
    TestCellNeighborhood(mesh);
    TestCellCacheConsistency(mesh);
    TestUnitConversions(mesh);

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