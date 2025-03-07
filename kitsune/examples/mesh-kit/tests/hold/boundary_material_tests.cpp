#include "BoundaryFactory.h"
#include "BoundaryTestHelpers.h"
#include "Cell.h"
#include "CellTestHelpers.h"
#include "DirichletBoundary.h"
#include "InflowBoundary.h"
#include "Material.h"
#include "NeumannBoundary.h"
#include "NoSlipBoundary.h"
#include "SlipBoundary.h"
#include <gtest/gtest.h>

class BoundaryMaterialTest : public ::testing::Test {
protected:
  void SetUp() override {
    // Register boundary types
    BoundaryFactory::registerBoundaryTypes();

    // Create test materials
    m_water = Material::createPredefined("water");
    m_air = Material::createPredefined("air");

    // Create test cells with helper functions
    auto cells = createBoundaryTestCells(m_water, m_air);
    m_boundaryCell = std::move(cells.first);
    m_fluidCell = std::move(cells.second);

    // Create a mock grid for tests that need it
    m_mockGrid = std::make_unique<SimpleMockGrid>(0.01, 0.01);
  }

  std::shared_ptr<Material> m_water;
  std::shared_ptr<Material> m_air;
  std::unique_ptr<Cell> m_boundaryCell;
  std::unique_ptr<Cell> m_fluidCell;
  std::unique_ptr<SimpleMockGrid> m_mockGrid;
};

// Test DirichletBoundary with temperature-dependent material
TEST_F(BoundaryMaterialTest, DirichletBoundaryWithMaterial) {
  // Create Dirichlet boundary with fixed temperature
  auto dirichlet = std::make_shared<DirichletBoundary>("Test Dirichlet");
  dirichlet->setTemperature(350.0); // 76.85°C

  // Apply boundary to the cell
  m_boundaryCell->setBoundaryCondition(dirichlet);
  std::vector<Cell *> neighbors = {m_fluidCell.get()};

  // Apply the boundary condition
  dirichlet->apply(*m_boundaryCell, 0.0, 0.0, 0.0, &neighbors);

  // Check that temperature is set correctly
  EXPECT_DOUBLE_EQ(m_boundaryCell->getTemperature(), 350.0)
      << "Dirichlet boundary should set temperature";

  // Verify that material properties reflect the new temperature
  double waterDensity = m_water->getPropertyAtTemperature(
      Material::MaterialProperty::DENSITY, 350.0);
  EXPECT_DOUBLE_EQ(m_boundaryCell->getEffectiveDensity(), waterDensity)
      << "Material density should reflect the boundary temperature";

  // Test that viscosity has changed with temperature
  double refViscosity =
      m_water->getProperty(Material::MaterialProperty::DYNAMIC_VISCOSITY);
  double hotViscosity = m_water->getPropertyAtTemperature(
      Material::MaterialProperty::DYNAMIC_VISCOSITY, 350.0);
  EXPECT_LT(hotViscosity, refViscosity)
      << "Water viscosity should decrease with temperature";
}

// Test NoSlipBoundary with heat flux and material conductivity
TEST_F(BoundaryMaterialTest, NoSlipBoundaryWithHeatFlux) {
  // Create NoSlip boundary with heat flux
  auto noSlip = std::make_shared<NoSlipBoundary>("Test NoSlip");
  double heatFlux = 1000.0; // W/m²
  noSlip->setHeatFlux(heatFlux);

  // Set interior cell temperature
  m_fluidCell->setTemperature(293.15); // 20°C

  // Apply boundary to the cell
  m_boundaryCell->setBoundaryCondition(noSlip);
  std::vector<Cell *> neighbors = {m_fluidCell.get()};

  // Use the helper to patch the boundary cell with mock grid access
  patchCellWithMockGrid(*m_boundaryCell, m_mockGrid.get());

  // Apply the boundary condition
  noSlip->apply(*m_boundaryCell, 0.0, 0.0, 0.0, &neighbors);

  // Expected temperature change based on heat flux, distance, and material
  // conductivity
  double k =
      m_water->getProperty(Material::MaterialProperty::THERMAL_CONDUCTIVITY);
  double distance = 0.01; // 1cm grid spacing
  double expectedTemp = 293.15 + (heatFlux * distance / k);

  // Check that temperature is calculated correctly based on material
  // conductivity
  EXPECT_NEAR(m_boundaryCell->getTemperature(), expectedTemp, 1e-6)
      << "NoSlip boundary should set temperature based on heat flux and "
         "material conductivity";
}

// Test InflowBoundary with temperature-dependent material properties
TEST_F(BoundaryMaterialTest, InflowBoundaryWithMaterial) {
  // Create a custom temperature profile for inflow
  auto tempProfileFunc = [](double x, double y, double t) {
    return 293.15 + 10.0 * sin(t); // Oscillating temperature
  };

  // Create Inflow boundary with temperature profile
  auto inflow = std::make_shared<InflowBoundary>(10.0, 0.0);
  inflow->setTemperatureProfile(tempProfileFunc);

  // Apply boundary to the cell
  m_boundaryCell->setBoundaryCondition(inflow);
  std::vector<Cell *> neighbors = {m_fluidCell.get()};

  // Apply at t=0
  inflow->apply(*m_boundaryCell, 0.0, 0.0, 0.0, &neighbors);
  double temp0 = m_boundaryCell->getTemperature();
  double visc0 = m_water->getPropertyAtTemperature(
      Material::MaterialProperty::DYNAMIC_VISCOSITY, temp0);

  // Apply at t=π/2 (maximum temperature)
  inflow->updateTime(M_PI / 2);
  inflow->apply(*m_boundaryCell, 0.0, 0.0, 0.0, &neighbors);
  double tempMax = m_boundaryCell->getTemperature();
  double viscMax = m_water->getPropertyAtTemperature(
      Material::MaterialProperty::DYNAMIC_VISCOSITY, tempMax);

  // Check that temperature and viscosity vary correctly with time
  EXPECT_GT(tempMax, temp0) << "Temperature should increase with time";
  EXPECT_LT(viscMax, visc0)
      << "Viscosity should decrease as temperature increases";
}

// Test multiple materials with boundaries
TEST_F(BoundaryMaterialTest, MultipleMaterialsWithBoundaries) {
  // Create cells with different materials
  auto cell1 = std::make_unique<Cell>(Cell::CellType::BOUNDARY);
  auto cell2 = std::make_unique<Cell>(Cell::CellType::BOUNDARY);

  cell1->setMaterial(m_water);
  cell2->setMaterial(m_air);

  // Create a boundary condition
  auto dirichlet = std::make_shared<DirichletBoundary>("Test Dirichlet");
  dirichlet->setTemperature(350.0);

  // Apply to both cells
  cell1->setBoundaryCondition(dirichlet);
  cell2->setBoundaryCondition(dirichlet);

  std::vector<Cell *> neighbors = {m_fluidCell.get()};
  dirichlet->apply(*cell1, 0.0, 0.0, 0.0, &neighbors);
  dirichlet->apply(*cell2, 0.0, 0.0, 0.0, &neighbors);

  // Both cells should have the same temperature
  EXPECT_DOUBLE_EQ(cell1->getTemperature(), cell2->getTemperature())
      << "Both cells should have the same temperature from Dirichlet boundary";

  // But different material-dependent properties
  EXPECT_NE(cell1->getEffectiveDensity(), cell2->getEffectiveDensity())
      << "Cells with different materials should have different densities";

  // Water thermal conductivity should be higher than air
  double waterK = m_water->getPropertyAtTemperature(
      Material::MaterialProperty::THERMAL_CONDUCTIVITY, 350.0);
  double airK = m_air->getPropertyAtTemperature(
      Material::MaterialProperty::THERMAL_CONDUCTIVITY, 350.0);

  EXPECT_GT(waterK, airK)
      << "Water thermal conductivity should be higher than air";
}

// Test material serialization through boundaries
TEST_F(BoundaryMaterialTest, BoundaryMaterialSerialization) {
  // Create a Dirichlet boundary
  auto dirichlet = std::make_shared<DirichletBoundary>("Test Dirichlet");
  dirichlet->setTemperature(320.0);

  // Apply to cell with material
  m_boundaryCell->setBoundaryCondition(dirichlet);
  std::vector<Cell *> neighbors = {m_fluidCell.get()};
  dirichlet->apply(*m_boundaryCell, 0.0, 0.0, 0.0, &neighbors);

  // Serialize cell
  std::string serialized = m_boundaryCell->serialize();

  // Create a new cell and deserialize
  Cell newCell;
  bool success = newCell.deserialize(serialized);

  EXPECT_TRUE(success) << "Deserialization should succeed";

  // Verify boundary condition type can be retrieved
  EXPECT_TRUE(newCell.hasBoundaryCondition())
      << "Deserialized cell should have boundary condition";

  // In a real implementation, boundary conditions would be deserialized
  // For this test, just verify that boundary type is included in serialization
  EXPECT_NE(serialized.find("TEMP="), std::string::npos)
      << "Serialized data should include temperature";
  EXPECT_NE(serialized.find("MAT="), std::string::npos)
      << "Serialized data should include material";
}

// Test factory creation with materials
TEST_F(BoundaryMaterialTest, BoundaryFactoryWithMaterials) {
  // Create a boundary through the factory
  auto boundary =
      BoundaryFactory::createBoundary("Dirichlet", "Factory Boundary");
  EXPECT_NE(boundary, nullptr) << "Factory should create boundary";

  // Apply to cell with material
  m_boundaryCell->setBoundaryCondition(boundary);

  // If it's a Dirichlet boundary, set temperature
  auto dirichlet = std::dynamic_pointer_cast<DirichletBoundary>(boundary);
  if (dirichlet) {
    dirichlet->setTemperature(400.0);
  }

  // Apply the boundary condition
  std::vector<Cell *> neighbors = {m_fluidCell.get()};
  boundary->apply(*m_boundaryCell, 0.0, 0.0, 0.0, &neighbors);

  // Verify temperature was set and material properties reflect this
  EXPECT_EQ(m_boundaryCell->getTemperature(), 400.0)
      << "Factory-created boundary should set temperature";

  // Test that created boundary interacts correctly with material
  double hotWaterDensity = m_water->getPropertyAtTemperature(
      Material::MaterialProperty::DENSITY, 400.0);
  EXPECT_NEAR(m_boundaryCell->getEffectiveDensity(), hotWaterDensity, 1e-6)
      << "Material density should reflect the boundary temperature";
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}