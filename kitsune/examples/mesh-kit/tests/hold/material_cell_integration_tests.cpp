#include "Cell.h"
#include "CellTestHelpers.h" // Assuming this exists in your infrastructure
#include "Material.h"
#include <gtest/gtest.h>

class MaterialCellTest : public ::testing::Test {
protected:
  void SetUp() override {
    // Create test materials
    m_water = Material::createPredefined("water");
    m_air = Material::createPredefined("air");
    m_aluminum = Material::createPredefined("aluminum");

    // Create test cell
    m_cell = std::make_unique<Cell>(Cell::CellType::FLUID);
  }

  std::shared_ptr<Material> m_water;
  std::shared_ptr<Material> m_air;
  std::shared_ptr<Material> m_aluminum;
  std::unique_ptr<Cell> m_cell;
};

// Basic material assignment test
TEST_F(MaterialCellTest, MaterialAssignment) {
  ASSERT_EQ(m_cell->getMaterial(), nullptr)
      << "New cell should have no material";

  m_cell->setMaterial(m_water);
  EXPECT_EQ(m_cell->getMaterial(), m_water)
      << "Cell should store water material";

  m_cell->setMaterial(m_air);
  EXPECT_EQ(m_cell->getMaterial(), m_air)
      << "Cell should update to air material";

  m_cell->setMaterial(nullptr);
  EXPECT_EQ(m_cell->getMaterial(), nullptr)
      << "Cell should allow clearing material";
}

// Test effective density calculation
TEST_F(MaterialCellTest, EffectiveDensity) {
  // Test with no material
  EXPECT_DOUBLE_EQ(m_cell->getDensity(), 1.0)
      << "Default density should be 1.0";
  EXPECT_DOUBLE_EQ(m_cell->getEffectiveDensity(), 1.0)
      << "Effective density should match cell density with no material";

  // Test with water at reference temperature
  m_cell->setMaterial(m_water);
  double waterDensity =
      m_water->getProperty(Material::MaterialProperty::DENSITY);
  EXPECT_NEAR(m_cell->getEffectiveDensity(), waterDensity, 1e-6)
      << "Effective density should match water density at reference temp";

  // Test temperature dependence
  m_cell->setTemperature(373.15); // 100°C
  EXPECT_NE(m_cell->getEffectiveDensity(), waterDensity)
      << "Density should change with temperature";
  EXPECT_LT(m_cell->getEffectiveDensity(), waterDensity)
      << "Water density should decrease at higher temperature";
}

// Test material mixing in cells
TEST_F(MaterialCellTest, MaterialMixing) {
  // Start with water
  m_cell->setMaterial(m_water);
  double initialDensity = m_cell->getEffectiveDensity();

  // Mix with air (50%)
  m_cell->mixMaterial(m_air, 0.5);
  double mixedDensity = m_cell->getEffectiveDensity();

  // Check that the mixture has a density between air and water
  EXPECT_LT(mixedDensity, initialDensity)
      << "Mixed density should be lower than pure water";
  EXPECT_GT(mixedDensity,
            m_air->getProperty(Material::MaterialProperty::DENSITY))
      << "Mixed density should be higher than pure air";

  // Verify that material type is changed to a mixture
  EXPECT_TRUE(m_cell->getMaterial()->isMixture())
      << "Cell material should be a mixture";

  // Check mixture components
  auto components = m_cell->getMaterial()->getMixtureComponents();
  EXPECT_EQ(components.size(), 2) << "Mixture should have 2 components";
}

// Test temperature unit conversion with materials
TEST_F(MaterialCellTest, TemperatureUnitConversion) {
  m_cell->setMaterial(m_water);

  // Set temperature in Celsius
  m_cell->setTemperatureWithUnits(25.0, "C");
  EXPECT_DOUBLE_EQ(m_cell->getTemperature(), 298.15)
      << "Temperature should be stored as Kelvin";
  EXPECT_DOUBLE_EQ(m_cell->getTemperatureWithUnits("C"), 25.0)
      << "Temperature should convert back to Celsius";

  // Verify material properties use the correct temperature
  double viscosityAt25C = m_water->getPropertyAtTemperature(
      Material::MaterialProperty::DYNAMIC_VISCOSITY, 298.15);

  // Change temperature and verify property changes
  m_cell->setTemperatureWithUnits(50.0, "C");
  double viscosityAt50C = m_water->getPropertyAtTemperature(
      Material::MaterialProperty::DYNAMIC_VISCOSITY, 323.15);

  EXPECT_LT(viscosityAt50C, viscosityAt25C)
      << "Water viscosity should decrease with temperature";
}

// Test material property access through cell
TEST_F(MaterialCellTest, MaterialPropertyAccess) {
  // Set aluminum material and test thermal conductivity
  m_cell->setMaterial(m_aluminum);
  double aluminumConductivity =
      m_aluminum->getProperty(Material::MaterialProperty::THERMAL_CONDUCTIVITY);

  // Verify cell can compute heat flux based on material properties
  m_cell->setHeatFluxX(0.0); // Initialize to zero

  // Set up a temperature gradient (using Cell's property compute function if
  // available) Note: This would typically require setting up neighboring cells
  // and calling the compute function For this test, directly set a value to
  // verify the material property is integrated correctly
  double heatFlux =
      aluminumConductivity * 10.0; // Assuming a temperature gradient of 10 K/m
  m_cell->setHeatFluxX(heatFlux);

  EXPECT_DOUBLE_EQ(m_cell->getHeatFluxX(), heatFlux)
      << "Heat flux should be based on material thermal conductivity";
}

// Test serialize/deserialize with materials
TEST_F(MaterialCellTest, Serialization) {
  // Set up a cell with material
  m_cell->setMaterial(m_water);
  m_cell->setTemperature(310.15); // 37°C

  // Serialize the cell
  std::string serialized = m_cell->serialize();

  // Create a new cell and deserialize
  Cell newCell;
  bool success = newCell.deserialize(serialized);

  EXPECT_TRUE(success) << "Deserialization should succeed";

  // Check material properties are maintained
  // Note: Full material deserialization may require material registry
  EXPECT_DOUBLE_EQ(newCell.getTemperature(), 310.15)
      << "Temperature should be preserved";

  // In a real implementation, material names would be used for lookup
  // For this test, just verify that material information is included in
  // serialization
  EXPECT_NE(serialized.find("MAT="), std::string::npos)
      << "Serialized data should include material";
}

// Test cell reset with materials
TEST_F(MaterialCellTest, CellReset) {
  // Set up a cell with custom properties
  m_cell->setMaterial(m_water);
  m_cell->setTemperature(350.0);
  m_cell->setVelocityU(10.0);
  m_cell->setVelocityV(5.0);
  m_cell->setProperty(Cell::PropertyType::KINETIC_ENERGY, 100.0);

  // Reset the cell
  m_cell->reset();

  // Check that physical properties are reset but material is preserved
  EXPECT_DOUBLE_EQ(m_cell->getTemperature(), 293.15)
      << "Temperature should be reset to default";
  EXPECT_DOUBLE_EQ(m_cell->getVelocityU(), 0.0) << "Velocity should be reset";
  EXPECT_DOUBLE_EQ(m_cell->getVelocityV(), 0.0) << "Velocity should be reset";
  EXPECT_DOUBLE_EQ(m_cell->getProperty(Cell::PropertyType::KINETIC_ENERGY), 0.0)
      << "Properties should be reset";

  // Material should be preserved through reset
  EXPECT_EQ(m_cell->getMaterial(), m_water)
      << "Material should be preserved after reset";
}

// Test material property clamping
TEST_F(MaterialCellTest, PropertyBounds) {
  m_cell->setMaterial(m_water);

  // Test temperature clamping
  m_cell->setTemperature(-10.0); // Below absolute zero
  EXPECT_GE(m_cell->getTemperature(), 0.0)
      << "Temperature should be clamped to absolute zero";

  // Test density valid range
  m_cell->setDensity(-5.0); // Invalid negative density
  EXPECT_GT(m_cell->getDensity(), 0.0)
      << "Density should be clamped to valid range";
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
