/**
 * ====================================================================
 * Material Move/Copy Semantics Tests
 * ====================================================================
 *
 * These tests verify the correct behavior of Material class instances
 * when they are copied or moved, ensuring proper:
 *
 * 1. Copy Construction
 *    - Properties are properly duplicated
 *    - IDs are unique
 *    - Temperature models are preserved
 *    - Custom functions are correctly copied
 *
 * 2. Move Construction
 *    - Properties are transferred correctly
 *    - IDs are preserved
 *    - Registry entries are updated
 *
 * 3. Copy Assignment
 *    - Properties are properly copied
 *    - Original objects remain valid
 *
 * 4. Move Assignment
 *    - Properties are transferred correctly
 *    - Source objects are properly cleared
 *
 * 5. Registry Behavior
 *    - Registry entries are properly updated during copy/move
 *    - Lookup by ID works correctly
 *    - Entries are removed when materials are destroyed
 */

#include "Material.h"
#include "Units.h"
#include <gtest/gtest.h>
#include <memory>

// Test fixture for move/copy semantics tests
class MaterialMoveAndCopyTest : public ::testing::Test {
protected:
  void SetUp() override {
    // Create basic materials for testing
    m_water = Material::createPredefined("water");
    m_oil = Material::createPredefined("oil");

    // Create a custom material with specific properties
    m_custom = std::make_shared<Material>(Material::MaterialType::FLUID,
                                          "CustomMaterial");
    m_custom->setProperty(Material::MaterialProperty::DENSITY, 850.0);
    m_custom->setProperty(Material::MaterialProperty::DYNAMIC_VISCOSITY, 0.05);
    m_custom->setProperty(Material::MaterialProperty::THERMAL_CONDUCTIVITY,
                          0.12);
    m_custom->setProperty(Material::MaterialProperty::SPECIFIC_HEAT, 2000.0);

    // Set up temperature dependence
    m_custom->setPropertyModel(Material::MaterialProperty::DYNAMIC_VISCOSITY,
                               Material::PropertyModel::EXPONENTIAL,
                               {-0.015} // Viscosity decreases with temperature
    );
    m_custom->setUseTempDependentProps(true);
  }

  std::shared_ptr<Material> m_water;
  std::shared_ptr<Material> m_oil;
  std::shared_ptr<Material> m_custom;
};

// Test copy constructor
TEST_F(MaterialMoveAndCopyTest, CopyConstructor) {
  // Create a copy of the custom material
  Material copiedMaterial(*m_custom);

  // Check that IDs are different
  EXPECT_NE(copiedMaterial.getID(), m_custom->getID())
      << "Copied material should have a different ID";

  // Check that properties are the same
  EXPECT_DOUBLE_EQ(
      copiedMaterial.getProperty(Material::MaterialProperty::DENSITY),
      m_custom->getProperty(Material::MaterialProperty::DENSITY))
      << "Density should be copied correctly";

  EXPECT_DOUBLE_EQ(
      copiedMaterial.getProperty(Material::MaterialProperty::DYNAMIC_VISCOSITY),
      m_custom->getProperty(Material::MaterialProperty::DYNAMIC_VISCOSITY))
      << "Viscosity should be copied correctly";

  // Check temperature dependence was copied
  EXPECT_TRUE(copiedMaterial.isUsingTempDependentProps())
      << "Temperature dependence flag should be copied";

  // Check property model behavior
  double origViscAtTemp = m_custom->getPropertyAtTemperature(
      Material::MaterialProperty::DYNAMIC_VISCOSITY, 323.15);
  double copyViscAtTemp = copiedMaterial.getPropertyAtTemperature(
      Material::MaterialProperty::DYNAMIC_VISCOSITY, 323.15);

  EXPECT_DOUBLE_EQ(origViscAtTemp, copyViscAtTemp)
      << "Temperature-dependent property calculation should be identical";
}

// Test move constructor
TEST_F(MaterialMoveAndCopyTest, MoveConstructor) {
  // Create a material to move from
  Material originalMaterial(Material::MaterialType::FLUID, "OriginalMaterial");
  originalMaterial.setProperty(Material::MaterialProperty::DENSITY, 1200.0);
  originalMaterial.setProperty(Material::MaterialProperty::SPECIFIC_HEAT,
                               1500.0);

  // Save the original ID and properties
  uint32_t originalID = originalMaterial.getID();
  double originalDensity =
      originalMaterial.getProperty(Material::MaterialProperty::DENSITY);

  // Move construct a new material
  Material movedMaterial(std::move(originalMaterial));

  // Check that ID was transferred
  EXPECT_EQ(movedMaterial.getID(), originalID)
      << "Moved material should have the same ID as the original";

  // Check that properties were moved
  EXPECT_DOUBLE_EQ(
      movedMaterial.getProperty(Material::MaterialProperty::DENSITY),
      originalDensity)
      << "Density should be moved correctly";

  // Check that original material's ID was cleared
  EXPECT_EQ(originalMaterial.getID(), 0)
      << "Original material ID should be cleared after move";
}

// Test copy assignment operator
TEST_F(MaterialMoveAndCopyTest, CopyAssignmentOperator) {
  // Create a material to copy to
  Material targetMaterial(Material::MaterialType::FLUID, "TargetMaterial");
  uint32_t targetID = targetMaterial.getID();

  // Perform copy assignment
  targetMaterial = *m_custom;

  // Check that properties are copied but ID remains different
  EXPECT_NE(targetMaterial.getID(), m_custom->getID())
      << "Copy assignment should assign a new ID";
  EXPECT_NE(targetMaterial.getID(), targetID)
      << "Copy assignment should change the ID";

  // Check that properties match
  EXPECT_DOUBLE_EQ(
      targetMaterial.getProperty(Material::MaterialProperty::DENSITY),
      m_custom->getProperty(Material::MaterialProperty::DENSITY))
      << "Density should be copied correctly";

  EXPECT_DOUBLE_EQ(
      targetMaterial.getProperty(
          Material::MaterialProperty::THERMAL_CONDUCTIVITY),
      m_custom->getProperty(Material::MaterialProperty::THERMAL_CONDUCTIVITY))
      << "Thermal conductivity should be copied correctly";

  // Check that name is copied
  EXPECT_EQ(targetMaterial.getName(), m_custom->getName())
      << "Name should be copied correctly";
}

// Test move assignment operator
TEST_F(MaterialMoveAndCopyTest, MoveAssignmentOperator) {
  // Create materials for move assignment
  Material sourceMaterial(Material::MaterialType::FLUID, "SourceMaterial");
  sourceMaterial.setProperty(Material::MaterialProperty::DENSITY, 950.0);
  sourceMaterial.setProperty(Material::MaterialProperty::DYNAMIC_VISCOSITY,
                             0.025);
  uint32_t sourceID = sourceMaterial.getID();

  Material targetMaterial(Material::MaterialType::FLUID, "TargetMaterial");
  uint32_t targetOriginalID = targetMaterial.getID();

  // Perform move assignment
  targetMaterial = std::move(sourceMaterial);

  // Check that ID was transferred
  EXPECT_EQ(targetMaterial.getID(), sourceID)
      << "Target should have the source's ID after move assignment";

  // Check that properties were moved
  EXPECT_DOUBLE_EQ(
      targetMaterial.getProperty(Material::MaterialProperty::DENSITY), 950.0)
      << "Density should be moved correctly";

  EXPECT_DOUBLE_EQ(
      targetMaterial.getProperty(Material::MaterialProperty::DYNAMIC_VISCOSITY),
      0.025)
      << "Viscosity should be moved correctly";

  // Check that source material was cleared
  EXPECT_EQ(sourceMaterial.getID(), 0)
      << "Source material ID should be cleared after move";

  // Original target material ID should be unregistered
  EXPECT_EQ(Material::getByID(targetOriginalID), nullptr)
      << "Original target ID should be unregistered";
}

// Test registry behavior during copy/move operations
TEST_F(MaterialMoveAndCopyTest, RegistryBehavior) {
  // Create a material
  auto originalMaterial = std::make_shared<Material>(
      Material::MaterialType::FLUID, "OriginalMaterial");
  uint32_t originalID = originalMaterial->getID();

  // Check that it's in the registry
  EXPECT_EQ(Material::getByID(originalID), originalMaterial.get())
      << "Original material should be in the registry";

  // Create a copy
  auto copiedMaterial = std::make_shared<Material>(*originalMaterial);
  uint32_t copiedID = copiedMaterial->getID();

  // Check that both are in the registry with different IDs
  EXPECT_NE(originalID, copiedID)
      << "Copied material should have a different ID";
  EXPECT_EQ(Material::getByID(originalID), originalMaterial.get())
      << "Original material should still be in the registry";
  EXPECT_EQ(Material::getByID(copiedID), copiedMaterial.get())
      << "Copied material should be in the registry";

  // Create a moved-to material
  auto movedMaterial = std::make_shared<Material>(std::move(*originalMaterial));

  // Check that the moved-to material has the original's ID
  EXPECT_EQ(movedMaterial->getID(), originalID)
      << "Moved material should keep the original ID";
  EXPECT_EQ(Material::getByID(originalID), movedMaterial.get())
      << "Registry should now point to the moved-to material";

  // Destroy materials and check registry cleanup
  copiedMaterial.reset();
  EXPECT_EQ(Material::getByID(copiedID), nullptr)
      << "Registry should remove the entry when material is destroyed";

  movedMaterial.reset();
  EXPECT_EQ(Material::getByID(originalID), nullptr)
      << "Registry should remove the entry when moved-to material is destroyed";
}

// Test copying of custom property functions
TEST_F(MaterialMoveAndCopyTest, CustomFunctionCopy) {
  // Create a material with a custom property function
  auto originalMaterial = std::make_shared<Material>(
      Material::MaterialType::FLUID, "CustomFunctionMaterial");

  // Define and set a custom density function based on temperature
  originalMaterial->setProperty(Material::MaterialProperty::DENSITY, 1000.0);
  originalMaterial->setCustomPropertyFunction(
      Material::MaterialProperty::DENSITY, [](double T) -> double {
        double T_C = T - 273.15; // Convert to Celsius
        // Parabolic density curve with peak at 4°C (similar to water)
        return 1000.0 - 0.1 * (T_C - 4.0) * (T_C - 4.0);
      });
  originalMaterial->setUseTempDependentProps(true);

  // Create a copy
  auto copiedMaterial = std::make_shared<Material>(*originalMaterial);

  // Check that the custom function behavior is preserved
  double originalAt0C = originalMaterial->getPropertyAtTemperature(
      Material::MaterialProperty::DENSITY, 273.15); // 0°C
  double copiedAt0C = copiedMaterial->getPropertyAtTemperature(
      Material::MaterialProperty::DENSITY, 273.15); // 0°C

  EXPECT_DOUBLE_EQ(originalAt0C, copiedAt0C)
      << "Custom function behavior should be preserved at 0°C";

  double originalAt4C = originalMaterial->getPropertyAtTemperature(
      Material::MaterialProperty::DENSITY, 277.15); // 4°C
  double copiedAt4C = copiedMaterial->getPropertyAtTemperature(
      Material::MaterialProperty::DENSITY, 277.15); // 4°C

  EXPECT_DOUBLE_EQ(originalAt4C, copiedAt4C)
      << "Custom function behavior should be preserved at 4°C";

  // Check that the function has the expected behavior
  EXPECT_GT(copiedAt4C, copiedAt0C)
      << "Density at 4°C should be higher than at 0°C (Water-like behavior)";
}

// Test copying of a material with mixture components
TEST_F(MaterialMoveAndCopyTest, MixtureCopy) {
  // Create a mixture
  auto mixture = m_water->createMixture(m_oil, 0.3); // 70% water, 30% oil

  // Get original properties and components
  double mixtureDensity =
      mixture->getProperty(Material::MaterialProperty::DENSITY);
  auto originalComponents = mixture->getMixtureComponents();
  ASSERT_EQ(originalComponents.size(), 2) << "Mixture should have 2 components";

  // Copy the mixture
  auto copiedMixture = std::make_shared<Material>(*mixture);

  // Check that properties match
  EXPECT_DOUBLE_EQ(
      copiedMixture->getProperty(Material::MaterialProperty::DENSITY),
      mixtureDensity)
      << "Density should be copied correctly";

  // Check that components were copied correctly
  auto copiedComponents = copiedMixture->getMixtureComponents();
  ASSERT_EQ(copiedComponents.size(), 2)
      << "Copied mixture should have 2 components";

  // Components should reference the same original materials
  EXPECT_EQ(copiedComponents[0].first->getID(),
            originalComponents[0].first->getID())
      << "First component should reference the same material";
  EXPECT_EQ(copiedComponents[1].first->getID(),
            originalComponents[1].first->getID())
      << "Second component should reference the same material";

  // Fractions should be the same
  EXPECT_DOUBLE_EQ(copiedComponents[0].second, originalComponents[0].second)
      << "First component fraction should be copied correctly";
  EXPECT_DOUBLE_EQ(copiedComponents[1].second, originalComponents[1].second)
      << "Second component fraction should be copied correctly";
}
