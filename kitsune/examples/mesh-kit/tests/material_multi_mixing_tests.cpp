/**
 * ====================================================================
 * Material Multi-Component Mixing Tests
 * ====================================================================
 *
 * These tests verify the functionality of multi-component material
 * mixing capabilities in the Material class.
 *
 * Test Coverage:
 * --------------
 * 1. Multi-Component Mixing:
 *    - Three or more component mixtures
 *    - Verification of component fractions and properties
 *
 * 2. Complex Mixing Scenarios:
 *    - Nested mixtures (mixing mixtures with other materials)
 *    - Mixtures with zero or near-zero fractions
 *    - Mixtures of different material types (fluids and solids)
 *
 * 3. Special Cases:
 *    - Temperature dependence in multi-component mixtures
 *    - Default mixing rules for various properties
 *    - Error handling and boundary conditions
 *    - Maximum component handling
 */

#include "Material.h"
#include "Units.h"
#include <cmath>
#include <gtest/gtest.h>
#include <memory>
#include <numeric>

// Test fixture for multi-component mixing tests
class MaterialMultiComponentMixingTest : public ::testing::Test {
protected:
  void SetUp() override {
    // Create materials for multi-component mixing tests
    m_water = Material::createPredefined("water");
    m_oil = Material::createPredefined("oil");
    m_air = Material::createPredefined("air");
    m_aluminum = Material::createPredefined("aluminum");

    // Create a custom material
    m_custom = std::make_shared<Material>(Material::MaterialType::FLUID,
                                          "CustomMaterial");
    m_custom->setProperty(Material::MaterialProperty::DENSITY, 850.0);
    m_custom->setProperty(Material::MaterialProperty::DYNAMIC_VISCOSITY, 0.05);
    m_custom->setProperty(Material::MaterialProperty::THERMAL_CONDUCTIVITY,
                          0.12);
    m_custom->setProperty(Material::MaterialProperty::SPECIFIC_HEAT, 2000.0);
    m_custom->setReferenceTemperature(293.15);
  }

  std::shared_ptr<Material> m_water;
  std::shared_ptr<Material> m_oil;
  std::shared_ptr<Material> m_air;
  std::shared_ptr<Material> m_aluminum;
  std::shared_ptr<Material> m_custom;
};

// Test creating a three-component mixture
TEST_F(MaterialMultiComponentMixingTest, ThreeComponentMixture) {
  // Create a three-component mixture (water, oil, air)
  std::vector<std::shared_ptr<Material>> materials = {m_water, m_oil, m_air};
  std::vector<double> fractions = {0.5, 0.3,
                                   0.2}; // 50% water, 30% oil, 20% air

  auto mixture = Material::createMixture(materials, fractions);

  // Check that the mixture has the correct components
  auto components = mixture->getMixtureComponents();
  ASSERT_EQ(components.size(), 3) << "Mixture should have 3 components";

  // Check component fractions (may be in different order)
  double waterFraction = 0.0;
  double oilFraction = 0.0;
  double airFraction = 0.0;

  for (const auto &comp : components) {
    if (comp.first->getName() == "Water") {
      waterFraction = comp.second;
    } else if (comp.first->getName() == "Oil") {
      oilFraction = comp.second;
    } else if (comp.first->getName() == "Air") {
      airFraction = comp.second;
    }
  }

  EXPECT_NEAR(waterFraction, 0.5, 1e-6) << "Water fraction should be 0.5";
  EXPECT_NEAR(oilFraction, 0.3, 1e-6) << "Oil fraction should be 0.3";
  EXPECT_NEAR(airFraction, 0.2, 1e-6) << "Air fraction should be 0.2";

  // Check that the properties are a weighted combination
  double waterDensity =
      m_water->getProperty(Material::MaterialProperty::DENSITY);
  double oilDensity = m_oil->getProperty(Material::MaterialProperty::DENSITY);
  double airDensity = m_air->getProperty(Material::MaterialProperty::DENSITY);

  double expectedDensity =
      waterDensity * 0.5 + oilDensity * 0.3 + airDensity * 0.2;
  double mixtureDensity =
      mixture->getProperty(Material::MaterialProperty::DENSITY);

  EXPECT_NEAR(mixtureDensity, expectedDensity, 1.5)
      << "Mixture density should be a weighted average (tolerance: 1.5 kg/m³)";
}

// Test different mixing rules with multi-component mixtures
TEST_F(MaterialMultiComponentMixingTest, MixingRules) {
  // Create mixtures with different mixing rules
  std::vector<std::shared_ptr<Material>> materials = {m_water, m_oil, m_air};
  std::vector<double> fractions = {0.4, 0.4,
                                   0.2}; // 40% water, 40% oil, 20% air

  // Create mixtures with different rules
  auto linearMix = Material::createMixture(materials, fractions,
                                           Material::MixingRuleType::LINEAR);
  auto logMix = Material::createMixture(materials, fractions,
                                        Material::MixingRuleType::LOGARITHMIC);
  auto harmonicMix = Material::createMixture(
      materials, fractions, Material::MixingRuleType::HARMONIC);

  // Test LINEAR mixing rule (should be weighted average)
  double waterVisc =
      m_water->getProperty(Material::MaterialProperty::DYNAMIC_VISCOSITY);
  double oilVisc =
      m_oil->getProperty(Material::MaterialProperty::DYNAMIC_VISCOSITY);
  double airVisc =
      m_air->getProperty(Material::MaterialProperty::DYNAMIC_VISCOSITY);

  double expectedLinearVisc = waterVisc * 0.4 + oilVisc * 0.4 + airVisc * 0.2;
  double linearVisc =
      linearMix->getProperty(Material::MaterialProperty::DYNAMIC_VISCOSITY);

  EXPECT_NEAR(linearVisc, expectedLinearVisc, 1e-6)
      << "Linear mixing should be weighted average";

  // Test LOGARITHMIC mixing rule (common for viscosity)
  double expectedLogVisc =
      std::exp(0.4 * std::log(waterVisc) + 0.4 * std::log(oilVisc) +
               0.2 * std::log(airVisc));
  double logVisc =
      logMix->getProperty(Material::MaterialProperty::DYNAMIC_VISCOSITY);

  EXPECT_NEAR(logVisc, expectedLogVisc, 1e-6)
      << "Logarithmic mixing should use exponential average";

  // Test HARMONIC mixing rule (common for thermal conductivity)
  double waterK =
      m_water->getProperty(Material::MaterialProperty::THERMAL_CONDUCTIVITY);
  double oilK =
      m_oil->getProperty(Material::MaterialProperty::THERMAL_CONDUCTIVITY);
  double airK =
      m_air->getProperty(Material::MaterialProperty::THERMAL_CONDUCTIVITY);

  double expectedHarmonicK = 1.0 / (0.4 / waterK + 0.4 / oilK + 0.2 / airK);
  double harmonicK = harmonicMix->getProperty(
      Material::MaterialProperty::THERMAL_CONDUCTIVITY);

  EXPECT_NEAR(harmonicK, expectedHarmonicK, 1e-6)
      << "Harmonic mixing should use harmonic mean";

  // Test GEOMETRIC mixing rule
  auto geometricMix = Material::createMixture(
      materials, fractions, Material::MixingRuleType::GEOMETRIC);

  double expectedGeometricVisc = std::pow(waterVisc, 0.4) *
                                 std::pow(oilVisc, 0.4) *
                                 std::pow(airVisc, 0.2);
  double geometricVisc =
      geometricMix->getProperty(Material::MaterialProperty::DYNAMIC_VISCOSITY);

  EXPECT_NEAR(geometricVisc, expectedGeometricVisc, 1e-6)
      << "Geometric mixing should use geometric mean";
}

// Test mixing with a mixture as one of the components
TEST_F(MaterialMultiComponentMixingTest, NestedMixture) {
  // First create a binary mixture
  auto waterOilMix = m_water->createMixture(m_oil, 0.4); // 60% water, 40% oil

  // Now mix this with air
  std::vector<std::shared_ptr<Material>> materials = {waterOilMix, m_air};
  std::vector<double> fractions = {0.8, 0.2}; // 80% water-oil mix, 20% air

  auto nestedMix = Material::createMixture(materials, fractions);

  // Check the final components (should have 3 components, not 2)
  auto components = nestedMix->getMixtureComponents();
  ASSERT_EQ(components.size(), 3)
      << "Nested mixture should have 3 components, not 2";

  // Calculate expected fractions:
  // 0.8 * 0.6 = 0.48 water, 0.8 * 0.4 = 0.32 oil, 0.2 air
  double waterFraction = 0.0;
  double oilFraction = 0.0;
  double airFraction = 0.0;

  for (const auto &comp : components) {
    if (comp.first->getName() == "Water") {
      waterFraction = comp.second;
    } else if (comp.first->getName() == "Oil") {
      oilFraction = comp.second;
    } else if (comp.first->getName() == "Air") {
      airFraction = comp.second;
    }
  }

  EXPECT_NEAR(waterFraction, 0.48, 1e-6) << "Water fraction should be 0.48";
  EXPECT_NEAR(oilFraction, 0.32, 1e-6) << "Oil fraction should be 0.32";
  EXPECT_NEAR(airFraction, 0.2, 1e-6) << "Air fraction should be 0.2";

  // Calculate expected properties for the nested mixture
  double waterDensity =
      m_water->getProperty(Material::MaterialProperty::DENSITY);
  double oilDensity = m_oil->getProperty(Material::MaterialProperty::DENSITY);
  double airDensity = m_air->getProperty(Material::MaterialProperty::DENSITY);

  double expectedDensity =
      waterDensity * 0.48 + oilDensity * 0.32 + airDensity * 0.2;
  double mixtureDensity =
      nestedMix->getProperty(Material::MaterialProperty::DENSITY);

  EXPECT_NEAR(mixtureDensity, expectedDensity, 1.5)
      << "Nested mixture density should match expected value (tolerance: 1.5 "
         "kg/m³)";
}

// Test mixing with zero and near-zero fractions
TEST_F(MaterialMultiComponentMixingTest, ZeroFractions) {
  // Create a mixture with some zero fractions
  std::vector<std::shared_ptr<Material>> materials = {m_water, m_oil, m_air};
  std::vector<double> fractions = {0.7, 0.0, 0.3}; // 70% water, 0% oil, 30% air

  auto mixture = Material::createMixture(materials, fractions);

  // Check that only non-zero components are included
  auto components = mixture->getMixtureComponents();
  ASSERT_EQ(components.size(), 2)
      << "Mixture should have only 2 components with non-zero fractions";

  // Check effective fractions (should be normalized)
  double waterFraction = 0.0;
  double airFraction = 0.0;

  for (const auto &comp : components) {
    if (comp.first->getName() == "Water") {
      waterFraction = comp.second;
    } else if (comp.first->getName() == "Air") {
      airFraction = comp.second;
    }
  }

  EXPECT_NEAR(waterFraction, 0.7, 1e-6)
      << "Water fraction should be normalized to 0.7";
  EXPECT_NEAR(airFraction, 0.3, 1e-6)
      << "Air fraction should be normalized to 0.3";

  // Test with a very small fraction
  fractions = {0.699, 0.001, 0.3}; // 69.9% water, 0.1% oil, 30% air

  auto mixture2 = Material::createMixture(materials, fractions);
  components = mixture2->getMixtureComponents();

  // Small but non-zero fractions should be included
  ASSERT_EQ(components.size(), 3)
      << "Mixture should include components with small fractions";
}

// Test mixing of fluids and solids
TEST_F(MaterialMultiComponentMixingTest, FluidSolidMixing) {
  // Create a mixture of water, air, and aluminum
  std::vector<std::shared_ptr<Material>> materials = {m_water, m_air,
                                                      m_aluminum};
  std::vector<double> fractions = {0.4, 0.3,
                                   0.3}; // 40% water, 30% air, 30% aluminum

  auto mixture = Material::createMixture(materials, fractions);

  // The mixture material type should be FLUID (the default)
  EXPECT_EQ(mixture->getType(), Material::MaterialType::FLUID)
      << "Mixed fluid and solid should default to fluid type";

  // Thermal conductivity should reflect the high conductivity of aluminum
  double waterK =
      m_water->getProperty(Material::MaterialProperty::THERMAL_CONDUCTIVITY);
  double airK =
      m_air->getProperty(Material::MaterialProperty::THERMAL_CONDUCTIVITY);
  double alK =
      m_aluminum->getProperty(Material::MaterialProperty::THERMAL_CONDUCTIVITY);

  double mixtureK =
      mixture->getProperty(Material::MaterialProperty::THERMAL_CONDUCTIVITY);

  // Due to harmonic mixing rule being used for thermal conductivity,
  // the mixture conductivity might actually be lower than expected with linear
  // mixing. Instead of testing absolute values, let's check that it behaves
  // according to the mixing rule
  double harmonicK = 1.0 / (0.4 / waterK + 0.3 / airK + 0.3 / alK);

  EXPECT_NEAR(mixtureK, harmonicK, 1e-6)
      << "Mixture thermal conductivity should follow harmonic mixing rule";
}

// Test temperature-dependent behavior in multi-component mixtures
TEST_F(MaterialMultiComponentMixingTest, TemperatureDependentMixing) {
  // Create a mixture
  std::vector<std::shared_ptr<Material>> materials = {m_water, m_oil, m_air};
  std::vector<double> fractions = {0.4, 0.4,
                                   0.2}; // 40% water, 40% oil, 20% air

  auto mixture = Material::createMixture(materials, fractions);

  // Ensure temperature dependence is enabled
  EXPECT_TRUE(mixture->isUsingTempDependentProps())
      << "Mixture should use temperature-dependent properties";

  // Test properties at different temperatures
  double refTemp = mixture->getReferenceTemperature();
  double hotTemp = refTemp + 50.0; // 50K hotter

  // Get density at reference temperature and higher temperature
  double refDensity = mixture->getProperty(Material::MaterialProperty::DENSITY);
  double hotDensity = mixture->getPropertyAtTemperature(
      Material::MaterialProperty::DENSITY, hotTemp);

  // Density should decrease with temperature
  EXPECT_LT(hotDensity, refDensity)
      << "Mixture density should decrease with temperature";

  // Get viscosity at reference temperature and higher temperature
  double refVisc =
      mixture->getProperty(Material::MaterialProperty::DYNAMIC_VISCOSITY);
  double hotVisc = mixture->getPropertyAtTemperature(
      Material::MaterialProperty::DYNAMIC_VISCOSITY, hotTemp);

  // Viscosity should decrease with temperature
  EXPECT_LT(hotVisc, refVisc)
      << "Mixture viscosity should decrease with temperature";

  // Calculate individual component values at hot temperature
  double waterDensityHot = m_water->getPropertyAtTemperature(
      Material::MaterialProperty::DENSITY, hotTemp);
  double oilDensityHot = m_oil->getPropertyAtTemperature(
      Material::MaterialProperty::DENSITY, hotTemp);
  double airDensityHot = m_air->getPropertyAtTemperature(
      Material::MaterialProperty::DENSITY, hotTemp);

  // Expected hot density should be a weighted average
  double expectedDensityHot =
      waterDensityHot * 0.4 + oilDensityHot * 0.4 + airDensityHot * 0.2;

  EXPECT_NEAR(hotDensity, expectedDensityHot, 1.5)
      << "Hot mixture density should match expected value (tolerance: 1.5 "
         "kg/m³)";
}

// Test mixing with default rule types
TEST_F(MaterialMultiComponentMixingTest, DefaultMixingRules) {
  // Create a mixture with the DEFAULT mixing rule
  std::vector<std::shared_ptr<Material>> materials = {m_water, m_oil, m_air};
  std::vector<double> fractions = {0.4, 0.4,
                                   0.2}; // 40% water, 40% oil, 20% air

  auto mixture = Material::createMixture(materials, fractions,
                                         Material::MixingRuleType::DEFAULT);

  // Default for viscosity should be logarithmic
  double waterVisc =
      m_water->getProperty(Material::MaterialProperty::DYNAMIC_VISCOSITY);
  double oilVisc =
      m_oil->getProperty(Material::MaterialProperty::DYNAMIC_VISCOSITY);
  double airVisc =
      m_air->getProperty(Material::MaterialProperty::DYNAMIC_VISCOSITY);

  double expectedLogVisc =
      std::exp(0.4 * std::log(waterVisc) + 0.4 * std::log(oilVisc) +
               0.2 * std::log(airVisc));
  double mixtureVisc =
      mixture->getProperty(Material::MaterialProperty::DYNAMIC_VISCOSITY);

  EXPECT_NEAR(mixtureVisc, expectedLogVisc, 1e-6)
      << "Default mixing rule for viscosity should be logarithmic";

  // Default for thermal conductivity should be harmonic
  double waterK =
      m_water->getProperty(Material::MaterialProperty::THERMAL_CONDUCTIVITY);
  double oilK =
      m_oil->getProperty(Material::MaterialProperty::THERMAL_CONDUCTIVITY);
  double airK =
      m_air->getProperty(Material::MaterialProperty::THERMAL_CONDUCTIVITY);

  double expectedHarmonicK = 1.0 / (0.4 / waterK + 0.4 / oilK + 0.2 / airK);
  double mixtureK =
      mixture->getProperty(Material::MaterialProperty::THERMAL_CONDUCTIVITY);

  EXPECT_NEAR(mixtureK, expectedHarmonicK, 1e-6)
      << "Default mixing rule for thermal conductivity should be harmonic";

  // Default for density should be linear
  double waterDensity =
      m_water->getProperty(Material::MaterialProperty::DENSITY);
  double oilDensity = m_oil->getProperty(Material::MaterialProperty::DENSITY);
  double airDensity = m_air->getProperty(Material::MaterialProperty::DENSITY);

  double expectedLinearDensity =
      waterDensity * 0.4 + oilDensity * 0.4 + airDensity * 0.2;
  double mixtureDensity =
      mixture->getProperty(Material::MaterialProperty::DENSITY);

  EXPECT_NEAR(mixtureDensity, expectedLinearDensity, 1.5)
      << "Default mixing rule for density should be linear";
}

// Test error/boundary conditions for multi-component mixing
TEST_F(MaterialMultiComponentMixingTest, ErrorConditions) {
  // Test with empty materials list (should throw)
  std::vector<std::shared_ptr<Material>> emptyMaterials;
  std::vector<double> emptyFractions;

  EXPECT_THROW(Material::createMixture(emptyMaterials, emptyFractions),
               std::invalid_argument)
      << "Empty materials list should throw exception";

  // Test with mismatched sizes (should throw)
  std::vector<std::shared_ptr<Material>> materials = {m_water, m_oil, m_air};
  std::vector<double> shortFractions = {
      0.5, 0.5}; // Only 2 fractions for 3 materials

  EXPECT_THROW(Material::createMixture(materials, shortFractions),
               std::invalid_argument)
      << "Mismatched sizes should throw exception";

  // Test with fractions that don't sum to 1.0 (should throw)
  std::vector<double> badFractions = {0.4, 0.4, 0.4}; // Sum to 1.2

  EXPECT_THROW(Material::createMixture(materials, badFractions),
               std::invalid_argument)
      << "Fractions not summing to 1.0 should throw exception";

  // Test with single material (should return a copy of that material)
  std::vector<std::shared_ptr<Material>> singleMaterial = {m_water};
  std::vector<double> singleFraction = {1.0};

  auto mixtureSingle = Material::createMixture(singleMaterial, singleFraction);

  EXPECT_DOUBLE_EQ(
      mixtureSingle->getProperty(Material::MaterialProperty::DENSITY),
      m_water->getProperty(Material::MaterialProperty::DENSITY))
      << "Single material mixture should be a copy of that material";
}

// Test maximum number of components
TEST_F(MaterialMultiComponentMixingTest, MaxComponents) {
  // Create a mixture with the maximum allowed number of components
  // MAX_MIXTURE_COMPONENTS is defined as 16 in Material.h
  const size_t maxComponents = Material::MAX_MIXTURE_COMPONENTS;

  // Create many identical materials for testing (we'll use water repeatedly)
  std::vector<std::shared_ptr<Material>> materials;
  std::vector<double> fractions;

  // Create maxComponents identical water materials
  for (size_t i = 0; i < maxComponents; i++) {
    materials.push_back(m_water);
    fractions.push_back(1.0 / maxComponents);
  }

  // Create mixture with max components
  auto mixture = Material::createMixture(materials, fractions);

  // Check that the mixture has maxComponents components
  auto components = mixture->getMixtureComponents();
  ASSERT_EQ(components.size(), maxComponents)
      << "Mixture should have maximum number of components";

  // All components should be water with equal fractions
  for (const auto &comp : components) {
    EXPECT_EQ(comp.first->getName(), "Water") << "Component should be water";
    EXPECT_NEAR(comp.second, 1.0 / maxComponents, 1e-6)
        << "Component fraction should be 1/maxComponents";
  }

  // Properties should match water's properties (within a small tolerance)
  EXPECT_NEAR(mixture->getProperty(Material::MaterialProperty::DENSITY),
              m_water->getProperty(Material::MaterialProperty::DENSITY),
              2.0 // Allow tolerance of 2 kg/m³
              )
      << "Mixture of identical materials should have nearly the same "
         "properties";
}

// Test string-based mixing rule API
TEST_F(MaterialMultiComponentMixingTest, StringBasedAPI) {
  // Create mixtures with different mixing rules using string API
  std::vector<std::shared_ptr<Material>> materials = {m_water, m_oil, m_air};
  std::vector<double> fractions = {0.4, 0.4, 0.2};

  auto linearMix = Material::createMixture(materials, fractions, "linear");
  auto logMix = Material::createMixture(materials, fractions, "logarithmic");
  auto harmonicMix = Material::createMixture(materials, fractions, "harmonic");
  auto defaultMix = Material::createMixture(materials, fractions, "default");

  // All should create valid mixtures
  EXPECT_NE(linearMix, nullptr) << "Linear mixing by string should work";
  EXPECT_NE(logMix, nullptr) << "Logarithmic mixing by string should work";
  EXPECT_NE(harmonicMix, nullptr) << "Harmonic mixing by string should work";
  EXPECT_NE(defaultMix, nullptr) << "Default mixing by string should work";

  // Unknown mixing rule should default to linear
  auto unknownMix =
      Material::createMixture(materials, fractions, "nonexistent");

  double waterVisc =
      m_water->getProperty(Material::MaterialProperty::DYNAMIC_VISCOSITY);
  double oilVisc =
      m_oil->getProperty(Material::MaterialProperty::DYNAMIC_VISCOSITY);
  double airVisc =
      m_air->getProperty(Material::MaterialProperty::DYNAMIC_VISCOSITY);

  double expectedLinearVisc = waterVisc * 0.4 + oilVisc * 0.4 + airVisc * 0.2;

  EXPECT_NEAR(
      unknownMix->getProperty(Material::MaterialProperty::DYNAMIC_VISCOSITY),
      expectedLinearVisc, 1e-6)
      << "Unknown mixing rule should default to linear";
}
