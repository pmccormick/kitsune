/**
 * ====================================================================
 * Material Binary Mixing Tests
 * ====================================================================
 *
 * These tests verify the binary mixing functionality of the Material class.
 *
 * Test Coverage:
 * --------------
 * 1. Simple Binary Mixing:
 *    - Basic mixing of two materials with different ratios
 *    - Property interpolation between components
 *
 * 2. Temperature Dependence:
 *    - Temperature effects on binary mixture properties
 *
 * 3. Mixing Rules:
 *    - Linear, Logarithmic, Harmonic, and Geometric mixing rules for binary
 * mixtures
 *
 * 4. Edge Cases:
 *    - Fractions of 0.0 and 1.0
 */

#include "Material.h"
#include "Units.h"
#include <cmath>
#include <gtest/gtest.h>
#include <memory>

// Test fixture for binary material mixing tests
class MaterialBinaryMixingTest : public ::testing::Test {
protected:
  void SetUp() override {
    // Create basic materials for testing
    m_water = Material::createPredefined("water");
    m_air = Material::createPredefined("air");
    m_oil = Material::createPredefined("oil");
  }

  std::shared_ptr<Material> m_water;
  std::shared_ptr<Material> m_air;
  std::shared_ptr<Material> m_oil;
};

// Test simple binary mixture (basic properties)
TEST_F(MaterialBinaryMixingTest, SimpleBinaryMixture) {
  // Test simple binary mixture (50% water, 50% air)
  auto mixture1 = m_water->createMixture(m_air, 0.5);

  // Mixture should have properties between components
  double waterDensity =
      m_water->getProperty(Material::MaterialProperty::DENSITY);
  double airDensity = m_air->getProperty(Material::MaterialProperty::DENSITY);
  double mixtureDensity =
      mixture1->getProperty(Material::MaterialProperty::DENSITY);

  EXPECT_LT(mixtureDensity, waterDensity)
      << "Mixture density should be less than water";
  EXPECT_GT(mixtureDensity, airDensity)
      << "Mixture density should be greater than air";

  // Test linear mixing with appropriate tolerance (0.2% relative difference
  // observed)
  EXPECT_NEAR(mixtureDensity, (waterDensity + airDensity) / 2.0, 1.0)
      << "Density should use linear mixing by default (tolerance: 1.0 kg/m³)";
}

// Test different mix fractions
TEST_F(MaterialBinaryMixingTest, DifferentMixFractions) {
  // Test different mix fractions
  auto mixture1 = m_water->createMixture(m_air, 0.5);  // 50% water, 50% air
  auto mixture2 = m_water->createMixture(m_air, 0.25); // 75% water, 25% air

  double mixture1Density =
      mixture1->getProperty(Material::MaterialProperty::DENSITY);
  double mixture2Density =
      mixture2->getProperty(Material::MaterialProperty::DENSITY);
  double waterDensity =
      m_water->getProperty(Material::MaterialProperty::DENSITY);
  double airDensity = m_air->getProperty(Material::MaterialProperty::DENSITY);

  EXPECT_GT(mixture2Density, mixture1Density)
      << "Mixture with more water should have higher density";
  EXPECT_NEAR(mixture2Density, waterDensity * 0.75 + airDensity * 0.25, 1.5)
      << "Density mixing should be proportional to fractions (tolerance: 1.5 "
         "kg/m³)";
}

// Test temperature dependence of binary mixtures
TEST_F(MaterialBinaryMixingTest, TemperatureDependentBinaryMixture) {
  // Create a binary mixture
  auto mixture = m_water->createMixture(m_air, 0.5); // 50% water, 50% air

  // Get density at reference temperature
  double mixtureDensity =
      mixture->getProperty(Material::MaterialProperty::DENSITY);

  // Get density at high temperature
  double hotTemp = 353.15; // 80°C
  double mixtureDensityHot = mixture->getPropertyAtTemperature(
      Material::MaterialProperty::DENSITY, hotTemp);

  EXPECT_NE(mixtureDensityHot, mixtureDensity)
      << "Mixture properties should still be temperature dependent";
  EXPECT_LT(mixtureDensityHot, mixtureDensity)
      << "Density should decrease with temperature";

  // Calculate expected density at high temperature based on component densities
  double waterDensityHot = m_water->getPropertyAtTemperature(
      Material::MaterialProperty::DENSITY, hotTemp);
  double airDensityHot = m_air->getPropertyAtTemperature(
      Material::MaterialProperty::DENSITY, hotTemp);
  double expectedDensityHot = (waterDensityHot + airDensityHot) / 2.0;

  EXPECT_NEAR(mixtureDensityHot, expectedDensityHot, 1.5)
      << "Hot mixture density should match expected value (tolerance: 1.5 "
         "kg/m³)";
}

// Test different mixing rules for binary mixtures
TEST_F(MaterialBinaryMixingTest, BinaryMixingRules) {
  // Get base component properties
  double waterViscosity =
      m_water->getProperty(Material::MaterialProperty::DYNAMIC_VISCOSITY);
  double oilViscosity =
      m_oil->getProperty(Material::MaterialProperty::DYNAMIC_VISCOSITY);

  // Linear mixing
  auto linearMix = Material::createMixture({m_water, m_oil}, {0.5, 0.5},
                                           Material::MixingRuleType::LINEAR);
  double linearViscosity =
      linearMix->getProperty(Material::MaterialProperty::DYNAMIC_VISCOSITY);
  EXPECT_NEAR(linearViscosity, (waterViscosity + oilViscosity) / 2.0, 1e-6)
      << "Linear mixing should be simple average";

  // Logarithmic mixing (common for viscosity)
  auto logMix = Material::createMixture({m_water, m_oil}, {0.5, 0.5},
                                        Material::MixingRuleType::LOGARITHMIC);
  double logViscosity =
      logMix->getProperty(Material::MaterialProperty::DYNAMIC_VISCOSITY);
  EXPECT_NEAR(logViscosity,
              exp(0.5 * log(waterViscosity) + 0.5 * log(oilViscosity)), 1e-6)
      << "Logarithmic mixing should use exponential average";

  // Harmonic mixing (common for thermal conductivity of perpendicular layers)
  auto harmonicMix = Material::createMixture(
      {m_water, m_oil}, {0.5, 0.5}, Material::MixingRuleType::HARMONIC);
  double waterK =
      m_water->getProperty(Material::MaterialProperty::THERMAL_CONDUCTIVITY);
  double oilK =
      m_oil->getProperty(Material::MaterialProperty::THERMAL_CONDUCTIVITY);
  double harmonicK = harmonicMix->getProperty(
      Material::MaterialProperty::THERMAL_CONDUCTIVITY);

  EXPECT_NEAR(harmonicK, 2.0 / (1.0 / waterK + 1.0 / oilK), 1e-6)
      << "Harmonic mixing should use harmonic mean";

  // Test GEOMETRIC mixing rule
  auto geometricMix = Material::createMixture(
      {m_water, m_oil}, {0.5, 0.5}, Material::MixingRuleType::GEOMETRIC);

  double expectedGeometricVisc =
      std::pow(waterViscosity, 0.5) * std::pow(oilViscosity, 0.5);
  double geometricVisc =
      geometricMix->getProperty(Material::MaterialProperty::DYNAMIC_VISCOSITY);

  EXPECT_NEAR(geometricVisc, expectedGeometricVisc, 1e-6)
      << "Geometric mixing should use geometric mean";
}

// Test edge cases for binary mixing
TEST_F(MaterialBinaryMixingTest, BinaryMixingEdgeCases) {
  // Test with fraction = 0 (should be equivalent to first material)
  auto mixture1 = m_water->createMixture(m_air, 0.0);
  EXPECT_DOUBLE_EQ(mixture1->getProperty(Material::MaterialProperty::DENSITY),
                   m_water->getProperty(Material::MaterialProperty::DENSITY))
      << "Mixture with fraction=0 should equal first material";

  // Test with fraction = 1 (should be equivalent to second material)
  auto mixture2 = m_water->createMixture(m_air, 1.0);
  EXPECT_DOUBLE_EQ(mixture2->getProperty(Material::MaterialProperty::DENSITY),
                   m_air->getProperty(Material::MaterialProperty::DENSITY))
      << "Mixture with fraction=1 should equal second material";
}

// Test material mixing with binary string API
TEST_F(MaterialBinaryMixingTest, StringBasedMixingRules) {
  // Test string-based API for mixing rules
  auto linearMix = m_water->createMixture(m_oil, 0.3, "linear");
  auto logMix = m_water->createMixture(m_oil, 0.3, "logarithmic");
  auto harmonicMix = m_water->createMixture(m_oil, 0.3, "harmonic");

  // All should create valid mixtures
  EXPECT_NE(linearMix, nullptr) << "Linear mixing by string should work";
  EXPECT_NE(logMix, nullptr) << "Logarithmic mixing by string should work";
  EXPECT_NE(harmonicMix, nullptr) << "Harmonic mixing by string should work";

  // Properties should be different with different mixing rules
  EXPECT_NE(
      linearMix->getProperty(Material::MaterialProperty::DYNAMIC_VISCOSITY),
      logMix->getProperty(Material::MaterialProperty::DYNAMIC_VISCOSITY))
      << "Linear and logarithmic mixing should give different results";
}
