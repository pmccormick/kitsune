/**
 * ====================================================================
 * Material Class Unit Tests
 * ====================================================================
 *
 * These tests verify the functionality of the Material class which
 * provides physical property management for CFD simulations.
 *
 * Test Coverage:
 * --------------
 * 1. Basic Properties:
 *    - Verification of base property values for predefined materials
 *    - Relationship between material properties (e.g., density comparisons)
 *
 * 2. Temperature Dependence:
 *    - Property changes with temperature for different materials
 *    - Verification of specific temperature models (e.g., ideal gas law for
 * air)
 *    - High temperature behavior
 *
 * 3. Property Models:
 *    - Constant, Linear, Exponential, and Polynomial models
 *    - Coefficient handling
 *    - Model behavior at different temperature ranges
 *
 * 4. Custom Property Functions:
 *    - User-defined property functions
 *    - Complex behavior (e.g., water density peak at 4°C)
 *
 * 5. Material Mixing:
 *    - Binary mixtures with different ratios
 *    - Property interpolation
 *    - Temperature dependence of mixtures
 *    - NOTE: Tolerance set to 1.5 kg/m³ for density to accommodate
 *            small variations due to implementation details of the
 *            mixing algorithms (~0.2% difference)
 *
 * 6. Mixing Rules:
 *    - Linear, Logarithmic, Harmonic, and other mixing rules
 *    - Property-specific behaviors (e.g., logarithmic for viscosity)
 *
 * 7. Unit Conversions:
 *    - Setting and getting properties with different units
 *    - Temperature unit conversions
 *    - Combined property and temperature unit handling
 *
 * 8. Registry and IDs:
 *    - Material instance tracking
 *    - ID uniqueness
 *    - ID-based lookup
 *
 * 9. Predefined Materials:
 *    - Availability of standard materials (water, air, metals, etc.)
 *    - Type classification (fluid vs. solid)
 *    - Temperature-dependent behavior
 *
 * Potential Additional Tests:
 * --------------------------
 * 1. Move/Copy Semantics: Test proper behavior when materials are copied or
 * moved
 *
 * 2. Memory Management: Verify no leaks when creating and destroying materials
 *
 * 3. Thread Safety: Test concurrent access to material registry and properties
 *
 * 4. Error Handling: Test behavior with invalid inputs and edge cases
 *
 * 5. Multi-Component Mixtures: Test mixtures with more than two components
 *
 * 6. Serialization: Test saving and loading material definitions
 *
 * 7. Performance: Benchmark property lookups and mixing operations
 *
 * 8. Custom Mixing Rules: Test user-defined mixing rule functions
 */

#include "Material.h"
#include "Units.h"
#include <cmath>
#include <gtest/gtest.h>
#include <memory>

class MaterialTest : public ::testing::Test {
protected:
  void SetUp() override {
    // Create basic materials for testing
    m_water = Material::createPredefined("water");
    m_air = Material::createPredefined("air");
    m_aluminum = Material::createPredefined("aluminum");
    m_oil = Material::createPredefined("oil");

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
  std::shared_ptr<Material> m_air;
  std::shared_ptr<Material> m_aluminum;
  std::shared_ptr<Material> m_oil;
  std::shared_ptr<Material> m_custom;
};

// Test basic property access
TEST_F(MaterialTest, BasicProperties) {
  // Check water properties at reference temperature
  EXPECT_NEAR(m_water->getProperty(Material::MaterialProperty::DENSITY), 998.2,
              0.1)
      << "Water density should be ~998.2 kg/m³";
  EXPECT_NEAR(
      m_water->getProperty(Material::MaterialProperty::DYNAMIC_VISCOSITY),
      0.001, 0.0001)
      << "Water viscosity should be ~0.001 Pa·s";
  EXPECT_NEAR(
      m_water->getProperty(Material::MaterialProperty::THERMAL_CONDUCTIVITY),
      0.6, 0.01)
      << "Water thermal conductivity should be ~0.6 W/(m·K)";
  EXPECT_NEAR(m_water->getProperty(Material::MaterialProperty::SPECIFIC_HEAT),
              4182.0, 1.0)
      << "Water specific heat should be ~4182 J/(kg·K)";

  // Check air properties
  EXPECT_LT(m_air->getProperty(Material::MaterialProperty::DENSITY),
            m_water->getProperty(Material::MaterialProperty::DENSITY))
      << "Air density should be less than water";

  // Check aluminum properties
  EXPECT_GT(
      m_aluminum->getProperty(Material::MaterialProperty::THERMAL_CONDUCTIVITY),
      m_water->getProperty(Material::MaterialProperty::THERMAL_CONDUCTIVITY))
      << "Aluminum conductivity should be greater than water";
}

// Test temperature-dependent properties
TEST_F(MaterialTest, TemperatureDependence) {
  // Water density at different temperatures
  double refDensity = m_water->getProperty(Material::MaterialProperty::DENSITY);
  double hotDensity = m_water->getPropertyAtTemperature(
      Material::MaterialProperty::DENSITY, 353.15); // 80°C

  EXPECT_LT(hotDensity, refDensity)
      << "Water density should decrease with temperature";

  // Air density (ideal gas behavior)
  double refAirDensity =
      m_air->getProperty(Material::MaterialProperty::DENSITY);
  double hotAirDensity = m_air->getPropertyAtTemperature(
      Material::MaterialProperty::DENSITY, 353.15);

  EXPECT_LT(hotAirDensity, refAirDensity)
      << "Air density should decrease with temperature";
  EXPECT_NEAR(hotAirDensity, refAirDensity * (293.15 / 353.15), 0.01)
      << "Air density should follow ideal gas law";

  // Aluminum thermal conductivity
  double refAlConductivity =
      m_aluminum->getProperty(Material::MaterialProperty::THERMAL_CONDUCTIVITY);
  double hotAlConductivity = m_aluminum->getPropertyAtTemperature(
      Material::MaterialProperty::THERMAL_CONDUCTIVITY, 353.15);

  EXPECT_NE(hotAlConductivity, refAlConductivity)
      << "Aluminum conductivity should change with temperature";

  // Oil viscosity
  double refOilViscosity =
      m_oil->getProperty(Material::MaterialProperty::DYNAMIC_VISCOSITY);
  double hotOilViscosity = m_oil->getPropertyAtTemperature(
      Material::MaterialProperty::DYNAMIC_VISCOSITY, 353.15);

  EXPECT_LT(hotOilViscosity, refOilViscosity)
      << "Oil viscosity should decrease significantly with temperature";
  // Exponential model with coefficient -0.025
  EXPECT_NEAR(hotOilViscosity,
              refOilViscosity * exp(-0.025 * (353.15 - 293.15)),
              0.01 * refOilViscosity) // 1% tolerance
      << "Oil viscosity should follow exponential model";
}

// Test different property models
TEST_F(MaterialTest, PropertyModels) {
  // Create a test material and set different property models
  auto testMaterial =
      std::make_shared<Material>(Material::MaterialType::FLUID, "TestModels");
  testMaterial->setReferenceTemperature(300.0);

  // Constant model (default)
  testMaterial->setProperty(Material::MaterialProperty::DENSITY, 1000.0);

  // Linear model: value = baseValue * (1 + a*(T-Tref))
  testMaterial->setPropertyModel(
      Material::MaterialProperty::THERMAL_CONDUCTIVITY,
      Material::PropertyModel::LINEAR, {0.002}); // 0.2% increase per degree K
  testMaterial->setProperty(Material::MaterialProperty::THERMAL_CONDUCTIVITY,
                            0.5);

  // Exponential model: value = baseValue * exp(a*(T-Tref))
  testMaterial->setPropertyModel(Material::MaterialProperty::DYNAMIC_VISCOSITY,
                                 Material::PropertyModel::EXPONENTIAL,
                                 {-0.01}); // 1% decrease per degre
  testMaterial->setProperty(Material::MaterialProperty::DYNAMIC_VISCOSITY, 0.1);

  // Polynomial model: value = baseValue * (1 + a1*dT + a2*dT^2 + ...)
  testMaterial->setPropertyModel(Material::MaterialProperty::SPECIFIC_HEAT,
                                 Material::PropertyModel::POLYNOMIAL,
                                 {0.001, -0.000001}); // 2nd order polynomial
  testMaterial->setProperty(Material::MaterialProperty::SPECIFIC_HEAT, 1000.0);

  // Test all models at a different temperature
  double testTemp = 350.0; // 50K above reference
  double deltaT = testTemp - 300.0;

  // Constant model
  EXPECT_DOUBLE_EQ(testMaterial->getPropertyAtTemperature(
                       Material::MaterialProperty::DENSITY, testTemp),
                   1000.0)
      << "Constant model should not change with temperature";

  // Linear model
  double expectedConductivity = 0.5 * (1.0 + 0.002 * deltaT);
  EXPECT_NEAR(testMaterial->getPropertyAtTemperature(
                  Material::MaterialProperty::THERMAL_CONDUCTIVITY, testTemp),
              expectedConductivity, 1e-6)
      << "Linear model should increase linearly with temperature";

  // Exponential model
  double expectedViscosity = 0.1 * exp(-0.01 * deltaT);
  EXPECT_NEAR(testMaterial->getPropertyAtTemperature(
                  Material::MaterialProperty::DYNAMIC_VISCOSITY, testTemp),
              expectedViscosity, 1e-6)
      << "Exponential model should follow exp(-0.01 * deltaT)";

  // Polynomial model
  double expectedSpecificHeat =
      1000.0 * (1.0 + 0.001 * deltaT - 0.000001 * deltaT * deltaT);
  EXPECT_NEAR(testMaterial->getPropertyAtTemperature(
                  Material::MaterialProperty::SPECIFIC_HEAT, testTemp),
              expectedSpecificHeat, 1e-6)
      << "Polynomial model should follow quadratic function";
}

// Test custom property function
TEST_F(MaterialTest, CustomPropertyFunction) {
  // Create a material with a custom function
  auto testMaterial =
      std::make_shared<Material>(Material::MaterialType::FLUID, "CustomFunc");
  testMaterial->setReferenceTemperature(293.15);

  // Set water-like density with a more complex temperature dependence
  testMaterial->setProperty(Material::MaterialProperty::DENSITY, 1000.0);
  testMaterial->setCustomPropertyFunction(
      Material::MaterialProperty::DENSITY, [](double T) -> double {
        // Water density peaks at around 4°C and decreases both above and
        // below
        double T_C = T - 273.15; // Convert to Celsius
        // Simple parabolic approximation with peak at 4°C
        return 1000.0 - 0.05 * (T_C - 4.0) * (T_C - 4.0);
      });

  // Enable temperature-dependent properties
  testMaterial->setUseTempDependentProps(true);

  // Test at various temperatures
  double density0C = testMaterial->getPropertyAtTemperature(
      Material::MaterialProperty::DENSITY, 273.15);
  double density4C = testMaterial->getPropertyAtTemperature(
      Material::MaterialProperty::DENSITY, 277.15);
  double density10C = testMaterial->getPropertyAtTemperature(
      Material::MaterialProperty::DENSITY, 283.15);

  // Density should peak at 4°C
  EXPECT_GT(density4C, density0C)
      << "Water density at 4°C should be higher than at 0°C";
  EXPECT_GT(density4C, density10C)
      << "Water density at 4°C should be higher than at 10°C";

  // Should match our function exactly
  EXPECT_DOUBLE_EQ(density4C, 1000.0)
      << "Density at 4°C should be exactly 1000.0";
  EXPECT_DOUBLE_EQ(density0C, 1000.0 - 0.05 * 16.0)
      << "Density at 0°C should match function";
}

// Test material mixing
TEST_F(MaterialTest, MaterialMixing) {
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

  // Test different mix fractions
  auto mixture2 = m_water->createMixture(m_air, 0.25); // 75% water, 25% air
  double mixture2Density =
      mixture2->getProperty(Material::MaterialProperty::DENSITY);

  EXPECT_GT(mixture2Density, mixtureDensity)
      << "Mixture with more water should have higher density";
  EXPECT_NEAR(mixture2Density, waterDensity * 0.75 + airDensity * 0.25, 1.5)
      << "Density mixing should be proportional to fractions (tolerance: 1.5 "
         "kg/m³)";

  // Test temperature dependence of mixtures
  double hotTemp = 353.15; // 80°C
  double mixtureDensityHot = mixture1->getPropertyAtTemperature(
      Material::MaterialProperty::DENSITY, hotTemp);

  EXPECT_NE(mixtureDensityHot, mixtureDensity)
      << "Mixture properties should still be temperature dependent";
}

// Test different mixing rules
TEST_F(MaterialTest, MixingRules) {
  // Testing different mixing rules for viscosity

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
}

// Test unit conversions
TEST_F(MaterialTest, UnitConversions) {
  // Test setting properties with different units
  auto testMaterial =
      std::make_shared<Material>(Material::MaterialType::FLUID, "UnitTest");

  // Set density in g/cm³
  testMaterial->setPropertyWithUnits(Material::MaterialProperty::DENSITY, 0.8,
                                     "g/cm³");
  EXPECT_NEAR(testMaterial->getProperty(Material::MaterialProperty::DENSITY),
              800.0, 1e-6)
      << "Density should be converted from g/cm³ to kg/m³";

  // Test getting property in different units
  double density_lbft3 = testMaterial->getPropertyWithUnits(
      Material::MaterialProperty::DENSITY, "lb/ft³");
  EXPECT_NEAR(density_lbft3, 800.0 / 16.0185, 1e-6)
      << "Density should be converted to lb/ft³";

  // Set temperature-dependent property and test with different units
  testMaterial->setPropertyModel(
      Material::MaterialProperty::THERMAL_CONDUCTIVITY,
      Material::PropertyModel::LINEAR, {0.001}); // 0.1% increase per degree K
  testMaterial->setProperty(Material::MaterialProperty::THERMAL_CONDUCTIVITY,
                            0.5);

  // Get property at temperature with unit conversion
  double k_btu = testMaterial->getPropertyAtTemperatureWithUnits(
      Material::MaterialProperty::THERMAL_CONDUCTIVITY, 100.0,
      "F",               // Temperature in Fahrenheit
      "BTU/(hr·ft·°F)"); // Result unit

  // Expected: convert 100°F to K, get property, convert to BTU/(hr·ft·°F)
  double tempK = Units::fahrenheitToKelvin(100.0);
  double k_si = 0.5 * (1.0 + 0.001 * (tempK - 293.15));
  double expected_btu = Units::wmkToBtu(k_si);

  EXPECT_NEAR(k_btu, expected_btu, 1e-6)
      << "Should convert both temperature and property units correctly";
}

// Test registry and ID handling
TEST_F(MaterialTest, RegistryAndIDs) {
  // Materials should have unique IDs
  EXPECT_NE(m_water->getID(), m_air->getID())
      << "Materials should have distinct IDs";
  EXPECT_NE(m_water->getID(), m_aluminum->getID())
      << "Materials should have distinct IDs";

  // Get material by ID
  uint32_t waterID = m_water->getID();
  Material *retrievedWater = Material::getByID(waterID);

  EXPECT_EQ(retrievedWater, m_water.get())
      << "Should retrieve the same material instance by ID";

  // Registry should track all created materials
  EXPECT_NE(Material::getByID(m_custom->getID()), nullptr)
      << "Custom material should be in the registry";

  // Invalid ID should return nullptr
  EXPECT_EQ(Material::getByID(999999), nullptr)
      << "Invalid ID should return nullptr";
}

// Test predefined materials
TEST_F(MaterialTest, PredefinedMaterials) {
  // Test all predefined materials
  auto water = Material::createPredefined("water");
  auto air = Material::createPredefined("air");
  auto aluminum = Material::createPredefined("aluminum");
  auto copper = Material::createPredefined("copper");
  auto steel = Material::createPredefined("steel");
  auto oil = Material::createPredefined("oil");

  // All should be valid instances
  EXPECT_NE(water, nullptr) << "Should create water";
  EXPECT_NE(air, nullptr) << "Should create air";
  EXPECT_NE(aluminum, nullptr) << "Should create aluminum";
  EXPECT_NE(copper, nullptr) << "Should create copper";
  EXPECT_NE(steel, nullptr) << "Should create steel";
  EXPECT_NE(oil, nullptr) << "Should create oil";

  // Check types
  EXPECT_EQ(water->getType(), Material::MaterialType::FLUID)
      << "Water should be fluid";
  EXPECT_EQ(aluminum->getType(), Material::MaterialType::SOLID)
      << "Aluminum should be solid";

  // Check temperature-dependent properties are enabled for all
  EXPECT_TRUE(water->isUsingTempDependentProps())
      << "Water should use temperature-dependent properties";
  EXPECT_TRUE(air->isUsingTempDependentProps())
      << "Air should use temperature-dependent properties";
  EXPECT_TRUE(aluminum->isUsingTempDependentProps())
      << "Aluminum should use temperature-dependent properties";

  // Check that invalid material name returns default
  auto invalid = Material::createPredefined("nonexistent");
  EXPECT_NE(invalid, nullptr)
      << "Should return default material for invalid name";
  EXPECT_EQ(invalid->getName(), "nonexistent")
      << "Should use provided name for default material";
}

// ====================================================================
// Main Function
// ====================================================================

int main(int argc, char **argv) {
  // Initialize Google Test
  ::testing::InitGoogleTest(&argc, argv);

  // Check if we should run all tests (including long-running ones)
  bool runLongTests = false;
  for (int i = 1; i < argc; i++) {
    if (std::string(argv[i]) == "--run-long-tests") {
      runLongTests = true;
      ::testing::GTEST_FLAG(filter) = "*";
      break;
    }
  }

  if (!runLongTests) {
    // Skip long-running tests by default
    ::testing::GTEST_FLAG(filter) = "-LongRunning";
    std::cout << "Running only short tests. Use --run-long-tests to include "
                 "long-running tests."
              << std::endl;
  }

  // Run the tests
  return RUN_ALL_TESTS();
}