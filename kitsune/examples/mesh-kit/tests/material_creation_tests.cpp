#include "Material.h"
#include "Units.h"
#include <gtest/gtest.h>
#include <memory>
#include <vector>

class MaterialFactoryTest : public ::testing::Test {
protected:
  void SetUp() override {
    // Nothing needed here
  }
};

// Test creating a material with createWithUnits
TEST_F(MaterialFactoryTest, CreateWithUnits) {
  // Create a material with non-SI units
  auto material =
      Material::createWithUnits(Material::MaterialType::FLUID, "Test Material",
                                0.9,              // Density
                                "g/cm³",          // Density units
                                10.0,             // Dynamic viscosity
                                "cP",             // Viscosity units
                                0.3,              // Thermal conductivity
                                "BTU/(hr·ft·°F)", // Conductivity units
                                0.5,              // Specific heat
                                "BTU/(lb·°F)",    // Specific heat units
                                68.0,             // Reference temperature
                                "F"               // Temperature units
      );

  // Verify that all units were converted correctly to SI
  EXPECT_NEAR(material->getProperty(Material::MaterialProperty::DENSITY), 900.0,
              1e-6)
      << "Density should be converted from g/cm³ to kg/m³";

  EXPECT_NEAR(
      material->getProperty(Material::MaterialProperty::DYNAMIC_VISCOSITY),
      0.01, 1e-6)
      << "Viscosity should be converted from cP to Pa·s";

  EXPECT_NEAR(
      material->getProperty(Material::MaterialProperty::THERMAL_CONDUCTIVITY),
      Units::btuToWmk(0.3), 1e-6)
      << "Conductivity should be converted from BTU/(hr·ft·°F) to W/(m·K)";

  EXPECT_NEAR(material->getProperty(Material::MaterialProperty::SPECIFIC_HEAT),
              Units::btuToJkgk(0.5), 1e-6)
      << "Specific heat should be converted from BTU/(lb·°F) to J/(kg·K)";

  EXPECT_NEAR(material->getReferenceTemperature(),
              Units::fahrenheitToKelvin(68.0), 1e-6)
      << "Temperature should be converted from F to K";
}

// Test material factory for advanced mixtures
TEST_F(MaterialFactoryTest, CreateMixture) {
  // Create component materials
  auto water = Material::createPredefined("water");
  auto air = Material::createPredefined("air");
  auto oil = Material::createPredefined("oil");

  // Create multi-component mixture
  std::vector<std::shared_ptr<Material>> components = {water, air, oil};
  std::vector<double> fractions = {0.6, 0.1, 0.3};

  auto mixture = Material::createMixture(components, fractions);

  // Check basic properties
  EXPECT_TRUE(mixture->isMixture()) << "Result should be a mixture";
  EXPECT_EQ(mixture->getType(), Material::MaterialType::FLUID)
      << "Mixture of fluids should be fluid";

  // Verify mixture name contains all components
  std::string name = mixture->getName();
  EXPECT_NE(name.find("Water"), std::string::npos)
      << "Mixture name should contain Water";
  EXPECT_NE(name.find("Air"), std::string::npos)
      << "Mixture name should contain Air";
  EXPECT_NE(name.find("Oil"), std::string::npos)
      << "Mixture name should contain Oil";

  // Verify mixture components
  auto mixComponents = mixture->getMixtureComponents();
  EXPECT_EQ(mixComponents.size(), 3) << "Mixture should have 3 components";

  // Check normalized fractions sum to 1.0
  double sum = 0.0;
  for (const auto &comp : mixComponents) {
    sum += comp.second;
  }
  EXPECT_NEAR(sum, 1.0, 1e-6) << "Component fractions should sum to 1.0";

  // Verify that mixture properties are weighted correctly
  double expectedDensity =
      water->getProperty(Material::MaterialProperty::DENSITY) * 0.6 +
      air->getProperty(Material::MaterialProperty::DENSITY) * 0.1 +
      oil->getProperty(Material::MaterialProperty::DENSITY) * 0.3;

  EXPECT_NEAR(mixture->getProperty(Material::MaterialProperty::DENSITY),
              expectedDensity, 1.0)
      << "Mixture density should be weighted average";
}

// Test boundary cases for mixture creation
TEST_F(MaterialFactoryTest, MixtureBoundaryCases) {
  auto water = Material::createPredefined("water");
  auto oil = Material::createPredefined("oil");

  // Test mixture with 0% fraction (should return pure water)
  auto mixture1 = water->createMixture(oil, 0.0);
  EXPECT_FALSE(mixture1->isMixture())
      << "0% mixture should be pure first component";
  EXPECT_EQ(mixture1->getProperty(Material::MaterialProperty::DENSITY),
            water->getProperty(Material::MaterialProperty::DENSITY))
      << "0% mixture should have same density as first component";

  // Test mixture with 100% fraction (should return pure oil)
  auto mixture2 = water->createMixture(oil, 1.0);
  EXPECT_FALSE(mixture2->isMixture())
      << "100% mixture should be pure second component";
  EXPECT_EQ(mixture2->getProperty(Material::MaterialProperty::DENSITY),
            oil->getProperty(Material::MaterialProperty::DENSITY))
      << "100% mixture should have same density as second component";

  // Test invalid fractions
  auto mixture3 = water->createMixture(oil, -0.5); // Should clamp to 0.0
  EXPECT_EQ(mixture3->getProperty(Material::MaterialProperty::DENSITY),
            water->getProperty(Material::MaterialProperty::DENSITY))
      << "Negative fraction should be clamped to 0.0";

  auto mixture4 = water->createMixture(oil, 1.5); // Should clamp to 1.0
  EXPECT_EQ(mixture4->getProperty(Material::MaterialProperty::DENSITY),
            oil->getProperty(Material::MaterialProperty::DENSITY))
      << "Fraction > 1.0 should be clamped to 1.0";
}

// Test createMixture with multiple component vector but only one non-zero
// component
TEST_F(MaterialFactoryTest, SingleComponentMixture) {
  auto water = Material::createPredefined("water");
  auto air = Material::createPredefined("air");
  auto oil = Material::createPredefined("oil");

  // Only water has non-zero fraction
  std::vector<std::shared_ptr<Material>> components = {water, air, oil};
  std::vector<double> fractions = {1.0, 0.0, 0.0};

  auto mixture = Material::createMixture(components, fractions);

  // Should behave like pure water
  EXPECT_FALSE(mixture->isMixture())
      << "Single component mixture should not be marked as mixture";
  EXPECT_EQ(mixture->getProperty(Material::MaterialProperty::DENSITY),
            water->getProperty(Material::MaterialProperty::DENSITY))
      << "Single component mixture should have same properties as component";
}

// Test nested mixtures
TEST_F(MaterialFactoryTest, NestedMixtures) {
  auto water = Material::createPredefined("water");
  auto air = Material::createPredefined("air");
  auto oil = Material::createPredefined("oil");

  // Create first mixture: 70% water, 30% air
  auto mixture1 = Material::createMixture({water, air}, {0.7, 0.3});

  // Create nested mixture: 60% mixture1, 40% oil
  auto mixture2 = Material::createMixture({mixture1, oil}, {0.6, 0.4});

  // Verify that mixture2 has all three original components
  auto components = mixture2->getMixtureComponents();

  // Would need to compute expected fractions: 42% water, 18% air, 40% oil
  double waterFraction = 0.7 * 0.6; // 42%
  double airFraction = 0.3 * 0.6;   // 18%
  double oilFraction = 0.4;         // 40%

  double expectedDensity =
      water->getProperty(Material::MaterialProperty::DENSITY) * waterFraction +
      air->getProperty(Material::MaterialProperty::DENSITY) * airFraction +
      oil->getProperty(Material::MaterialProperty::DENSITY) * oilFraction;

  EXPECT_NEAR(mixture2->getProperty(Material::MaterialProperty::DENSITY),
              expectedDensity, 1.0)
      << "Nested mixture should properly combine all components";
}

// Test mixing rules compatibility with property types
TEST_F(MaterialFactoryTest, MixingRuleCompatibility) {
  auto water = Material::createPredefined("water");
  auto oil = Material::createPredefined("oil");

  // Test that mixing rules are applied correctly for different properties

  // Linear mixing for density (default for most properties)
  auto linearMix = Material::createMixture({water, oil}, {0.5, 0.5}, "linear");
  double waterDensity = water->getProperty(Material::MaterialProperty::DENSITY);
  double oilDensity = oil->getProperty(Material::MaterialProperty::DENSITY);
  double mixDensity =
      linearMix->getProperty(Material::MaterialProperty::DENSITY);

  EXPECT_NEAR(mixDensity, (waterDensity + oilDensity) / 2.0, 1.0)
      << "Density should use linear mixing";

  // Logarithmic mixing for viscosity
  double waterVisc =
      water->getProperty(Material::MaterialProperty::DYNAMIC_VISCOSITY);
  double oilVisc =
      oil->getProperty(Material::MaterialProperty::DYNAMIC_VISCOSITY);
  double mixVisc =
      linearMix->getProperty(Material::MaterialProperty::DYNAMIC_VISCOSITY);

  // Viscosity often uses logarithmic mixing by default
  double logMixVisc = exp(0.5 * log(waterVisc) + 0.5 * log(oilVisc));
  EXPECT_NEAR(mixVisc, logMixVisc, 1e-5)
      << "Viscosity should use logarithmic mixing by default";

  // Harmonic mixing for thermal conductivity
  double waterK =
      water->getProperty(Material::MaterialProperty::THERMAL_CONDUCTIVITY);
  double oilK =
      oil->getProperty(Material::MaterialProperty::THERMAL_CONDUCTIVITY);
  double mixK =
      linearMix->getProperty(Material::MaterialProperty::THERMAL_CONDUCTIVITY);

  // Thermal conductivity often uses harmonic mixing by default
  double harmonicK = 2.0 / (1.0 / waterK + 1.0 / oilK);
  EXPECT_NEAR(mixK, harmonicK, 1e-5)
      << "Thermal conductivity should use harmonic mixing by default";
}

// Test temperature-dependent properties in mixtures
TEST_F(MaterialFactoryTest, MixtureTemperatureDependence) {
  auto water = Material::createPredefined("water");
  auto oil = Material::createPredefined("oil");

  // Create mixture
  auto mixture = Material::createMixture({water, oil}, {0.7, 0.3});

  // Test that the mixture has temperature-dependent properties
  EXPECT_TRUE(mixture->isUsingTempDependentProps())
      << "Mixture should inherit temperature dependence";

  // Check that properties vary with temperature
  double refDensity = mixture->getProperty(Material::MaterialProperty::DENSITY);
  double hotDensity = mixture->getPropertyAtTemperature(
      Material::MaterialProperty::DENSITY, 353.15); // 80°C

  EXPECT_NE(refDensity, hotDensity)
      << "Mixture density should change with temperature";
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}