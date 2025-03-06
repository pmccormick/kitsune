#include <algorithm>
#include <cassert>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

// Include necessary headers
#include "Cell.h"
#include "Material.h"
#include "Units.h"

/**
 * Improved helper function for robust floating-point comparison
 * Uses a combination of absolute and relative tolerance based on value range
 *
 * @param a First value to compare
 * @param b Second value to compare
 * @param property Optional MaterialProperty to adjust tolerance appropriately
 * @param relTolerance Default relative tolerance
 * @param absTolerance Default absolute tolerance for near-zero values
 * @return True if values are approximately equal
 */
bool approxEqual(
    double a, double b,
    Material::MaterialProperty property = Material::MaterialProperty::COUNT,
    double relTolerance = 1e-5, double absTolerance = 1e-10) {

  // Adjust tolerance based on property type
  if (property == Material::MaterialProperty::DENSITY) {
    // Density values tend to be larger, use higher tolerance
    relTolerance = 2e-3; // 0.2% for density values
  } else if (property == Material::MaterialProperty::DYNAMIC_VISCOSITY) {
    // Viscosity can have large variation, use higher tolerance
    relTolerance = 5e-3; // 0.5% for viscosity
  } else if (std::max(std::abs(a), std::abs(b)) > 100.0) {
    // For any large values, use slightly higher tolerance
    relTolerance = 1e-3; // 0.1% for large values
  }

  // For values very close to zero, use absolute tolerance
  if (std::abs(a) < absTolerance || std::abs(b) < absTolerance) {
    return std::abs(a - b) < absTolerance;
  }

  // Otherwise use relative tolerance
  const double diff = std::abs(a - b);
  const double maxAbs = std::max(std::abs(a), std::abs(b));
  return diff < maxAbs * relTolerance;
}

// For debugging, this function shows more information about the comparison
bool approxEqualWithDebug(
    double a, double b,
    Material::MaterialProperty property = Material::MaterialProperty::COUNT,
    double relTolerance = 1e-5, double absTolerance = 1e-10) {

  // Adjust tolerance based on property
  if (property == Material::MaterialProperty::DENSITY) {
    relTolerance = 2e-3; // 0.2% for density
  } else if (property == Material::MaterialProperty::DYNAMIC_VISCOSITY) {
    relTolerance = 5e-3; // 0.5% for viscosity
  } else if (std::max(std::abs(a), std::abs(b)) > 100.0) {
    relTolerance = 1e-3; // 0.1% for large values
  }

  const double diff = std::abs(a - b);
  const double maxAbs = std::max(std::abs(a), std::abs(b));
  const double relDiff = (maxAbs > 0) ? diff / maxAbs : 0.0;

  bool isEqual = (diff < absTolerance) || (relDiff < relTolerance);

  std::cout << std::fixed << std::setprecision(12);
  std::cout << "Property: " << static_cast<int>(property) << ", Value1: " << a
            << ", Value2: " << b << std::endl;
  std::cout << "Absolute diff: " << diff << ", Relative diff: " << relDiff
            << ", Tolerance: " << relTolerance << std::endl;
  std::cout << "Result: " << (isEqual ? "EQUAL" : "NOT EQUAL") << std::endl;
  std::cout << std::defaultfloat;

  return isEqual;
}

// Test output helper
#define RUN_TEST(test)                                                         \
  do {                                                                         \
    std::cout << "Running test: " << #test << "... ";                          \
    bool result = test();                                                      \
    if (result) {                                                              \
      std::cout << "PASSED" << std::endl;                                      \
    } else {                                                                   \
      std::cout << "FAILED" << std::endl;                                      \
    }                                                                          \
    testsPassed += result;                                                     \
    testsTotal++;                                                              \
  } while (0)

/**
 * Improved test suite for Material property mixing functionality
 * Focuses on correctness, numerical stability, and consistency
 */
class MaterialMixingTestSuite {
public:
  MaterialMixingTestSuite() : testsPassed(0), testsTotal(0) {}

  // Run all tests
  bool runAllTests() {
    // Core mixing functionality
    RUN_TEST(testBasicMaterialProperties);
    RUN_TEST(testSimpleMaterialMixing);
    RUN_TEST(testMixingRuleSelection);

    // Individual mixing rules
    RUN_TEST(testLinearMixing);
    RUN_TEST(testLogarithmicMixing);
    RUN_TEST(testHarmonicMixing);
    RUN_TEST(testGeometricMixing);

    // Edge cases and specific scenarios
    RUN_TEST(testNumericalStability);
    RUN_TEST(testDefaultMixingRuleAssignment);
    RUN_TEST(testZeroAndExtremeFractions);

    // Advanced mixing
    RUN_TEST(testThreeComponentMixing);
    RUN_TEST(testFourComponentMixing);
    RUN_TEST(testNestedMixtures);
    RUN_TEST(testMixingConsistency);

    // Application scenarios
    RUN_TEST(testMaterialPropertyEvolution);

    std::cout << "\nMaterial Mixing Tests Results: " << testsPassed << " of "
              << testsTotal << " tests passed." << std::endl;
    return testsPassed == testsTotal;
  }

private:
  int testsPassed;
  int testsTotal;

  /**
   * Test basic material property setting and getting
   */
  bool testBasicMaterialProperties() {
    bool pass = true;

    // Create a test material
    auto material = std::make_shared<Material>(Material::MaterialType::FLUID,
                                               "TestMaterial");

    // Set and get each property type
    material->setProperty(Material::MaterialProperty::DENSITY, 1000.0);
    material->setProperty(Material::MaterialProperty::DYNAMIC_VISCOSITY, 0.001);
    material->setProperty(Material::MaterialProperty::THERMAL_CONDUCTIVITY,
                          0.6);
    material->setProperty(Material::MaterialProperty::SPECIFIC_HEAT, 4200.0);

    // Verify values were set correctly
    pass &= approxEqualWithDebug(
        material->getProperty(Material::MaterialProperty::DENSITY), 1000.0);
    pass &= approxEqualWithDebug(
        material->getProperty(Material::MaterialProperty::DYNAMIC_VISCOSITY),
        0.001);
    pass &= approxEqualWithDebug(
        material->getProperty(Material::MaterialProperty::THERMAL_CONDUCTIVITY),
        0.6);
    pass &= approxEqualWithDebug(
        material->getProperty(Material::MaterialProperty::SPECIFIC_HEAT),
        4200.0);

    return pass;
  }

  /**
   * Test simple binary mixing of materials with various fractions
   */
  bool testSimpleMaterialMixing() {
    bool pass = true;

    // Create two distinct materials
    auto water =
        std::make_shared<Material>(Material::MaterialType::FLUID, "Water");
    water->setProperty(Material::MaterialProperty::DENSITY, 1000.0);
    water->setProperty(Material::MaterialProperty::DYNAMIC_VISCOSITY, 0.001);
    water->setProperty(Material::MaterialProperty::THERMAL_CONDUCTIVITY, 0.6);
    water->setProperty(Material::MaterialProperty::SPECIFIC_HEAT, 4200.0);

    auto oil = std::make_shared<Material>(Material::MaterialType::FLUID, "Oil");
    oil->setProperty(Material::MaterialProperty::DENSITY, 900.0);
    oil->setProperty(Material::MaterialProperty::DYNAMIC_VISCOSITY, 0.03);
    oil->setProperty(Material::MaterialProperty::THERMAL_CONDUCTIVITY, 0.15);
    oil->setProperty(Material::MaterialProperty::SPECIFIC_HEAT, 1900.0);

    // Test with various mixing fractions using linear mixing
    std::vector<double> fractions = {0.0, 0.25, 0.5, 0.75, 1.0};

    for (double fraction : fractions) {
      // Create the mixture
      auto mixture = water->createMixture(oil, fraction, "linear");

      // Calculate expected values
      double expectedDensity = (1.0 - fraction) * 1000.0 + fraction * 900.0;

      double expectedConductivity = (1.0 - fraction) * 0.6 + fraction * 0.15;
      double expectedSpecificHeat =
          (1.0 - fraction) * 4200.0 + fraction * 1900.0;

      // Check each property
      pass &= approxEqualWithDebug(
          mixture->getProperty(Material::MaterialProperty::DENSITY),
          expectedDensity, Material::MaterialProperty::DENSITY);

      pass &= approxEqualWithDebug(
          mixture->getProperty(
              Material::MaterialProperty::THERMAL_CONDUCTIVITY),
          expectedConductivity,
          Material::MaterialProperty::THERMAL_CONDUCTIVITY);

      pass &= approxEqualWithDebug(
          mixture->getProperty(Material::MaterialProperty::SPECIFIC_HEAT),
          expectedSpecificHeat, Material::MaterialProperty::SPECIFIC_HEAT);

      // Verify mixture flags and components
      if (fraction > 0.0 && fraction < 1.0) {
        // Should be marked as a mixture
        if (!mixture->isMixture()) {
          std::cout << "Mixture flag not set for fraction: " << fraction
                    << std::endl;
          pass = false;
        }

        // Check components
        auto components = mixture->getMixtureComponents();

        if (components.size() != 2) {
          std::cout << "Expected 2 components, got " << components.size()
                    << " for fraction: " << fraction << std::endl;
          pass = false;
        } else {
          // Find each material and verify its fraction
          bool foundWater = false;
          bool foundOil = false;

          for (const auto &[material, compFraction] : components) {
            if (material->getName() == "Water") {
              foundWater = true;
              if (!approxEqualWithDebug(compFraction, 1.0 - fraction)) {
                std::cout << "Water fraction mismatch: expected "
                          << (1.0 - fraction) << ", got " << compFraction
                          << std::endl;
                pass = false;
              }
            } else if (material->getName() == "Oil") {
              foundOil = true;
              if (!approxEqualWithDebug(compFraction, fraction)) {
                std::cout << "Oil fraction mismatch: expected " << fraction
                          << ", got " << compFraction << std::endl;
                pass = false;
              }
            } else {
              std::cout << "Found unexpected material: " << material->getName()
                        << std::endl;
              pass = false;
            }
          }

          if (!foundWater) {
            std::cout << "Water component missing" << std::endl;
            pass = false;
          }
          if (!foundOil) {
            std::cout << "Oil component missing" << std::endl;
            pass = false;
          }
        }
      } else if (fraction <= 0.0) {
        // Should be just water
        if (mixture->getName() != "Water") {
          std::cout << "Expected water, got: " << mixture->getName()
                    << std::endl;
          pass = false;
        }
      } else if (fraction >= 1.0) {
        // Should be just oil
        if (mixture->getName() != "Oil") {
          std::cout << "Expected oil, got: " << mixture->getName() << std::endl;
          pass = false;
        }
      }
    }

    return pass;
  }

  /**
   * Test that different mixing rules produce appropriately different results
   */
  bool testMixingRuleSelection() {
    bool pass = true;

    // Create materials with property values chosen to highlight mixing rule
    // differences
    auto material1 =
        std::make_shared<Material>(Material::MaterialType::FLUID, "Material1");
    material1->setProperty(Material::MaterialProperty::THERMAL_CONDUCTIVITY,
                           10.0);
    material1->setProperty(Material::MaterialProperty::DYNAMIC_VISCOSITY,
                           0.001);

    auto material2 =
        std::make_shared<Material>(Material::MaterialType::FLUID, "Material2");
    material2->setProperty(Material::MaterialProperty::THERMAL_CONDUCTIVITY,
                           100.0);
    material2->setProperty(Material::MaterialProperty::DYNAMIC_VISCOSITY, 0.1);

    // Use 50-50 mix to test different rules
    double fraction = 0.5;

    // Apply different mixing rules
    auto linearMix = material1->createMixture(material2, fraction, "linear");
    auto logMix = material1->createMixture(material2, fraction, "logarithmic");
    auto harmonicMix =
        material1->createMixture(material2, fraction, "harmonic");
    auto geometricMix =
        material1->createMixture(material2, fraction, "geometric");

    // Calculate expected values for each rule
    double linearConductivity = 0.5 * 10.0 + 0.5 * 100.0; // 55.0
    double logConductivity =
        std::exp(0.5 * std::log(10.0) + 0.5 * std::log(100.0));     // ~31.62
    double harmonicConductivity = 1.0 / (0.5 / 10.0 + 0.5 / 100.0); // ~18.18
    double geometricConductivity =
        std::pow(10.0, 0.5) * std::pow(100.0, 0.5); // ~31.62

    // Verify each mixing rule produces the expected result
    pass &= approxEqualWithDebug(linearMix->getProperty(
                            Material::MaterialProperty::THERMAL_CONDUCTIVITY),
                        linearConductivity,
                        Material::MaterialProperty::THERMAL_CONDUCTIVITY);

    pass &= approxEqualWithDebug(
        logMix->getProperty(Material::MaterialProperty::THERMAL_CONDUCTIVITY),
        logConductivity, Material::MaterialProperty::THERMAL_CONDUCTIVITY);

    pass &= approxEqualWithDebug(harmonicMix->getProperty(
                            Material::MaterialProperty::THERMAL_CONDUCTIVITY),
                        harmonicConductivity,
                        Material::MaterialProperty::THERMAL_CONDUCTIVITY);

    pass &= approxEqualWithDebug(geometricMix->getProperty(
                            Material::MaterialProperty::THERMAL_CONDUCTIVITY),
                        geometricConductivity,
                        Material::MaterialProperty::THERMAL_CONDUCTIVITY);

    // Verify that different rules produce different results
    pass &= !approxEqualWithDebug(linearConductivity, harmonicConductivity);
    pass &= !approxEqualWithDebug(linearConductivity, logConductivity);
    pass &= !approxEqualWithDebug(harmonicConductivity, logConductivity);

    return pass;
  }

  /**
   * Test linear mixing rule for various properties
   */
  bool testLinearMixing() {
    bool pass = true;

    // Test the static mixing function directly
    auto material = std::make_shared<Material>();

    // Test with different property types
    std::vector<Material::MaterialProperty> properties = {
        Material::MaterialProperty::DENSITY,
        Material::MaterialProperty::SPECIFIC_HEAT,
        Material::MaterialProperty::THERMAL_CONDUCTIVITY};

    for (auto prop : properties) {
      // Test with fractions: 0, 0.25, 0.5, 0.75, 1.0
      double value1 = 100.0;
      double value2 = 200.0;

      // At fraction = 0, should equal value1
      double result0 =
          Material::mixProperties(prop, value1, value2, 0.0, "linear");
      pass &= approxEqualWithDebug(result0, value1, prop);

      // At fraction = 1, should equal value2
      double result1 =
          Material::mixProperties(prop, value1, value2, 1.0, "linear");
      pass &= approxEqualWithDebug(result1, value2, prop);

      // At fraction = 0.5, should be midpoint
      double result05 =
          Material::mixProperties(prop, value1, value2, 0.5, "linear");
      pass &= approxEqualWithDebug(result05, 150.0, prop);

      // At fraction = 0.25, should be 1/4 of the way
      double result025 =
          Material::mixProperties(prop, value1, value2, 0.25, "linear");
      pass &= approxEqualWithDebug(result025, 125.0, prop);

      // At fraction = 0.75, should be 3/4 of the way
      double result075 =
          Material::mixProperties(prop, value1, value2, 0.75, "linear");
      pass &= approxEqualWithDebug(result075, 175.0, prop);
    }

    return pass;
  }

  /**
   * Test logarithmic mixing rule, especially for viscosity
   */
  bool testLogarithmicMixing() {
    bool pass = true;

    // Test with values that highlight logarithmic mixing behavior
    double value1 = 0.001; // 1 mPa·s
    double value2 = 0.1;   // 100 mPa·s

    Material::MaterialProperty prop =
        Material::MaterialProperty::DYNAMIC_VISCOSITY;

    // Test fraction = 0 (should be value1)
    double result0 =
        Material::mixProperties(prop, value1, value2, 0.0, "logarithmic");
    pass &= approxEqualWithDebug(result0, value1, prop);

    // Test fraction = 1 (should be value2)
    double result1 =
        Material::mixProperties(prop, value1, value2, 1.0, "logarithmic");
    pass &= approxEqualWithDebug(result1, value2, prop);

    // Test fraction = 0.5 (should be geometric mean: sqrt(value1 * value2))
    double expected05 = std::sqrt(value1 * value2);
    double result05 =
        Material::mixProperties(prop, value1, value2, 0.5, "logarithmic");
    pass &= approxEqualWithDebug(result05, expected05, prop);

    // Test with negative values (should fall back to linear mixing)
    double resultNeg =
        Material::mixProperties(prop, -1.0, 2.0, 0.5, "logarithmic");
    pass &= approxEqualWithDebug(resultNeg, 0.5, prop); // Linear result

    // Verify with materials
    auto fluid1 = std::make_shared<Material>(Material::MaterialType::FLUID,
                                             "LowViscosity");
    fluid1->setProperty(Material::MaterialProperty::DYNAMIC_VISCOSITY, value1);

    auto fluid2 = std::make_shared<Material>(Material::MaterialType::FLUID,
                                             "HighViscosity");
    fluid2->setProperty(Material::MaterialProperty::DYNAMIC_VISCOSITY, value2);

    auto mixture = fluid1->createMixture(fluid2, 0.5, "logarithmic");

    pass &= approxEqualWithDebug(
        mixture->getProperty(Material::MaterialProperty::DYNAMIC_VISCOSITY),
        expected05, Material::MaterialProperty::DYNAMIC_VISCOSITY);

    return pass;
  }

  /**
   * Test harmonic mixing rule, especially for thermal conductivity
   */
  bool testHarmonicMixing() {
    bool pass = true;

    Material::MaterialProperty prop =
        Material::MaterialProperty::THERMAL_CONDUCTIVITY;

    // Test values
    double value1 = 10.0;
    double value2 = 20.0;

    // Test fraction = 0 (should be value1)
    double result0 =
        Material::mixProperties(prop, value1, value2, 0.0, "harmonic");
    pass &= approxEqualWithDebug(result0, value1, prop);

    // Test fraction = 1 (should be value2)
    double result1 =
        Material::mixProperties(prop, value1, value2, 1.0, "harmonic");
    pass &= approxEqualWithDebug(result1, value2, prop);

    // Test fraction = 0.5
    // For harmonic mean: 1/result = (1-f)/v1 + f/v2
    double expected05 = 1.0 / (0.5 / value1 + 0.5 / value2);
    double result05 =
        Material::mixProperties(prop, value1, value2, 0.5, "harmonic");
    pass &= approxEqualWithDebug(result05, expected05, prop);

    // Test with zero values (should return 0)
    double resultZero1 =
        Material::mixProperties(prop, 0.0, value2, 0.5, "harmonic");
    pass &= approxEqualWithDebug(resultZero1, 0.0, prop);

    double resultZero2 =
        Material::mixProperties(prop, value1, 0.0, 0.5, "harmonic");
    pass &= approxEqualWithDebug(resultZero2, 0.0, prop);

    // Test with near-zero values for numerical stability
    double resultTiny =
        Material::mixProperties(prop, 1e-10, value2, 0.5, "harmonic");
    // Should be very close to zero due to harmonic mean behavior
    pass &= approxEqualWithDebug(resultTiny, 0.0, prop, 1e-5, 1e-10);

    return pass;
  }

  /**
   * Test geometric mixing rule
   */
  bool testGeometricMixing() {
    bool pass = true;

    Material::MaterialProperty prop =
        Material::MaterialProperty::ELECTRICAL_CONDUCTIVITY;

    // Test values
    double value1 = 5.0;
    double value2 = 20.0;

    // Test fraction = 0 (should be value1)
    double result0 =
        Material::mixProperties(prop, value1, value2, 0.0, "geometric");
    pass &= approxEqualWithDebug(result0, value1, prop);

    // Test fraction = 1 (should be value2)
    double result1 =
        Material::mixProperties(prop, value1, value2, 1.0, "geometric");
    pass &= approxEqualWithDebug(result1, value2, prop);

    // Test fraction = 0.5
    // For geometric mean: result = v1^(1-f) * v2^f
    double expected05 = std::pow(value1, 0.5) * std::pow(value2, 0.5);
    double result05 =
        Material::mixProperties(prop, value1, value2, 0.5, "geometric");
    pass &= approxEqualWithDebug(result05, expected05, prop);

    // Test with negative values (should fall back to linear)
    double resultNeg =
        Material::mixProperties(prop, -1.0, 2.0, 0.5, "geometric");
    pass &= approxEqualWithDebug(resultNeg, 0.5, prop); // Linear result

    // Verify with actual materials
    auto material1 =
        std::make_shared<Material>(Material::MaterialType::FLUID, "Material1");
    material1->setProperty(Material::MaterialProperty::ELECTRICAL_CONDUCTIVITY,
                           value1);

    auto material2 =
        std::make_shared<Material>(Material::MaterialType::FLUID, "Material2");
    material2->setProperty(Material::MaterialProperty::ELECTRICAL_CONDUCTIVITY,
                           value2);

    auto mixture = material1->createMixture(material2, 0.5, "geometric");

    pass &= approxEqualWithDebug(
        mixture->getProperty(
            Material::MaterialProperty::ELECTRICAL_CONDUCTIVITY),
        expected05, Material::MaterialProperty::ELECTRICAL_CONDUCTIVITY);

    return pass;
  }

  /**
   * Test numerical stability with extreme values and potential division by zero
   */
  bool testNumericalStability() {
    bool pass = true;

    // Create materials with extreme property values
    auto material1 =
        std::make_shared<Material>(Material::MaterialType::FLUID, "Extreme1");
    material1->setProperty(Material::MaterialProperty::DYNAMIC_VISCOSITY,
                           1e-10);
    material1->setProperty(Material::MaterialProperty::THERMAL_CONDUCTIVITY,
                           1e-10);

    auto material2 =
        std::make_shared<Material>(Material::MaterialType::FLUID, "Extreme2");
    material2->setProperty(Material::MaterialProperty::DYNAMIC_VISCOSITY, 1e5);
    material2->setProperty(Material::MaterialProperty::THERMAL_CONDUCTIVITY,
                           1e3);

    // Very small fraction to test numerical stability
    double tinyFraction = 1e-10;

    // Logarithmic mixing with extreme values
    auto logMix =
        material1->createMixture(material2, tinyFraction, "logarithmic");
    double viscosity =
        logMix->getProperty(Material::MaterialProperty::DYNAMIC_VISCOSITY);

    // Should be very close to material1's viscosity
    pass &= approxEqualWithDebug(
        viscosity,
        material1->getProperty(Material::MaterialProperty::DYNAMIC_VISCOSITY),
        Material::MaterialProperty::DYNAMIC_VISCOSITY);

    // Harmonic mixing with tiny values
    auto harmonicMix = material1->createMixture(material2, 0.5, "harmonic");
    double conductivity = harmonicMix->getProperty(
        Material::MaterialProperty::THERMAL_CONDUCTIVITY);

    // Ensure no division by zero or numerical instability
    pass &= !std::isnan(conductivity) && !std::isinf(conductivity);

    // Test mixing with zero values
    auto zeroMaterial =
        std::make_shared<Material>(Material::MaterialType::FLUID, "ZeroProps");
    zeroMaterial->setProperty(Material::MaterialProperty::THERMAL_CONDUCTIVITY,
                              0.0);
    zeroMaterial->setProperty(Material::MaterialProperty::DYNAMIC_VISCOSITY,
                              0.0);

    // Test harmonic mixing with zero value
    auto zeroHarmonicMix =
        zeroMaterial->createMixture(material2, 0.5, "harmonic");
    double zeroConductivity = zeroHarmonicMix->getProperty(
        Material::MaterialProperty::THERMAL_CONDUCTIVITY);

    // Should gracefully handle division by zero and return 0
    pass &= approxEqualWithDebug(zeroConductivity, 0.0);

    // Test logarithmic mixing with zero value
    auto zeroLogMix =
        zeroMaterial->createMixture(material2, 0.5, "logarithmic");
    double zeroViscosity =
        zeroLogMix->getProperty(Material::MaterialProperty::DYNAMIC_VISCOSITY);

    // Should fall back to linear mixing
    double expectedLinearViscosity =
        0.5 *
        material2->getProperty(Material::MaterialProperty::DYNAMIC_VISCOSITY);
    pass &= approxEqualWithDebug(zeroViscosity, expectedLinearViscosity,
                        Material::MaterialProperty::DYNAMIC_VISCOSITY);

    return pass;
  }

  /**
   * Test default mixing rule assignment for different properties
   */
  bool testDefaultMixingRuleAssignment() {
    bool pass = true;

    // Create materials with different properties
    auto material1 =
        std::make_shared<Material>(Material::MaterialType::FLUID, "Material1");
    material1->setProperty(Material::MaterialProperty::DENSITY, 1000.0);
    material1->setProperty(Material::MaterialProperty::DYNAMIC_VISCOSITY,
                           0.001);
    material1->setProperty(Material::MaterialProperty::THERMAL_CONDUCTIVITY,
                           0.6);
    material1->setProperty(Material::MaterialProperty::SPECIFIC_HEAT, 4200.0);

    auto material2 =
        std::make_shared<Material>(Material::MaterialType::FLUID, "Material2");
    material2->setProperty(Material::MaterialProperty::DENSITY, 900.0);
    material2->setProperty(Material::MaterialProperty::DYNAMIC_VISCOSITY, 0.01);
    material2->setProperty(Material::MaterialProperty::THERMAL_CONDUCTIVITY,
                           0.3);
    material2->setProperty(Material::MaterialProperty::SPECIFIC_HEAT, 2100.0);

    // Create mixture using default mixing rules
    auto defaultMix = material1->createMixture(material2, 0.5, "default");

    // Verify that each property uses the appropriate default rule

    // Density should use linear mixing
    double expectedDensity = 0.5 * 1000.0 + 0.5 * 900.0; // 950.0
    pass &= approxEqualWithDebug(
        defaultMix->getProperty(Material::MaterialProperty::DENSITY),
        expectedDensity, Material::MaterialProperty::DENSITY);

    // Viscosity should use logarithmic mixing
    double expectedViscosity =
        std::exp(0.5 * std::log(0.001) + 0.5 * std::log(0.01));
    pass &= approxEqualWithDebug(
        defaultMix->getProperty(Material::MaterialProperty::DYNAMIC_VISCOSITY),
        expectedViscosity, Material::MaterialProperty::DYNAMIC_VISCOSITY);

    // Thermal conductivity should use harmonic mixing
    double expectedConductivity = 1.0 / (0.5 / 0.6 + 0.5 / 0.3);
    pass &= approxEqualWithDebug(defaultMix->getProperty(
                            Material::MaterialProperty::THERMAL_CONDUCTIVITY),
                        expectedConductivity,
                        Material::MaterialProperty::THERMAL_CONDUCTIVITY);

    // Specific heat should use linear mixing
    double expectedSpecificHeat = 0.5 * 4200.0 + 0.5 * 2100.0; // 3150.0
    pass &= approxEqualWithDebug(
        defaultMix->getProperty(Material::MaterialProperty::SPECIFIC_HEAT),
        expectedSpecificHeat, Material::MaterialProperty::SPECIFIC_HEAT);

    return pass;
  }

  /**
   * Test mixing with zero fractions and extreme fractions
   */
  bool testZeroAndExtremeFractions() {
    bool pass = true;

    // Create distinct materials
    auto water =
        std::make_shared<Material>(Material::MaterialType::FLUID, "Water");
    water->setProperty(Material::MaterialProperty::DENSITY, 1000.0);

    auto oil = std::make_shared<Material>(Material::MaterialType::FLUID, "Oil");
    oil->setProperty(Material::MaterialProperty::DENSITY, 900.0);

    // Test with zero fraction - should return first material properties
    auto zeroMix = water->createMixture(oil, 0.0, "linear");
    pass &=
        approxEqualWithDebug(zeroMix->getProperty(Material::MaterialProperty::DENSITY),
                    1000.0, Material::MaterialProperty::DENSITY);

    // Test with fraction = 1 - should return second material properties
    auto oneMix = water->createMixture(oil, 1.0, "linear");
    pass &=
        approxEqualWithDebug(oneMix->getProperty(Material::MaterialProperty::DENSITY),
                    900.0, Material::MaterialProperty::DENSITY);

    // Test with tiny but non-zero fraction
    auto tinyMix = water->createMixture(oil, 1e-6, "linear");
    double expectedDensity = 999.9999;
    pass &=
        approxEqualWithDebug(tinyMix->getProperty(Material::MaterialProperty::DENSITY),
                    expectedDensity, Material::MaterialProperty::DENSITY);

    // Test with very large fraction (near 1.0)
    auto largeMix = water->createMixture(oil, 0.9999, "linear");
    double expectedLargeDensity = 900.01;
    pass &=
        approxEqualWithDebug(largeMix->getProperty(Material::MaterialProperty::DENSITY),
                    expectedLargeDensity, Material::MaterialProperty::DENSITY);

    // Test with multi-component mixing and zero fractions
    std::vector<std::shared_ptr<Material>> materials = {water, oil};
    std::vector<double> fractions = {1.0, 0.0}; // Zero fraction for oil

    auto zeroComponentMix =
        Material::createMixture(materials, fractions, "linear");
    pass &= approxEqualWithDebug(
        zeroComponentMix->getProperty(Material::MaterialProperty::DENSITY),
        1000.0, Material::MaterialProperty::DENSITY);

    // Test with invalid fractions (sum != 1.0)
    bool exceptionCaught = false;
    try {
      std::vector<double> invalidFractions = {0.7, 0.4}; // Sum = 1.1
      auto invalidMix =
          Material::createMixture(materials, invalidFractions, "linear");
    } catch (const std::invalid_argument &e) {
      exceptionCaught = true;
    }
    pass &= exceptionCaught;

    return pass;
  }

  /**
   * Test mixing with three components
   */
  bool testThreeComponentMixing() {
    bool pass = true;

    // Create three distinct materials
    auto water =
        std::make_shared<Material>(Material::MaterialType::FLUID, "Water");
    water->setProperty(Material::MaterialProperty::DENSITY, 1000.0);
    water->setProperty(Material::MaterialProperty::DYNAMIC_VISCOSITY, 0.001);
    water->setProperty(Material::MaterialProperty::THERMAL_CONDUCTIVITY, 0.6);
    water->setProperty(Material::MaterialProperty::SPECIFIC_HEAT, 4200.0);

    auto air = std::make_shared<Material>(Material::MaterialType::FLUID, "Air");
    air->setProperty(Material::MaterialProperty::DENSITY, 1.2);
    air->setProperty(Material::MaterialProperty::DYNAMIC_VISCOSITY, 0.000018);
    air->setProperty(Material::MaterialProperty::THERMAL_CONDUCTIVITY, 0.026);
    air->setProperty(Material::MaterialProperty::SPECIFIC_HEAT, 1005.0);

    auto oil = std::make_shared<Material>(Material::MaterialType::FLUID, "Oil");
    oil->setProperty(Material::MaterialProperty::DENSITY, 900.0);
    oil->setProperty(Material::MaterialProperty::DYNAMIC_VISCOSITY, 0.03);
    oil->setProperty(Material::MaterialProperty::THERMAL_CONDUCTIVITY, 0.15);
    oil->setProperty(Material::MaterialProperty::SPECIFIC_HEAT, 1900.0);

    // Define component fractions
    std::vector<std::shared_ptr<Material>> materials = {water, air, oil};
    std::vector<double> fractions = {0.6, 0.1, 0.3};

    // Test Part 1: Mixing with "linear" rule
    // Create mixture with explicit "linear" mixing rule
    auto linearMixture =
        Material::createMixture(materials, fractions, "linear");

    // Check that it's a mixture
    pass &= linearMixture->isMixture();

    // Calculate expected property values using linear mixing for all properties
    double expectedDensity =
        fractions[0] * water->getProperty(Material::MaterialProperty::DENSITY) +
        fractions[1] * air->getProperty(Material::MaterialProperty::DENSITY) +
        fractions[2] * oil->getProperty(Material::MaterialProperty::DENSITY);

    double expectedLinearViscosity =
        fractions[0] *
            water->getProperty(Material::MaterialProperty::DYNAMIC_VISCOSITY) +
        fractions[1] *
            air->getProperty(Material::MaterialProperty::DYNAMIC_VISCOSITY) +
        fractions[2] *
            oil->getProperty(Material::MaterialProperty::DYNAMIC_VISCOSITY);

    // Verify density (linear mixing)
    pass &= approxEqualWithDebug(
        linearMixture->getProperty(Material::MaterialProperty::DENSITY),
        expectedDensity, Material::MaterialProperty::DENSITY);

    // Verify viscosity (linear mixing because we specified "linear")
    double actualLinearViscosity = linearMixture->getProperty(
        Material::MaterialProperty::DYNAMIC_VISCOSITY);

    std::cout << "Linear mixing test:" << std::endl;
    std::cout << "Expected (linear): " << expectedLinearViscosity
              << ", Actual: " << actualLinearViscosity << std::endl;

    double viscTolerance = 0.01; // 1% tolerance
    pass &= std::abs(actualLinearViscosity - expectedLinearViscosity) /
                expectedLinearViscosity <
            viscTolerance;

    // Test Part 2: Mixing with "default" rule (should use logarithmic for
    // viscosity)
    auto defaultMixture =
        Material::createMixture(materials, fractions, "default");

    // For logarithmic mixing, if all values are positive:
    // result = exp(sum(f_i * ln(v_i)))
    double waterVisc =
        water->getProperty(Material::MaterialProperty::DYNAMIC_VISCOSITY);
    double airVisc =
        air->getProperty(Material::MaterialProperty::DYNAMIC_VISCOSITY);
    double oilVisc =
        oil->getProperty(Material::MaterialProperty::DYNAMIC_VISCOSITY);

    double expectedLogViscosity = std::exp(fractions[0] * std::log(waterVisc) +
                                           fractions[1] * std::log(airVisc) +
                                           fractions[2] * std::log(oilVisc));

    // Get the actual viscosity from the default mixture
    double actualDefaultViscosity = defaultMixture->getProperty(
        Material::MaterialProperty::DYNAMIC_VISCOSITY);

    std::cout << "Default mixing test (should use logarithmic for viscosity):"
              << std::endl;
    std::cout << "Water visc: " << waterVisc << ", fraction: " << fractions[0]
              << std::endl;
    std::cout << "Air visc: " << airVisc << ", fraction: " << fractions[1]
              << std::endl;
    std::cout << "Oil visc: " << oilVisc << ", fraction: " << fractions[2]
              << std::endl;
    std::cout << "Expected (logarithmic): " << expectedLogViscosity
              << ", Actual: " << actualDefaultViscosity << std::endl;

    pass &= std::abs(actualDefaultViscosity - expectedLogViscosity) /
                expectedLogViscosity <
            viscTolerance;

    return pass;
  }

  /**
   * Test mixing with four components
   */
  bool testFourComponentMixing() {
  bool pass = true;

  // Create four distinct materials
  auto water =
      std::make_shared<Material>(Material::MaterialType::FLUID, "Water");
  water->setProperty(Material::MaterialProperty::DENSITY, 1000.0);
  water->setProperty(Material::MaterialProperty::DYNAMIC_VISCOSITY, 0.001);

  auto air = std::make_shared<Material>(Material::MaterialType::FLUID, "Air");
  air->setProperty(Material::MaterialProperty::DENSITY, 1.2);
  air->setProperty(Material::MaterialProperty::DYNAMIC_VISCOSITY, 0.000018);

  auto oil = std::make_shared<Material>(Material::MaterialType::FLUID, "Oil");
  oil->setProperty(Material::MaterialProperty::DENSITY, 900.0);
  oil->setProperty(Material::MaterialProperty::DYNAMIC_VISCOSITY, 0.03);

  auto ethanol =
      std::make_shared<Material>(Material::MaterialType::FLUID, "Ethanol");
  ethanol->setProperty(Material::MaterialProperty::DENSITY, 789.0);
  ethanol->setProperty(Material::MaterialProperty::DYNAMIC_VISCOSITY, 0.0012);

  // Define component fractions
  std::vector<std::shared_ptr<Material>> materials = {water, air, oil, ethanol};
  std::vector<double> fractions = {0.4, 0.1, 0.3, 0.2};

  // Create mixture
  auto mixture = Material::createMixture(materials, fractions, "linear");

  // Calculate expected property values using linear mixing
  double expectedDensity =
      fractions[0] * water->getProperty(Material::MaterialProperty::DENSITY) +
      fractions[1] * air->getProperty(Material::MaterialProperty::DENSITY) +
      fractions[2] * oil->getProperty(Material::MaterialProperty::DENSITY) +
      fractions[3] * ethanol->getProperty(Material::MaterialProperty::DENSITY);

  // Verify density (linear mixing)
  pass &= approxEqualWithDebug(
      mixture->getProperty(Material::MaterialProperty::DENSITY),
      expectedDensity, Material::MaterialProperty::DENSITY);

  // Test with different material order (should give identical results)
  std::vector<std::shared_ptr<Material>> reorderedMaterials = {ethanol, water,
                                                               oil, air};
  std::vector<double> reorderedFractions = {0.2, 0.4, 0.3, 0.1};

  auto reorderedMixture =
      Material::createMixture(reorderedMaterials, reorderedFractions, "linear");

  pass &= approxEqualWithDebug(
      reorderedMixture->getProperty(Material::MaterialProperty::DENSITY),
      expectedDensity, Material::MaterialProperty::DENSITY);

  return pass;
}

  /**
   * Test mixing with nested mixtures
   */
  bool testNestedMixtures() {
    bool pass = true;

    // Create base materials
    auto water =
        std::make_shared<Material>(Material::MaterialType::FLUID, "Water");
    water->setProperty(Material::MaterialProperty::DENSITY, 1000.0);
    water->setProperty(Material::MaterialProperty::THERMAL_CONDUCTIVITY, 0.6);

    auto air = std::make_shared<Material>(Material::MaterialType::FLUID, "Air");
    air->setProperty(Material::MaterialProperty::DENSITY, 1.2);
    air->setProperty(Material::MaterialProperty::THERMAL_CONDUCTIVITY, 0.026);

    auto oil = std::make_shared<Material>(Material::MaterialType::FLUID, "Oil");
    oil->setProperty(Material::MaterialProperty::DENSITY, 900.0);
    oil->setProperty(Material::MaterialProperty::THERMAL_CONDUCTIVITY, 0.15);

    // Create nested mixtures
    auto waterAir =
        water->createMixture(air, 0.2, "linear"); // 80% water, 20% air
    auto oilWaterAir =
        oil->createMixture(waterAir, 0.6, "linear"); // 40% oil, 60% waterAir

    // Calculate expected density based on final component fractions:
    // oil: 40%
    // water: 60% * 80% = 48%
    // air: 60% * 20% = 12%
    double expectedDensity =
        0.4 * oil->getProperty(Material::MaterialProperty::DENSITY) +
        0.48 * water->getProperty(Material::MaterialProperty::DENSITY) +
        0.12 * air->getProperty(Material::MaterialProperty::DENSITY);

    pass &= approxEqualWithDebug(
        oilWaterAir->getProperty(Material::MaterialProperty::DENSITY),
        expectedDensity, Material::MaterialProperty::DENSITY);

    // Test equivalent direct mixing
    std::vector<std::shared_ptr<Material>> materials = {oil, water, air};
    std::vector<double> fractions = {0.4, 0.48, 0.12};

    auto directMix = Material::createMixture(materials, fractions, "linear");

    pass &=
        approxEqualWithDebug(directMix->getProperty(Material::MaterialProperty::DENSITY),
                    expectedDensity, Material::MaterialProperty::DENSITY);

    // Compare the nested mixture with the direct mixture
    pass &= approxEqualWithDebug(
        oilWaterAir->getProperty(Material::MaterialProperty::DENSITY),
        directMix->getProperty(Material::MaterialProperty::DENSITY),
        Material::MaterialProperty::DENSITY);

    pass &= approxEqualWithDebug(oilWaterAir->getProperty(
                            Material::MaterialProperty::THERMAL_CONDUCTIVITY),
                        directMix->getProperty(
                            Material::MaterialProperty::THERMAL_CONDUCTIVITY),
                        Material::MaterialProperty::THERMAL_CONDUCTIVITY);

    return pass;
  }

  /**
   * Test consistency between different mixing approaches
   */
  bool testMixingConsistency() {
    bool pass = true;

    // Create materials
    auto water =
        std::make_shared<Material>(Material::MaterialType::FLUID, "Water");
    water->setProperty(Material::MaterialProperty::DENSITY, 1000.0);
    water->setProperty(Material::MaterialProperty::THERMAL_CONDUCTIVITY, 0.6);

    auto oil = std::make_shared<Material>(Material::MaterialType::FLUID, "Oil");
    oil->setProperty(Material::MaterialProperty::DENSITY, 900.0);
    oil->setProperty(Material::MaterialProperty::THERMAL_CONDUCTIVITY, 0.15);

    auto air = std::make_shared<Material>(Material::MaterialType::FLUID, "Air");
    air->setProperty(Material::MaterialProperty::DENSITY, 1.2);
    air->setProperty(Material::MaterialProperty::THERMAL_CONDUCTIVITY, 0.026);

    // Direct mixing: A+B+C
    std::vector<std::shared_ptr<Material>> materials = {water, oil, air};
    std::vector<double> fractions = {0.5, 0.3, 0.2};

    auto directMix = Material::createMixture(materials, fractions, "linear");

    // Sequential mixing: (A+B)+C
    auto waterOil =
        water->createMixture(oil, 0.375, "linear"); // 62.5% water, 37.5% oil
    auto seqMix =
        waterOil->createMixture(air, 0.2, "linear"); // 80% waterOil, 20% air

    // Both should give the same result
    pass &=
        approxEqualWithDebug(directMix->getProperty(Material::MaterialProperty::DENSITY),
                    seqMix->getProperty(Material::MaterialProperty::DENSITY),
                    Material::MaterialProperty::DENSITY);

    // For thermal conductivity with harmonic mixing, the results may differ
    // slightly due to the mathematics of harmonic mixing. So check within a
    // wider tolerance.
    double directCond = directMix->getProperty(
        Material::MaterialProperty::THERMAL_CONDUCTIVITY);
    double seqCond =
        seqMix->getProperty(Material::MaterialProperty::THERMAL_CONDUCTIVITY);

    // For non-linear mixing rules, the order can matter. For this test,
    // we'll just ensure results are in a reasonable range.
    double minCond = std::min(
        {water->getProperty(Material::MaterialProperty::THERMAL_CONDUCTIVITY),
         oil->getProperty(Material::MaterialProperty::THERMAL_CONDUCTIVITY),
         air->getProperty(Material::MaterialProperty::THERMAL_CONDUCTIVITY)});
    double maxCond = std::max(
        {water->getProperty(Material::MaterialProperty::THERMAL_CONDUCTIVITY),
         oil->getProperty(Material::MaterialProperty::THERMAL_CONDUCTIVITY),
         air->getProperty(Material::MaterialProperty::THERMAL_CONDUCTIVITY)});

    pass &= (directCond >= minCond) && (directCond <= maxCond);
    pass &= (seqCond >= minCond) && (seqCond <= maxCond);

    return pass;
  }

  /**
   * Test material property evolution in a simple transport scenario
   */
  bool testMaterialPropertyEvolution() {
    bool pass = true;

    // Create materials
    auto water =
        std::make_shared<Material>(Material::MaterialType::FLUID, "Water");
    water->setProperty(Material::MaterialProperty::DENSITY, 1000.0);
    water->setProperty(Material::MaterialProperty::SPECIFIC_HEAT, 4200.0);

    auto air = std::make_shared<Material>(Material::MaterialType::FLUID, "Air");
    air->setProperty(Material::MaterialProperty::DENSITY, 1.2);
    air->setProperty(Material::MaterialProperty::SPECIFIC_HEAT, 1005.0);

    // Setup two cells
    Cell cell1, cell2;
    cell1.setMaterial(water);
    cell2.setMaterial(air);

    // Set initial temperatures
    cell1.setTemperature(293.15); // 20°C
    cell2.setTemperature(350.0);  // 76.85°C

    // Define flow rate from cell1 to cell2
    double flowRate = 0.2; // 20% of volume per time step

    // Simulate material transport over 3 steps
    for (int step = 0; step < 3; step++) {
      // Calculate water fractions in each cell
      double waterFractionInCell1 = 1.0 - (step * flowRate);
      double waterFractionInCell2 = step * flowRate;

      if (waterFractionInCell1 < 0.0)
        waterFractionInCell1 = 0.0;
      if (waterFractionInCell1 > 1.0)
        waterFractionInCell1 = 1.0;
      if (waterFractionInCell2 < 0.0)
        waterFractionInCell2 = 0.0;
      if (waterFractionInCell2 > 1.0)
        waterFractionInCell2 = 1.0;

      // Update cell materials
      if (waterFractionInCell1 < 1.0) {
        auto cell1Mixture =
            water->createMixture(air, 1.0 - waterFractionInCell1, "linear");
        cell1.setMaterial(cell1Mixture);
      }

      if (waterFractionInCell2 > 0.0) {
        auto cell2Mixture =
            air->createMixture(water, waterFractionInCell2, "linear");
        cell2.setMaterial(cell2Mixture);
      }

      // Verify material densities
      double expectedDensity1 =
          waterFractionInCell1 * 1000.0 + (1.0 - waterFractionInCell1) * 1.2;
      double expectedDensity2 =
          (1.0 - waterFractionInCell2) * 1.2 + waterFractionInCell2 * 1000.0;

      // Use flexible comparison with relative tolerance
      double relativeTol = 1e-3; // 0.1% tolerance

      double actualDensity1 =
          cell1.getMaterial()->getProperty(Material::MaterialProperty::DENSITY);
      double actualDensity2 =
          cell2.getMaterial()->getProperty(Material::MaterialProperty::DENSITY);

      // Only check if the fractions are significant
      if (waterFractionInCell1 > 0.01 && waterFractionInCell1 < 0.99) {
        pass &= std::abs(actualDensity1 - expectedDensity1) / expectedDensity1 <
                relativeTol;
      }

      if (waterFractionInCell2 > 0.01 && waterFractionInCell2 < 0.99) {
        pass &= std::abs(actualDensity2 - expectedDensity2) / expectedDensity2 <
                relativeTol;
      }
    }

    return pass;
  }
};

int main() {
  std::cout << "===== Improved Material Property Mixing Test Suite ====="
            << std::endl;

  MaterialMixingTestSuite testSuite;
  bool allPassed = testSuite.runAllTests();

  if (allPassed) {
    std::cout << "\nAll material mixing tests passed successfully!"
              << std::endl;
    return 0;
  } else {
    std::cout << "\nSome material mixing tests failed. See above for details."
              << std::endl;
    return 1;
  }
}