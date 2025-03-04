#include <iostream>
#include <cassert>
#include <memory>
#include <cmath>
#include <vector>
#include <string>

// Include necessary headers
#include "Material.h"
#include "Units.h"

// Helper function to check if two doubles are approximately equal
bool approxEqual(double a, double b, double epsilon = 1e-6) {
    return std::fabs(a - b) < epsilon;
}

// Test output helper
#define RUN_TEST(test) \
    do { \
        std::cout << "Running test: " << #test << "... "; \
        bool result = test(); \
        std::cout << (result ? "PASSED" : "FAILED") << std::endl; \
        testsPassed += result; \
        testsTotal++; \
    } while(0)

/**
 * Test suite for Material class
 */
class MaterialTestSuite {
public:
    MaterialTestSuite() : testsPassed(0), testsTotal(0) {}
    
    // Run all tests
    bool runAllTests() {
        RUN_TEST(testMaterialConstruction);
        RUN_TEST(testMaterialTypes);
        RUN_TEST(testBaseProperties);
        RUN_TEST(testPropertyModels);
        RUN_TEST(testCustomPropertyFunction);
        RUN_TEST(testTemperatureDependentProperties);
        RUN_TEST(testPredefinedMaterials);
        RUN_TEST(testMaterialMixing);
        RUN_TEST(testReferenceTemperature);
        RUN_TEST(testPropertyNameLookup);
        
        std::cout << "\nMaterial Tests Results: " << testsPassed << " of " << testsTotal << " tests passed." << std::endl;
        return testsPassed == testsTotal;
    }
    
private:
    int testsPassed;
    int testsTotal;
    
    // Test basic material construction
    bool testMaterialConstruction() {
        bool pass = true;
        
        // Default constructor
        Material defaultMaterial;
        pass &= defaultMaterial.getType() == Material::MaterialType::FLUID;
        pass &= defaultMaterial.getName() == "DefaultMaterial";
        pass &= !defaultMaterial.isMixture();
        
        // Parameterized constructor
        Material solidMaterial(Material::MaterialType::SOLID, "TestSolid");
        pass &= solidMaterial.getType() == Material::MaterialType::SOLID;
        pass &= solidMaterial.getName() == "TestSolid";
        pass &= !solidMaterial.isMixture();
        
        // Test properties default to zero
        pass &= solidMaterial.getProperty(Material::MaterialProperty::DENSITY) == 0.0;
        pass &= solidMaterial.getProperty(Material::MaterialProperty::THERMAL_CONDUCTIVITY) == 0.0;
        
        return pass;
    }
    
    // Test material types
    bool testMaterialTypes() {
        bool pass = true;
        
        Material fluid(Material::MaterialType::FLUID, "TestFluid");
        pass &= fluid.getType() == Material::MaterialType::FLUID;
        
        Material solid(Material::MaterialType::SOLID, "TestSolid");
        pass &= solid.getType() == Material::MaterialType::SOLID;
        
        Material interface(Material::MaterialType::INTERFACE, "TestInterface");
        pass &= interface.getType() == Material::MaterialType::INTERFACE;
        
        return pass;
    }
    
    // Test setting and getting base properties
    bool testBaseProperties() {
        bool pass = true;
        
        Material material;
        
        // Test all property types
        for (int i = 0; i < static_cast<int>(Material::MaterialProperty::COUNT); ++i) {
            Material::MaterialProperty prop = static_cast<Material::MaterialProperty>(i);
            double value = 10.0 * (i + 1); // Unique value for each property
            
            material.setProperty(prop, value);
            pass &= approxEqual(material.getProperty(prop), value);
        }
        
        // Specific property tests
        material.setProperty(Material::MaterialProperty::DENSITY, 998.2);
        pass &= approxEqual(material.getProperty(Material::MaterialProperty::DENSITY), 998.2);
        
        material.setProperty(Material::MaterialProperty::THERMAL_CONDUCTIVITY, 0.6);
        pass &= approxEqual(material.getProperty(Material::MaterialProperty::THERMAL_CONDUCTIVITY), 0.6);
        
        material.setProperty(Material::MaterialProperty::SPECIFIC_HEAT, 4180.0);
        pass &= approxEqual(material.getProperty(Material::MaterialProperty::SPECIFIC_HEAT), 4180.0);
        
        return pass;
    }
    
    // Test property models
    bool testPropertyModels() {
        bool pass = true;
        
        Material material;
        
        // Set base property
        material.setProperty(Material::MaterialProperty::DENSITY, 1000.0);
        
        // Set reference temperature
        material.setReferenceTemperature(293.15); // 20°C
        
        // Test constant model (default)
        pass &= approxEqual(
            material.getPropertyAtTemperature(Material::MaterialProperty::DENSITY, 293.15),
            1000.0
        );
        
        pass &= approxEqual(
            material.getPropertyAtTemperature(Material::MaterialProperty::DENSITY, 353.15), // 80°C
            1000.0 // No change with constant model
        );
        
        // Test linear model
        material.setPropertyModel(
            Material::MaterialProperty::DENSITY,
            Material::PropertyModel::LINEAR,
            {-0.0002} // Linear coefficient
        );
        
        // At reference temperature, should match base value
        pass &= approxEqual(
            material.getPropertyAtTemperature(Material::MaterialProperty::DENSITY, 293.15),
            1000.0
        );
        
        // At T = Tref + 50K, should be base * (1 + coeff*deltaT)
        pass &= approxEqual(
            material.getPropertyAtTemperature(Material::MaterialProperty::DENSITY, 343.15),
            1000.0 * (1.0 - 0.0002 * 50.0),
            0.01
        );
        
        // Test polynomial model
        material.setPropertyModel(
            Material::MaterialProperty::THERMAL_CONDUCTIVITY,
            Material::PropertyModel::POLYNOMIAL,
            {0.0015, -0.000001} // Linear and quadratic coefficients
        );
        
        material.setProperty(Material::MaterialProperty::THERMAL_CONDUCTIVITY, 0.6);
        
        // At T = Tref + 50K, should follow polynomial formula
        double deltaT = 50.0;
        double expectedPolynomial = 0.6 * (1.0 + 0.0015 * deltaT - 0.000001 * deltaT * deltaT);
        pass &= approxEqual(
            material.getPropertyAtTemperature(Material::MaterialProperty::THERMAL_CONDUCTIVITY, 343.15),
            expectedPolynomial,
            0.0001
        );
        
        // Test exponential model
        material.setPropertyModel(
            Material::MaterialProperty::DYNAMIC_VISCOSITY,
            Material::PropertyModel::EXPONENTIAL,
            {-0.02} // Exponential coefficient
        );
        
        material.setProperty(Material::MaterialProperty::DYNAMIC_VISCOSITY, 0.001);
        
        // At T = Tref + 50K, should follow exponential formula
        double expectedExponential = 0.001 * std::exp(-0.02 * 50.0);
        pass &= approxEqual(
            material.getPropertyAtTemperature(Material::MaterialProperty::DYNAMIC_VISCOSITY, 343.15),
            expectedExponential,
            0.0001
        );
        
        return pass;
    }
    
    // Test custom property functions
    bool testCustomPropertyFunction() {
        bool pass = true;
        
        Material material;
        material.setProperty(Material::MaterialProperty::DENSITY, 1000.0);
        material.setReferenceTemperature(293.15);
        
        // Define a custom function that gives density as a function of temperature
        // For water, density peaks at 4°C and decreases above and below that
        auto waterDensityFunction = [](double T) -> double {
            // Simplified model: quadratic function with peak at 277.15K (4°C)
            double dT = T - 277.15;
            return 1000.0 - 0.0005 * dT * dT;
        };
        
        material.setCustomPropertyFunction(
            Material::MaterialProperty::DENSITY,
            waterDensityFunction
        );
        
        // At 4°C, density should be 1000.0
        pass &= approxEqual(
            material.getPropertyAtTemperature(Material::MaterialProperty::DENSITY, 277.15),
            1000.0,
            0.01
        );
        
        // At 20°C, density should be less
        double expectedDensity = waterDensityFunction(293.15);
        pass &= approxEqual(
            material.getPropertyAtTemperature(Material::MaterialProperty::DENSITY, 293.15),
            expectedDensity,
            0.01
        );
        
        return pass;
    }
   

    // Test temperature-dependent properties
    bool testTemperatureDependentProperties() {
      bool pass = true;
    
      // Create a material with temperature-dependent viscosity
      Material material;
      material.setProperty(Material::MaterialProperty::DYNAMIC_VISCOSITY, 0.001); // 1 mPa·s at 20°C
      material.setReferenceTemperature(293.15);
    
      // Enable temperature dependence explicitly
      material.setUseTempDependentProps(true);
    
      // Set viscosity to use exponential model (similar to Andrade equation)
      // This decreases viscosity with increasing temperature
      material.setPropertyModel(
        Material::MaterialProperty::DYNAMIC_VISCOSITY,
        Material::PropertyModel::EXPONENTIAL,
        {-0.02} // Coefficient for exponential decrease with temperature
      );
    
      // Get viscosity at different temperatures
      double viscAt20C = material.getPropertyAtTemperature(
        Material::MaterialProperty::DYNAMIC_VISCOSITY, 293.15
      );
    
      double viscAt50C = material.getPropertyAtTemperature(
        Material::MaterialProperty::DYNAMIC_VISCOSITY, 323.15
      );
    
      // Viscosity should decrease with temperature
      pass &= viscAt50C < viscAt20C;
    
      // Check that the ratio is reasonable (viscosity typically drops by ~50% for a 30°C increase)
      double ratio = viscAt50C / viscAt20C;
      pass &= (ratio > 0.3 && ratio < 0.7);
    
      return pass;
    } 
    
    // Test predefined materials
    bool testPredefinedMaterials() {
        bool pass = true;
        
        // Test water
        auto water = Material::createPredefined("water");
        pass &= water->getName() == "Water";
        pass &= water->getType() == Material::MaterialType::FLUID;
        
        // Check water properties
        double waterDensity = water->getProperty(Material::MaterialProperty::DENSITY);
        pass &= approxEqual(waterDensity, 998.2, 0.1);
        
        double waterViscosity = water->getProperty(Material::MaterialProperty::DYNAMIC_VISCOSITY);
        pass &= approxEqual(waterViscosity, 0.001, 0.0001);
        
        // Test air
        auto air = Material::createPredefined("air");
        pass &= air->getName() == "Air";
        pass &= air->getType() == Material::MaterialType::FLUID;
        
        // Check air properties
        double airDensity = air->getProperty(Material::MaterialProperty::DENSITY);
        pass &= approxEqual(airDensity, 1.204, 0.001);
        
        // Test aluminum
        auto aluminum = Material::createPredefined("aluminum");
        pass &= aluminum->getName() == "Aluminum";
        pass &= aluminum->getType() == Material::MaterialType::SOLID;
        
        // Check aluminum properties
        double alDensity = aluminum->getProperty(Material::MaterialProperty::DENSITY);
        pass &= approxEqual(alDensity, 2700.0, 0.1);
        
        double alConductivity = aluminum->getProperty(Material::MaterialProperty::THERMAL_CONDUCTIVITY);
        pass &= approxEqual(alConductivity, 237.0, 0.1);
        
        return pass;
    }
    
    // Test material mixing
    bool testMaterialMixing() {
        bool pass = true;
        
        // Create two materials to mix
        auto water = std::make_shared<Material>(Material::MaterialType::FLUID, "Water");
        water->setProperty(Material::MaterialProperty::DENSITY, 1000.0);
        water->setProperty(Material::MaterialProperty::DYNAMIC_VISCOSITY, 0.001);
        water->setProperty(Material::MaterialProperty::THERMAL_CONDUCTIVITY, 0.6);
        
        auto glycol = std::make_shared<Material>(Material::MaterialType::FLUID, "Glycol");
        glycol->setProperty(Material::MaterialProperty::DENSITY, 1100.0);
        glycol->setProperty(Material::MaterialProperty::DYNAMIC_VISCOSITY, 0.02);
        glycol->setProperty(Material::MaterialProperty::THERMAL_CONDUCTIVITY, 0.3);
        
        // Mix 70% water, 30% glycol
        auto mixture = water->createMixture(glycol, 0.3, "linear");
        
        // Check mixture properties
        pass &= mixture->isMixture();
        pass &= mixture->getName().find("Water") != std::string::npos;
        pass &= mixture->getName().find("Glycol") != std::string::npos;
        
        // Check linear mixing for density
        double expectedDensity = 0.7 * 1000.0 + 0.3 * 1100.0; // Linear mixing
        pass &= approxEqual(
            mixture->getProperty(Material::MaterialProperty::DENSITY),
            expectedDensity,
            0.1
        );
        
        // Check components
        auto components = mixture->getMixtureComponents();
        pass &= components.size() == 2;
        
        // Can't directly check component pointers, but we can check fractions
        bool foundWater = false;
        bool foundGlycol = false;
        
        for (const auto& [material, fraction] : components) {
            if (material->getName() == "Water") {
                pass &= approxEqual(fraction, 0.7, 0.01);
                foundWater = true;
            }
            else if (material->getName() == "Glycol") {
                pass &= approxEqual(fraction, 0.3, 0.01);
                foundGlycol = true;
            }
        }
        
        pass &= foundWater;
        pass &= foundGlycol;
        
        return pass;
    }
    
    // Test reference temperature
    bool testReferenceTemperature() {
        bool pass = true;
        
        Material material;
        
        // Default reference temperature should be around 20°C (293.15K)
        pass &= approxEqual(material.getReferenceTemperature(), 293.15, 0.1);
        
        // Set new reference temperature
        material.setReferenceTemperature(303.15); // 30°C
        pass &= approxEqual(material.getReferenceTemperature(), 303.15);
        
        return pass;
    }
    
    // Test property name lookup
    bool testPropertyNameLookup() {
        bool pass = true;
        
        // Test property name lookup
        std::string densityName = Material::getPropertyName(Material::MaterialProperty::DENSITY);
        pass &= densityName == "Density";
        
        std::string viscosityName = Material::getPropertyName(Material::MaterialProperty::DYNAMIC_VISCOSITY);
        pass &= viscosityName == "DynamicViscosity";
        
        // Test model name lookup
        std::string constModelName = Material::getModelName(Material::PropertyModel::CONSTANT);
        pass &= constModelName == "Constant";
        
        std::string linearModelName = Material::getModelName(Material::PropertyModel::LINEAR);
        pass &= linearModelName == "Linear";
        
        return pass;
    }
};

// Main function
int main() {
    std::cout << "===== Material Class Test Suite =====" << std::endl;
    
    MaterialTestSuite testSuite;
    bool allPassed = testSuite.runAllTests();
    
    if (allPassed) {
        std::cout << "\nAll material tests passed successfully!" << std::endl;
        return 0;
    } else {
        std::cout << "\nSome material tests failed. See above for details." << std::endl;
        return 1;
    }
}

