#include <algorithm>
#include <cassert>
#include <cmath>
#include <initializer_list>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

// Include necessary headers
#include "Cell.h"
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
        RUN_TEST(testReferenceTemperature);
        RUN_TEST(testPropertyNameLookup);
        RUN_TEST(testMaterialUnitConversions);

        std::cout << "\nMaterial Tests Results: " << testsPassed << " of "
                  << testsTotal << " tests passed." << std::endl;
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

    bool testMaterialUnitConversions() {
      bool pass = true;

      // Test with units we know are used in createWithUnits method, which
      // should be supported

      // Test density units - kg/m³ is the SI unit used internally
      auto material = std::make_shared<Material>(Material::MaterialType::FLUID,
                                                 "TestMaterial");

      try {
        // First set a known value in SI units
        material->setProperty(Material::MaterialProperty::DENSITY,
                              1000.0); // 1000 kg/m³
        std::cout << "Set density to 1000 kg/m³" << std::endl;

        // Try to get it in the same units used in createWithUnits
        double density = material->getPropertyWithUnits(
            Material::MaterialProperty::DENSITY, "kg/m³");
        std::cout << "Retrieved density in kg/m³: " << density << std::endl;
        pass &= approxEqual(density, 1000.0);
      } catch (const std::exception &e) {
        std::cout << "Exception when testing density units: " << e.what()
                  << std::endl;
        pass = false;
      }

      // Test viscosity units - Pa·s is the SI unit used internally
      try {
        material->setProperty(Material::MaterialProperty::DYNAMIC_VISCOSITY,
                              0.001); // 0.001 Pa·s
        std::cout << "Set viscosity to 0.001 Pa·s" << std::endl;

        // Try to get it in the same units used in createWithUnits
        double viscosity = material->getPropertyWithUnits(
            Material::MaterialProperty::DYNAMIC_VISCOSITY, "Pa·s");
        std::cout << "Retrieved viscosity in Pa·s: " << viscosity << std::endl;
        pass &= approxEqual(viscosity, 0.001);
      } catch (const std::exception &e) {
        std::cout << "Exception when testing viscosity units: " << e.what()
                  << std::endl;
        pass = false;
      }

      // Test thermal conductivity units - W/(m·K) is the SI unit used
      // internally
      try {
        material->setProperty(Material::MaterialProperty::THERMAL_CONDUCTIVITY,
                              0.6); // 0.6 W/(m·K)
        std::cout << "Set thermal conductivity to 0.6 W/(m·K)" << std::endl;

        // Try to get it in the same units used in createWithUnits
        double conductivity = material->getPropertyWithUnits(
            Material::MaterialProperty::THERMAL_CONDUCTIVITY, "W/(m·K)");
        std::cout << "Retrieved thermal conductivity in W/(m·K): "
                  << conductivity << std::endl;
        pass &= approxEqual(conductivity, 0.6);
      } catch (const std::exception &e) {
        std::cout << "Exception when testing thermal conductivity units: "
                  << e.what() << std::endl;
        pass = false;
      }

      // Test specific heat units - J/(kg·K) is the SI unit used internally
      try {
        material->setProperty(Material::MaterialProperty::SPECIFIC_HEAT,
                              4200.0); // 4200 J/(kg·K)
        std::cout << "Set specific heat to 4200 J/(kg·K)" << std::endl;

        // Try to get it in the same units used in createWithUnits
        double specificHeat = material->getPropertyWithUnits(
            Material::MaterialProperty::SPECIFIC_HEAT, "J/(kg·K)");
        std::cout << "Retrieved specific heat in J/(kg·K): " << specificHeat
                  << std::endl;
        pass &= approxEqual(specificHeat, 4200.0);
      } catch (const std::exception &e) {
        std::cout << "Exception when testing specific heat units: " << e.what()
                  << std::endl;
        pass = false;
      }

      // Test temperature units - K is the SI unit used internally
      try {
        material->setReferenceTemperature(293.15); // 293.15 K (20°C)
        std::cout << "Set reference temperature to 293.15 K" << std::endl;

        // Try to get it in Celsius
        double tempC = material->getReferenceTemperatureWithUnits("C");
        std::cout << "Retrieved temperature in °C: " << tempC << std::endl;
        pass &= approxEqual(tempC, 20.0);

        // Try to get it in Fahrenheit
        double tempF = material->getReferenceTemperatureWithUnits("F");
        std::cout << "Retrieved temperature in °F: " << tempF << std::endl;
        pass &= approxEqual(tempF, 68.0);

        // Try setting in Celsius and getting in Kelvin
        material->setReferenceTemperatureWithUnits(100.0, "C");
        std::cout << "Set reference temperature to 100.0 °C" << std::endl;

        double tempK = material->getReferenceTemperature();
        std::cout << "Retrieved temperature in K: " << tempK << std::endl;
        pass &= approxEqual(tempK, 373.15);
      } catch (const std::exception &e) {
        std::cout << "Exception when testing temperature units: " << e.what()
                  << std::endl;
        pass = false;
      }

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

