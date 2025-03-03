#include "Material.h"
#include <gtest/gtest.h>
#include <memory>
#include <cmath>

class MaterialTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create test materials
        water = std::make_shared<Material>(Material::MaterialType::FLUID, "Water");
        water->setProperty(Material::MaterialProperty::DENSITY, 998.0);
        water->setProperty(Material::MaterialProperty::DYNAMIC_VISCOSITY, 0.001);
        water->setProperty(Material::MaterialProperty::THERMAL_CONDUCTIVITY, 0.6);
        water->setProperty(Material::MaterialProperty::SPECIFIC_HEAT, 4182.0);
        
        air = std::make_shared<Material>(Material::MaterialType::FLUID, "Air");
        air->setProperty(Material::MaterialProperty::DENSITY, 1.2);
        air->setProperty(Material::MaterialProperty::DYNAMIC_VISCOSITY, 1.8e-5);
        air->setProperty(Material::MaterialProperty::THERMAL_CONDUCTIVITY, 0.026);
        air->setProperty(Material::MaterialProperty::SPECIFIC_HEAT, 1005.0);
    }

    std::shared_ptr<Material> water;
    std::shared_ptr<Material> air;
};

// Test basic material properties
TEST_F(MaterialTest, BasicProperties) {
    Material material(Material::MaterialType::FLUID, "TestMaterial");
    
    // Test default values
    EXPECT_EQ(material.getType(), Material::MaterialType::FLUID);
    EXPECT_EQ(material.getName(), "TestMaterial");
    EXPECT_DOUBLE_EQ(material.getReferenceTemperature(), 293.15);
    
    // Test default property values (should be zero)
    EXPECT_DOUBLE_EQ(material.getProperty(Material::MaterialProperty::DENSITY), 0.0);
    EXPECT_DOUBLE_EQ(material.getProperty(Material::MaterialProperty::THERMAL_CONDUCTIVITY), 0.0);
    
    // Test setting and getting properties
    material.setProperty(Material::MaterialProperty::DENSITY, 2.5);
    EXPECT_DOUBLE_EQ(material.getProperty(Material::MaterialProperty::DENSITY), 2.5);
    
    material.setProperty(Material::MaterialProperty::THERMAL_CONDUCTIVITY, 0.75);
    EXPECT_DOUBLE_EQ(material.getProperty(Material::MaterialProperty::THERMAL_CONDUCTIVITY), 0.75);
    
    // Test all properties are independent
    for (int i = 0; i < static_cast<int>(Material::MaterialProperty::COUNT); ++i) {
        Material::MaterialProperty prop = static_cast<Material::MaterialProperty>(i);
        double value = i * 10.0 + 5.0;
        material.setProperty(prop, value);
        EXPECT_DOUBLE_EQ(material.getProperty(prop), value);
    }
    
    // Test reference temperature
    material.setReferenceTemperature(300.0);
    EXPECT_DOUBLE_EQ(material.getReferenceTemperature(), 300.0);
}

// Test constant property model
TEST_F(MaterialTest, ConstantPropertyModel) {
    Material material(Material::MaterialType::FLUID, "TestMaterial");
    material.setProperty(Material::MaterialProperty::DENSITY, 1000.0);
    
    // By default, property model is constant
    EXPECT_DOUBLE_EQ(material.getPropertyAtTemperature(Material::MaterialProperty::DENSITY, 293.15), 1000.0);
    EXPECT_DOUBLE_EQ(material.getPropertyAtTemperature(Material::MaterialProperty::DENSITY, 373.15), 1000.0);
    
    // Explicitly set constant model
    material.setPropertyModel(Material::MaterialProperty::DENSITY, Material::PropertyModel::CONSTANT);
    EXPECT_DOUBLE_EQ(material.getPropertyAtTemperature(Material::MaterialProperty::DENSITY, 293.15), 1000.0);
    EXPECT_DOUBLE_EQ(material.getPropertyAtTemperature(Material::MaterialProperty::DENSITY, 373.15), 1000.0);
}

// Test linear property model
TEST_F(MaterialTest, LinearPropertyModel) {
    Material material(Material::MaterialType::FLUID, "TestMaterial");
    material.setReferenceTemperature(300.0);
    material.setProperty(Material::MaterialProperty::DENSITY, 1000.0);
    
    // Set linear model with coefficient of -0.0002 (similar to water)
    material.setPropertyModel(Material::MaterialProperty::DENSITY, Material::PropertyModel::LINEAR, {-0.0002});
    
    // Test at reference temperature
    EXPECT_DOUBLE_EQ(material.getPropertyAtTemperature(Material::MaterialProperty::DENSITY, 300.0), 1000.0);
    
    // Test at other temperatures: value = baseValue * (1 + a*(T-Tref))
    EXPECT_DOUBLE_EQ(material.getPropertyAtTemperature(Material::MaterialProperty::DENSITY, 320.0), 
                     1000.0 * (1.0 + (-0.0002) * (320.0 - 300.0)));
                     
    EXPECT_DOUBLE_EQ(material.getPropertyAtTemperature(Material::MaterialProperty::DENSITY, 280.0), 
                     1000.0 * (1.0 + (-0.0002) * (280.0 - 300.0)));
}

// Test polynomial property model
TEST_F(MaterialTest, PolynomialPropertyModel) {
    Material material(Material::MaterialType::FLUID, "TestMaterial");
    material.setReferenceTemperature(300.0);
    material.setProperty(Material::MaterialProperty::SPECIFIC_HEAT, 4000.0);
    
    // Set polynomial model with coefficients for quadratic variation
    material.setPropertyModel(Material::MaterialProperty::SPECIFIC_HEAT, 
                              Material::PropertyModel::POLYNOMIAL, 
                              {0.0005, -0.000001});
    
    // Test at reference temperature
    EXPECT_DOUBLE_EQ(material.getPropertyAtTemperature(Material::MaterialProperty::SPECIFIC_HEAT, 300.0), 4000.0);
    
    // Test at other temperatures: value = baseValue * (1 + a1*dT + a2*dT^2 + ...)
    double deltaT = 20.0;
    EXPECT_DOUBLE_EQ(material.getPropertyAtTemperature(Material::MaterialProperty::SPECIFIC_HEAT, 320.0), 
                     4000.0 * (1.0 + 0.0005 * deltaT + (-0.000001) * deltaT * deltaT));
                     
    deltaT = -20.0;
    EXPECT_DOUBLE_EQ(material.getPropertyAtTemperature(Material::MaterialProperty::SPECIFIC_HEAT, 280.0), 
                     4000.0 * (1.0 + 0.0005 * deltaT + (-0.000001) * deltaT * deltaT));
}

// Test exponential property model
TEST_F(MaterialTest, ExponentialPropertyModel) {
    Material material(Material::MaterialType::FLUID, "TestMaterial");
    material.setReferenceTemperature(300.0);
    material.setProperty(Material::MaterialProperty::DYNAMIC_VISCOSITY, 0.001);
    
    // Set exponential model with coefficient (similar to water viscosity)
    material.setPropertyModel(Material::MaterialProperty::DYNAMIC_VISCOSITY, 
                             Material::PropertyModel::EXPONENTIAL, 
                             {-0.02});
    
    // Test at reference temperature
    EXPECT_DOUBLE_EQ(material.getPropertyAtTemperature(Material::MaterialProperty::DYNAMIC_VISCOSITY, 300.0), 0.001);
    
    // Test at other temperatures: value = baseValue * exp(a*(T-Tref))
    EXPECT_DOUBLE_EQ(material.getPropertyAtTemperature(Material::MaterialProperty::DYNAMIC_VISCOSITY, 320.0), 
                     0.001 * std::exp(-0.02 * (320.0 - 300.0)));
                     
    EXPECT_DOUBLE_EQ(material.getPropertyAtTemperature(Material::MaterialProperty::DYNAMIC_VISCOSITY, 280.0), 
                     0.001 * std::exp(-0.02 * (280.0 - 300.0)));
}

// Test custom property function
TEST_F(MaterialTest, CustomPropertyFunction) {
    Material material(Material::MaterialType::FLUID, "TestMaterial");
    material.setReferenceTemperature(300.0);
    material.setProperty(Material::MaterialProperty::DENSITY, 1.0);
    
    // Set custom function for ideal gas: rho = rho_ref * (T_ref / T)
    material.setCustomPropertyFunction(Material::MaterialProperty::DENSITY,
                                    [&material](double temp) -> double {
                                        double rho0 = material.getProperty(Material::MaterialProperty::DENSITY);
                                        double T0 = material.getReferenceTemperature();
                                        return rho0 * (T0 / temp);
                                    });
    
    // Test at reference temperature
    EXPECT_DOUBLE_EQ(material.getPropertyAtTemperature(Material::MaterialProperty::DENSITY, 300.0), 1.0);
    
    // Test at other temperatures
    EXPECT_DOUBLE_EQ(material.getPropertyAtTemperature(Material::MaterialProperty::DENSITY, 600.0), 0.5);
    EXPECT_DOUBLE_EQ(material.getPropertyAtTemperature(Material::MaterialProperty::DENSITY, 150.0), 2.0);
}

// Test material mixing
TEST_F(MaterialTest, MaterialMixing) {
    // Simple test with 50% water, 50% air
    auto mixture = water->createMixture(air, 0.5);
    
    // Check mixture properties and type
    EXPECT_TRUE(mixture->isMixture());
    EXPECT_EQ(mixture->getType(), Material::MaterialType::FLUID);
    EXPECT_EQ(mixture->getName(), "Water-Air-Mixture");
    
    // Check mixture components
    const auto& components = mixture->getMixtureComponents();
    EXPECT_EQ(components.size(), 2);
    EXPECT_EQ(components[0].first->getName(), "Water");
    EXPECT_DOUBLE_EQ(components[0].second, 0.5); // 50% water
    EXPECT_EQ(components[1].first->getName(), "Air");
    EXPECT_DOUBLE_EQ(components[1].second, 0.5); // 50% air
    
    // Check mixed properties (with linear mixing)
    double expectedDensity = 0.5 * 998.0 + 0.5 * 1.2; // Linear mix of densities
    EXPECT_DOUBLE_EQ(mixture->getProperty(Material::MaterialProperty::DENSITY), expectedDensity);
    
    double expectedViscosity = 0.5 * 0.001 + 0.5 * 1.8e-5; // Linear mix of viscosities
    EXPECT_DOUBLE_EQ(mixture->getProperty(Material::MaterialProperty::DYNAMIC_VISCOSITY), expectedViscosity);
}

// Test predefined materials
TEST_F(MaterialTest, PredefinedMaterials) {
    // Test creating water
    auto water = Material::createPredefined("water");
    EXPECT_EQ(water->getName(), "Water");
    EXPECT_EQ(water->getType(), Material::MaterialType::FLUID);
    EXPECT_DOUBLE_EQ(water->getProperty(Material::MaterialProperty::DENSITY), 998.2);
    EXPECT_DOUBLE_EQ(water->getProperty(Material::MaterialProperty::DYNAMIC_VISCOSITY), 1.0016e-3);
    
    // Test temperature dependence of water
    double baseViscosity = water->getProperty(Material::MaterialProperty::DYNAMIC_VISCOSITY);
    double hotViscosity = water->getPropertyAtTemperature(Material::MaterialProperty::DYNAMIC_VISCOSITY, 353.15); // 80°C
    EXPECT_LT(hotViscosity, baseViscosity); // Viscosity decreases with temperature
    
    // Test creating air
    auto air = Material::createPredefined("air");
    EXPECT_EQ(air->getName(), "Air");
    EXPECT_EQ(air->getType(), Material::MaterialType::FLUID);
    EXPECT_DOUBLE_EQ(air->getProperty(Material::MaterialProperty::DENSITY), 1.204);
    
    // Test temperature dependence of air density (should decrease with temperature)
    double baseDensity = air->getProperty(Material::MaterialProperty::DENSITY);
    double hotDensity = air->getPropertyAtTemperature(Material::MaterialProperty::DENSITY, 353.15); // 80°C
    EXPECT_LT(hotDensity, baseDensity); // Density decreases with temperature
    
    // Test creating solid material
    auto aluminum = Material::createPredefined("aluminum");
    EXPECT_EQ(aluminum->getName(), "Aluminum");
    EXPECT_EQ(aluminum->getType(), Material::MaterialType::SOLID);
    EXPECT_DOUBLE_EQ(aluminum->getProperty(Material::MaterialProperty::DENSITY), 2700.0);
    EXPECT_DOUBLE_EQ(aluminum->getProperty(Material::MaterialProperty::THERMAL_CONDUCTIVITY), 237.0);
}

// Test different mixing rules
TEST_F(MaterialTest, MixingRules) {
    // Create materials with very different viscosities to better test rules
    auto material1 = std::make_shared<Material>(Material::MaterialType::FLUID, "Material1");
    material1->setProperty(Material::MaterialProperty::DYNAMIC_VISCOSITY, 0.001);
    
    auto material2 = std::make_shared<Material>(Material::MaterialType::FLUID, "Material2");
    material2->setProperty(Material::MaterialProperty::DYNAMIC_VISCOSITY, 0.1);
    
    // Test linear mixing (50/50)
    auto linearMix = material1->createMixture(material2, 0.5, "linear");
    EXPECT_DOUBLE_EQ(linearMix->getProperty(Material::MaterialProperty::DYNAMIC_VISCOSITY),
                    0.5 * 0.001 + 0.5 * 0.1);
    
    // Test logarithmic mixing (50/50)
    auto logMix = material1->createMixture(material2, 0.5, "logarithmic");
    EXPECT_DOUBLE_EQ(logMix->getProperty(Material::MaterialProperty::DYNAMIC_VISCOSITY),
                    std::exp(0.5 * std::log(0.001) + 0.5 * std::log(0.1)));
    
    // Test harmonic mixing (50/50)
    auto harmonicMix = material1->createMixture(material2, 0.5, "harmonic");
    EXPECT_DOUBLE_EQ(harmonicMix->getProperty(Material::MaterialProperty::DYNAMIC_VISCOSITY),
                    1.0 / (0.5 / 0.001 + 0.5 / 0.1));
    
    // Test min mixing (should take minimum of the two)
    auto minMix = material1->createMixture(material2, 0.5, "min");
    EXPECT_DOUBLE_EQ(minMix->getProperty(Material::MaterialProperty::DYNAMIC_VISCOSITY), 0.001);
    
    // Test max mixing (should take maximum of the two)
    auto maxMix = material1->createMixture(material2, 0.5, "max");
    EXPECT_DOUBLE_EQ(maxMix->getProperty(Material::MaterialProperty::DYNAMIC_VISCOSITY), 0.1);
}

// Test mixing with different fractions
TEST_F(MaterialTest, MixingFractions) {
    double waterDensity = water->getProperty(Material::MaterialProperty::DENSITY);
    double airDensity = air->getProperty(Material::MaterialProperty::DENSITY);
    
    // Test 25% water, 75% air
    auto mix1 = water->createMixture(air, 0.75);
    EXPECT_DOUBLE_EQ(mix1->getProperty(Material::MaterialProperty::DENSITY),
                    0.25 * waterDensity + 0.75 * airDensity);
    
    // Test 75% water, 25% air
    auto mix2 = water->createMixture(air, 0.25);
    EXPECT_DOUBLE_EQ(mix2->getProperty(Material::MaterialProperty::DENSITY),
                    0.75 * waterDensity + 0.25 * airDensity);
    
    // Test boundary cases
    // 100% water, 0% air
    auto mix3 = water->createMixture(air, 0.0);
    EXPECT_DOUBLE_EQ(mix3->getProperty(Material::MaterialProperty::DENSITY), waterDensity);
    
    // 0% water, 100% air
    auto mix4 = water->createMixture(air, 1.0);
    EXPECT_DOUBLE_EQ(mix4->getProperty(Material::MaterialProperty::DENSITY), airDensity);
}

// Test complex mixtures (mixture of mixtures)
TEST_F(MaterialTest, ComplexMixtures) {
    // Create initial mixtures
    auto waterAir = water->createMixture(air, 0.5); // 50% water, 50% air
    
    // Create a third material
    auto oil = std::make_shared<Material>(Material::MaterialType::FLUID, "Oil");
    oil->setProperty(Material::MaterialProperty::DENSITY, 900.0);
    
    // Create mixture of mixture with new material
    auto complexMix = waterAir->createMixture(oil, 0.3); // 30% oil, 70% waterAir
    
    // Check the complex mixture
    EXPECT_TRUE(complexMix->isMixture());
    
    // Expected density calculation:
    // - waterAir is 50% water (998.0), 50% air (1.2) -> density = 499.6
    // - complexMix is 70% waterAir (499.6), 30% oil (900.0)
    // - Expected density = 0.7 * 499.6 + 0.3 * 900.0 = 349.72 + 270.0 = 619.72
    double expectedDensity = 0.7 * (0.5 * 998.0 + 0.5 * 1.2) + 0.3 * 900.0;
    EXPECT_DOUBLE_EQ(complexMix->getProperty(Material::MaterialProperty::DENSITY), expectedDensity);
    
    // Check components - should have 3 components with appropriate fractions
    const auto& components = complexMix->getMixtureComponents();
    EXPECT_EQ(components.size(), 3);
   
   // Check components - should have 3 components with appropriate fractions
    const auto& components = complexMix->getMixtureComponents();
    EXPECT_EQ(components.size(), 3);

    // Component fractions should be:
    // - water: 0.7 * 0.5 = 0.35 (35%)
    // - air: 0.7 * 0.5 = 0.35 (35%)
    // - oil: 0.3 (30%)

    // Find each component by name and verify its fraction
    for (const auto& comp : components) {
        if (comp.first->getName() == "Water") {
            EXPECT_NEAR(comp.second, 0.35, 1e-6);
        }
        else if (comp.first->getName() == "Air") {
            EXPECT_NEAR(comp.second, 0.35, 1e-6);
        }
        else if (comp.first->getName() == "Oil") {
            EXPECT_NEAR(comp.second, 0.3, 1e-6);
        }
    }
}

// Main function to run all tests
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
