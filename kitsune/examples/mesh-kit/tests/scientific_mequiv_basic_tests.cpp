/**
 * @file scientific_mequiv_basic_tests.cpp
 * @brief Basic tests for Scientific Material Equivalence implementation
 */
#include "ScientificMaterialEquivalence.h"
#include "Material.h"
#include "TestMaterialHelpers.h"
#include <gtest/gtest.h>
#include <memory>
#include <iostream>

// Test fixture for basic Scientific Equivalence key tests
class ScientificEquivBasicTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create default scientific equivalence with standard settings
        key = std::make_shared<ScientificMaterialEquivalence>();
    }

    // Helper method to create a simple test material
    std::shared_ptr<Material> createSimpleMaterial(
            double density = 1000.0,
            double viscosity = 0.001,
            double thermalConductivity = 0.5) {
        
        auto material = std::make_shared<Material>(Material::MaterialType::FLUID, "Test Material");
        material->setProperty(Material::MaterialProperty::DENSITY, density);
        material->setProperty(Material::MaterialProperty::DYNAMIC_VISCOSITY, viscosity);
        material->setProperty(Material::MaterialProperty::THERMAL_CONDUCTIVITY, thermalConductivity);
        return material;
    }
    
    std::shared_ptr<ScientificMaterialEquivalence> key;
};

// Test basic hash generation
TEST_F(ScientificEquivBasicTest, BasicHash) {
    auto material = createSimpleMaterial();
    
    // Hash should be non-zero
    uint64_t hash = key->hash(material);
    EXPECT_NE(hash, 0ULL) << "Hash should not be zero";
    
    // Hash should be consistent
    EXPECT_EQ(hash, key->hash(material)) << "Hash should be consistent for the same material";
    
    // Different materials should have different hashes
    auto differentMaterial = createSimpleMaterial(1100.0, 0.002, 0.6);
    EXPECT_NE(hash, key->hash(differentMaterial)) << "Different materials should have different hashes";
}

// Test basic equivalence checks
TEST_F(ScientificEquivBasicTest, BasicEquivalence) {
    auto material1 = createSimpleMaterial();
    auto material2 = createSimpleMaterial();
    
    // Identical materials should be equivalent
    EXPECT_TRUE(key->areEquivalent(material1, material2)) << "Identical materials should be equivalent";
    
    // Same instance should be equivalent to itself
    EXPECT_TRUE(key->areEquivalent(material1, material1)) << "Same material instance should be equivalent to itself";
    
    // Different materials should not be equivalent
    auto differentMaterial = createSimpleMaterial(1100.0, 0.002, 0.6);
    EXPECT_FALSE(key->areEquivalent(material1, differentMaterial)) << "Different materials should not be equivalent";
}

// Test small variations within tolerance
TEST_F(ScientificEquivBasicTest, ToleranceTests) {
    auto baseMaterial = createSimpleMaterial();
    
    // Create a material with a very small variation (likely within tolerance)
    auto slightVariation = createSimpleMaterial(
        baseMaterial->getProperty(Material::MaterialProperty::DENSITY) * 1.0001,
        baseMaterial->getProperty(Material::MaterialProperty::DYNAMIC_VISCOSITY),
        baseMaterial->getProperty(Material::MaterialProperty::THERMAL_CONDUCTIVITY)
    );
    
    // Check if small variation is within default tolerance
    bool areEquiv = key->areEquivalent(baseMaterial, slightVariation);
    std::cout << "Materials with 0.01% density difference are " 
              << (areEquiv ? "equivalent" : "not equivalent") 
              << " with default tolerance" << std::endl;
    
    // Create a material with a larger variation (likely outside tolerance)
    auto largerVariation = createSimpleMaterial(
        baseMaterial->getProperty(Material::MaterialProperty::DENSITY) * 1.02,
        baseMaterial->getProperty(Material::MaterialProperty::DYNAMIC_VISCOSITY),
        baseMaterial->getProperty(Material::MaterialProperty::THERMAL_CONDUCTIVITY)
    );
    
    // Materials with larger variations should not be equivalent
    EXPECT_FALSE(key->areEquivalent(baseMaterial, largerVariation)) 
        << "Materials with 2% density difference should not be equivalent";
}

// Test with standard materials
TEST_F(ScientificEquivBasicTest, StandardMaterials) {
    // Create standard materials using the helpers
    auto water = TestMaterialHelpers::createWaterMaterial();
    auto aluminum = TestMaterialHelpers::createAluminumMaterial();
    
    // Different material types should not be equivalent
    EXPECT_FALSE(key->areEquivalent(water, aluminum)) 
        << "Water and aluminum should not be equivalent";
        
    // Test hash for standard materials
    uint64_t waterHash = key->hash(water);
    uint64_t aluminumHash = key->hash(aluminum);
    EXPECT_NE(waterHash, aluminumHash) << "Water and aluminum should have different hashes";
    
    // Create a second instance of the same material
    auto water2 = TestMaterialHelpers::createWaterMaterial();
    
    // Two instances of the same material should be equivalent
    EXPECT_TRUE(key->areEquivalent(water, water2)) 
        << "Two instances of water should be equivalent";
        
    // Two instances of the same material should have the same hash
    EXPECT_EQ(waterHash, key->hash(water2)) 
        << "Two instances of water should have the same hash";
}

// Test empty/missing properties
TEST_F(ScientificEquivBasicTest, EmptyProperties) {
    // Create a material with only one property set
    auto sparseProperties = std::make_shared<Material>(Material::MaterialType::FLUID, "Sparse Material");
    sparseProperties->setProperty(Material::MaterialProperty::DENSITY, 1000.0);
    
    // Create another material with different properties set
    auto differentProperties = std::make_shared<Material>(Material::MaterialType::FLUID, "Different Properties");
    differentProperties->setProperty(Material::MaterialProperty::THERMAL_CONDUCTIVITY, 0.5);
    
    // Create material with all standard properties
    auto fullMaterial = createSimpleMaterial();
    
    // Test hash generation for sparse materials
    uint64_t sparseHash = key->hash(sparseProperties);
    uint64_t diffHash = key->hash(differentProperties);
    
    EXPECT_NE(sparseHash, 0ULL) << "Hash should be generated even with sparse properties";
    EXPECT_NE(sparseHash, diffHash) << "Materials with different sparse properties should have different hashes";
    
    // Materials with different properties set should not be equivalent
    EXPECT_FALSE(key->areEquivalent(sparseProperties, differentProperties)) 
        << "Materials with different sparse properties should not be equivalent";
        
    // Full material should not be equivalent to sparse material
    EXPECT_FALSE(key->areEquivalent(fullMaterial, sparseProperties)) 
        << "Full material should not be equivalent to sparse material";
}

// Test effect of different tolerance settings (if supported)
TEST_F(ScientificEquivBasicTest, ToleranceEffects) {
    auto baseMaterial = createSimpleMaterial();
    
    // Create materials with increasing differences in density
    auto diff0_1pct = createSimpleMaterial(
        baseMaterial->getProperty(Material::MaterialProperty::DENSITY) * 1.001, // 0.1% difference
        baseMaterial->getProperty(Material::MaterialProperty::DYNAMIC_VISCOSITY),
        baseMaterial->getProperty(Material::MaterialProperty::THERMAL_CONDUCTIVITY)
    );
    
    auto diff1pct = createSimpleMaterial(
        baseMaterial->getProperty(Material::MaterialProperty::DENSITY) * 1.01, // 1% difference
        baseMaterial->getProperty(Material::MaterialProperty::DYNAMIC_VISCOSITY),
        baseMaterial->getProperty(Material::MaterialProperty::THERMAL_CONDUCTIVITY)
    );
    
    auto diff5pct = createSimpleMaterial(
        baseMaterial->getProperty(Material::MaterialProperty::DENSITY) * 1.05, // 5% difference
        baseMaterial->getProperty(Material::MaterialProperty::DYNAMIC_VISCOSITY),
        baseMaterial->getProperty(Material::MaterialProperty::THERMAL_CONDUCTIVITY)
    );
    
    // Test with default key
    bool equiv0_1pct = key->areEquivalent(baseMaterial, diff0_1pct);
    bool equiv1pct = key->areEquivalent(baseMaterial, diff1pct);
    bool equiv5pct = key->areEquivalent(baseMaterial, diff5pct);
    
    std::cout << "Materials with density differences compared to default (water-like) material:" << std::endl;
    std::cout << "  - 0.1% difference: " << (equiv0_1pct ? "Equivalent" : "Not equivalent") << std::endl;
    std::cout << "  - 1% difference: " << (equiv1pct ? "Equivalent" : "Not equivalent") << std::endl;
    std::cout << "  - 5% difference: " << (equiv5pct ? "Equivalent" : "Not equivalent") << std::endl;
    
    // We expect smaller differences to be more likely to be considered equivalent
    if (equiv5pct) {
        EXPECT_TRUE(equiv1pct) << "If 5% difference is equivalent, 1% should also be equivalent";
    }
    if (equiv1pct) {
        EXPECT_TRUE(equiv0_1pct) << "If 1% difference is equivalent, 0.1% should also be equivalent";
    }
    
    // Directly test the threshold where equivalence changes
    if (!equiv5pct) {
        // Try to find approximate threshold
        double threshold = 0.0;
        for (double pct = 1.0; pct < 5.0; pct += 0.5) {
            auto testMaterial = createSimpleMaterial(
                baseMaterial->getProperty(Material::MaterialProperty::DENSITY) * (1.0 + pct/100.0),
                baseMaterial->getProperty(Material::MaterialProperty::DYNAMIC_VISCOSITY),
                baseMaterial->getProperty(Material::MaterialProperty::THERMAL_CONDUCTIVITY)
            );
            
            if (!key->areEquivalent(baseMaterial, testMaterial)) {
                threshold = pct;
                break;
            }
        }
        
        if (threshold > 0.0) {
            std::cout << "  Approximate equivalence threshold for density: " << threshold << "%" << std::endl;
        }
    }
}

// Test basic usage with simple material property changes
TEST_F(ScientificEquivBasicTest, BasicUsage) {
    // Create a base water-like material
    auto baseMaterial = createSimpleMaterial(1000.0, 0.001, 0.58);
    
    // Test modifying density by increasing amounts
    auto density1pct = createSimpleMaterial(1000.0 * 1.01, 0.001, 0.58);
    auto density5pct = createSimpleMaterial(1000.0 * 1.05, 0.001, 0.58);
    auto density10pct = createSimpleMaterial(1000.0 * 1.10, 0.001, 0.58);
    
    std::cout << "Basic usage test with density changes:" << std::endl;
    std::cout << "  - 1% density change: " 
              << (key->areEquivalent(baseMaterial, density1pct) ? "Equivalent" : "Not equivalent") << std::endl;
    std::cout << "  - 5% density change: " 
              << (key->areEquivalent(baseMaterial, density5pct) ? "Equivalent" : "Not equivalent") << std::endl;
    std::cout << "  - 10% density change: " 
              << (key->areEquivalent(baseMaterial, density10pct) ? "Equivalent" : "Not equivalent") << std::endl;
    
    // Test modifying viscosity by increasing amounts
    auto visc1pct = createSimpleMaterial(1000.0, 0.001 * 1.01, 0.58);
    auto visc5pct = createSimpleMaterial(1000.0, 0.001 * 1.05, 0.58);
    auto visc10pct = createSimpleMaterial(1000.0, 0.001 * 1.10, 0.58);
    
    std::cout << "Basic usage test with viscosity changes:" << std::endl;
    std::cout << "  - 1% viscosity change: " 
              << (key->areEquivalent(baseMaterial, visc1pct) ? "Equivalent" : "Not equivalent") << std::endl;
    std::cout << "  - 5% viscosity change: " 
              << (key->areEquivalent(baseMaterial, visc5pct) ? "Equivalent" : "Not equivalent") << std::endl;
    std::cout << "  - 10% viscosity change: " 
              << (key->areEquivalent(baseMaterial, visc10pct) ? "Equivalent" : "Not equivalent") << std::endl;
    
    // Test that variations to multiple properties have combined effect
    auto combined = createSimpleMaterial(1000.0 * 1.01, 0.001 * 1.01, 0.58 * 1.01);
    std::cout << "  - 1% change to all properties: " 
              << (key->areEquivalent(baseMaterial, combined) ? "Equivalent" : "Not equivalent") << std::endl;
}

// Test setting and getting property weights
TEST_F(ScientificEquivBasicTest, PropertyWeights) {
    // Create a custom key with modified property weights
    auto customKey = std::make_shared<ScientificMaterialEquivalence>();
    
    // Emphasize density and de-emphasize thermal conductivity
    customKey->setPropertyBits(Material::MaterialProperty::DENSITY, 12);              // More bits for density
    customKey->setPropertyBits(Material::MaterialProperty::THERMAL_CONDUCTIVITY, 4);  // Fewer bits for thermal conductivity
    
    // Create test materials with variations in different properties
    auto baseMaterial = createSimpleMaterial();
    
    auto densityVariation = createSimpleMaterial(
        baseMaterial->getProperty(Material::MaterialProperty::DENSITY) * 1.01, // 1% change
        baseMaterial->getProperty(Material::MaterialProperty::DYNAMIC_VISCOSITY),
        baseMaterial->getProperty(Material::MaterialProperty::THERMAL_CONDUCTIVITY)
    );
    
    auto thermalVariation = createSimpleMaterial(
        baseMaterial->getProperty(Material::MaterialProperty::DENSITY),
        baseMaterial->getProperty(Material::MaterialProperty::DYNAMIC_VISCOSITY),
        baseMaterial->getProperty(Material::MaterialProperty::THERMAL_CONDUCTIVITY) * 1.01 // 1% change
    );
    
    // Test with default key
    bool defaultDensityEquiv = key->areEquivalent(baseMaterial, densityVariation);
    bool defaultThermalEquiv = key->areEquivalent(baseMaterial, thermalVariation);
    
    // Test with custom key
    bool customDensityEquiv = customKey->areEquivalent(baseMaterial, densityVariation);
    bool customThermalEquiv = customKey->areEquivalent(baseMaterial, thermalVariation);
    
    std::cout << "Property weighting test:" << std::endl;
    std::cout << "  Default key:" << std::endl;
    std::cout << "    - 1% density change: " 
              << (defaultDensityEquiv ? "Equivalent" : "Not equivalent") << std::endl;
    std::cout << "    - 1% thermal conductivity change: " 
              << (defaultThermalEquiv ? "Equivalent" : "Not equivalent") << std::endl;
    std::cout << "  Custom key (more bits for density, fewer for thermal conductivity):" << std::endl;
    std::cout << "    - 1% density change: " 
              << (customDensityEquiv ? "Equivalent" : "Not equivalent") << std::endl;
    std::cout << "    - 1% thermal conductivity change: " 
              << (customThermalEquiv ? "Equivalent" : "Not equivalent") << std::endl;
    
    // With our custom bit allocation, we expect the density change to be less likely
    // to be considered equivalent and the thermal conductivity change to be more likely
    if (defaultDensityEquiv && !customDensityEquiv) {
        std::cout << "  Bit allocation had expected effect on density (more sensitive)" << std::endl;
    }
    if (!defaultThermalEquiv && customThermalEquiv) {
        std::cout << "  Bit allocation had expected effect on thermal conductivity (less sensitive)" << std::endl;
    }
}


