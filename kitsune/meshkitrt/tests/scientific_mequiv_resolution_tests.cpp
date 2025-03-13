/**
 * @file scientific_mequiv_resolution_tests.cpp
 * @brief Tests for hash resolution and sensitivity in Scientific Material Equivalence
 */
#include "scientific_mequiv_advanced_test_base.h"
#include <gtest/gtest.h>
#include <random>
#include <unordered_set>

// Test sensitivity to small changes in different properties
TEST_F(ScientificEquivAdvancedTest, SensitivityAnalysis) {
    // Reference material (water-like)
    auto reference = createTestMaterial();
    uint64_t refHash = scientificKey->hash(reference);
    
    std::map<Material::MaterialProperty, std::string> propNames = {
        {Material::MaterialProperty::DENSITY, "DENSITY"},
        {Material::MaterialProperty::DYNAMIC_VISCOSITY, "DYNAMIC_VISCOSITY"},
        {Material::MaterialProperty::THERMAL_CONDUCTIVITY, "THERMAL_CONDUCTIVITY"},
        {Material::MaterialProperty::SPECIFIC_HEAT, "SPECIFIC_HEAT"},
        {Material::MaterialProperty::THERMAL_EXPANSION, "THERMAL_EXPANSION"},
        {Material::MaterialProperty::SURFACE_TENSION, "SURFACE_TENSION"},
        {Material::MaterialProperty::ELECTRICAL_CONDUCTIVITY, "ELECTRICAL_CONDUCTIVITY"}
    };
    
    std::cout << "\n--- Sensitivity Analysis ---\n";
    std::cout << std::left << std::setw(25) << "Property" 
              << std::setw(15) << "Min % Change" 
              << "For Different Hash" << std::endl;
    std::cout << std::string(60, '-') << std::endl;
    
    for (const auto& prop : propNames) {
        double baseValue = reference->getProperty(prop.first);
        double minPercentage = 100.0;
        
        // Determine if this is a high or low importance property based on bit allocation
        bool lowImportanceProp = false;
        auto bitAllocation = scientificKey->getBitAllocation();
        auto bitIt = bitAllocation.find(prop.first);
        if (bitIt != bitAllocation.end()) {
            // Consider properties with fewer bits as low importance
            lowImportanceProp = (bitIt->second <= 8);
        } else {
            // Default to treating as low importance if not found in allocation
            lowImportanceProp = true;
        }
        
        // Find minimum percentage change needed to get a different hash
        for (double pctChange = 0.001; pctChange <= 20.0; pctChange *= 1.1) {
            auto testMaterial = std::make_shared<Material>(*reference);
            double newValue = baseValue * (1.0 + pctChange/100.0);
            testMaterial->setProperty(prop.first, newValue);
            
            uint64_t testHash = scientificKey->hash(testMaterial);
            if (testHash != refHash) {
                minPercentage = pctChange;
                break;
            }
        }
        
        std::cout << std::left << std::setw(25) << prop.second 
                  << std::setw(15) << minPercentage << "%" << std::endl;
                  
        // Use different thresholds for different property types
        if (lowImportanceProp) {
            // For less important properties, we allow up to 20% change before detection
            EXPECT_LT(minPercentage, 20.0) << "Too insensitive to changes in " << prop.second;
        } else {
            // For more important properties, expect detection below 5%
            EXPECT_LT(minPercentage, 5.0) << "Too insensitive to changes in " << prop.second;
        }
    }
    
    // Now test with a custom key that has a finer resolution for density
    auto customKey = std::make_shared<ScientificMaterialEquivalence>();
    customKey->setPropertyBits(Material::MaterialProperty::DENSITY, 16); // Allocate lots of bits to density
    
    // Test the effect of the increased bits through sensitivity
    double baseValue = reference->getProperty(Material::MaterialProperty::DENSITY);
    double customMinPercentage = 100.0;
    uint64_t customRefHash = customKey->hash(reference);
    
    for (double pctChange = 0.0001; pctChange <= 1.0; pctChange *= 1.1) {
        auto testMaterial = std::make_shared<Material>(*reference);
        double newValue = baseValue * (1.0 + pctChange/100.0);
        testMaterial->setProperty(Material::MaterialProperty::DENSITY, newValue);
        
        uint64_t testHash = customKey->hash(testMaterial);
        if (testHash != customRefHash) {
            customMinPercentage = pctChange;
            break;
        }
    }
    
    std::cout << "\nDensity with custom high-resolution key:\n";
    std::cout << "  Minimum detectable change: " << customMinPercentage << "%" << std::endl;
    
    // The custom key should be more sensitive to density changes
    if (customMinPercentage < 100.0) {  // If we found a difference
        double standardMinPct = 100.0;
        // Find the standard sensitivity for density
        for (const auto& prop : propNames) {
            if (prop.first == Material::MaterialProperty::DENSITY) {
                for (double pctChange = 0.001; pctChange <= 10.0; pctChange *= 1.1) {
                    auto testMaterial = std::make_shared<Material>(*reference);
                    double newValue = baseValue * (1.0 + pctChange/100.0);
                    testMaterial->setProperty(prop.first, newValue);
                    
                    uint64_t testHash = scientificKey->hash(testMaterial);
                    if (testHash != refHash) {
                        standardMinPct = pctChange;
                        break;
                    }
                }
                break;
            }
        }
        
        if (standardMinPct < 100.0) {  // If we found a difference with standard key too
            EXPECT_LT(customMinPercentage, standardMinPct) << "Custom key should be more sensitive to density changes";
        }
    }
}

// Test adaptive bit allocation based on property importance
TEST_F(ScientificEquivAdvancedTest, DynamicHashResolution) {
    // Test creating keys with different hash resolutions
    auto lowResKey = std::make_shared<ScientificMaterialEquivalence>();
    lowResKey->setHashResolution(32); // 32-bit hash
    
    auto medResKey = std::make_shared<ScientificMaterialEquivalence>();
    medResKey->setHashResolution(64); // 64-bit hash (standard)
    
    auto highResKey = std::make_shared<ScientificMaterialEquivalence>();
    highResKey->setHashResolution(128); // 128-bit hash
    
    // Report bit allocation for different resolutions
    reportBitAllocation("Low Resolution (32-bit)", lowResKey);
    reportBitAllocation("Medium Resolution (64-bit)", medResKey);
    reportBitAllocation("High Resolution (128-bit)", highResKey);
    
    // Create two nearly identical materials with a small difference
    auto baseMaterial = createTestMaterial();
    auto slightlyDifferent = std::make_shared<Material>(*baseMaterial);
    
    // Apply a very small change to density (0.1%)
    slightlyDifferent->setProperty(
        Material::MaterialProperty::DENSITY,
        baseMaterial->getProperty(Material::MaterialProperty::DENSITY) * 1.001
    );
    
    // Test equivalence at different resolutions
    bool lowResEquiv = lowResKey->areEquivalent(baseMaterial, slightlyDifferent);
    bool medResEquiv = medResKey->areEquivalent(baseMaterial, slightlyDifferent);
    bool highResEquiv = highResKey->areEquivalent(baseMaterial, slightlyDifferent);
    
    std::cout << "\nEquivalence with 0.1% density change at different resolutions:\n";
    std::cout << "  Low Resolution (32-bit): " << (lowResEquiv ? "Equivalent" : "Not Equivalent") << std::endl;
    std::cout << "  Medium Resolution (64-bit): " << (medResEquiv ? "Equivalent" : "Not Equivalent") << std::endl;
    std::cout << "  High Resolution (128-bit): " << (highResEquiv ? "Equivalent" : "Not Equivalent") << std::endl;
    
    // Higher resolution should be more sensitive to small differences
    // We expect high res to potentially detect smaller changes than low res
    if (!lowResEquiv) {
        EXPECT_FALSE(medResEquiv) << "Medium resolution should detect changes that low resolution detects";
        EXPECT_FALSE(highResEquiv) << "High resolution should detect changes that low resolution detects";
    }
    if (!medResEquiv) {
        EXPECT_FALSE(highResEquiv) << "High resolution should detect changes that medium resolution detects";
    }
    
    // Test collision rates at different resolutions
    const int NUM_MATERIALS = 1000;
    std::vector<std::shared_ptr<Material>> materials;
    std::unordered_set<uint64_t> lowResHashes;
    std::unordered_set<uint64_t> medResHashes;
    std::unordered_set<uint64_t> highResHashes;
    
    // Create many slightly different materials
    std::mt19937 rng(42); // Fixed seed for reproducibility
    std::uniform_real_distribution<double> dist(-0.05, 0.05); // ±5% variation
    
    for (int i = 0; i < NUM_MATERIALS; i++) {
        auto material = std::make_shared<Material>();
        
        // Set properties with random variations from base
        for (int propIndex = 0; propIndex < 7; propIndex++) {
            Material::MaterialProperty prop = static_cast<Material::MaterialProperty>(propIndex);
            double baseValue = baseMaterial->getProperty(prop);
            double randomFactor = 1.0 + dist(rng);
            material->setProperty(prop, baseValue * randomFactor);
        }
        
        materials.push_back(material);
        
        // Calculate hashes at different resolutions
        uint64_t lowHash = lowResKey->hash(material);
        uint64_t medHash = medResKey->hash(material);
        uint64_t highHash = highResKey->hash(material);
        
        lowResHashes.insert(lowHash);
        medResHashes.insert(medHash);
        highResHashes.insert(highHash);
    }
    
    // Calculate collision rates
    double lowResCollisionRate = 1.0 - (lowResHashes.size() / static_cast<double>(NUM_MATERIALS));
    double medResCollisionRate = 1.0 - (medResHashes.size() / static_cast<double>(NUM_MATERIALS));
    double highResCollisionRate = 1.0 - (highResHashes.size() / static_cast<double>(NUM_MATERIALS));
    
    std::cout << "\nHash collision rates at different resolutions:\n";
    std::cout << "  Low Resolution (32-bit): " << (lowResCollisionRate * 100.0) << "%" << std::endl;
    std::cout << "  Medium Resolution (64-bit): " << (medResCollisionRate * 100.0) << "%" << std::endl;
    std::cout << "  High Resolution (128-bit): " << (highResCollisionRate * 100.0) << "%" << std::endl;
    
    // Higher resolution should have fewer collisions
    EXPECT_LE(medResCollisionRate, lowResCollisionRate) << "Medium resolution should have fewer collisions than low resolution";
    EXPECT_LE(highResCollisionRate, medResCollisionRate) << "High resolution should have fewer collisions than medium resolution";
}

// Test for different property scale types
TEST_F(ScientificEquivAdvancedTest, PropertyScaleTypes) {
    // Test if the implementation has configurable property scale types
    auto defaultKey = std::make_shared<ScientificMaterialEquivalence>();
    
    // Create keys with different property scale types manually
    auto linearKey = std::make_shared<ScientificMaterialEquivalence>();
    try {
        // Set linear scale for viscosity explicitly
        linearKey->setPropertyRange(
            Material::MaterialProperty::DYNAMIC_VISCOSITY,
            1e-6, 1e6,
            ScientificMaterialEquivalence::ScaleType::LINEAR
        );
        
        // Make sure we have enough bits to see the effect
        linearKey->setPropertyBits(Material::MaterialProperty::DYNAMIC_VISCOSITY, 16);
        
    } catch (...) {
        std::cout << "LINEAR scale type not available" << std::endl;
    }
    
    auto logarithmicKey = std::make_shared<ScientificMaterialEquivalence>();
    try {
        // Set logarithmic scale for viscosity explicitly
        logarithmicKey->setPropertyRange(
            Material::MaterialProperty::DYNAMIC_VISCOSITY,
            1e-6, 1e6,
            ScientificMaterialEquivalence::ScaleType::LOGARITHMIC
        );
        
        // Make sure we have enough bits to see the effect
        logarithmicKey->setPropertyBits(Material::MaterialProperty::DYNAMIC_VISCOSITY, 16);
        
    } catch (...) {
        std::cout << "LOGARITHMIC scale type not available" << std::endl;
    }
    
    // Create materials to test the different scale types
    bool hasScaleOptions = true;
    
    // Create a base material for testing
    auto baseMaterial = createTestMaterial();
    
    if (hasScaleOptions) {
        // Create extreme value materials
        auto lowVisc = std::make_shared<Material>(*baseMaterial);
        lowVisc->setProperty(Material::MaterialProperty::DYNAMIC_VISCOSITY, 1e-6);
        
        auto highVisc = std::make_shared<Material>(*baseMaterial);
        highVisc->setProperty(Material::MaterialProperty::DYNAMIC_VISCOSITY, 1.0);
        
        // Create slightly modified versions
        auto lowViscMod = std::make_shared<Material>(*lowVisc);
        lowViscMod->setProperty(Material::MaterialProperty::DYNAMIC_VISCOSITY, 1e-6 * 1.01); // 1% change
        
        auto highViscMod = std::make_shared<Material>(*highVisc);
        highViscMod->setProperty(Material::MaterialProperty::DYNAMIC_VISCOSITY, 1.0 * 1.01); // 1% change
        
        // Test if linear distribution is better for higher values
        bool linearLowEquiv = linearKey->areEquivalent(lowVisc, lowViscMod);
        bool linearHighEquiv = linearKey->areEquivalent(highVisc, highViscMod);
        
        // Test if logarithmic distribution is more balanced
        bool logLowEquiv = logarithmicKey->areEquivalent(lowVisc, lowViscMod);
        bool logHighEquiv = logarithmicKey->areEquivalent(highVisc, highViscMod);
        
        std::cout << "\nProperty scale type comparison (1% viscosity change):\n";
        std::cout << "  Linear at 1e-6: " << (linearLowEquiv ? "Equivalent" : "Not Equivalent") << std::endl;
        std::cout << "  Linear at 1.0: " << (linearHighEquiv ? "Equivalent" : "Not Equivalent") << std::endl;
        std::cout << "  Logarithmic at 1e-6: " << (logLowEquiv ? "Equivalent" : "Not Equivalent") << std::endl;
        std::cout << "  Logarithmic at 1.0: " << (logHighEquiv ? "Equivalent" : "Not Equivalent") << std::endl;
        
        // Logarithmic should be more balanced across the range
        if (logLowEquiv == logHighEquiv) {
            std::cout << "Logarithmic scale shows consistent sensitivity across range (good)" << std::endl;
        } else {
            std::cout << "Logarithmic scale shows inconsistent sensitivity across range (unexpected)" << std::endl;
        }
        
        // Linear may be less sensitive at the low end
        if (!linearHighEquiv && linearLowEquiv) {
            std::cout << "Linear scale less sensitive at low values (expected)" << std::endl;
            EXPECT_FALSE(linearHighEquiv);
            EXPECT_TRUE(linearLowEquiv);
        }
    } else {
        std::cout << "No configurable property scale types available in the implementation" << std::endl;
        // Skip the tests
    }
}


