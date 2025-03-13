/**
 * @file scientific_mequiv_hash_distribution_tests.cpp
 * @brief Tests for hash distribution analysis of Scientific Material Equivalence
 */
#include "scientific_mequiv_advanced_test_base.h"
#include <gtest/gtest.h>
#include <random>
#include <unordered_map>
#include <unordered_set>

// Test hash collision probability with different materials
TEST_F(ScientificEquivAdvancedTest, HashDistributionAnalysis) {
    // Generate a diverse set of materials (not just variations of one base material)
    std::vector<std::shared_ptr<Material>> diverseMaterials;
    const int NUM_DIVERSE_MATERIALS = 1000;
    
    // Use fixed seed for reproducibility
    std::mt19937 rng(42);
    
    // Define reasonable ranges for each property
    struct PropertyRange {
        double min;
        double max;
        bool useLog; // Use logarithmic distribution if true
    };
    
    std::map<Material::MaterialProperty, PropertyRange> ranges = {
        {Material::MaterialProperty::DENSITY, {10.0, 20000.0, true}},
        {Material::MaterialProperty::DYNAMIC_VISCOSITY, {1e-6, 1e6, true}},
        {Material::MaterialProperty::THERMAL_CONDUCTIVITY, {0.01, 1000.0, true}},
        {Material::MaterialProperty::SPECIFIC_HEAT, {10.0, 10000.0, false}},
        {Material::MaterialProperty::THERMAL_EXPANSION, {1e-7, 1e-4, true}},
        {Material::MaterialProperty::SURFACE_TENSION, {0.001, 10.0, false}},
        {Material::MaterialProperty::ELECTRICAL_CONDUCTIVITY, {1e-16, 1e7, true}}
    };
    
    // Generate random materials
    for (int i = 0; i < NUM_DIVERSE_MATERIALS; i++) {
        auto material = std::make_shared<Material>();
        
        for (const auto& range : ranges) {
            Material::MaterialProperty prop = range.first;
            const PropertyRange& r = range.second;
            
            double value;
            if (r.useLog) {
                // Generate value with logarithmic distribution
                std::uniform_real_distribution<double> logDist(std::log10(r.min), std::log10(r.max));
                value = std::pow(10.0, logDist(rng));
            } else {
                // Generate value with uniform distribution
                std::uniform_real_distribution<double> uniformDist(r.min, r.max);
                value = uniformDist(rng);
            }
            
            material->setProperty(prop, value);
        }
        
        diverseMaterials.push_back(material);
    }
    
    // Calculate hashes for all materials
    std::unordered_map<uint64_t, int> hashCounts;
    for (const auto& material : diverseMaterials) {
        uint64_t hash = scientificKey->hash(material);
        hashCounts[hash]++;
    }
    
    // Analyze hash distribution
    int uniqueHashes = hashCounts.size();
    int maxCollisions = 0;
    uint64_t mostCommonHash = 0;
    
    for (const auto& pair : hashCounts) {
        if (pair.second > maxCollisions) {
            maxCollisions = pair.second;
            mostCommonHash = pair.first;
        }
    }
    
    double collisionRate = 1.0 - (uniqueHashes / static_cast<double>(NUM_DIVERSE_MATERIALS));
    
    std::cout << "\n--- Hash Distribution Analysis ---\n";
    std::cout << "Total materials: " << NUM_DIVERSE_MATERIALS << std::endl;
    std::cout << "Unique hashes: " << uniqueHashes << std::endl;
    std::cout << "Collision rate: " << (collisionRate * 100.0) << "%" << std::endl;
    std::cout << "Maximum collisions for a single hash: " << maxCollisions << std::endl;
    
    if (maxCollisions > 1) {
        std::cout << "Most common hash: 0x" << std::hex << mostCommonHash << std::dec << std::endl;
        
        // Analyze materials that share this hash
        std::cout << "\nProperties of colliding materials:" << std::endl;
        std::vector<std::shared_ptr<Material>> collidingMaterials;
        
        for (size_t i = 0; i < diverseMaterials.size(); i++) {
            if (scientificKey->hash(diverseMaterials[i]) == mostCommonHash) {
                collidingMaterials.push_back(diverseMaterials[i]);
            }
        }
        
        // Print property ranges for colliding materials
        for (int propIndex = 0; propIndex < 7; propIndex++) {
            Material::MaterialProperty prop = static_cast<Material::MaterialProperty>(propIndex);
            double minVal = std::numeric_limits<double>::max();
            double maxVal = std::numeric_limits<double>::lowest();
            
            for (const auto& material : collidingMaterials) {
                double val = material->getProperty(prop);
                minVal = std::min(minVal, val);
                maxVal = std::max(maxVal, val);
            }
            
            std::string propName;
            switch (prop) {
                case Material::MaterialProperty::DENSITY: propName = "DENSITY"; break;
                case Material::MaterialProperty::DYNAMIC_VISCOSITY: propName = "DYNAMIC_VISCOSITY"; break;
                case Material::MaterialProperty::THERMAL_CONDUCTIVITY: propName = "THERMAL_CONDUCTIVITY"; break;
                case Material::MaterialProperty::SPECIFIC_HEAT: propName = "SPECIFIC_HEAT"; break;
                case Material::MaterialProperty::THERMAL_EXPANSION: propName = "THERMAL_EXPANSION"; break;
                case Material::MaterialProperty::SURFACE_TENSION: propName = "SURFACE_TENSION"; break;
                case Material::MaterialProperty::ELECTRICAL_CONDUCTIVITY: propName = "ELECTRICAL_CONDUCTIVITY"; break;
                default: propName = "UNKNOWN"; break;
            }
            
            double range = 100.0 * (maxVal - minVal) / ((minVal + maxVal) / 2.0);
            
            std::cout << std::left << std::setw(25) << propName
                      << "Range: " << minVal << " to " << maxVal
                      << " (" << range << "% variation)" << std::endl;
        }
    }
    
    // Expect a low collision rate for diverse materials
    EXPECT_LT(collisionRate, 0.05) << "Collision rate should be under 5% for diverse materials";
    EXPECT_LT(maxCollisions, 5) << "No hash should be shared by more than 4 materials";
}

// Test hash distribution for similar materials
TEST_F(ScientificEquivAdvancedTest, SimilarMaterialsDistribution) {
    // Generate a set of similar materials with small variations
    std::vector<std::shared_ptr<Material>> similarMaterials;
    const int NUM_SIMILAR_MATERIALS = 1000;
    
    // Use fixed seed for reproducibility
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> smallDist(-0.02, 0.02); // ±2% variation
    
    // Create base material (water-like)
    auto baseMaterial = createTestMaterial();
    
    // Create variations of the base material
    for (int i = 0; i < NUM_SIMILAR_MATERIALS; i++) {
        auto material = std::make_shared<Material>();
        
        // Set properties with small random variations
        for (int propIndex = 0; propIndex < 7; propIndex++) {
            Material::MaterialProperty prop = static_cast<Material::MaterialProperty>(propIndex);
            double baseValue = baseMaterial->getProperty(prop);
            double randomFactor = 1.0 + smallDist(rng);
            material->setProperty(prop, baseValue * randomFactor);
        }
        
        similarMaterials.push_back(material);
    }
    
    // Calculate hashes for all materials
    std::unordered_map<uint64_t, int> hashCounts;
    for (const auto& material : similarMaterials) {
        uint64_t hash = scientificKey->hash(material);
        hashCounts[hash]++;
    }
    
    // Analyze hash distribution
    int uniqueHashes = hashCounts.size();
    int maxCollisions = 0;
    
    for (const auto& pair : hashCounts) {
        if (pair.second > maxCollisions) {
            maxCollisions = pair.second;
        }
    }
    
    double collisionRate = 1.0 - (uniqueHashes / static_cast<double>(NUM_SIMILAR_MATERIALS));
    
    std::cout << "\n--- Similar Materials Hash Distribution Analysis ---\n";
    std::cout << "Total materials: " << NUM_SIMILAR_MATERIALS << std::endl;
    std::cout << "Unique hashes: " << uniqueHashes << std::endl;
    std::cout << "Collision rate: " << (collisionRate * 100.0) << "%" << std::endl;
    std::cout << "Maximum collisions for a single hash: " << maxCollisions << std::endl;
    
    // For similar materials, we expect more collisions
    // But there should still be reasonable differentiation
    EXPECT_GT(uniqueHashes, NUM_SIMILAR_MATERIALS / 2) << "At least 50% of similar materials should have unique hashes";
    
    // Test how varying only one property affects hash distribution
    std::vector<double> densityDistribution;
    
    // Create materials with only density varying
    for (int i = 0; i < 100; i++) {
        auto material = std::make_shared<Material>(*baseMaterial);
        double densityFactor = 0.8 + (i * 0.004); // 0.8x to 1.2x base density
        material->setProperty(
            Material::MaterialProperty::DENSITY, 
            baseMaterial->getProperty(Material::MaterialProperty::DENSITY) * densityFactor
        );
        
        uint64_t hash = scientificKey->hash(material);
        densityDistribution.push_back(hash);
    }
    
    // Count unique hashes in the distribution
    std::unordered_set<uint64_t> uniqueDensityHashes(densityDistribution.begin(), densityDistribution.end());
    
    std::cout << "Density-only variation: " << uniqueDensityHashes.size() 
              << " unique hashes out of 100 materials" << std::endl;
    
    // We expect a reasonable distribution of hashes even when varying only one property
    EXPECT_GT(uniqueDensityHashes.size(), 10) << "Should have at least 10 unique hashes for density-only variation";
}

// Test hash distribution for different material types
TEST_F(ScientificEquivAdvancedTest, MaterialTypeDistribution) {
    // Test distribution across standard material types
    std::vector<std::shared_ptr<Material>> standardMaterials;
    
    // Create a diverse set of standard materials
    standardMaterials.push_back(TestMaterialHelpers::createWaterMaterial());
    standardMaterials.push_back(TestMaterialHelpers::createAirMaterial());
    standardMaterials.push_back(TestMaterialHelpers::createAluminumMaterial());
    standardMaterials.push_back(TestMaterialHelpers::createSteelMaterial());
    standardMaterials.push_back(TestMaterialHelpers::createOilMaterial());
    standardMaterials.push_back(TestMaterialHelpers::createMercuryMaterial());
    standardMaterials.push_back(TestMaterialHelpers::createConcreteMaterial());
    standardMaterials.push_back(TestMaterialHelpers::createCopperMaterial());
    standardMaterials.push_back(TestMaterialHelpers::createGlycerinMaterial());
    standardMaterials.push_back(TestMaterialHelpers::createSaltWaterMaterial());
    standardMaterials.push_back(TestMaterialHelpers::createGasolineMaterial());
    
    // Calculate hashes for these materials
    std::cout << "\n--- Standard Material Hash Distribution ---\n";
    std::cout << std::left << std::setw(20) << "Material" << "Hash" << std::endl;
    std::cout << std::string(40, '-') << std::endl;
    
    for (const auto& material : standardMaterials) {
        uint64_t hash = scientificKey->hash(material);
        std::string materialName = material->getName();
        std::cout << std::left << std::setw(20) << materialName 
                  << "0x" << std::hex << hash << std::dec << std::endl;
    }
    
    // Check if all hashes are unique
    std::unordered_set<uint64_t> uniqueStandardHashes;
    for (const auto& material : standardMaterials) {
        uniqueStandardHashes.insert(scientificKey->hash(material));
    }
    
    std::cout << "Standard materials: " << uniqueStandardHashes.size() 
              << " unique hashes out of " << standardMaterials.size() << " materials" << std::endl;
    
    // We expect all standard materials to have unique hashes
    EXPECT_EQ(uniqueStandardHashes.size(), standardMaterials.size()) 
        << "All standard materials should have unique hashes";
}


