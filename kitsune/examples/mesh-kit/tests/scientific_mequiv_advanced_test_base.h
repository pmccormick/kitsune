/**
 * @file scientific_mequiv_advanced_test_base.h
 * @brief Base fixture for advanced tests for Scientific Material Equivalence implementation
 */
#include "ScientificMaterialEquivalence.h"
#include "Material.h"
#include "TestMaterialHelpers.h"
#include <gtest/gtest.h>
#include <memory>
#include <map>
#include <string>
#include <iostream>
#include <iomanip>
#include <vector>
#include <limits>

// Base test fixture for scientific equivalence advanced tests
class ScientificEquivAdvancedTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create default scientific equivalence with reasonable defaults
        scientificKey = std::make_shared<ScientificMaterialEquivalence>();
    }
    
    // Helper to print a comparison of bit allocations
    void reportBitAllocation(const std::string& title, 
                           const std::shared_ptr<ScientificMaterialEquivalence>& key) {
        std::cout << "\n--- " << title << " Bit Allocation ---\n";
        auto bitAllocation = key->getBitAllocation();
        
        // Map for property names
        std::map<Material::MaterialProperty, std::string> propNames = {
            {Material::MaterialProperty::DENSITY, "DENSITY"},
            {Material::MaterialProperty::DYNAMIC_VISCOSITY, "DYNAMIC_VISCOSITY"},
            {Material::MaterialProperty::THERMAL_CONDUCTIVITY, "THERMAL_CONDUCTIVITY"},
            {Material::MaterialProperty::SPECIFIC_HEAT, "SPECIFIC_HEAT"},
            {Material::MaterialProperty::THERMAL_EXPANSION, "THERMAL_EXPANSION"},
            {Material::MaterialProperty::SURFACE_TENSION, "SURFACE_TENSION"},
            {Material::MaterialProperty::ELECTRICAL_CONDUCTIVITY, "ELECTRICAL_CONDUCTIVITY"}
        };
        
        // Print each property's allocation
        for (const auto& pair : bitAllocation) {
            const Material::MaterialProperty& prop = pair.first;
            const size_t bits = pair.second;
            
            auto nameIt = propNames.find(prop);
            std::string propName = nameIt != propNames.end() ? nameIt->second : "UNKNOWN";
            std::cout << std::left << std::setw(25) << propName 
                      << bits << " bits" << std::endl;
        }
        
        // Calculate total bits
        size_t totalBits = 0;
        for (const auto& pair : bitAllocation) {
            totalBits += pair.second;
        }
        std::cout << std::string(40, '-') << "\n";
        std::cout << std::left << std::setw(25) << "TOTAL" 
                  << totalBits << " bits" << std::endl;
    }
    
    // Helper to create materials with varying properties for testing
    std::shared_ptr<Material> createTestMaterial(
            double density = 1000.0,
            double viscosity = 0.001,
            double thermalConductivity = 0.6,
            double specificHeat = 4200.0,
            double thermalExpansion = 0.0002,
            double surfaceTension = 0.072,
            double electricalConductivity = 5.0) {
        
        auto material = std::make_shared<Material>();
        material->setProperty(Material::MaterialProperty::DENSITY, density);
        material->setProperty(Material::MaterialProperty::DYNAMIC_VISCOSITY, viscosity);
        material->setProperty(Material::MaterialProperty::THERMAL_CONDUCTIVITY, thermalConductivity);
        material->setProperty(Material::MaterialProperty::SPECIFIC_HEAT, specificHeat);
        material->setProperty(Material::MaterialProperty::THERMAL_EXPANSION, thermalExpansion);
        material->setProperty(Material::MaterialProperty::SURFACE_TENSION, surfaceTension);
        material->setProperty(Material::MaterialProperty::ELECTRICAL_CONDUCTIVITY, electricalConductivity);
        
        return material;
    }
    
    // Helper to compare two materials with detailed reporting
    void compareWithDetails(const std::shared_ptr<Material>& a, 
                           const std::shared_ptr<Material>& b, 
                           bool expectEquivalent = true) {
        bool result = scientificKey->areEquivalent(a, b);
        
        if (result != expectEquivalent || ::testing::Test::HasFailure()) {
            std::cout << "\n--- Material Comparison Details ---\n";
            std::map<Material::MaterialProperty, std::string> propNames = {
                {Material::MaterialProperty::DENSITY, "DENSITY"},
                {Material::MaterialProperty::DYNAMIC_VISCOSITY, "DYNAMIC_VISCOSITY"},
                {Material::MaterialProperty::THERMAL_CONDUCTIVITY, "THERMAL_CONDUCTIVITY"},
                {Material::MaterialProperty::SPECIFIC_HEAT, "SPECIFIC_HEAT"},
                {Material::MaterialProperty::THERMAL_EXPANSION, "THERMAL_EXPANSION"},
                {Material::MaterialProperty::SURFACE_TENSION, "SURFACE_TENSION"},
                {Material::MaterialProperty::ELECTRICAL_CONDUCTIVITY, "ELECTRICAL_CONDUCTIVITY"}
            };
            
            for (const auto& prop : propNames) {
                double valA = a->getProperty(prop.first);
                double valB = b->getProperty(prop.first);
                double diff = std::abs(valA - valB);
                double pctDiff = valA != 0.0 ? (diff / std::abs(valA)) * 100.0 : 0.0;
                
                std::cout << std::left << std::setw(25) << prop.second 
                          << "A: " << std::setw(15) << valA 
                          << "B: " << std::setw(15) << valB
                          << "Diff: " << std::setw(15) << diff
                          << "% Diff: " << pctDiff << "%" << std::endl;
            }
            
            uint64_t hashA = scientificKey->hash(a);
            uint64_t hashB = scientificKey->hash(b);
            std::cout << "Hash A: 0x" << std::hex << hashA << std::dec << std::endl;
            std::cout << "Hash B: 0x" << std::hex << hashB << std::dec << std::endl;
            std::cout << "Hash Diff: 0x" << std::hex << (hashA ^ hashB) << std::dec << std::endl;
            
            // Calculate hamming distance
            uint64_t xorResult = hashA ^ hashB;
            int hammingDist = 0;
            for (uint64_t i = xorResult; i > 0; i >>= 1) {
                if (i & 1) hammingDist++;
            }
            std::cout << "Hamming Distance: " << hammingDist << " bits" << std::endl;
        }
    }
    
    std::shared_ptr<ScientificMaterialEquivalence> scientificKey;
};

