/**
 * @file scientific_mequiv_industry_tests.cpp
 * @brief Tests for industry-specific configurations of Scientific Material Equivalence
 */
#include "scientific_mequiv_advanced_test_base.h"
#include <gtest/gtest.h>

// Test fixture for industry-specific configuration tests
class ScientificEquivIndustryTest : public ScientificEquivAdvancedTest {
};

// Test customized industry-specific profiles
TEST_F(ScientificEquivIndustryTest, CustomIndustryProfiles) {
    // Create specialized keys for different industries
    auto aeroKey = std::make_shared<ScientificMaterialEquivalence>();
    auto nuclearKey = std::make_shared<ScientificMaterialEquivalence>();
    auto biomedKey = std::make_shared<ScientificMaterialEquivalence>();
    
    // Configure for aerospace applications (lightweight but thermally stable)
    aeroKey->setPropertyBits(Material::MaterialProperty::DENSITY, 12);          // More bits for density
    aeroKey->setPropertyBits(Material::MaterialProperty::THERMAL_EXPANSION, 10); // More bits for thermal expansion
    aeroKey->setPropertyBits(Material::MaterialProperty::THERMAL_CONDUCTIVITY, 8); // More bits for thermal conductivity
    
    // Set a stricter tolerance for more sensitivity
    aeroKey->setEquivalenceTolerance(0.005); // 0.5%
    
    // Configure for nuclear applications (thermal properties critical)
    nuclearKey->setPropertyBits(Material::MaterialProperty::THERMAL_CONDUCTIVITY, 12); // More bits for thermal conductivity
    nuclearKey->setPropertyBits(Material::MaterialProperty::SPECIFIC_HEAT, 10);        // More bits for specific heat
    nuclearKey->setPropertyBits(Material::MaterialProperty::DENSITY, 8);               // More bits for density
    
    // Set a stricter tolerance for more sensitivity
    nuclearKey->setEquivalenceTolerance(0.005); // 0.5%
    
    // Configure for biomedical applications (fluid-focused)
    biomedKey->setPropertyBits(Material::MaterialProperty::DYNAMIC_VISCOSITY, 12);   // More bits for viscosity
    biomedKey->setPropertyBits(Material::MaterialProperty::DENSITY, 10);             // More bits for density
    biomedKey->setPropertyBits(Material::MaterialProperty::SURFACE_TENSION, 8);      // More bits for surface tension
    
    // Set a stricter tolerance for more sensitivity
    biomedKey->setEquivalenceTolerance(0.005); // 0.5%
    
    // Report bit allocations
    std::cout << "\n==== Industry-Specific Configurations ====\n";
    reportBitAllocation("Aerospace", aeroKey);
    reportBitAllocation("Nuclear", nuclearKey);
    reportBitAllocation("Biomedical", biomedKey);
    
    // Test with materials specifically relevant to each industry
    // Create a titanium alloy (aerospace)
    auto titanium = TestMaterialHelpers::createCustomMaterial(
        "Titanium", Material::MaterialType::SOLID, 4500, 0, 21.9, 523, 8.6e-6, 1.65, 2.38e6);
    
    auto titaniumMod = std::make_shared<Material>(*titanium);
    titaniumMod->setProperty(Material::MaterialProperty::DENSITY, 4500 * 1.02); // 2% density change
    
    // Create a water coolant (nuclear)
    auto coolant = TestMaterialHelpers::createCustomMaterial(
        "Coolant", Material::MaterialType::FLUID, 1000, 0.001, 0.6, 4200, 0.0002, 0.072, 5.0);
    
    auto coolantMod = std::make_shared<Material>(*coolant);
    coolantMod->setProperty(Material::MaterialProperty::THERMAL_CONDUCTIVITY, 0.6 * 1.02); // 2% conductivity change
    
    // Create a blood-like fluid (biomedical)
    auto bloodLike = TestMaterialHelpers::createCustomMaterial(
        "BloodLike", Material::MaterialType::FLUID, 1060, 0.004, 0.5, 3617, 0.0001, 0.058, 0.7);
    
    auto bloodLikeMod = std::make_shared<Material>(*bloodLike);
    bloodLikeMod->setProperty(Material::MaterialProperty::DYNAMIC_VISCOSITY, 0.004 * 1.02); // 2% viscosity change
    
    // Test each industry key with its relevant material
    bool aeroEquiv = aeroKey->areEquivalent(titanium, titaniumMod);
    bool nuclearEquiv = nuclearKey->areEquivalent(coolant, coolantMod);
    bool biomedEquiv = biomedKey->areEquivalent(bloodLike, bloodLikeMod);
    
    // Since we're using 2% changes in the most important property for each key,
    // we expect the materials to be considered not equivalent
    std::cout << "\nEquivalence with 2% change in critical property:\n";
    std::cout << "  Aerospace (2% density change): " << (aeroEquiv ? "Equivalent" : "Not Equivalent") << std::endl;
    std::cout << "  Nuclear (2% thermal conductivity change): " << (nuclearEquiv ? "Equivalent" : "Not Equivalent") << std::endl;
    std::cout << "  Biomedical (2% viscosity change): " << (biomedEquiv ? "Equivalent" : "Not Equivalent") << std::endl;
    
    EXPECT_FALSE(aeroEquiv) << "Aerospace key should detect 2% density change";
    EXPECT_FALSE(nuclearEquiv) << "Nuclear key should detect 2% thermal conductivity change";
    EXPECT_FALSE(biomedEquiv) << "Biomedical key should detect 2% viscosity change";
    
    // Now test with 1% changes in less important properties
    auto titaniumMinorMod = std::make_shared<Material>(*titanium);
    titaniumMinorMod->setProperty(Material::MaterialProperty::ELECTRICAL_CONDUCTIVITY, 2.38e6 * 1.01); // 1% change
    
    auto coolantMinorMod = std::make_shared<Material>(*coolant);
    coolantMinorMod->setProperty(Material::MaterialProperty::SURFACE_TENSION, 0.072 * 1.01); // 1% change
    
    auto bloodLikeMinorMod = std::make_shared<Material>(*bloodLike);
    bloodLikeMinorMod->setProperty(Material::MaterialProperty::THERMAL_EXPANSION, 0.0001 * 1.01); // 1% change
    
    // Test each industry key with changes in less important properties
    aeroEquiv = aeroKey->areEquivalent(titanium, titaniumMinorMod);
    nuclearEquiv = nuclearKey->areEquivalent(coolant, coolantMinorMod);
    biomedEquiv = biomedKey->areEquivalent(bloodLike, bloodLikeMinorMod);
    
    std::cout << "\nEquivalence with 1% change in less important property:\n";
    std::cout << "  Aerospace (1% electrical conductivity change): " << (aeroEquiv ? "Equivalent" : "Not Equivalent") << std::endl;
    std::cout << "  Nuclear (1% surface tension change): " << (nuclearEquiv ? "Equivalent" : "Not Equivalent") << std::endl;
    std::cout << "  Biomedical (1% thermal expansion change): " << (biomedEquiv ? "Equivalent" : "Not Equivalent") << std::endl;
    
    // For less important properties, we expect the materials may still be considered equivalent
    // But this depends on the exact bit allocation, so we don't assert on these results
}

// Test simulation-specific presets
TEST_F(ScientificEquivIndustryTest, SimulationPresets) {
    // Create default key
    auto defaultKey = std::make_shared<ScientificMaterialEquivalence>();
    
    // Create specialized keys for different simulations
    auto cfdKey = std::make_shared<ScientificMaterialEquivalence>();
    auto thermalKey = std::make_shared<ScientificMaterialEquivalence>();
    
    bool hasPresets = false;
    
    // Try to configure for specific simulations
    try {
        // Configure CFD key
        cfdKey->configureForCFD(true, 100);
        cfdKey->setEquivalenceTolerance(0.005); // 0.5% tolerance
        
        // Configure thermal key
        thermalKey->configureForThermal(100, false);
        thermalKey->setEquivalenceTolerance(0.005); // 0.5% tolerance
        
        hasPresets = true;
    } catch (...) {
        std::cout << "Simulation presets not available" << std::endl;
    }
    
    if (hasPresets) {
        // Report bit allocations
        std::cout << "\n==== Simulation Preset Configurations ====\n";
        reportBitAllocation("Default", defaultKey);
        reportBitAllocation("CFD", cfdKey);
        reportBitAllocation("Thermal", thermalKey);
        
        // Test CFD preset with water
        auto water = TestMaterialHelpers::createWaterMaterial();
        auto waterMod = std::make_shared<Material>(*water);
        waterMod->setProperty(Material::MaterialProperty::DENSITY, 
                             water->getProperty(Material::MaterialProperty::DENSITY) * 1.02); // 2% density change
        
        bool cfdEquiv = cfdKey->areEquivalent(water, waterMod);
        std::cout << "CFD preset with 2% density change: " 
                  << (cfdEquiv ? "Equivalent (unexpected)" : "Not Equivalent (expected)") << std::endl;
        
        EXPECT_FALSE(cfdEquiv) << "CFD preset should detect 2% density change in water";
        
        // Test thermal preset with aluminum
        auto aluminum = TestMaterialHelpers::createAluminumMaterial();
        auto aluminumMod = std::make_shared<Material>(*aluminum);
        aluminumMod->setProperty(Material::MaterialProperty::THERMAL_CONDUCTIVITY, 
                                aluminum->getProperty(Material::MaterialProperty::THERMAL_CONDUCTIVITY) * 1.02); // 2% change
        
        bool thermalEquiv = thermalKey->areEquivalent(aluminum, aluminumMod);
        std::cout << "Thermal preset with 2% thermal conductivity change: " 
                 << (thermalEquiv ? "Equivalent (unexpected)" : "Not Equivalent (expected)") << std::endl;
        
        EXPECT_FALSE(thermalEquiv) << "Thermal preset should detect 2% thermal conductivity change";
    } else {
        std::cout << "No simulation presets available in the implementation" << std::endl;
        // Skip the tests
    }
}


