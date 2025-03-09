/**
 * @file scientific_mequiv_robustness_tests.cpp
 * @brief Tests for robustness against numerical edge cases
 */
#include "scientific_mequiv_advanced_test_base.h"
#include <gtest/gtest.h>

// Test for robustness against numerical edge cases and anomalies
TEST_F(ScientificEquivAdvancedTest, NumericalRobustness) {
    // Test with extreme property values beyond typical ranges
    auto extremeValuesMaterial = createTestMaterial(
        std::numeric_limits<double>::min(),  // min density
        std::numeric_limits<double>::min(),  // min viscosity
        std::numeric_limits<double>::max(),  // max thermal conductivity
        std::numeric_limits<double>::max(),  // max specific heat
        std::numeric_limits<double>::epsilon(), // near-zero thermal expansion
        std::numeric_limits<double>::max(),  // max surface tension
        std::numeric_limits<double>::max()   // max electrical conductivity
    );
    
    // Test handling of NaN and Infinity
    auto nanMaterial = createTestMaterial();
    nanMaterial->setProperty(Material::MaterialProperty::SPECIFIC_HEAT, 
                            std::numeric_limits<double>::quiet_NaN());
    
    auto infMaterial = createTestMaterial();
    infMaterial->setProperty(Material::MaterialProperty::ELECTRICAL_CONDUCTIVITY, 
                            std::numeric_limits<double>::infinity());
    
    auto negInfMaterial = createTestMaterial();
    negInfMaterial->setProperty(Material::MaterialProperty::THERMAL_EXPANSION, 
                              -std::numeric_limits<double>::infinity());
    
    // Test the ability to calculate hashes for these edge cases
    bool extremeHashOk = true;
    bool nanHashOk = true;
    bool infHashOk = true;
    bool negInfHashOk = true;
    
    uint64_t extremeHash = 0, nanHash = 0, infHash = 0, negInfHash = 0;
    
    try {
        extremeHash = scientificKey->hash(extremeValuesMaterial);
    } catch (...) {
        extremeHashOk = false;
    }
    
    try {
        nanHash = scientificKey->hash(nanMaterial);
    } catch (...) {
        nanHashOk = false;
    }
    
    try {
        infHash = scientificKey->hash(infMaterial);
    } catch (...) {
        infHashOk = false;
    }
    
    try {
        negInfHash = scientificKey->hash(negInfMaterial);
    } catch (...) {
        negInfHashOk = false;
    }
    
    std::cout << "\n--- Numerical Robustness Tests ---\n";
    std::cout << "Extreme values hash: " << (extremeHashOk ? "OK" : "Failed") << std::endl;
    std::cout << "NaN value hash: " << (nanHashOk ? "OK" : "Failed") << std::endl;
    std::cout << "Infinity hash: " << (infHashOk ? "OK" : "Failed") << std::endl;
    std::cout << "Negative infinity hash: " << (negInfHashOk ? "OK" : "Failed") << std::endl;
    
    if (extremeHashOk) std::cout << "Extreme hash: 0x" << std::hex << extremeHash << std::dec << std::endl;
    if (nanHashOk) std::cout << "NaN hash: 0x" << std::hex << nanHash << std::dec << std::endl;
    if (infHashOk) std::cout << "Infinity hash: 0x" << std::hex << infHash << std::dec << std::endl;
    if (negInfHashOk) std::cout << "Negative infinity hash: 0x" << std::hex << negInfHash << std::dec << std::endl;
    
    // We expect the implementation to handle these cases gracefully
    EXPECT_TRUE(extremeHashOk) << "Should handle extreme values";
    EXPECT_TRUE(nanHashOk) << "Should handle NaN values";
    EXPECT_TRUE(infHashOk) << "Should handle infinity";
    EXPECT_TRUE(negInfHashOk) << "Should handle negative infinity";
    
    // Test comparison of materials with these special values
    if (nanHashOk && infHashOk) {
        bool nanInfEquiv = scientificKey->areEquivalent(nanMaterial, infMaterial);
        EXPECT_FALSE(nanInfEquiv) << "NaN and Infinity should not be equivalent";
    }
    
    if (infHashOk && negInfHashOk) {
        bool infNegInfEquiv = scientificKey->areEquivalent(infMaterial, negInfMaterial);
        EXPECT_FALSE(infNegInfEquiv) << "Positive and negative infinity should not be equivalent";
    }
    
    // Test denormalized values
    auto denormalMaterial = createTestMaterial();
    denormalMaterial->setProperty(Material::MaterialProperty::THERMAL_EXPANSION,
                                std::numeric_limits<double>::denorm_min());
    
    bool denormalHashOk = true;
    try {
        scientificKey->hash(denormalMaterial);
    } catch (...) {
        denormalHashOk = false;
    }
    
    std::cout << "Denormalized value hash: " << (denormalHashOk ? "OK" : "Failed") << std::endl;
    EXPECT_TRUE(denormalHashOk) << "Should handle denormalized values";
}

// Test for NaN propagation
TEST_F(ScientificEquivAdvancedTest, NaNPropagation) {
    // Create materials with NaN in different properties
    auto material1 = createTestMaterial();
    auto material2 = createTestMaterial();
    auto material3 = createTestMaterial();
    
    // Set one NaN in each material in a different property
    material1->setProperty(Material::MaterialProperty::DENSITY, 
                          std::numeric_limits<double>::quiet_NaN());
    
    material2->setProperty(Material::MaterialProperty::THERMAL_CONDUCTIVITY, 
                          std::numeric_limits<double>::quiet_NaN());
    
    material3->setProperty(Material::MaterialProperty::DYNAMIC_VISCOSITY, 
                          std::numeric_limits<double>::quiet_NaN());
    
    // Calculate hashes
    uint64_t hash1 = 0, hash2 = 0, hash3 = 0;
    bool hash1Ok = true, hash2Ok = true, hash3Ok = true;
    
    try {
        hash1 = scientificKey->hash(material1);
    } catch (...) {
        hash1Ok = false;
    }
    
    try {
        hash2 = scientificKey->hash(material2);
    } catch (...) {
        hash2Ok = false;
    }
    
    try {
        hash3 = scientificKey->hash(material3);
    } catch (...) {
        hash3Ok = false;
    }
    
    std::cout << "\n--- NaN Propagation Tests ---\n";
    std::cout << "Material with NaN density hash: " << (hash1Ok ? "OK" : "Failed") << std::endl;
    std::cout << "Material with NaN thermal conductivity hash: " << (hash2Ok ? "OK" : "Failed") << std::endl;
    std::cout << "Material with NaN viscosity hash: " << (hash3Ok ? "OK" : "Failed") << std::endl;
    
    if (hash1Ok && hash2Ok && hash3Ok) {
        // All hashes were calculated successfully
        // Check if all NaN materials have different hashes
        bool allDifferent = (hash1 != hash2) && (hash1 != hash3) && (hash2 != hash3);
        std::cout << "All NaN materials have different hashes: " << (allDifferent ? "Yes" : "No") << std::endl;
        
        // Verify hash consistency
        uint64_t hash1Repeat = scientificKey->hash(material1);
        bool hashesConsistent = (hash1 == hash1Repeat);
        std::cout << "NaN material hash consistency: " << (hashesConsistent ? "Yes" : "No") << std::endl;
        
        EXPECT_TRUE(hashesConsistent) << "Hash calculation should be consistent even with NaN";
    }
    
    // Test equivalence with NaN
    bool nanEquivalence = false;
    try {
        nanEquivalence = scientificKey->areEquivalent(material1, material1);
    } catch (...) {
        std::cout << "Equivalence check with same NaN material failed with exception" << std::endl;
    }
    
    std::cout << "NaN self-equivalence: " << (nanEquivalence ? "Equivalent" : "Not Equivalent") << std::endl;
    
    // Test NaN in different properties
    bool diffNanEquivalence = false;
    try {
        diffNanEquivalence = scientificKey->areEquivalent(material1, material2);
    } catch (...) {
        std::cout << "Equivalence check with different NaN materials failed with exception" << std::endl;
    }
    
    std::cout << "Different NaN properties equivalence: " << (diffNanEquivalence ? "Equivalent" : "Not Equivalent") << std::endl;
    EXPECT_FALSE(diffNanEquivalence) << "Materials with NaN in different properties should not be equivalent";
}

// Test for zero-valued properties
TEST_F(ScientificEquivAdvancedTest, ZeroValueProperties) {
    // Create materials with zero values in different properties
    auto standardMaterial = createTestMaterial();
    auto zeroVisc = std::make_shared<Material>(*standardMaterial);
    zeroVisc->setProperty(Material::MaterialProperty::DYNAMIC_VISCOSITY, 0.0);
    
    auto zeroExpansion = std::make_shared<Material>(*standardMaterial);
    zeroExpansion->setProperty(Material::MaterialProperty::THERMAL_EXPANSION, 0.0);
    
    auto zeroTension = std::make_shared<Material>(*standardMaterial);
    zeroTension->setProperty(Material::MaterialProperty::SURFACE_TENSION, 0.0);
    
    // Calculate hashes
    std::cout << "\n--- Zero Value Properties Tests ---\n";
    
    uint64_t standardHash = scientificKey->hash(standardMaterial);
    uint64_t zeroViscHash = scientificKey->hash(zeroVisc);
    uint64_t zeroExpansionHash = scientificKey->hash(zeroExpansion);
    uint64_t zeroTensionHash = scientificKey->hash(zeroTension);
    
    std::cout << "Standard material hash: 0x" << std::hex << standardHash << std::dec << std::endl;
    std::cout << "Zero viscosity hash: 0x" << std::hex << zeroViscHash << std::dec << std::endl;
    std::cout << "Zero thermal expansion hash: 0x" << std::hex << zeroExpansionHash << std::dec << std::endl;
    std::cout << "Zero surface tension hash: 0x" << std::hex << zeroTensionHash << std::dec << std::endl;
    
    // Zero values should result in different hashes
    EXPECT_NE(standardHash, zeroViscHash) << "Zero viscosity should have different hash";
    EXPECT_NE(standardHash, zeroExpansionHash) << "Zero thermal expansion should have different hash";
    EXPECT_NE(standardHash, zeroTensionHash) << "Zero surface tension should have different hash";
    
    // Test near-zero values vs. zero
    auto nearZeroVisc = std::make_shared<Material>(*standardMaterial);
    nearZeroVisc->setProperty(Material::MaterialProperty::DYNAMIC_VISCOSITY, 1e-15);
    
    // Compare equivalence
    bool zeroVsNearZero = scientificKey->areEquivalent(zeroVisc, nearZeroVisc);
    std::cout << "Zero vs. near-zero viscosity: " << (zeroVsNearZero ? "Equivalent" : "Not Equivalent") << std::endl;
    
    // Test with different zeros
    auto allZeros = createTestMaterial(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    uint64_t allZerosHash = scientificKey->hash(allZeros);
    std::cout << "All zeros hash: 0x" << std::hex << allZerosHash << std::dec << std::endl;
    
    // Even a material with all zeros should have a valid, consistent hash
    EXPECT_NE(allZerosHash, 0ULL) << "All zeros material should not hash to zero";
    EXPECT_EQ(allZerosHash, scientificKey->hash(allZeros)) << "All zeros hash should be consistent";
}

// Test for underflow and overflow conditions
TEST_F(ScientificEquivAdvancedTest, UnderflowOverflow) {
    auto baseMaterial = createTestMaterial();
    
    // Test with values approaching overflow
    auto overflowMaterial = std::make_shared<Material>(*baseMaterial);
    overflowMaterial->setProperty(
        Material::MaterialProperty::SPECIFIC_HEAT, 
        std::numeric_limits<double>::max() / 2.0
    );
    
    // Test with values approaching underflow
    auto underflowMaterial = std::make_shared<Material>(*baseMaterial);
    underflowMaterial->setProperty(
        Material::MaterialProperty::THERMAL_EXPANSION, 
        std::numeric_limits<double>::min() * 2.0
    );
    
    // Create materials with slight variations that might cause overflow/underflow
    auto overflowPlus = std::make_shared<Material>(*overflowMaterial);
    overflowPlus->setProperty(
        Material::MaterialProperty::SPECIFIC_HEAT, 
        overflowMaterial->getProperty(Material::MaterialProperty::SPECIFIC_HEAT) * 1.1
    );
    
    auto underflowMinus = std::make_shared<Material>(*underflowMaterial);
    underflowMinus->setProperty(
        Material::MaterialProperty::THERMAL_EXPANSION, 
        underflowMaterial->getProperty(Material::MaterialProperty::THERMAL_EXPANSION) * 0.9
    );
    
    std::cout << "\n--- Underflow/Overflow Tests ---\n";
    
    // Calculate hashes
    bool overflowHashOk = true;
    bool underflowHashOk = true;
    bool overflowPlusHashOk = true;
    bool underflowMinusHashOk = true;
    
    try {
        // Store the hashes in local variables inside the try blocks
        // to avoid unused variable warnings
        uint64_t overflowHash = scientificKey->hash(overflowMaterial);
        std::cout << "Near overflow hash value: 0x" << std::hex << overflowHash << std::dec << std::endl;
    } catch (...) {
        overflowHashOk = false;
    }
    
    try {
        uint64_t underflowHash = scientificKey->hash(underflowMaterial);
        std::cout << "Near underflow hash value: 0x" << std::hex << underflowHash << std::dec << std::endl;
    } catch (...) {
        underflowHashOk = false;
    }
    
    try {
        uint64_t overflowPlusHash = scientificKey->hash(overflowPlus);
        std::cout << "Potential overflow hash value: 0x" << std::hex << overflowPlusHash << std::dec << std::endl;
    } catch (...) {
        overflowPlusHashOk = false;
    }
    
    try {
        uint64_t underflowMinusHash = scientificKey->hash(underflowMinus);
        std::cout << "Potential underflow hash value: 0x" << std::hex << underflowMinusHash << std::dec << std::endl;
    } catch (...) {
        underflowMinusHashOk = false;
    }
    
    std::cout << "Near overflow hash: " << (overflowHashOk ? "OK" : "Failed") << std::endl;
    std::cout << "Near underflow hash: " << (underflowHashOk ? "OK" : "Failed") << std::endl;
    std::cout << "Potential overflow hash: " << (overflowPlusHashOk ? "OK" : "Failed") << std::endl;
    std::cout << "Potential underflow hash: " << (underflowMinusHashOk ? "OK" : "Failed") << std::endl;
    
    // The implementation should handle all these cases gracefully
    EXPECT_TRUE(overflowHashOk) << "Should handle near overflow values";
    EXPECT_TRUE(underflowHashOk) << "Should handle near underflow values";
    
    // Test equivalence at boundaries
    if (overflowHashOk && overflowPlusHashOk) {
        bool overflowEquiv = scientificKey->areEquivalent(overflowMaterial, overflowPlus);
        std::cout << "Near overflow equivalence: " << (overflowEquiv ? "Equivalent" : "Not Equivalent") << std::endl;
    }
    
    if (underflowHashOk && underflowMinusHashOk) {
        bool underflowEquiv = scientificKey->areEquivalent(underflowMaterial, underflowMinus);
        std::cout << "Near underflow equivalence: " << (underflowEquiv ? "Equivalent" : "Not Equivalent") << std::endl;
    }
}

