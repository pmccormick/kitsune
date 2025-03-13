/**
 * @file scientific_mequiv_enhanced_numerical_tests.cpp
 * @brief Enhanced numerical tests for Scientific Material Equivalence
 */
#include "scientific_mequiv_advanced_test_base.h"
#include <gtest/gtest.h>
#include <random>
#include <cmath>
#include <algorithm>
#include <vector>
#include <string>

// Test numerical precision loss scenarios
TEST_F(ScientificEquivAdvancedTest, PrecisionLossTests) {
    // Create a reference material
    auto baseMaterial = createTestMaterial();
    
    std::cout << "\n--- Numerical Precision Loss Tests ---\n";
    
    // Test 1: Adding very small values to large ones
    // Create a material with a very large density value
    auto largeDensityMaterial = std::make_shared<Material>(*baseMaterial);
    double largeValue = 1.0e15;
    largeDensityMaterial->setProperty(Material::MaterialProperty::DENSITY, largeValue);
    
    // Create two variants with small additions that should be below precision
    auto smallAddition1 = std::make_shared<Material>(*largeDensityMaterial);
    smallAddition1->setProperty(Material::MaterialProperty::DENSITY, largeValue + 0.1);
    
    auto smallAddition2 = std::make_shared<Material>(*largeDensityMaterial);
    smallAddition2->setProperty(Material::MaterialProperty::DENSITY, largeValue + 1.0);
    
    // Test hash equivalence
    uint64_t largeHash = scientificKey->hash(largeDensityMaterial);
    uint64_t smallAdd1Hash = scientificKey->hash(smallAddition1);
    uint64_t smallAdd2Hash = scientificKey->hash(smallAddition2);
    
    std::cout << "Large value (1e15) hash: 0x" << std::hex << largeHash << std::dec << std::endl;
    std::cout << "Large value + 0.1 hash: 0x" << std::hex << smallAdd1Hash << std::dec << std::endl;
    std::cout << "Large value + 1.0 hash: 0x" << std::hex << smallAdd2Hash << std::dec << std::endl;
    
    bool smallAdd1Equiv = scientificKey->areEquivalent(largeDensityMaterial, smallAddition1);
    bool smallAdd2Equiv = scientificKey->areEquivalent(largeDensityMaterial, smallAddition2);
    
    std::cout << "Large value vs. large value + 0.1 equivalence: " 
              << (smallAdd1Equiv ? "Equivalent" : "Not Equivalent") << std::endl;
    std::cout << "Large value vs. large value + 1.0 equivalence: " 
              << (smallAdd2Equiv ? "Equivalent" : "Not Equivalent") << std::endl;
    
    // We expect these to be equivalent due to precision loss
    EXPECT_TRUE(smallAdd1Equiv) << "Small additions to large values should be lost in precision";
    
    // Test 2: Subtracting nearly equal large numbers
    auto nearEqual1 = std::make_shared<Material>(*baseMaterial);
    auto nearEqual2 = std::make_shared<Material>(*baseMaterial);
    
    double val1 = 1.0e10;
    double val2 = val1 + 1.0e-10; // Very small difference
    
    nearEqual1->setProperty(Material::MaterialProperty::THERMAL_CONDUCTIVITY, val1);
    nearEqual2->setProperty(Material::MaterialProperty::THERMAL_CONDUCTIVITY, val2);
    
    bool nearEqualEquiv = scientificKey->areEquivalent(nearEqual1, nearEqual2);
    std::cout << "Near-equal large values equivalence: " 
              << (nearEqualEquiv ? "Equivalent" : "Not Equivalent") << std::endl;
    
    // We expect these to be equivalent due to precision loss
    EXPECT_TRUE(nearEqualEquiv) << "Near-equal large values should be equivalent due to precision limits";
    
    // Test 3: Small values compared to epsilon
    auto epsilonMaterial = std::make_shared<Material>(*baseMaterial);
    epsilonMaterial->setProperty(Material::MaterialProperty::SURFACE_TENSION, std::numeric_limits<double>::epsilon() * 10.0);
    
    auto doubleEpsilonMaterial = std::make_shared<Material>(*baseMaterial);
    doubleEpsilonMaterial->setProperty(Material::MaterialProperty::SURFACE_TENSION, std::numeric_limits<double>::epsilon() * 20.0);
    
    bool epsilonEquiv = scientificKey->areEquivalent(epsilonMaterial, doubleEpsilonMaterial);
    std::cout << "Epsilon-scale values equivalence: " 
              << (epsilonEquiv ? "Equivalent" : "Not Equivalent") << std::endl;
    
    // Check if hashing remains consistent with very small values
    uint64_t epsilonHash1 = scientificKey->hash(epsilonMaterial);
    uint64_t epsilonHash2 = scientificKey->hash(epsilonMaterial); // Same material, should get same hash
    
    std::cout << "Epsilon hash consistency: " << ((epsilonHash1 == epsilonHash2) ? "Consistent" : "Inconsistent") << std::endl;
    EXPECT_EQ(epsilonHash1, epsilonHash2) << "Hash should be consistent even with epsilon-scale values";
}

// Test property interaction effects in equivalence determination
TEST_F(ScientificEquivAdvancedTest, PropertyInteractionTests) {
    // Create a reference material
    auto baseMaterial = createTestMaterial();
    
    std::cout << "\n--- Property Interaction Tests ---\n";
    
    // Test 1: Counteracting changes (one property increased, one decreased)
    auto counteractingMaterial = std::make_shared<Material>(*baseMaterial);
    
    // Apply +2% to density and -2% to viscosity
    double baseDensity = baseMaterial->getProperty(Material::MaterialProperty::DENSITY);
    double baseViscosity = baseMaterial->getProperty(Material::MaterialProperty::DYNAMIC_VISCOSITY);
    
    counteractingMaterial->setProperty(Material::MaterialProperty::DENSITY, baseDensity * 1.02);
    counteractingMaterial->setProperty(Material::MaterialProperty::DYNAMIC_VISCOSITY, baseViscosity * 0.98);
    
    bool counteractEquiv = scientificKey->areEquivalent(baseMaterial, counteractingMaterial);
    std::cout << "Counteracting changes (density +2%, viscosity -2%): " 
              << (counteractEquiv ? "Equivalent" : "Not Equivalent") << std::endl;
    
    // Test 2: Amplifying changes (multiple properties changed in same direction)
    auto amplifyingMaterial = std::make_shared<Material>(*baseMaterial);
    
    // Apply small changes to multiple properties (all +0.5%)
    for (int propIndex = 0; propIndex < 7; propIndex++) {
        Material::MaterialProperty prop = static_cast<Material::MaterialProperty>(propIndex);
        double baseValue = baseMaterial->getProperty(prop);
        amplifyingMaterial->setProperty(prop, baseValue * 1.005);
    }
    
    bool amplifyEquiv = scientificKey->areEquivalent(baseMaterial, amplifyingMaterial);
    std::cout << "Amplifying changes (all properties +0.5%): " 
              << (amplifyEquiv ? "Equivalent" : "Not Equivalent") << std::endl;
    
    // Test 3: Cumulative effect of very small changes
    auto cumulativeMaterial = std::make_shared<Material>(*baseMaterial);
    
    // Apply very small changes to all properties (+0.2% each)
    for (int propIndex = 0; propIndex < 7; propIndex++) {
        Material::MaterialProperty prop = static_cast<Material::MaterialProperty>(propIndex);
        double baseValue = baseMaterial->getProperty(prop);
        cumulativeMaterial->setProperty(prop, baseValue * 1.002);
    }
    
    bool cumulativeEquiv = scientificKey->areEquivalent(baseMaterial, cumulativeMaterial);
    std::cout << "Cumulative small changes (all properties +0.2%): " 
              << (cumulativeEquiv ? "Equivalent" : "Not Equivalent") << std::endl;
    
    // Test hash differences
    uint64_t baseHash = scientificKey->hash(baseMaterial);
    uint64_t counterHash = scientificKey->hash(counteractingMaterial);
    uint64_t amplifyHash = scientificKey->hash(amplifyingMaterial);
    uint64_t cumulativeHash = scientificKey->hash(cumulativeMaterial);
    
    std::cout << "Base material hash: 0x" << std::hex << baseHash << std::dec << std::endl;
    std::cout << "Counteracting changes hash: 0x" << std::hex << counterHash << std::dec << std::endl;
    std::cout << "Amplifying changes hash: 0x" << std::hex << amplifyHash << std::dec << std::endl;
    std::cout << "Cumulative small changes hash: 0x" << std::hex << cumulativeHash << std::dec << std::endl;
}

// Test behavior at magnitude transition boundaries
TEST_F(ScientificEquivAdvancedTest, MagnitudeTransitionTests) {
    // Create reference material
    auto baseMaterial = createTestMaterial();
    
    std::cout << "\n--- Magnitude Transition Tests ---\n";
    
    // Test points around powers of 10 to check for transitional behavior
    std::vector<double> testPoints = {0.099, 0.1, 0.101, 0.999, 1.0, 1.001, 9.99, 10.0, 10.01};
    
    // Test viscosity transitions
    std::cout << "Testing viscosity magnitude transitions:" << std::endl;
    std::cout << std::left << std::setw(10) << "Value" 
              << std::setw(12) << "Hash"
              << std::setw(15) << "Equiv to 1.01x"
              << std::setw(15) << "Equiv to 0.99x" << std::endl;
    std::cout << std::string(52, '-') << std::endl;
    
    std::map<double, uint64_t> viscHashes;
    
    for (double val : testPoints) {
        auto material = std::make_shared<Material>(*baseMaterial);
        material->setProperty(Material::MaterialProperty::DYNAMIC_VISCOSITY, val);
        
        uint64_t hash = scientificKey->hash(material);
        viscHashes[val] = hash;
        
        // Create slightly varied materials (±1%)
        auto materialPlus = std::make_shared<Material>(*material);
        materialPlus->setProperty(Material::MaterialProperty::DYNAMIC_VISCOSITY, val * 1.01);
        
        auto materialMinus = std::make_shared<Material>(*material);
        materialMinus->setProperty(Material::MaterialProperty::DYNAMIC_VISCOSITY, val * 0.99);
        
        bool equivPlus = scientificKey->areEquivalent(material, materialPlus);
        bool equivMinus = scientificKey->areEquivalent(material, materialMinus);
        
        std::cout << std::left << std::setw(10) << val 
                  << std::setw(12) << std::hex << (hash & 0xFFFF) << std::dec
                  << std::setw(15) << (equivPlus ? "Equivalent" : "Different")
                  << std::setw(15) << (equivMinus ? "Equivalent" : "Different") << std::endl;
    }
    
    // Verify hash transitions
    std::cout << "\nHash differences between consecutive points:" << std::endl;
    for (size_t i = 0; i < testPoints.size() - 1; i++) {
        double val1 = testPoints[i];
        double val2 = testPoints[i+1];
        uint64_t hash1 = viscHashes[val1];
        uint64_t hash2 = viscHashes[val2];
        
        std::cout << val1 << " -> " << val2 << ": " 
                  << (hash1 == hash2 ? "Same hash" : "Different hash") << std::endl;
    }
    
    // Test logarithmic vs. linear at boundaries
    std::cout << "\nLogarithmic vs. Linear Scale at Boundaries:" << std::endl;
    
    auto logKey = std::make_shared<ScientificMaterialEquivalence>();
    try {
        logKey->setPropertyRange(
            Material::MaterialProperty::DYNAMIC_VISCOSITY,
            1e-6, 1e6,
            ScientificMaterialEquivalence::ScaleType::LOGARITHMIC
        );
    } catch (...) {
        std::cout << "Logarithmic scale not supported." << std::endl;
    }
    
    auto linearKey = std::make_shared<ScientificMaterialEquivalence>();
    try {
        linearKey->setPropertyRange(
            Material::MaterialProperty::DYNAMIC_VISCOSITY,
            1e-6, 1e6,
            ScientificMaterialEquivalence::ScaleType::LINEAR
        );
    } catch (...) {
        std::cout << "Linear scale not supported." << std::endl;
    }
    
    // Test points with same percentage difference but at different magnitudes
    std::vector<double> testMagnitudes = {1e-5, 1e-3, 1e-1, 1e1, 1e3, 1e5};
    
    std::cout << std::left << std::setw(12) << "Base Value"
              << std::setw(15) << "Log Scale Hash"
              << std::setw(15) << "Linear Scale Hash"
              << std::setw(18) << "Log 1% Difference"
              << std::setw(18) << "Linear 1% Difference" << std::endl;
    std::cout << std::string(78, '-') << std::endl;
    
    for (double baseVal : testMagnitudes) {
        // Create base material at this magnitude
        auto magBaseMaterial = std::make_shared<Material>(*baseMaterial);
        magBaseMaterial->setProperty(Material::MaterialProperty::DYNAMIC_VISCOSITY, baseVal);
        
        // Create modified material with 1% difference
        auto magModMaterial = std::make_shared<Material>(*magBaseMaterial);
        magModMaterial->setProperty(Material::MaterialProperty::DYNAMIC_VISCOSITY, baseVal * 1.01);
        
        // Get hashes with both keys
        uint64_t logBaseHash = logKey->hash(magBaseMaterial);
        uint64_t linearBaseHash = linearKey->hash(magBaseMaterial);
        
        // Check equivalence with both keys
        bool logEquiv = logKey->areEquivalent(magBaseMaterial, magModMaterial);
        bool linearEquiv = linearKey->areEquivalent(magBaseMaterial, magModMaterial);
        
        std::cout << std::left << std::setw(12) << baseVal
                  << std::setw(15) << std::hex << (logBaseHash & 0xFFFF) << std::dec
                  << std::setw(15) << std::hex << (linearBaseHash & 0xFFFF) << std::dec
                  << std::setw(18) << (logEquiv ? "Equivalent" : "Different")
                  << std::setw(18) << (linearEquiv ? "Equivalent" : "Different") << std::endl;
    }
}

// Test custom range impact on equivalence and hashing
TEST_F(ScientificEquivAdvancedTest, CustomRangeTests) {
    // Create reference material
    auto baseMaterial = createTestMaterial();
    
    std::cout << "\n--- Custom Range Tests ---\n";
    
    // Create keys with different custom ranges for density
    auto defaultKey = scientificKey;
    
    auto narrowRangeKey = std::make_shared<ScientificMaterialEquivalence>();
    narrowRangeKey->setPropertyRange(Material::MaterialProperty::DENSITY, 900.0, 1100.0);
    
    auto wideRangeKey = std::make_shared<ScientificMaterialEquivalence>();
    wideRangeKey->setPropertyRange(Material::MaterialProperty::DENSITY, 1.0, 20000.0);
    
    // Materials with increasing density differences
    std::vector<double> percentChanges = {0.1, 0.5, 1.0, 2.0, 5.0};
    
    std::cout << "Effect of custom property ranges on density sensitivity:" << std::endl;
    std::cout << std::left << std::setw(10) << "% Change" 
              << std::setw(20) << "Default Range"
              << std::setw(20) << "Narrow Range"
              << std::setw(20) << "Wide Range" << std::endl;
    std::cout << std::string(70, '-') << std::endl;
    
    for (double pct : percentChanges) {
        auto modifiedMaterial = std::make_shared<Material>(*baseMaterial);
        double baseDensity = baseMaterial->getProperty(Material::MaterialProperty::DENSITY);
        double newDensity = baseDensity * (1.0 + pct/100.0);
        modifiedMaterial->setProperty(Material::MaterialProperty::DENSITY, newDensity);
        
        bool defaultEquiv = defaultKey->areEquivalent(baseMaterial, modifiedMaterial);
        bool narrowEquiv = narrowRangeKey->areEquivalent(baseMaterial, modifiedMaterial);
        bool wideEquiv = wideRangeKey->areEquivalent(baseMaterial, modifiedMaterial);
        
        std::cout << std::left << std::setw(10) << pct
                  << std::setw(20) << (defaultEquiv ? "Equivalent" : "Different")
                  << std::setw(20) << (narrowEquiv ? "Equivalent" : "Different")
                  << std::setw(20) << (wideEquiv ? "Equivalent" : "Different") << std::endl;
    }
    
    // Test behavior when value is outside the configured range
    auto outOfRangeMaterial = std::make_shared<Material>(*baseMaterial);
    outOfRangeMaterial->setProperty(Material::MaterialProperty::DENSITY, 1200.0); // Outside narrow range
    
    bool defaultOutOfRange = defaultKey->areEquivalent(baseMaterial, outOfRangeMaterial);
    bool narrowOutOfRange = narrowRangeKey->areEquivalent(baseMaterial, outOfRangeMaterial);
    
    std::cout << "\nBehavior with out-of-range values:" << std::endl;
    std::cout << "Default range with value 1200 (20% change): " 
              << (defaultOutOfRange ? "Equivalent" : "Different") << std::endl;
    std::cout << "Narrow range (900-1100) with value 1200: " 
              << (narrowOutOfRange ? "Equivalent" : "Different") << std::endl;
    
    // Test hash stability at range boundaries
    auto atMinMaterial = std::make_shared<Material>(*baseMaterial);
    atMinMaterial->setProperty(Material::MaterialProperty::DENSITY, 900.0); // At narrow range min
    
    auto nearMinMaterial = std::make_shared<Material>(*baseMaterial);
    nearMinMaterial->setProperty(Material::MaterialProperty::DENSITY, 901.0); // Just above narrow range min
    
    auto atMaxMaterial = std::make_shared<Material>(*baseMaterial);
    atMaxMaterial->setProperty(Material::MaterialProperty::DENSITY, 1100.0); // At narrow range max
    
    auto nearMaxMaterial = std::make_shared<Material>(*baseMaterial);
    nearMaxMaterial->setProperty(Material::MaterialProperty::DENSITY, 1099.0); // Just below narrow range max
    
    // Get hashes at boundaries
    uint64_t minHash = narrowRangeKey->hash(atMinMaterial);
    uint64_t nearMinHash = narrowRangeKey->hash(nearMinMaterial);
    uint64_t maxHash = narrowRangeKey->hash(atMaxMaterial);
    uint64_t nearMaxHash = narrowRangeKey->hash(nearMaxMaterial);
    
    std::cout << "\nHashes at range boundaries:" << std::endl;
    std::cout << "At minimum (900): 0x" << std::hex << minHash << std::dec << std::endl;
    std::cout << "Near minimum (901): 0x" << std::hex << nearMinHash << std::dec << std::endl;
    std::cout << "At maximum (1100): 0x" << std::hex << maxHash << std::dec << std::endl;
    std::cout << "Near maximum (1099): 0x" << std::hex << nearMaxHash << std::dec << std::endl;
}

// Test tolerance scaling behavior across property values
TEST_F(ScientificEquivAdvancedTest, ToleranceScalingTests) {
    std::cout << "\n--- Tolerance Scaling Tests ---\n";
    
    // Create a standard material for testing
    auto baseMaterial = createTestMaterial();
    
    // Create keys with different tolerances
    auto tightKey = std::make_shared<ScientificMaterialEquivalence>();
    tightKey->setEquivalenceTolerance(0.005); // 0.5%
    
    auto defaultKey = scientificKey; // Default tolerance (likely 1%)
    
    auto looseKey = std::make_shared<ScientificMaterialEquivalence>();
    looseKey->setEquivalenceTolerance(0.05); // 5%
    
    // Test tolerance scaling across different orders of magnitude
    std::vector<double> densityMagnitudes = {0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0};
    
    std::cout << "Testing if tolerance scales proportionally across magnitudes:" << std::endl;
    std::cout << std::left << std::setw(15) << "Base Density"
              << std::setw(15) << "% Change"
              << std::setw(15) << "Tight (0.5%)"
              << std::setw(15) << "Default (~1%)"
              << std::setw(15) << "Loose (5%)" << std::endl;
    std::cout << std::string(75, '-') << std::endl;
    
    // For each base magnitude, test with different percentage changes
    for (double baseDensity : densityMagnitudes) {
        // Create a material with this base density
        auto testBaseMaterial = std::make_shared<Material>(*baseMaterial);
        testBaseMaterial->setProperty(Material::MaterialProperty::DENSITY, baseDensity);
        
        // Test different percentage changes
        std::vector<double> percentChanges = {0.1, 0.5, 1.0, 2.0, 5.0, 10.0};
        
        for (double pct : percentChanges) {
            auto testModMaterial = std::make_shared<Material>(*testBaseMaterial);
            testModMaterial->setProperty(Material::MaterialProperty::DENSITY, baseDensity * (1.0 + pct/100.0));
            
            bool tightEquiv = tightKey->areEquivalent(testBaseMaterial, testModMaterial);
            bool defaultEquiv = defaultKey->areEquivalent(testBaseMaterial, testModMaterial);
            bool looseEquiv = looseKey->areEquivalent(testBaseMaterial, testModMaterial);
            
            std::cout << std::left << std::setw(15) << baseDensity
                      << std::setw(15) << pct
                      << std::setw(15) << (tightEquiv ? "Equivalent" : "Different")
                      << std::setw(15) << (defaultEquiv ? "Equivalent" : "Different")
                      << std::setw(15) << (looseEquiv ? "Equivalent" : "Different") << std::endl;
        }
        
        std::cout << std::string(75, '-') << std::endl;
    }
    
    // Test tolerance behavior with fractional and percentage changes
    std::cout << "\nTolerance behavior with absolute changes vs percentage changes:" << std::endl;
    
    // Test with small fractional changes to very different base values
    auto smallBaseMaterial = std::make_shared<Material>(*baseMaterial);
    smallBaseMaterial->setProperty(Material::MaterialProperty::DENSITY, 0.01);
    
    auto largeBaseMaterial = std::make_shared<Material>(*baseMaterial);
    largeBaseMaterial->setProperty(Material::MaterialProperty::DENSITY, 10000.0);
    
    // Add the same absolute value to both
    auto smallPlusMaterial = std::make_shared<Material>(*smallBaseMaterial);
    smallPlusMaterial->setProperty(Material::MaterialProperty::DENSITY, 0.01 + 0.0001);
    
    auto largePlusMaterial = std::make_shared<Material>(*largeBaseMaterial);
    largePlusMaterial->setProperty(Material::MaterialProperty::DENSITY, 10000.0 + 0.0001);
    
    bool smallAbsChange = defaultKey->areEquivalent(smallBaseMaterial, smallPlusMaterial);
    bool largeAbsChange = defaultKey->areEquivalent(largeBaseMaterial, largePlusMaterial);
    
    std::cout << "0.01 + 0.0001 (1% change): " << (smallAbsChange ? "Equivalent" : "Different") << std::endl;
    std::cout << "10000.0 + 0.0001 (0.000001% change): " << (largeAbsChange ? "Equivalent" : "Different") << std::endl;
    
    // Now test with the same percentage change
    auto smallPctMaterial = std::make_shared<Material>(*smallBaseMaterial);
    smallPctMaterial->setProperty(Material::MaterialProperty::DENSITY, 0.01 * 1.01);
    
    auto largePctMaterial = std::make_shared<Material>(*largeBaseMaterial);
    largePctMaterial->setProperty(Material::MaterialProperty::DENSITY, 10000.0 * 1.01);
    
    bool smallPctChange = defaultKey->areEquivalent(smallBaseMaterial, smallPctMaterial);
    bool largePctChange = defaultKey->areEquivalent(largeBaseMaterial, largePctMaterial);
    
    std::cout << "0.01 * 1.01 (1% change): " << (smallPctChange ? "Equivalent" : "Different") << std::endl;
    std::cout << "10000.0 * 1.01 (1% change): " << (largePctChange ? "Equivalent" : "Different") << std::endl;
    
    // We expect percentage changes to be treated similarly regardless of magnitude,
    // but absolute changes to have very different effects depending on magnitude
    EXPECT_EQ(smallPctChange, largePctChange) << "Same percentage changes should be treated consistently";
}

