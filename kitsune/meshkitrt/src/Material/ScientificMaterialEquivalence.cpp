/**
 * @file ScientificMaterialEquivalence.cpp
 * @brief Implementation of Scientific Material Equivalence Class
 */
#include "ScientificMaterialEquivalence.h"
#include <cmath>
#include <limits>
#include <algorithm>
#include <iostream>
#include <bitset>

// Constructor
ScientificMaterialEquivalence::ScientificMaterialEquivalence() 
    : m_hashBits(64), m_compatibilityVersion(0), m_name("Scientific Key"), m_equivalenceTolerance(0.01) {
    initialize();
}

// Constructor with simulation type
ScientificMaterialEquivalence::ScientificMaterialEquivalence(SimulationType type, size_t hashBits) 
    : m_hashBits(hashBits), m_compatibilityVersion(0), m_equivalenceTolerance(0.01) {
    
    initialize();
    
    // Configure for specific simulation type
    switch (type) {
        case SimulationType::CFD_MULTIPHASE:
            configureForCFD(true, hashBits);
            break;
        case SimulationType::THERMAL:
            configureForThermal(hashBits);
            break;
        case SimulationType::STRUCTURAL:
            configureForStructural(hashBits);
            break;
        case SimulationType::GENERIC:
        default:
            // Already initialized with generic settings
            break;
    }
}

// Destructor
ScientificMaterialEquivalence::~ScientificMaterialEquivalence() {
    // Nothing to clean up
}

// Initialize default configuration
void ScientificMaterialEquivalence::initialize() {
    // Default bit allocation (total should sum to m_hashBits)
    m_propertyBits[Material::MaterialProperty::DENSITY] = 10;
    m_propertyBits[Material::MaterialProperty::DYNAMIC_VISCOSITY] = 10;
    m_propertyBits[Material::MaterialProperty::THERMAL_CONDUCTIVITY] = 10;
    m_propertyBits[Material::MaterialProperty::SPECIFIC_HEAT] = 10;
    m_propertyBits[Material::MaterialProperty::THERMAL_EXPANSION] = 8;
    m_propertyBits[Material::MaterialProperty::SURFACE_TENSION] = 8;
    m_propertyBits[Material::MaterialProperty::ELECTRICAL_CONDUCTIVITY] = 8;
    
    // Default property ranges - these are example values and should be adjusted
    // based on the specific application domain
    m_propertyRanges[Material::MaterialProperty::DENSITY] = {0.1, 20000.0};
    m_propertyRanges[Material::MaterialProperty::DYNAMIC_VISCOSITY] = {1e-6, 1e6};
    m_propertyRanges[Material::MaterialProperty::THERMAL_CONDUCTIVITY] = {0.01, 1000.0};
    m_propertyRanges[Material::MaterialProperty::SPECIFIC_HEAT] = {100.0, 10000.0};
    m_propertyRanges[Material::MaterialProperty::THERMAL_EXPANSION] = {1e-7, 1e-3};
    m_propertyRanges[Material::MaterialProperty::SURFACE_TENSION] = {0.001, 10.0};
    m_propertyRanges[Material::MaterialProperty::ELECTRICAL_CONDUCTIVITY] = {1e-16, 1e8};
    
    // Default scaling types
    m_propertyScales[Material::MaterialProperty::DENSITY] = ScaleType::LINEAR;
    m_propertyScales[Material::MaterialProperty::DYNAMIC_VISCOSITY] = ScaleType::LOGARITHMIC;
    m_propertyScales[Material::MaterialProperty::THERMAL_CONDUCTIVITY] = ScaleType::LOGARITHMIC;
    m_propertyScales[Material::MaterialProperty::SPECIFIC_HEAT] = ScaleType::LINEAR;
    m_propertyScales[Material::MaterialProperty::THERMAL_EXPANSION] = ScaleType::LOGARITHMIC;
    m_propertyScales[Material::MaterialProperty::SURFACE_TENSION] = ScaleType::LINEAR;
    m_propertyScales[Material::MaterialProperty::ELECTRICAL_CONDUCTIVITY] = ScaleType::LOGARITHMIC;
}

// Normalize property value based on defined range and scale type
double ScientificMaterialEquivalence::normalizeProperty(
    Material::MaterialProperty prop, double value) const {
    
    // Handle NaN or infinity
    if (std::isnan(value) || !std::isfinite(value)) {
        return 0.0; // Default value for invalid inputs
    }
    
    // Get range for this property
    auto rangeIt = m_propertyRanges.find(prop);
    if (rangeIt == m_propertyRanges.end()) {
        return 0.0; // No range defined
    }
    
    double min = rangeIt->second.first;
    double max = rangeIt->second.second;
    
    // Get scale type for this property
    auto scaleIt = m_propertyScales.find(prop);
    ScaleType scaleType = (scaleIt != m_propertyScales.end()) ? 
                         scaleIt->second : ScaleType::LINEAR;
    
    // Normalize based on scale type
    double normalizedValue = 0.0;
    
    if (scaleType == ScaleType::LOGARITHMIC) {
        // Ensure positive values for logarithm
        if (value <= 0.0 || min <= 0.0 || max <= 0.0) {
            return 0.0; // Cannot take logarithm of zero or negative numbers
        }
        
        double logValue = std::log10(value);
        double logMin = std::log10(min);
        double logMax = std::log10(max);
        
        // Clamp to range
        logValue = std::max(logMin, std::min(logValue, logMax));
        
        // Normalize to 0-1 range
        normalizedValue = (logValue - logMin) / (logMax - logMin);
    } else { // LINEAR
        // Clamp to range
        value = std::max(min, std::min(value, max));
        
        // Normalize to 0-1 range
        normalizedValue = (value - min) / (max - min);
    }
    
    return normalizedValue;
}

// Check if two property values are equivalent within tolerance
bool ScientificMaterialEquivalence::arePropertyValuesEquivalent(
    [[maybe_unused]] Material::MaterialProperty prop, double valueA, double valueB) const {
    
    // Handle NaN and infinities
    if (std::isnan(valueA) || std::isnan(valueB)) {
        return false; // NaN is never equivalent to anything, not even another NaN
    }
    
    if (!std::isfinite(valueA) || !std::isfinite(valueB)) {
        // Infinities are only equivalent if they have the same sign
        if (std::isinf(valueA) && std::isinf(valueB)) {
            return (valueA > 0.0) == (valueB > 0.0);
        }
        return false;
    }
    
    // Both values are finite, non-NaN numbers
    // Special case for zero values to avoid division by zero
    if (valueA == 0.0 || valueB == 0.0) {
        double absA = std::abs(valueA);
        double absB = std::abs(valueB);
        // If both are very close to zero, consider them equivalent
        if (absA < 1e-12 && absB < 1e-12) {
            return true;
        }
        // If one is zero and the other is not, use absolute comparison
        return std::abs(valueA - valueB) < m_equivalenceTolerance;
    }
    
    // For normal values, use relative comparison
    double avgValue = (std::abs(valueA) + std::abs(valueB)) / 2.0;
    double relDiff = std::abs(valueA - valueB) / avgValue;
    
    // Values are equivalent if relative difference is less than tolerance
    return relDiff < m_equivalenceTolerance;
}

// Check if two materials are equivalent
bool ScientificMaterialEquivalence::areEquivalent(
    const std::shared_ptr<Material>& a, const std::shared_ptr<Material>& b) const {
    
    // Fast path: if the same object, they are equivalent
    if (a.get() == b.get()) {
        return true;
    }
    
    // If either is null, they are not equivalent
    if (!a || !b) {
        return false;
    }
    
    // Both materials must have the same material type
    if (a->getType() != b->getType()) {
        return false;
    }
    
    // Check each property for equivalence
    for (int propIndex = 0; propIndex < static_cast<int>(Material::MaterialProperty::COUNT); ++propIndex) {
        Material::MaterialProperty prop = static_cast<Material::MaterialProperty>(propIndex);
        
        // Skip properties with 0 bits allocated (not important for this equivalence)
        auto bitsIt = m_propertyBits.find(prop);
        if (bitsIt == m_propertyBits.end() || bitsIt->second == 0) {
            continue;
        }
        
        double valueA = a->getProperty(prop);
        double valueB = b->getProperty(prop);
        
        if (!arePropertyValuesEquivalent(prop, valueA, valueB)) {
            return false;
        }
    }
    
    // If we get here, all properties are equivalent
    return true;
}

// Calculate hash for a material
//
//
// Updated hash method to improve distribution for small property variations
uint64_t ScientificMaterialEquivalence::hash(const std::shared_ptr<Material>& material) const {
    if (!material) {
        return 0; // Null pointer check
    }
    
    // Initial seed value - use a large prime number
    uint64_t hash = 14695981039346656037ULL; // FNV-1a initial value
    
    // For each property, calculate hash contribution independently
    for (int propIndex = 0; propIndex < static_cast<int>(Material::MaterialProperty::COUNT); ++propIndex) {
        Material::MaterialProperty prop = static_cast<Material::MaterialProperty>(propIndex);
        
        // Get number of bits for this property
        auto bitsIt = m_propertyBits.find(prop);
        if (bitsIt == m_propertyBits.end() || bitsIt->second == 0) {
            continue;
        }
        
        size_t bits = bitsIt->second;
        
        // Get raw property value
        double rawValue = material->getProperty(prop);
        
        // Generate a wider hash for this property (more than the allocated bits)
        // to increase randomness before we reduce it
        uint64_t propHash = hashProperty(prop, rawValue);
        
        // Extract the required number of bits using a mask
        uint64_t mask = (1ULL << bits) - 1;
        propHash &= mask;
        
        // Incorporate this property's hash into the overall hash
        // using multiplication and XOR (better mixing than bit shifting alone)
        hash ^= propHash;
        hash *= 1099511628211ULL; // FNV-1a prime multiplier for good distribution
    }
    
    // Apply an avalanche function to ensure small changes propagate
    hash = avalanche(hash);
    
    return hash;
}

#if 0
uint64_t ScientificMaterialEquivalence::hash(const std::shared_ptr<Material>& material) const {
    if (!material) {
        return 0; // Null pointer check
    }
    
    uint64_t hash = 0;
    size_t nextBitPosition = 0;
    
    // For each property, calculate hash contribution
    for (int propIndex = 0; propIndex < static_cast<int>(Material::MaterialProperty::COUNT); ++propIndex) {
        Material::MaterialProperty prop = static_cast<Material::MaterialProperty>(propIndex);
        
        // Get number of bits for this property
        auto bitsIt = m_propertyBits.find(prop);
        if (bitsIt == m_propertyBits.end() || bitsIt->second == 0) {
            continue;
        }
        
        size_t bits = bitsIt->second;
        
        // Get normalized property value (0.0-1.0)
        double normalizedValue = normalizeProperty(prop, material->getProperty(prop));
        
        // Convert to integer representation
        uint64_t maxValue = (1ULL << bits) - 1;
        uint64_t scaledValue = static_cast<uint64_t>(normalizedValue * maxValue);
        
        // Add to hash
        hash |= (scaledValue << nextBitPosition);
        nextBitPosition += bits;
    }
    
    return hash;
}
#endif 

uint64_t ScientificMaterialEquivalence::hashProperty(Material::MaterialProperty prop, double value) const {
    // Handle NaN or infinity
    if (std::isnan(value) || !std::isfinite(value)) {
        return 0; // Default value for invalid inputs
    }

    // Get range for this property
    auto rangeIt = m_propertyRanges.find(prop);
    if (rangeIt == m_propertyRanges.end()) {
        // If no range is defined, use the raw bits of the double as hash
        return *reinterpret_cast<uint64_t*>(&value);
    }

    double min = rangeIt->second.first;
    double max = rangeIt->second.second;

    // Get scale type for this property
    auto scaleIt = m_propertyScales.find(prop);
    ScaleType scaleType = (scaleIt != m_propertyScales.end()) ?
                         scaleIt->second : ScaleType::LINEAR;

    // Process value based on scale type
    uint64_t result = 0;

    if (scaleType == ScaleType::LOGARITHMIC) {
        // Ensure positive values for logarithm
        if (value <= 0.0 || min <= 0.0 || max <= 0.0) {
            return 0; // Cannot take logarithm of zero or negative numbers
        }

        double logValue = std::log10(value);
        double logMin = std::log10(min);
        double logMax = std::log10(max);

        // Scale to a range suitable for hashing
        logValue = std::max(logMin, std::min(logValue, logMax));
        double scaledValue = (logValue - logMin) / (logMax - logMin);

        // Split into integer and fractional parts for better hashing
        double intPart, fracPart;
        fracPart = std::modf(scaledValue * 1000000.0, &intPart);

        // Mix the parts
        result = static_cast<uint64_t>(intPart);
        result ^= static_cast<uint64_t>(fracPart * 1000000.0);
        result *= 1099511628211ULL; // FNV-1a prime
    } else { // LINEAR
        // Clamp to range
        value = std::max(min, std::min(value, max));

        // Normalize and scale
        double scaledValue = (value - min) / (max - min) * 1000000.0;

        // Split into integer and fractional parts
        double intPart, fracPart;
        fracPart = std::modf(scaledValue, &intPart);

        // Mix the parts
        result = static_cast<uint64_t>(intPart);
        result ^= static_cast<uint64_t>(fracPart * 1000000.0);
        result *= 1099511628211ULL; // FNV-1a prime
    }

    // Apply bit mixing for better distribution
    result = (result ^ (result >> 27)) * UINT64_C(0x94d049bb133111eb);
    result = (result ^ (result >> 31)) * UINT64_C(0xbf58476d1ce4e5b9);

    return result;
}

// New avalanche function to ensure small input changes cause large hash changes
uint64_t ScientificMaterialEquivalence::avalanche(uint64_t hash) const {
    // This is an implementation of the finalizer from MurmurHash3
    hash ^= hash >> 33;
    hash *= 0xff51afd7ed558ccdULL;
    hash ^= hash >> 33;
    hash *= 0xc4ceb9fe1a85ec53ULL;
    hash ^= hash >> 33;
    return hash;
}

// Get name of this equivalence key
std::string ScientificMaterialEquivalence::getName() const {
    return m_name;
}

// Get bit allocation for each property
std::map<Material::MaterialProperty, size_t> ScientificMaterialEquivalence::getBitAllocation() const {
    return m_propertyBits;
}

// Configure for CFD simulation
void ScientificMaterialEquivalence::configureForCFD(bool isMultiphase, size_t hashBits, bool includeHeatTransfer) {
    m_hashBits = hashBits;
    
    // Adjust name based on configuration
    m_name = isMultiphase ? "CFD Multiphase Key" : "CFD Single-Phase Key";
    if (includeHeatTransfer) {
        m_name += " with Heat Transfer";
    }
    
    // Reset bit allocation
    initialize();
    
    // For CFD simulations, density and viscosity are critical
    m_propertyBits[Material::MaterialProperty::DENSITY] = 12;
    m_propertyBits[Material::MaterialProperty::DYNAMIC_VISCOSITY] = 12;
    
    // Surface tension is critical for multiphase
    if (isMultiphase) {
        m_propertyBits[Material::MaterialProperty::SURFACE_TENSION] = 12;
        // Reduce bits for less important properties
        m_propertyBits[Material::MaterialProperty::ELECTRICAL_CONDUCTIVITY] = 4;
    } else {
        m_propertyBits[Material::MaterialProperty::SURFACE_TENSION] = 6;
        // Single phase needs more accuracy in other properties
        m_propertyBits[Material::MaterialProperty::THERMAL_EXPANSION] = 6;
    }
    
    // If heat transfer is included, thermal properties become important
    if (includeHeatTransfer) {
        m_propertyBits[Material::MaterialProperty::THERMAL_CONDUCTIVITY] = 10;
        m_propertyBits[Material::MaterialProperty::SPECIFIC_HEAT] = 10;
    } else {
        m_propertyBits[Material::MaterialProperty::THERMAL_CONDUCTIVITY] = 6;
        m_propertyBits[Material::MaterialProperty::SPECIFIC_HEAT] = 6;
    }
}

// Configure for thermal simulation
void ScientificMaterialEquivalence::configureForThermal(size_t hashBits, bool includeFluidFlow) {
    m_hashBits = hashBits;
    
    // Adjust name based on configuration
    m_name = includeFluidFlow ? "Thermal with Flow Key" : "Thermal Key";
    
    // Reset bit allocation
    initialize();
    
    // For thermal simulations, thermal conductivity and specific heat are critical
    m_propertyBits[Material::MaterialProperty::THERMAL_CONDUCTIVITY] = 14;
    m_propertyBits[Material::MaterialProperty::SPECIFIC_HEAT] = 14;
    m_propertyBits[Material::MaterialProperty::DENSITY] = 10;
    
    // If fluid flow is included, viscosity becomes important
    if (includeFluidFlow) {
        m_propertyBits[Material::MaterialProperty::DYNAMIC_VISCOSITY] = 12;
        // Surface tension might be relevant for some thermal flow problems
        m_propertyBits[Material::MaterialProperty::SURFACE_TENSION] = 6;
        // Reduce bits for less important properties
        m_propertyBits[Material::MaterialProperty::ELECTRICAL_CONDUCTIVITY] = 4;
    } else {
        m_propertyBits[Material::MaterialProperty::DYNAMIC_VISCOSITY] = 6;
        m_propertyBits[Material::MaterialProperty::SURFACE_TENSION] = 4;
        m_propertyBits[Material::MaterialProperty::ELECTRICAL_CONDUCTIVITY] = 4;
    }
}

// Configure for structural simulation
void ScientificMaterialEquivalence::configureForStructural(size_t hashBits, bool includeThermalStress) {
    m_hashBits = hashBits;
    
    // Adjust name based on configuration
    m_name = includeThermalStress ? "Structural with Thermal Stress Key" : "Structural Key";
    
    // Reset bit allocation
    initialize();
    
    // For structural simulations, density is critical
    m_propertyBits[Material::MaterialProperty::DENSITY] = 16;
    
    // If thermal stress is included, thermal expansion becomes critical
    if (includeThermalStress) {
        m_propertyBits[Material::MaterialProperty::THERMAL_EXPANSION] = 14;
        m_propertyBits[Material::MaterialProperty::THERMAL_CONDUCTIVITY] = 10;
        m_propertyBits[Material::MaterialProperty::SPECIFIC_HEAT] = 8;
    } else {
        m_propertyBits[Material::MaterialProperty::THERMAL_EXPANSION] = 8;
        m_propertyBits[Material::MaterialProperty::THERMAL_CONDUCTIVITY] = 6;
        m_propertyBits[Material::MaterialProperty::SPECIFIC_HEAT] = 6;
    }
    
    // These properties are less important for structural simulations
    m_propertyBits[Material::MaterialProperty::DYNAMIC_VISCOSITY] = 4;
    m_propertyBits[Material::MaterialProperty::SURFACE_TENSION] = 4;
    m_propertyBits[Material::MaterialProperty::ELECTRICAL_CONDUCTIVITY] = 4;
}

// Set number of bits to use for a property
void ScientificMaterialEquivalence::setPropertyBits(Material::MaterialProperty prop, size_t bits) {
    m_propertyBits[prop] = bits;
}

// Set range for a property
void ScientificMaterialEquivalence::setPropertyRange(
    Material::MaterialProperty prop, double min, double max, ScaleType scaleType) {
    
    m_propertyRanges[prop] = {min, max};
    m_propertyScales[prop] = scaleType;
}

// Set hash resolution in bits
void ScientificMaterialEquivalence::setHashResolution(size_t bits) {
    m_hashBits = bits;
    
    // Adjust bit allocation to fit within new resolution
    double totalBits = 0;
    for (const auto& pair : m_propertyBits) {
        totalBits += pair.second;
    }
    
    if (totalBits > 0) {
        double scaleFactor = static_cast<double>(bits) / totalBits;
        for (auto& pair : m_propertyBits) {
            pair.second = static_cast<size_t>(pair.second * scaleFactor);
            if (pair.second == 0 && scaleFactor > 0) {
                pair.second = 1; // Ensure at least 1 bit if non-zero before
            }
        }
    }
}

// Check if version compatibility mode is supported
bool ScientificMaterialEquivalence::supportsVersionCompatibilityMode() const {
    return true;
}

// Set version compatibility mode
void ScientificMaterialEquivalence::setVersionCompatibilityMode(int version) {
    m_compatibilityVersion = version;
}

// Set equivalence tolerance
void ScientificMaterialEquivalence::setEquivalenceTolerance(double tolerance) {
    if (tolerance < 0.0) {
        tolerance = 0.0;
    }
    m_equivalenceTolerance = tolerance;
}

// Get equivalence tolerance
double ScientificMaterialEquivalence::getEquivalenceTolerance() const {
    return m_equivalenceTolerance;
}

