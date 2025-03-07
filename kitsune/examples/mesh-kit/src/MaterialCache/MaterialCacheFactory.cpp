/**
 * @file MaterialCacheFactory.cpp
 * @brief Implementation of factory methods for material cache creation
 */

#include "MaterialCacheFactory.h"

std::unique_ptr<IMaterialCache> MaterialCacheFactory::createStandardCache(
    size_t maxSize, double tolerance, bool enableContaminationDetection) {
    return std::make_unique<MaterialCache>(maxSize, tolerance, enableContaminationDetection);
}

std::unique_ptr<IMaterialCache> MaterialCacheFactory::createFixedSizeCache(
    size_t size, double tolerance, [[maybe_unused]] bool enableContaminationDetection) {
    return std::make_unique<FixedSizeMaterialCache>(size, tolerance);
}

std::unique_ptr<IMaterialCache> MaterialCacheFactory::createAdaptiveCache(
    size_t initialSize, size_t maxSize, double initialTolerance,
    [[maybe_unused]] bool enableContaminationDetection) {
  return std::make_unique<AdaptiveMaterialCache>(initialSize, maxSize,
                                                 initialTolerance, 1000);
}

std::unique_ptr<IMaterialCache> MaterialCacheFactory::createDomainSpecificCache(
    DomainSpecificEquivalenceKey::DomainType domainType,
    size_t maxSize, double tolerance, bool enableContaminationDetection) {
    return std::make_unique<MaterialCache>(
        std::make_unique<DomainSpecificEquivalenceKey>(domainType, tolerance),
        std::make_unique<LRUEvictionPolicy>(),
        maxSize,
        enableContaminationDetection);
}

std::unique_ptr<IMaterialCache> MaterialCacheFactory::createHighPerformanceCache(
    size_t maxSize, bool enableContaminationDetection) {
    return std::make_unique<MaterialCache>(
        std::make_unique<StandardEquivalenceKey>(1e-5), // Slightly loose tolerance
        std::make_unique<LRUEvictionPolicy>(),
        maxSize,
        enableContaminationDetection);
}

std::unique_ptr<IMaterialCache> MaterialCacheFactory::createMemoryOptimizedCache(
    size_t maxSize, bool enableContaminationDetection) {
    auto equivalenceKey = std::make_unique<StandardEquivalenceKey>(1e-7); // Tight tolerance
    
    // Disable non-essential comparisons to save memory
    equivalenceKey->setCompareName(false);
    equivalenceKey->setCompareReferenceTemperature(false);
    
    return std::make_unique<MaterialCache>(
        std::move(equivalenceKey),
        std::make_unique<FrequencyEvictionPolicy>(), // Better for memory optimization
        maxSize,
        enableContaminationDetection);
}

std::unique_ptr<IMaterialCache> MaterialCacheFactory::createCustomCache(
    std::unique_ptr<IMaterialEquivalenceKey> equivalenceKey,
    std::unique_ptr<IEvictionPolicy> evictionPolicy,
    size_t maxSize,
    bool enableContaminationDetection) {
    return std::make_unique<MaterialCache>(
        std::move(equivalenceKey),
        std::move(evictionPolicy),
        maxSize,
        enableContaminationDetection);
}


