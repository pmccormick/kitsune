/**
 * @file MaterialCache.h
 * @brief Main implementation of the material caching system
 * @details
 * 
 * This file provides the primary implementation of the material caching
 * system, designed to reduce memory fragmentation and improve performance
 * in CFD simulations. It includes:
 * 
 * - MaterialCache: Standard cache implementation with configurable strategies
 * - FixedSizeMaterialCache: Cache with fixed size (cannot be resized)
 * - AdaptiveMaterialCache: Self-tuning cache that adapts to usage patterns
 * - MaterialCacheFactory: Factory methods for creating common cache configurations
 */

#pragma once

#include "MaterialCacheCore.h"
#include "EquivalenceKeys.h"
#include "EvictionPolicies.h"
#include <unordered_map>
#include <memory>
#include <mutex>
#include <chrono>
#include <atomic>

/**
 * @class MaterialCache
 * @brief Caches Material objects to avoid redundancy and memory fragmentation
 * @details
 * 
 * The MaterialCache class provides the primary implementation of the material caching
 * system, designed to reduce memory fragmentation and improve performance in CFD simulations.
 * It uses the Strategy Pattern to allow customization of both the material equivalence
 * determination and the cache eviction policy.
 * 
 * Key features:
 * 1. Configurable cache size: Limits memory usage while maximizing hit rate
 * 2. Pluggable equivalence strategy: Determines when materials are functionally equivalent
 * 3. Pluggable eviction policy: Determines which materials to remove when the cache is full
 * 4. Thread safety: Supports concurrent access from multiple threads
 * 5. Comprehensive statistics: Tracks hits, misses, and other performance metrics
 */
class MaterialCache : public IMaterialCache {
public:
    /**
     * @brief Constructor with default strategies
     * @param maxSize Maximum cache size (0 = unlimited)
     * @param tolerance Tolerance for material equivalence
     */
    MaterialCache(size_t maxSize = 100, double tolerance = 1e-6)
        : m_maxSize(maxSize) {
        // Create default strategies
        m_equivalenceKey = std::make_unique<StandardEquivalenceKey>(tolerance);
        m_evictionPolicy = std::make_unique<LRUEvictionPolicy>();
        
        // Initialize static pointers for hash table
        CacheKey::setEquivalenceKey(m_equivalenceKey.get());
        CacheKeyHash::s_equivalenceKeyPtr = m_equivalenceKey.get();
    }
    
    /**
     * @brief Constructor with explicit strategies
     * @param equivalenceKey Material equivalence strategy
     * @param evictionPolicy Cache eviction policy
     * @param maxSize Maximum cache size (0 = unlimited)
     */
    MaterialCache(std::unique_ptr<IMaterialEquivalenceKey> equivalenceKey,
                 std::unique_ptr<IEvictionPolicy> evictionPolicy,
                 size_t maxSize = 100)
        : m_equivalenceKey(std::move(equivalenceKey)),
          m_evictionPolicy(std::move(evictionPolicy)),
          m_maxSize(maxSize) {
        // Initialize static pointers for hash table
        CacheKey::setEquivalenceKey(m_equivalenceKey.get());
        CacheKeyHash::s_equivalenceKeyPtr = m_equivalenceKey.get();
    }
    
    /**
     * @brief Destructor
     */
    ~MaterialCache() {
        clear();
    }
    
    std::shared_ptr<Material> getOrAdd(const std::shared_ptr<Material>& material) override {
        // Measure query time
        auto startTime = std::chrono::high_resolution_clock::now();
        
        std::shared_ptr<Material> result;
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            
            // Try to find in cache
            CacheKey key{material};
            auto it = m_cache.find(key);
            
            if (it != m_cache.end()) {
                // Hit - update eviction policy and return cached material
                m_evictionPolicy->onAccess(it->second.get());
                m_stats.recordHit();
                result = it->second;
            } else {
                // Miss - add to cache
                m_stats.recordMiss();
                result = add(material);
            }
        }
        
        // Record query time
        auto endTime = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(endTime - startTime);
        m_stats.recordQueryTime(duration);
        
        return result;
    }
    
    std::shared_ptr<Material> find(const std::shared_ptr<Material>& material) override {
        // Measure query time
        auto startTime = std::chrono::high_resolution_clock::now();
        
        std::shared_ptr<Material> result;
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            
            // Try to find in cache
            CacheKey key{material};
            auto it = m_cache.find(key);
            
            if (it != m_cache.end()) {
                // Hit - update eviction policy and return cached material
                m_evictionPolicy->onAccess(it->second.get());
                m_stats.recordHit();
                result = it->second;
            } else {
                // Miss - return nullptr
                m_stats.recordMiss();
                result = nullptr;
            }
        }
        
        // Record query time
        auto endTime = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(endTime - startTime);
        m_stats.recordQueryTime(duration);
        
        return result;
    }
    
    std::shared_ptr<Material> add(const std::shared_ptr<Material>& material) override {
        std::lock_guard<std::mutex> lock(m_mutex);
        
        // If cache is full, evict an item
        if (m_maxSize > 0 && m_cache.size() >= m_maxSize) {
            evictItem();
        }
        
        // Add to cache (only if not full or eviction succeeded)
        if (m_maxSize == 0 || m_cache.size() < m_maxSize) {
            CacheKey key{material};
            m_cache[key] = material;
            m_evictionPolicy->addItem(material.get());
            updateMemoryUsage();
        }
        
        return material;
    }
    
    size_t size() const override {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_cache.size();
    }
    
    void clear() override {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_cache.clear();
        m_evictionPolicy->clear();
        updateMemoryUsage();
    }
    
    void preload(const std::vector<std::shared_ptr<Material>>& materials) override {
        for (const auto& material : materials) {
            add(material);
        }
    }
    
    const MaterialCacheStatistics& getStatistics() const override {
        return m_stats;
    }
    
    void resetStatistics() override {
        m_stats.reset();
    }
    
    const IMaterialEquivalenceKey& getEquivalenceKey() const override {
        return *m_equivalenceKey;
    }
    
    const IEvictionPolicy& getEvictionPolicy() const override {
        return *m_evictionPolicy;
    }
    
    std::string getName() const override {
        return "MaterialCache [Equivalence: " + m_equivalenceKey->getName() + 
               ", Policy: " + m_evictionPolicy->getName() + "]";
    }
    
    void setMaxSize(size_t size) override {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_maxSize = size;
        
        // If new size is smaller, evict excess items
        while (m_maxSize > 0 && m_cache.size() > m_maxSize) {
            evictItem();
        }
    }
    
    size_t getMaxSize() const override {
        return m_maxSize;
    }
    
    void setTolerance(double tolerance) override {
        std::lock_guard<std::mutex> lock(m_mutex);
        
        // If tolerance changes, we need to rebuild the cache
        if (m_equivalenceKey->getTolerance() != tolerance) {
            m_equivalenceKey->setTolerance(tolerance);
            rebuildCache();
        }
    }
    
    /**
     * @brief Set a new equivalence key strategy
     * @param equivalenceKey The new strategy
     */
    void setEquivalenceKey(std::unique_ptr<IMaterialEquivalenceKey> equivalenceKey) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_equivalenceKey = std::move(equivalenceKey);
        
        // Update static pointer used by the hash table
        CacheKey::setEquivalenceKey(m_equivalenceKey.get());
        CacheKeyHash::s_equivalenceKeyPtr = m_equivalenceKey.get();
        
        rebuildCache();
    }
    
    /**
     * @brief Set a new eviction policy strategy
     * @param evictionPolicy The new strategy
     */
    void setEvictionPolicy(std::unique_ptr<IEvictionPolicy> evictionPolicy) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_evictionPolicy = std::move(evictionPolicy);
        rebuildEvictionData();
    }
    
protected:
    // For access from derived classes
    std::mutex m_mutex;
    
private:
    // Cache key for material lookup
    struct CacheKey {
        std::shared_ptr<Material> material;
        
        bool operator==(const CacheKey& other) const {
            // Use the active equivalence key for comparison
            // Note: This requires thread safety precautions
            return s_equivalenceKeyPtr->areEquivalent(material, other.material);
        }
        
        // Allow static access to the current equivalence key for hashing and comparison
        // This is a bit of a hack but necessary for std::unordered_map
        static void setEquivalenceKey(IMaterialEquivalenceKey* key) {
            s_equivalenceKeyPtr = key;
        }
        
        static IMaterialEquivalenceKey* s_equivalenceKeyPtr;
    };
    
    // Hash function for CacheKey
    struct CacheKeyHash {
        size_t operator()(const CacheKey& key) const {
            // Use the active equivalence key for hashing
            return s_equivalenceKeyPtr->hash(key.material);
        }
        
        // Static pointer to the current equivalence key
        static IMaterialEquivalenceKey* s_equivalenceKeyPtr;
    };
    
    // Cache storage
    std::unordered_map<CacheKey, std::shared_ptr<Material>, CacheKeyHash> m_cache;
    
    // Strategies
    std::unique_ptr<IMaterialEquivalenceKey> m_equivalenceKey;
    std::unique_ptr<IEvictionPolicy> m_evictionPolicy;
    
    // Configuration
    size_t m_maxSize;
    
    // Statistics
    MaterialCacheStatistics m_stats;
    
    // Evict an item based on the current policy
    void evictItem() {
        // Select item to evict
        void* itemToEvict = m_evictionPolicy->selectForEviction();
        if (itemToEvict) {
            // Find the key for this item
            for (auto it = m_cache.begin(); it != m_cache.end(); ++it) {
                if (it->second.get() == itemToEvict) {
                    // Remove from cache
                    m_evictionPolicy->removeItem(itemToEvict);
                    m_cache.erase(it);
                    m_stats.recordEviction();
                    updateMemoryUsage();
                    break;
                }
            }
        }
    }
    
    // Rebuild the cache with new equivalence key
    void rebuildCache() {
        // Save all materials
        std::vector<std::shared_ptr<Material>> materials;
        materials.reserve(m_cache.size());
        
        for (const auto& pair : m_cache) {
            materials.push_back(pair.second);
        }
        
        // Clear and rebuild
        m_cache.clear();
        m_evictionPolicy->clear();
        
        // Re-add all materials
        for (const auto& material : materials) {
            CacheKey key{material};
            m_cache[key] = material;
            m_evictionPolicy->addItem(material.get());
        }
        
        updateMemoryUsage();
    }
    
    // Rebuild the eviction policy data
    void rebuildEvictionData() {
        // Clear the policy
        m_evictionPolicy->clear();
        
        // Re-add all materials to the policy
        for (const auto& pair : m_cache) {
            m_evictionPolicy->addItem(pair.second.get());
        }
    }
    
    // Update memory usage statistics
    void updateMemoryUsage() {
        // Estimate memory usage
        // This is a rough estimate - actual memory usage depends on many factors
        static const size_t MATERIAL_SIZE_ESTIMATE = 1024; // Rough estimate: 1KB per material
        size_t bytes = m_cache.size() * MATERIAL_SIZE_ESTIMATE;
        m_stats.updateMemoryUsage(bytes);
    }
};

// Initialize static members
IMaterialEquivalenceKey* MaterialCache::CacheKey::s_equivalenceKeyPtr = nullptr;
IMaterialEquivalenceKey* MaterialCache::CacheKeyHash::s_equivalenceKeyPtr = nullptr;

/**
 * @class FixedSizeMaterialCache
 * @brief Material cache with a fixed size
 * @details
 * 
 * This cache has a fixed size and cannot be resized. It's optimized for
 * performance in environments with limited memory.
 */
class FixedSizeMaterialCache : public MaterialCache {
public:
    /**
     * @brief Constructor with default strategies
     * @param size Fixed cache size
     * @param tolerance Tolerance for material equivalence
     */
    FixedSizeMaterialCache(size_t size, double tolerance = 1e-6)
        : MaterialCache(size, tolerance) {
    }
    
    /**
     * @brief Constructor with explicit strategies
     * @param size Fixed cache size
     * @param equivalenceKey Material equivalence strategy
     * @param evictionPolicy Cache eviction policy
     */
    FixedSizeMaterialCache(size_t size,
                         std::unique_ptr<IMaterialEquivalenceKey> equivalenceKey,
                         std::unique_ptr<IEvictionPolicy> evictionPolicy)
        : MaterialCache(std::move(equivalenceKey), std::move(evictionPolicy), size) {
    }
    
    void setMaxSize(size_t size) override {
        // Cannot change size of fixed cache
        // Could throw exception, but silently ignoring is probably safer
    }
    
    std::string getName() const override {
        return "FixedSizeMaterialCache [Size: " + std::to_string(getMaxSize()) + "]";
    }
};

/**
 * @class AdaptiveMaterialCache
 * @brief Material cache with adaptive sizing and strategies
 * @details
 * 
 * This cache automatically adjusts its size and strategies based on observed
 * usage patterns, optimizing for both memory usage and performance.
 */
class AdaptiveMaterialCache : public MaterialCache {
public:
    /**
     * @brief Constructor
     * @param initialSize Initial cache size
     * @param maxSize Maximum cache size (0 = unlimited)
     * @param initialTolerance Initial tolerance for material equivalence
     * @param adaptationInterval Number of operations between adaptations
     */
    AdaptiveMaterialCache(size_t initialSize = 100, size_t maxSize = 1000,
                        double initialTolerance = 1e-6,
                        size_t adaptationInterval = 1000)
        : MaterialCache(
            std::make_unique<AdaptiveEquivalenceKey>(initialTolerance, adaptationInterval),
            std::make_unique<AdaptiveEvictionPolicy>(adaptationInterval),
            initialSize),
          m_initialSize(initialSize),
          m_maxSize(maxSize),
          m_adaptationInterval(adaptationInterval),
          m_lastSizeAdaptation(0),
          m_lastConfigAdaptation(0),
          m_operationsSinceLastAdaptation(0) {
    }
    
    std::shared_ptr<Material> getOrAdd(const std::shared_ptr<Material>& material) override {
        // Call parent implementation
        auto result = MaterialCache::getOrAdd(material);
        
        // Check if adaptation is needed
        m_operationsSinceLastAdaptation++;
        if (m_operationsSinceLastAdaptation >= m_adaptationInterval) {
            adaptCache();
            m_operationsSinceLastAdaptation = 0;
        }
        
        return result;
    }
    
    std::string getName() const override {
        return "AdaptiveMaterialCache [Size: " + std::to_string(size()) + 
               "/" + std::to_string(getMaxSize()) + ", Adaptations: " + 
               std::to_string(m_adaptationCount) + "]";
    }
    
    /**
     * @brief Get additional statistics
     * @return Map of statistic name to value
     */
    std::unordered_map<std::string, double> getAdaptiveStatistics() const {
        return {
            {"Adaptation Count", static_cast<double>(m_adaptationCount)},
            {"Size Adaptations", static_cast<double>(m_sizeAdaptationCount)},
            {"Config Adaptations", static_cast<double>(m_configAdaptationCount)},
            {"Initial Size", static_cast<double>(m_initialSize)},
            {"Maximum Size", static_cast<double>(m_maxSize)},
            {"Adaptation Interval", static_cast<double>(m_adaptationInterval)}
        };
    }
    
private:
    size_t m_initialSize;
    size_t m_maxSize;
    size_t m_adaptationInterval;
    
    // Adaptation state
    std::atomic<size_t> m_operationsSinceLastAdaptation{0};
    size_t m_adaptationCount = 0;
    size_t m_sizeAdaptationCount = 0;
    size_t m_configAdaptationCount = 0;
    
    // History for hit rates
    std::vector<double> m_hitRateHistory;
    const size_t MAX_HISTORY = 10;
    
    // Last adaptation times
    size_t m_lastSizeAdaptation;
    size_t m_lastConfigAdaptation;
    
    void adaptCache() {
        std::lock_guard<std::mutex> lock(m_mutex);
        
        m_adaptationCount++;
        
        // Record current hit rate
        const double hitRate = getStatistics().getHitRate();
        m_hitRateHistory.push_back(hitRate);
        if (m_hitRateHistory.size() > MAX_HISTORY) {
            m_hitRateHistory.erase(m_hitRateHistory.begin());
        }
        
        // Adapt cache size
        adaptCacheSize();
        
        // Adapt cache configuration
        adaptCacheConfiguration();
    }
    
    void adaptCacheSize() {
        // Only adapt size after collecting enough history
        if (m_hitRateHistory.size() < 3) {
            return;
        }
        
        // Calculate average hit rate
        double avgHitRate = 0.0;
        for (double rate : m_hitRateHistory) {
            avgHitRate += rate;
        }
        avgHitRate /= m_hitRateHistory.size();
        
        // Get current size
        size_t currentSize = getMaxSize();
        size_t newSize = currentSize;
        
        // Adjust size based on hit rate
        if (avgHitRate < 20.0) {
            // Very low hit rate - increase size significantly
            newSize = std::min(currentSize * 2, m_maxSize);
        } else if (avgHitRate < 50.0) {
            // Moderate hit rate - increase size moderately
            newSize = std::min(currentSize * 3 / 2, m_maxSize);
        } else if (avgHitRate > 95.0) {
            // Very high hit rate - decrease size to save memory
            newSize = std::max(currentSize * 2 / 3, m_initialSize);
        } else if (avgHitRate > 80.0) {
            // High hit rate - slight decrease
            newSize = std::max(currentSize * 4 / 5, m_initialSize);
        }
        
        // Apply new size if changed
        if (newSize != currentSize) {
            MaterialCache::setMaxSize(newSize);
            m_lastSizeAdaptation = m_adaptationCount;
            m_sizeAdaptationCount++;
        }
    }
    
    void adaptCacheConfiguration() {
        // Adapt less frequently than size
        if (m_adaptationCount - m_lastConfigAdaptation < 5) {
            return;
        }
        
        // Get current hit rate trend (increasing, decreasing, or stable)
        double hitRateTrend = 0.0;
        if (m_hitRateHistory.size() >= 5) {
            double recentAvg = 0.0;
            double olderAvg = 0.0;
            
            // Average of 3 most recent rates
            for (size_t i = m_hitRateHistory.size() - 3; i < m_hitRateHistory.size(); i++) {
                recentAvg += m_hitRateHistory[i];
            }
            recentAvg /= 3.0;
            
            // Average of 3 older rates
            for (size_t i = m_hitRateHistory.size() - 6; i < m_hitRateHistory.size() - 3; i++) {
                olderAvg += m_hitRateHistory[i];
            }
            olderAvg /= 3.0;
            
            hitRateTrend = recentAvg - olderAvg;
        }
        
        // Get current configuration
        auto* adaptiveKey = dynamic_cast<AdaptiveEquivalenceKey*>(m_equivalenceKey.get());
        auto* adaptivePolicy = dynamic_cast<AdaptiveEvictionPolicy*>(m_evictionPolicy.get());
        
        // If we're using an adaptive equivalence key, update its tolerance based on the trend
        if (adaptiveKey) {
            double currentTolerance = adaptiveKey->getTolerance();
            double newTolerance = currentTolerance;
            
            if (hitRateTrend < -10.0) {
                // Hit rate decreasing - try more lenient tolerance
                newTolerance = currentTolerance * 2.0;
            } else if (hitRateTrend > 10.0) {
                // Hit rate increasing - maybe try more strict tolerance
                newTolerance = currentTolerance * 0.8;
            }
            
            // Don't change tolerance too drastically
            newTolerance = std::max(1e-12, std::min(1e-2, newTolerance));
            
            // Apply changes if significant
            if (std::abs(newTolerance - currentTolerance) / currentTolerance > 0.1) {
                adaptiveKey->setTolerance(newTolerance);
                m_lastConfigAdaptation = m_adaptationCount;
                m_configAdaptationCount++;
            }
        }
    }
};

/**
 * @class MaterialCacheFactory
 * @brief Factory for creating different types of material caches
 */
class MaterialCacheFactory {
public:
    /**
     * @brief Create a standard material cache
     * @param maxSize Maximum cache size (0 = unlimited)
     * @param tolerance Tolerance for material equivalence
     * @return New material cache instance
     */
    static std::unique_ptr<IMaterialCache> createStandardCache(
        size_t maxSize = 100, double tolerance = 1e-6) {
        return std::make_unique<MaterialCache>(maxSize, tolerance);
    }
    
    /**
     * @brief Create a fixed size material cache
     * @param size Fixed cache size
     * @param tolerance Tolerance for material equivalence
     * @return New material cache instance
     */
    static std::unique_ptr<IMaterialCache> createFixedSizeCache(
        size_t size, double tolerance = 1e-6) {
        return std::make_unique<FixedSizeMaterialCache>(size, tolerance);
    }
    
    /**
     * @brief Create an adaptive material cache
     * @param initialSize Initial cache size
     * @param maxSize Maximum cache size (0 = unlimited)
     * @param initialTolerance Initial tolerance for material equivalence
     * @return New material cache instance
     */
    static std::unique_ptr<IMaterialCache> createAdaptiveCache(
        size_t initialSize = 100, size_t maxSize = 1000,
        double initialTolerance = 1e-6) {
        return std::make_unique<AdaptiveMaterialCache>(
            initialSize, maxSize, initialTolerance);
    }
    
    /**
     * @brief Create a domain-specific cache for CFD simulations
     * @param domainType The type of CFD simulation
     * @param maxSize Maximum cache size
     * @param tolerance Tolerance for material equivalence
     * @return New material cache instance
     */
    static std::unique_ptr<IMaterialCache> createDomainSpecificCache(
        DomainSpecificEquivalenceKey::DomainType domainType,
        size_t maxSize = 100, double tolerance = 1e-6) {
        return std::make_unique<MaterialCache>(
            std::make_unique<DomainSpecificEquivalenceKey>(domainType, tolerance),
            std::make_unique<LRUEvictionPolicy>(),
            maxSize);
    }
    
    /**
     * @brief Create a high-performance cache for performance-critical simulations
     * @param maxSize Maximum cache size
     * @return New material cache instance optimized for performance
     */
    static std::unique_ptr<IMaterialCache> createHighPerformanceCache(
        size_t maxSize = 100) {
        return std::make_unique<MaterialCache>(
            std::make_unique<StandardEquivalenceKey>(1e-5), // Slightly loose tolerance
            std::make_unique<LRUEvictionPolicy>(),
            maxSize);
    }
    
    /**
     * @brief Create a memory-optimized cache for large simulations
     * @param maxSize Maximum cache size
     * @return New material cache instance optimized for memory efficiency
     */
    static std::unique_ptr<IMaterialCache> createMemoryOptimizedCache(
        size_t maxSize = 100) {
        auto equivalenceKey = std::make_unique<StandardEquivalenceKey>(1e-7); // Tight tolerance
        
        // Disable non-essential comparisons to save memory
        equivalenceKey->setCompareName(false);
        equivalenceKey->setCompareReferenceTemperature(false);
        
        return std::make_unique<MaterialCache>(
            std::move(equivalenceKey),
            std::make_unique<FrequencyEvictionPolicy>(), // Better for memory optimization
            maxSize);
    }
    
    /**
     * @brief Create a custom cache with specific strategies
     * @param equivalenceKey Material equivalence strategy
     * @param evictionPolicy Cache eviction policy
     * @param maxSize Maximum cache size
     * @return New material cache instance with custom strategies
     */
    static std::unique_ptr<IMaterialCache> createCustomCache(
        std::unique_ptr<IMaterialEquivalenceKey> equivalenceKey,
        std::unique_ptr<IEvictionPolicy> evictionPolicy,
        size_t maxSize = 100) {
        return std::make_unique<MaterialCache>(
            std::move(equivalenceKey),
            std::move(evictionPolicy),
            maxSize);
    }
};

