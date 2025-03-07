/**
 * @file MaterialCache.cpp
 * @brief Implementation of the main material cache system
 */

#include "MaterialCache.h"
#include "EnhancedStatistics.h"
#include "MaterialBehaviorTracker.h"
#include <algorithm>


//==============================================================================
// MaterialCache Implementation
//==============================================================================

MaterialCache::MaterialCache(size_t maxSize, double tolerance, bool enableContaminationDetection)
    : m_maxSize(maxSize),
      m_enableContaminationDetection(enableContaminationDetection),
      m_equivalenceKey(std::make_unique<StandardEquivalenceKey>(tolerance)),
      m_evictionPolicy(std::make_unique<LRUEvictionPolicy>()),
      m_cache(10, MaterialPtrHash(m_equivalenceKey.get()), MaterialPtrEqual(m_equivalenceKey.get())) {
    // Constructor body can be empty now
}

MaterialCache::MaterialCache(std::unique_ptr<IMaterialEquivalenceKey> equivalenceKey,
                     std::unique_ptr<IEvictionPolicy> evictionPolicy,
                     size_t maxSize,
                     bool enableContaminationDetection)
    : m_equivalenceKey(std::move(equivalenceKey)),
      m_evictionPolicy(std::move(evictionPolicy)),
      m_maxSize(maxSize),
      m_enableContaminationDetection(enableContaminationDetection),
      m_cache(10, MaterialPtrHash(m_equivalenceKey.get()), MaterialPtrEqual(m_equivalenceKey.get())) {
    // Constructor body can be empty now
}

MaterialCache::~MaterialCache() {
    clear();
}

std::shared_ptr<Material> MaterialCache::getOrAdd(const std::shared_ptr<Material>& material) {
    // Measure query time
    auto startTime = std::chrono::high_resolution_clock::now();
    
    // First try to find the material without holding the lock for too long
    auto found = find(material);
    if (found) {
        // Hit - material was found, return it
        return found;
    }
    
    // Material wasn't found, add it
    auto result = add(material);
    
    // Record query time
    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(endTime - startTime);
    m_stats.recordQueryTime(duration);
    
    return result;
}

std::shared_ptr<Material> MaterialCache::find(const std::shared_ptr<Material>& material) {
    std::lock_guard<std::mutex> lock(m_mutex);
    
    // First try direct lookup via hash
    auto it = m_cache.find(material);
    if (it != m_cache.end()) {
        // Hit via hash
        m_evictionPolicy->onAccess(it->second.get());
        m_stats.recordHit();
        
        // Enhanced stats tracking
        if (m_enableContaminationDetection) {
            m_enhancedStats.recordHit(it->second);
            m_behaviorTracker.recordAccess(it->second, true);
        }
        
        return it->second;
    }
    
    // If the hash-based lookup fails, we should do a linear scan 
    // in case there's a hash collision or hash function inconsistency
    for (const auto& pair : m_cache) {
        if (m_equivalenceKey->areEquivalent(material, pair.first)) {
            // Found via linear scan
            m_evictionPolicy->onAccess(pair.second.get());
            m_stats.recordHit();
            
            // Enhanced stats tracking
            if (m_enableContaminationDetection) {
                m_enhancedStats.recordHit(pair.second);
                m_behaviorTracker.recordAccess(pair.second, true);
            }
            
            return pair.second;
        }
    }
    
    // Miss - not found
    m_stats.recordMiss();
    
    // Enhanced stats tracking
    if (m_enableContaminationDetection) {
        m_enhancedStats.recordMiss(material);
        m_behaviorTracker.recordAccess(material, false);
    }
    
    return nullptr;
}

std::shared_ptr<Material> MaterialCache::add(const std::shared_ptr<Material>& material) {
    std::lock_guard<std::mutex> lock(m_mutex);

    // If cache is full, evict an item
    if (m_maxSize > 0 && m_cache.size() >= m_maxSize) {
        evictItem();
    }

    // Add to cache
    m_cache[material] = material;  // Use material directly
    m_evictionPolicy->addItem(material.get());
    updateMemoryUsage();

    return material;
}

size_t MaterialCache::size() const {
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_cache.size();
}

void MaterialCache::clear() {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_cache.clear();
    m_evictionPolicy->clear();
    updateMemoryUsage();
}

void MaterialCache::preload(const std::vector<std::shared_ptr<Material>>& materials) {
    for (const auto& material : materials) {
        add(material);
    }
}

const MaterialCacheStatistics& MaterialCache::getStatistics() const {
    return m_stats;
}

void MaterialCache::resetStatistics() {
    m_stats.reset();
    if (m_enableContaminationDetection) {
        m_enhancedStats.reset();
        m_behaviorTracker.reset();
    }
}

const IMaterialEquivalenceKey& MaterialCache::getEquivalenceKey() const {
    return *m_equivalenceKey;
}

const IEvictionPolicy& MaterialCache::getEvictionPolicy() const {
    return *m_evictionPolicy;
}

std::string MaterialCache::getName() const {
    return "MaterialCache [Equivalence: " + m_equivalenceKey->getName() + 
           ", Policy: " + m_evictionPolicy->getName() + "]";
}

void MaterialCache::setMaxSize(size_t size) {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_maxSize = size;
    
    // If new size is smaller, evict excess items
    while (m_maxSize > 0 && m_cache.size() > m_maxSize) {
        evictItem();
    }
}

size_t MaterialCache::getMaxSize() const {
    return m_maxSize;
}

void MaterialCache::setTolerance(double tolerance) {
    std::lock_guard<std::mutex> lock(m_mutex);
    
    // If tolerance changes, we need to rebuild the cache
    if (m_equivalenceKey->getTolerance() != tolerance) {
        m_equivalenceKey->setTolerance(tolerance);
        rebuildCache();
    }
}

void MaterialCache::setEquivalenceKey(std::unique_ptr<IMaterialEquivalenceKey> equivalenceKey) {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_equivalenceKey = std::move(equivalenceKey);
    
    // We need to rebuild the cache since the hash and equality functions have changed
    rebuildCache();
}

void MaterialCache::setEvictionPolicy(std::unique_ptr<IEvictionPolicy> evictionPolicy) {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_evictionPolicy = std::move(evictionPolicy);
    rebuildEvictionData();
}

void MaterialCache::setEnableContaminationDetection(bool enable) {
    m_enableContaminationDetection = enable;
    if (!enable) {
        m_enhancedStats.reset();
        m_behaviorTracker.reset();
    }
}

bool MaterialCache::isContaminationDetectionEnabled() const {
    return m_enableContaminationDetection;
}

const EnhancedMaterialCacheStatistics& MaterialCache::getEnhancedStatistics() const {
    return m_enhancedStats;
}

const MaterialCacheBehaviorTracker& MaterialCache::getBehaviorTracker() const {
    return m_behaviorTracker;
}

std::vector<uint32_t> MaterialCache::getContaminatorMaterials() const {
    if (!m_enableContaminationDetection) {
        return {};
    }
    return m_behaviorTracker.getContaminators();
}

std::vector<MaterialCacheBehaviorTracker::MaterialBehaviorInfo> 
MaterialCache::getMaterialsWithBehavior(MaterialCacheBehaviorTracker::BehaviorType behavior) const {
    if (!m_enableContaminationDetection) {
        return {};
    }
    return m_behaviorTracker.getMaterialDetails(m_behaviorTracker.getMaterialsByBehavior(behavior));
}

std::vector<MaterialCacheBehaviorTracker::MaterialBehaviorInfo> MaterialCache::getAllMaterialBehaviors() const {
    if (!m_enableContaminationDetection) {
        return {};
    }
    return m_behaviorTracker.getMaterialDetails();
}

std::vector<MaterialCacheBehaviorTracker::MaterialBehaviorInfo> MaterialCache::detectContamination() const {
    if (!m_enableContaminationDetection) {
        return {};
    }
    return m_behaviorTracker.getMaterialDetails(m_behaviorTracker.getContaminators());
}

size_t MaterialCache::evictContaminators(size_t maxToEvict) {
    if (!m_enableContaminationDetection) {
        return 0;
    }
    
    std::lock_guard<std::mutex> lock(m_mutex);
    
    auto contaminators = m_behaviorTracker.getContaminators();
    size_t evicted = 0;
    
    for (uint32_t id : contaminators) {
        if (maxToEvict > 0 && evicted >= maxToEvict) {
            break;
        }
        
        // Find material with this ID
        for (auto it = m_cache.begin(); it != m_cache.end(); ++it) {
            if (it->second->getID() == id) {
                // Remove from eviction policy
                m_evictionPolicy->removeItem(it->second.get());
                
                // Record eviction
                m_stats.recordEviction();
                if (m_enableContaminationDetection) {
                    m_enhancedStats.recordEviction(it->second);
                    m_behaviorTracker.recordEviction(it->second);
                }
                
                // Remove from cache
                m_cache.erase(it);
                evicted++;
                break;
            }
        }
    }
    
    updateMemoryUsage();
    return evicted;
}

std::unordered_map<std::string, double> MaterialCache::getDetailedStatistics() const {
    std::unordered_map<std::string, double> result = m_stats.getStatistics();
    
    if (m_enableContaminationDetection) {
        // Add enhanced statistics
        auto enhancedStats = m_enhancedStats.getDetailedStatistics();
        result.insert(enhancedStats.begin(), enhancedStats.end());
        
        // Add behavior statistics
        auto behaviorStats = m_behaviorTracker.getStatistics();
        result.insert(behaviorStats.begin(), behaviorStats.end());
    }
    
    return result;
}

void MaterialCache::evictItem() {
    // Select item to evict
    void* itemToEvict = m_evictionPolicy->selectForEviction();
    if (itemToEvict) {
        // Find the key for this item
        for (auto it = m_cache.begin(); it != m_cache.end(); ++it) {
            if (it->second.get() == itemToEvict) {
                // Record eviction
                m_stats.recordEviction();
                if (m_enableContaminationDetection) {
                    m_enhancedStats.recordEviction(it->second);
                    m_behaviorTracker.recordEviction(it->second);
                }
                
                // Remove from cache
                m_evictionPolicy->removeItem(itemToEvict);
                m_cache.erase(it);
                break;
            }
        }
    }
    
    updateMemoryUsage();
}

void MaterialCache::rebuildCache() {
    // Save all materials
    std::vector<std::shared_ptr<Material>> materials;
    materials.reserve(m_cache.size());
    
    for (const auto& pair : m_cache) {
        materials.push_back(pair.second);
    }
    
    // Clear the eviction policy
    m_evictionPolicy->clear();
    
    // Create a new cache with the updated equivalence key
    std::unordered_map<std::shared_ptr<Material>, std::shared_ptr<Material>, 
                     MaterialPtrHash, MaterialPtrEqual> newCache(
        materials.size() * 2, 
        MaterialPtrHash(m_equivalenceKey.get()), 
        MaterialPtrEqual(m_equivalenceKey.get()));
    
    // Add all materials to the new cache
    for (const auto& material : materials) {
        newCache[material] = material;
        m_evictionPolicy->addItem(material.get());
    }
    
    // Swap with the old cache
    m_cache.swap(newCache);
    
    updateMemoryUsage();
}

void MaterialCache::rebuildEvictionData() {
    // Clear the policy
    m_evictionPolicy->clear();
    
    // Re-add all materials to the policy
    for (const auto& pair : m_cache) {
        m_evictionPolicy->addItem(pair.second.get());
    }
}

void MaterialCache::updateMemoryUsage() {
    // Estimate memory usage
    // This is a rough estimate - actual memory usage depends on many factors
    static const size_t MATERIAL_SIZE_ESTIMATE = 1024; // Rough estimate: 1KB per material
    size_t bytes = m_cache.size() * MATERIAL_SIZE_ESTIMATE;
    m_stats.updateMemoryUsage(bytes);
}

