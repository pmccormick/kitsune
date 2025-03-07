/**
 * @file MaterialBehaviorTracker.cpp
 * @brief Implementation of material cache behavior tracking
 */

#include "MaterialBehaviorTracker.h"

MaterialCacheBehaviorTracker::MaterialCacheBehaviorTracker(size_t analysisWindowMinutes)
    : m_analysisWindow(std::chrono::minutes(analysisWindowMinutes)) {
}

void MaterialCacheBehaviorTracker::recordAccess(const std::shared_ptr<Material>& material, bool isHit) {
    if (!material) return;
    
    auto now = std::chrono::steady_clock::now();
    auto id = material->getID();
    
    auto& behavior = m_behaviors[id];
    
    // Initialize if first time
    if (behavior.accessCount == 0) {
        behavior.materialID = id;
        behavior.materialName = material->getName();
        behavior.materialType = material->getType();
        behavior.firstSeen = now;
    }
    
    // Update counts
    behavior.accessCount++;
    if (isHit) behavior.hitCount++;
    else behavior.missCount++;
    
    behavior.lastSeen = now;
    
    // Update overall stats
    m_totalAccesses++;
    if (isHit) m_totalHits++;
    
    // Update behavior classification periodically
    if (m_totalAccesses % 1000 == 0) {
        classifyAllMaterials();
    }
}

void MaterialCacheBehaviorTracker::recordEviction(const std::shared_ptr<Material>& material) {
    if (!material) return;
    
    auto id = material->getID();
    
    auto it = m_behaviors.find(id);
    if (it != m_behaviors.end()) {
        it->second.evictionCount++;
    }
}

std::vector<uint32_t> MaterialCacheBehaviorTracker::getMaterialsByBehavior(BehaviorType type) const {
    std::vector<uint32_t> result;
    
    for (const auto& pair : m_behaviors) {
        if (pair.second.currentBehavior == type) {
            result.push_back(pair.first);
        }
    }
    
    return result;
}

std::vector<uint32_t> MaterialCacheBehaviorTracker::getContaminators() const {
    return getMaterialsByBehavior(BehaviorType::CONTAMINATOR);
}

std::unordered_map<std::string, double> MaterialCacheBehaviorTracker::getStatistics() const {
    std::unordered_map<std::string, double> result;
    
    // Count materials in each behavior category
    std::unordered_map<BehaviorType, size_t> behaviorCounts;
    for (const auto& pair : m_behaviors) {
        behaviorCounts[pair.second.currentBehavior]++;
    }
    
    // Add behavior counts
    result["Materials_Hot"] = static_cast<double>(behaviorCounts[BehaviorType::HOT]);
    result["Materials_Cold"] = static_cast<double>(behaviorCounts[BehaviorType::COLD]);
    result["Materials_Volatile"] = static_cast<double>(behaviorCounts[BehaviorType::VOLATILE]);
    result["Materials_Stable"] = static_cast<double>(behaviorCounts[BehaviorType::STABLE]);
    result["Materials_Contaminator"] = static_cast<double>(behaviorCounts[BehaviorType::CONTAMINATOR]);
    result["Materials_Churner"] = static_cast<double>(behaviorCounts[BehaviorType::CHURNER]);
    
    // Add overall stats
    result["Total_Accesses"] = static_cast<double>(m_totalAccesses);
    result["Total_Hits"] = static_cast<double>(m_totalHits);
    result["Overall_HitRate"] = m_totalAccesses > 0 ? 
        static_cast<double>(m_totalHits) / m_totalAccesses * 100.0 : 0.0;
    
    return result;
}

std::vector<MaterialCacheBehaviorTracker::MaterialBehaviorInfo> 
MaterialCacheBehaviorTracker::getMaterialDetails(const std::vector<uint32_t>& materialIDs) const {
    std::vector<MaterialBehaviorInfo> result;
    
    if (materialIDs.empty()) {
        // Return all materials
        for (const auto& pair : m_behaviors) {
            MaterialBehaviorInfo info;
            info.materialID = pair.second.materialID;
            info.materialName = pair.second.materialName;
            info.behaviorType = pair.second.currentBehavior;
            info.accessCount = pair.second.accessCount;
            info.hitCount = pair.second.hitCount;
            info.missCount = pair.second.missCount;
            info.evictionCount = pair.second.evictionCount;
            info.hitRate = pair.second.accessCount > 0 ?
                static_cast<double>(pair.second.hitCount) / pair.second.accessCount * 100.0 : 0.0;
            
            result.push_back(info);
        }
    } else {
        // Return specific materials
        for (uint32_t id : materialIDs) {
            auto it = m_behaviors.find(id);
            if (it != m_behaviors.end()) {
                MaterialBehaviorInfo info;
                info.materialID = it->second.materialID;
                info.materialName = it->second.materialName;
                info.behaviorType = it->second.currentBehavior;
                info.accessCount = it->second.accessCount;
                info.hitCount = it->second.hitCount;
                info.missCount = it->second.missCount;
                info.evictionCount = it->second.evictionCount;
                info.hitRate = it->second.accessCount > 0 ?
                    static_cast<double>(it->second.hitCount) / it->second.accessCount * 100.0 : 0.0;
                
                result.push_back(info);
            }
        }
    }
    
    return result;
}

void MaterialCacheBehaviorTracker::reset() {
    m_behaviors.clear();
    m_totalAccesses = 0;
    m_totalHits = 0;
}

const char* MaterialCacheBehaviorTracker::behaviorTypeToString(BehaviorType type) {
    switch (type) {
        case BehaviorType::HOT: return "Hot";
        case BehaviorType::COLD: return "Cold";
        case BehaviorType::VOLATILE: return "Volatile";
        case BehaviorType::STABLE: return "Stable";
        case BehaviorType::CONTAMINATOR: return "Contaminator";
        case BehaviorType::CHURNER: return "Churner";
        default: return "Unknown";
    }
}

void MaterialCacheBehaviorTracker::classifyAllMaterials() {
    auto now = std::chrono::steady_clock::now();
    
    // Global cache statistics
    double globalHitRate = m_totalAccesses > 0 ? 
        static_cast<double>(m_totalHits) / m_totalAccesses : 0.0;
    
    for (auto& pair : m_behaviors) {
        auto& behavior = pair.second;
        
        // Calculate metrics
        double hitRate = behavior.accessCount > 0 ? 
            static_cast<double>(behavior.hitCount) / behavior.accessCount : 0.0;
            
        double relativeAccessFrequency = m_totalAccesses > 0 ?
            static_cast<double>(behavior.accessCount) / m_totalAccesses : 0.0;
            
        double ageHours = std::chrono::duration_cast<std::chrono::hours>(
            now - behavior.firstSeen).count();
            
        double idleTimeMinutes = std::chrono::duration_cast<std::chrono::minutes>(
            now - behavior.lastSeen).count();
        
        // Classify behavior
        if (relativeAccessFrequency < 0.001 && idleTimeMinutes > 30) {
            // Rarely accessed and not recently used
            behavior.currentBehavior = BehaviorType::CONTAMINATOR;
        }
        else if (hitRate < 0.5 * globalHitRate && relativeAccessFrequency > 0.01) {
            // Poor hit rate but accessed frequently
            behavior.currentBehavior = BehaviorType::CHURNER;
        }
        else if (relativeAccessFrequency > 0.05) {
            // Very frequently accessed
            behavior.currentBehavior = BehaviorType::HOT;
        }
        else if (ageHours > 1 && behavior.evictionCount == 0) {
            // Long-lived, never evicted
            behavior.currentBehavior = BehaviorType::STABLE;
        }
        else if (behavior.evictionCount > behavior.accessCount * 0.5) {
            // Frequently evicted relative to access count
            behavior.currentBehavior = BehaviorType::VOLATILE;
        }
        else {
            // Default
            behavior.currentBehavior = BehaviorType::COLD;
        }
    }
}

