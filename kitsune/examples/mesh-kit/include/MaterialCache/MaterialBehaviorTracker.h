/**
 * @file MaterialBehaviorTracker.h
 * @brief Material cache behavior tracking and analysis
 * @details
 *
 * This file provides the MaterialCacheBehaviorTracker class which analyzes
 * cache access patterns and classifies materials based on their behavior.
 */

#pragma once

#include "Material.h"
#include <chrono>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include <set>

/**
 * @class MaterialCacheBehaviorTracker
 * @brief Tracks and classifies materials based on their cache access patterns
 * @details
 *
 * This class monitors how materials are used in the cache and classifies them
 * into behavior categories, which helps identify potential issues like
 * cache contamination and provides insights for cache optimization.
 */
class MaterialCacheBehaviorTracker {
public:
    /**
     * @enum BehaviorType
     * @brief Classification of material cache behaviors
     */
    enum class BehaviorType {
        HOT,          ///< Frequently accessed, high hit rate
        COLD,         ///< Infrequently accessed
        VOLATILE,     ///< Frequently created and evicted
        STABLE,       ///< Long-lived with consistent access
        CONTAMINATOR, ///< Takes cache space but rarely used (likely cache contamination)
        CHURNER       ///< Frequently misses despite previous accesses (poor locality)
    };
    
    /**
     * @struct MaterialBehaviorInfo
     * @brief Public information about material cache behavior
     */
    struct MaterialBehaviorInfo {
        uint32_t materialID;
        std::string materialName;
        BehaviorType behaviorType;
        size_t accessCount = 0;
        size_t hitCount = 0;
        size_t missCount = 0;
        size_t evictionCount = 0;
        double hitRate = 0.0;
    };
    
    /**
     * @brief Constructor
     * @param analysisWindowMinutes Time window for behavior analysis in minutes (default: 60)
     */
    MaterialCacheBehaviorTracker(size_t analysisWindowMinutes = 60);
    
    /**
     * @brief Record a material access (hit or miss)
     * @param material The material that was accessed
     * @param isHit Whether the access was a hit (true) or miss (false)
     */
    void recordAccess(const std::shared_ptr<Material>& material, bool isHit);
    
    /**
     * @brief Record an eviction
     * @param material The material that was evicted
     */
    void recordEviction(const std::shared_ptr<Material>& material);
    
    /**
     * @brief Get all materials classified with a specific behavior type
     * @param type The behavior type to query
     * @return Vector of material IDs with the specified behavior
     */
    std::vector<uint32_t> getMaterialsByBehavior(BehaviorType type) const;
    
    /**
     * @brief Get materials classified as contaminators
     * @return Vector of material IDs identified as contaminators
     */
    std::vector<uint32_t> getContaminators() const;
    
    /**
     * @brief Get detailed analysis of material behaviors
     * @return Map of statistic name to value
     */
    std::unordered_map<std::string, double> getStatistics() const;
    
    /**
     * @brief Get detailed information about specific materials
     * @param materialIDs The material IDs to get information for, or empty for all
     * @return Vector of material behavior details
     */
    std::vector<MaterialBehaviorInfo> getMaterialDetails(
        const std::vector<uint32_t>& materialIDs = {}) const;
    
    /**
     * @brief Reset all behavior tracking
     */
    void reset();
    
    /**
     * @brief Convert behavior type to string
     * @param type The behavior type
     * @return String representation
     */
    static const char* behaviorTypeToString(BehaviorType type);
    
private:
    /**
     * @struct MaterialBehavior
     * @brief Internal tracking of material behavior
     */
    struct MaterialBehavior {
        uint32_t materialID;
        std::string materialName;
        Material::MaterialType materialType;
        size_t accessCount = 0;
        size_t hitCount = 0;
        size_t missCount = 0;
        size_t evictionCount = 0;
        std::chrono::time_point<std::chrono::steady_clock> firstSeen;
        std::chrono::time_point<std::chrono::steady_clock> lastSeen;
        BehaviorType currentBehavior = BehaviorType::COLD;
    };
    
    std::unordered_map<uint32_t, MaterialBehavior> m_behaviors;
    std::chrono::minutes m_analysisWindow;
    size_t m_totalAccesses = 0;
    size_t m_totalHits = 0;
    
    /**
     * @brief Classify all materials based on their access patterns
     */
    void classifyAllMaterials();
};

