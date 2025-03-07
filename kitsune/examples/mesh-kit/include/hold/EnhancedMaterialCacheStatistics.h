/**
 * @class EnhancedMaterialCacheStatistics
 * @brief Extended statistics for material cache with material-specific insights
 * @details
 *
 * This class extends the base MaterialCacheStatistics to provide
 * material-specific insights, including tracking per-material type
 * statistics, identifying top hit/miss materials, and other
 * detailed metrics useful for diagnosing cache behavior.
 */
class EnhancedMaterialCacheStatistics : public MaterialCacheStatistics {
public:
    /**
     * @brief Constructor
     */
    EnhancedMaterialCacheStatistics() : MaterialCacheStatistics() {}

    /**
     * @brief Record a cache hit with material information
     * @param material The material that was accessed
     */
    void recordHit(const std::shared_ptr<Material>& material) {
        MaterialCacheStatistics::recordHit();
        
        if (material) {
            // Update type-specific stats
            m_typeStats[material->getType()].hits++;
            
            // Update name-specific stats
            m_materialNameStats[material->getName()].hits++;
            
            // Update top materials
            updateTopMaterials(m_topHitMaterials, material, 10);
        }
    }
    
    /**
     * @brief Record a cache miss with material information
     * @param material The material that was not found
     */
    void recordMiss(const std::shared_ptr<Material>& material) {
        MaterialCacheStatistics::recordMiss();
        
        if (material) {
            // Update type-specific stats
            m_typeStats[material->getType()].misses++;
            
            // Update name-specific stats
            m_materialNameStats[material->getName()].misses++;
            
            // Update top materials
            updateTopMaterials(m_topMissMaterials, material, 10);
        }
    }
    
    /**
     * @brief Record an eviction with material information
     * @param material The material that was evicted
     */
    void recordEviction(const std::shared_ptr<Material>& material) {
        MaterialCacheStatistics::recordEviction();
        
        if (material) {
            // Update type-specific stats
            m_typeStats[material->getType()].evictions++;
            
            // Update name-specific stats
            m_materialNameStats[material->getName()].evictions++;
        }
    }
    
    /**
     * @brief Get hit rate for a specific material type
     * @param type The material type
     * @return Hit rate percentage (0-100)
     */
    double getHitRateForType(Material::MaterialType type) const {
        const auto& stats = m_typeStats.find(type);
        if (stats == m_typeStats.end()) return 0.0;
        
        size_t total = stats->second.hits + stats->second.misses;
        return total > 0 ? static_cast<double>(stats->second.hits) / total * 100.0 : 0.0;
    }
    
    /**
     * @brief Get hit rate for a specific material name
     * @param name The material name
     * @return Hit rate percentage (0-100)
     */
    double getHitRateForName(const std::string& name) const {
        const auto& stats = m_materialNameStats.find(name);
        if (stats == m_materialNameStats.end()) return 0.0;
        
        size_t total = stats->second.hits + stats->second.misses;
        return total > 0 ? static_cast<double>(stats->second.hits) / total * 100.0 : 0.0;
    }
    
    /**
     * @brief Get detailed statistics including material-specific metrics
     * @return Map of statistic name to value
     */
    std::unordered_map<std::string, double> getDetailedStatistics() const {
        auto baseStats = MaterialCacheStatistics::getStatistics();
        std::unordered_map<std::string, double> result(baseStats.begin(), baseStats.end());
        
        // Add type stats
        for (const auto& pair : m_typeStats) {
            std::string typeStr;
            switch (pair.first) {
                case Material::MaterialType::FLUID: typeStr = "Fluid"; break;
                case Material::MaterialType::SOLID: typeStr = "Solid"; break;
                case Material::MaterialType::INTERFACE: typeStr = "Interface"; break;
                default: typeStr = "Unknown";
            }
            
            result["HitRate_" + typeStr] = 
                (pair.second.hits + pair.second.misses > 0) ? 
                static_cast<double>(pair.second.hits) / (pair.second.hits + pair.second.misses) * 100.0 : 0.0;
            
            result["Hits_" + typeStr] = static_cast<double>(pair.second.hits);
            result["Misses_" + typeStr] = static_cast<double>(pair.second.misses);
            result["Evictions_" + typeStr] = static_cast<double>(pair.second.evictions);
        }
        
        // Add top materials information
        // We'll use a separate method to get material names as they're not easily convertible to double
        for (size_t i = 0; i < m_topHitMaterials.size(); i++) {
            result["TopHit_" + std::to_string(i+1) + "_Count"] = 
                static_cast<double>(m_topHitMaterials[i].count);
        }
        
        for (size_t i = 0; i < m_topMissMaterials.size(); i++) {
            result["TopMiss_" + std::to_string(i+1) + "_Count"] = 
                static_cast<double>(m_topMissMaterials[i].count);
        }
        
        return result;
    }
    
    /**
     * @brief Get the top materials by hit count
     * @return Vector of material info sorted by hit count (highest first)
     */
    std::vector<std::pair<std::string, size_t>> getTopHitMaterials() const {
        std::vector<std::pair<std::string, size_t>> result;
        for (const auto& item : m_topHitMaterials) {
            result.emplace_back(item.name, item.count);
        }
        return result;
    }
    
    /**
     * @brief Get the top materials by miss count
     * @return Vector of material info sorted by miss count (highest first)
     */
    std::vector<std::pair<std::string, size_t>> getTopMissMaterials() const {
        std::vector<std::pair<std::string, size_t>> result;
        for (const auto& item : m_topMissMaterials) {
            result.emplace_back(item.name, item.count);
        }
        return result;
    }
    
    /**
     * @brief Reset all statistics
     */
    void reset() override {
        MaterialCacheStatistics::reset();
        m_typeStats.clear();
        m_materialNameStats.clear();
        m_topHitMaterials.clear();
        m_topMissMaterials.clear();
    }
    
private:
    /**
     * @struct MaterialTypeStats
     * @brief Statistics specific to a material type
     */
    struct MaterialTypeStats {
        size_t hits = 0;
        size_t misses = 0;
        size_t evictions = 0;
    };
    
    /**
     * @struct MaterialCounter
     * @brief Tracks access counts for individual materials
     */
    struct MaterialCounter {
        std::string name;
        uint32_t id;
        size_t count;
    };
    
    std::unordered_map<Material::MaterialType, MaterialTypeStats> m_typeStats;
    std::unordered_map<std::string, MaterialTypeStats> m_materialNameStats;
    std::vector<MaterialCounter> m_topHitMaterials;
    std::vector<MaterialCounter> m_topMissMaterials;
    
    /**
     * @brief Update the top materials tracking list
     * @param topList The list to update
     * @param material The material to add or update
     * @param maxItems The maximum number of items to track
     */
    void updateTopMaterials(
        std::vector<MaterialCounter>& topList, 
        const std::shared_ptr<Material>& material, 
        size_t maxItems) {
        
        // Find if material already in list
        auto it = std::find_if(topList.begin(), topList.end(), 
            [&material](const MaterialCounter& counter) {
                return counter.id == material->getID();
            });
            
        if (it != topList.end()) {
            // Update existing entry
            it->count++;
            
            // Re-sort to maintain order
            std::sort(topList.begin(), topList.end(),
                [](const MaterialCounter& a, const MaterialCounter& b) {
                    return a.count > b.count;
                });
        } else if (topList.size() < maxItems) {
            // Add new entry if list not full
            topList.push_back({material->getName(), material->getID(), 1});
            
            // Sort by count
            std::sort(topList.begin(), topList.end(),
                [](const MaterialCounter& a, const MaterialCounter& b) {
                    return a.count > b.count;
                });
        } else if (topList.back().count < 1) {
            // Replace last entry if new material has higher count
            topList.back() = {material->getName(), material->getID(), 1};
            
            // Sort by count
            std::sort(topList.begin(), topList.end(),
                [](const MaterialCounter& a, const MaterialCounter& b) {
                    return a.count > b.count;
                });
        }
    }
};

