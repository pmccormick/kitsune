/**
 * @file EnhancedStatistics.cpp
 * @brief Implementation of enhanced material cache statistics
 */

#include "EnhancedStatistics.h"

EnhancedMaterialCacheStatistics::EnhancedMaterialCacheStatistics() 
    : MaterialCacheStatistics() {
}

void EnhancedMaterialCacheStatistics::recordHit(const std::shared_ptr<Material>& material) {
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

void EnhancedMaterialCacheStatistics::recordMiss(const std::shared_ptr<Material>& material) {
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

void EnhancedMaterialCacheStatistics::recordEviction(const std::shared_ptr<Material>& material) {
    MaterialCacheStatistics::recordEviction();
    
    if (material) {
        // Update type-specific stats
        m_typeStats[material->getType()].evictions++;
        
        // Update name-specific stats
        m_materialNameStats[material->getName()].evictions++;
    }
}

double EnhancedMaterialCacheStatistics::getHitRateForType(Material::MaterialType type) const {
    const auto& stats = m_typeStats.find(type);
    if (stats == m_typeStats.end()) return 0.0;
    
    size_t total = stats->second.hits + stats->second.misses;
    return total > 0 ? static_cast<double>(stats->second.hits) / total * 100.0 : 0.0;
}

double EnhancedMaterialCacheStatistics::getHitRateForName(const std::string& name) const {
    const auto& stats = m_materialNameStats.find(name);
    if (stats == m_materialNameStats.end()) return 0.0;
    
    size_t total = stats->second.hits + stats->second.misses;
    return total > 0 ? static_cast<double>(stats->second.hits) / total * 100.0 : 0.0;
}

std::unordered_map<std::string, double> EnhancedMaterialCacheStatistics::getDetailedStatistics() const {
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

std::vector<std::pair<std::string, size_t>> EnhancedMaterialCacheStatistics::getTopHitMaterials() const {
    std::vector<std::pair<std::string, size_t>> result;
    for (const auto& item : m_topHitMaterials) {
        result.emplace_back(item.name, item.count);
    }
    return result;
}

std::vector<std::pair<std::string, size_t>> EnhancedMaterialCacheStatistics::getTopMissMaterials() const {
    std::vector<std::pair<std::string, size_t>> result;
    for (const auto& item : m_topMissMaterials) {
        result.emplace_back(item.name, item.count);
    }
    return result;
}

void EnhancedMaterialCacheStatistics::reset() {
    MaterialCacheStatistics::reset();
    m_typeStats.clear();
    m_materialNameStats.clear();
    m_topHitMaterials.clear();
    m_topMissMaterials.clear();
}

void EnhancedMaterialCacheStatistics::updateTopMaterials(
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


