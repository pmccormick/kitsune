/**
 * @file ContaminationReport.cpp
 * @brief Implementation of cache contamination reporting utilities
 */

#include "ContaminationReport.h"
#include "MaterialBehaviorTracker.h"
#include "EnhancedStatistics.h"

std::string CacheContaminationReport::generateReport(const MaterialCache& cache, bool verbose) {
    if (!cache.isContaminationDetectionEnabled()) {
        return "Contamination detection is not enabled for this cache.\n";
    }
    
    std::stringstream report;
    report << "=== Material Cache Contamination Report ===\n\n";
    
    // Basic cache statistics
    const auto& stats = cache.getStatistics();
    report << "Cache Name: " << cache.getName() << "\n";
    report << "Cache Size: " << cache.size() << " / " << cache.getMaxSize() << "\n";
    report << "Hit Rate: " << stats.getHitRate() << "%\n";
    report << "Hits: " << stats.getHits() << "\n";
    report << "Misses: " << stats.getMisses() << "\n";
    report << "Evictions: " << stats.getEvictions() << "\n";
    report << "Memory Usage: " << formatBytes(stats.getMemoryUsageBytes()) << "\n\n";
    
    // Enhanced statistics
    const auto& enhancedStats = cache.getEnhancedStatistics();
    const auto& behaviorTracker = cache.getBehaviorTracker();
    
    // Add material type statistics
    report << "--- Material Type Statistics ---\n";
    auto detailedStats = enhancedStats.getDetailedStatistics();
    
    // Extract type stats
    for (const auto& type : { "Fluid", "Solid", "Interface" }) {
        auto hitRateIt = detailedStats.find(std::string("HitRate_") + type);
        auto hitsIt = detailedStats.find(std::string("Hits_") + type);
        auto missesIt = detailedStats.find(std::string("Misses_") + type);
        
        if (hitRateIt != detailedStats.end() && hitsIt != detailedStats.end() && missesIt != detailedStats.end()) {
            report << type << " Hit Rate: " << hitRateIt->second << "%\n";
            report << type << " Hits: " << hitsIt->second << "\n";
            report << type << " Misses: " << missesIt->second << "\n";
        }
    }
    report << "\n";
    
    // Contamination analysis
    auto contaminators = cache.detectContamination();
    report << "--- Contamination Analysis ---\n";
    report << "Detected Contaminators: " << contaminators.size() << "\n";
    
    if (!contaminators.empty()) {
        // Calculate potential memory savings
        size_t estimatedMemorySavings = contaminators.size() * 1024; // 1KB per material (rough estimate)
        report << "Estimated Memory Savings: " << formatBytes(estimatedMemorySavings) << "\n";
        
        if (verbose || contaminators.size() <= 10) {
            report << "\nContaminator Materials:\n";
            for (const auto& contam : contaminators) {
                report << "  - " << contam.materialName << " (ID: " << contam.materialID << ")\n";
                report << "    Access Count: " << contam.accessCount << "\n";
                report << "    Hit Rate: " << contam.hitRate << "%\n";
                report << "    Last Access: " << "Too long ago" << "\n";
            }
            report << "\n";
        } else {
            report << "Top 10 Contaminators (use verbose mode to see all):\n";
            for (size_t i = 0; i < 10; i++) {
                report << "  - " << contaminators[i].materialName << " (ID: " << contaminators[i].materialID << ")\n";
            }
            report << "\n";
        }
    }
    
    // Material behavior statistics
    auto behaviorStats = behaviorTracker.getStatistics();
    report << "--- Material Behavior Statistics ---\n";
    report << "Hot Materials: " << behaviorStats["Materials_Hot"] << "\n";
    report << "Cold Materials: " << behaviorStats["Materials_Cold"] << "\n";
    report << "Volatile Materials: " << behaviorStats["Materials_Volatile"] << "\n";
    report << "Stable Materials: " << behaviorStats["Materials_Stable"] << "\n";
    report << "Contaminator Materials: " << behaviorStats["Materials_Contaminator"] << "\n";
    report << "Churner Materials: " << behaviorStats["Materials_Churner"] << "\n\n";
    
    // Top hit and miss materials
    if (verbose) {
        report << "--- Top Materials by Hit Count ---\n";
        auto topHitMaterials = enhancedStats.getTopHitMaterials();
        for (const auto& material : topHitMaterials) {
            report << "  - " << material.first << ": " << material.second << " hits\n";
        }
        report << "\n";
        
        report << "--- Top Materials by Miss Count ---\n";
        auto topMissMaterials = enhancedStats.getTopMissMaterials();
        for (const auto& material : topMissMaterials) {
            report << "  - " << material.first << ": " << material.second << " misses\n";
        }
        report << "\n";
    }
    
    // Recommendations
    report << "--- Recommendations ---\n";
    if (contaminators.size() > cache.size() * 0.25) {
        report << "- HIGH PRIORITY: Significant cache contamination detected (" 
               << contaminators.size() << " materials, " 
               << (100.0 * contaminators.size() / cache.size()) << "% of cache).\n";
        report << "  Consider calling 'evictContaminators()' to free up space.\n";
    } else if (!contaminators.empty()) {
        report << "- MEDIUM PRIORITY: Some cache contamination detected (" 
               << contaminators.size() << " materials).\n";
        report << "  Consider periodic cleanup with 'evictContaminators()'.\n";
    } else {
        report << "- LOW PRIORITY: No cache contamination detected.\n";
    }
    
    // Additional recommendations based on other metrics
    if (stats.getHitRate() < 50.0) {
        report << "- Consider increasing cache size to improve hit rate.\n";
    }
    
    double volatilesRatio = behaviorStats["Materials_Volatile"] / cache.size();
    if (volatilesRatio > 0.3) {
        report << "- High ratio of volatile materials detected. Consider tuning the eviction policy.\n";
    }
    
    if (behaviorStats["Materials_Churner"] > 0) {
        report << "- Churner materials detected. These frequently miss despite previous access.\n";
        report << "  Consider reviewing their property tolerance settings.\n";
    }
    
    report << "\n";
    return report.str();
}

std::string CacheContaminationReport::generateContaminationSummary(const MaterialCache& cache) {
    if (!cache.isContaminationDetectionEnabled()) {
        return "Contamination detection is not enabled for this cache.\n";
    }
    
    auto contaminators = cache.detectContamination();
    
    std::stringstream report;
    report << "Cache Contamination Summary:\n";
    report << "- Cache Size: " << cache.size() << " / " << cache.getMaxSize() << "\n";
    report << "- Hit Rate: " << cache.getStatistics().getHitRate() << "%\n";
    report << "- Contaminators: " << contaminators.size() << " materials";
    
    if (!contaminators.empty()) {
        size_t estimatedMemorySavings = contaminators.size() * 1024; // 1KB per material (estimate)
        report << " (" << formatBytes(estimatedMemorySavings) << " potential savings)\n";
        
        if (contaminators.size() <= 5) {
            report << "- Top Contaminators:\n";
            for (const auto& contam : contaminators) {
                report << "  * " << contam.materialName << " (accessed " << contam.accessCount << " times)\n";
            }
        } else {
            report << "- Top 5 Contaminators:\n";
            for (size_t i = 0; i < 5; i++) {
                report << "  * " << contaminators[i].materialName << " (accessed " 
                       << contaminators[i].accessCount << " times)\n";
            }
        }
    } else {
        report << "\n- No contamination detected.\n";
    }
    
    return report.str();
}

std::string CacheContaminationReport::generatePerformanceReport(const MaterialCache& cache) {
    std::stringstream report;
    report << "=== Material Cache Performance Report ===\n\n";
    
    // Basic cache statistics
    const auto& stats = cache.getStatistics();
    report << "Cache Configuration:\n";
    report << "- Name: " << cache.getName() << "\n";
    report << "- Size: " << cache.size() << " / " << cache.getMaxSize() << "\n";
    report << "- Equivalence Strategy: " << cache.getEquivalenceKey().getName() << "\n";
    report << "- Equivalence Tolerance: " << cache.getEquivalenceKey().getTolerance() << "\n";
    report << "- Eviction Policy: " << cache.getEvictionPolicy().getName() << "\n\n";
    
    report << "Performance Metrics:\n";
    report << "- Hit Rate: " << stats.getHitRate() << "%\n";
    report << "- Average Query Time: " << stats.getAvgQueryTimeNs() << " ns\n";
    report << "- Memory Usage: " << formatBytes(stats.getMemoryUsageBytes()) << "\n\n";
    
    if (cache.isContaminationDetectionEnabled()) {
        const auto& behaviorTracker = cache.getBehaviorTracker();
        auto behaviorStats = behaviorTracker.getStatistics();
        
        // Calculate performance metrics
        double contaminationRatio = behaviorStats["Materials_Contaminator"] / cache.size();
        double churnRatio = behaviorStats["Materials_Churner"] / cache.size();
        double volatileRatio = behaviorStats["Materials_Volatile"] / cache.size();
        double stableRatio = behaviorStats["Materials_Stable"] / cache.size();
        
        report << "Cache Health Indicators:\n";
        report << "- Contamination Ratio: " << (contaminationRatio * 100.0) << "%\n";
        report << "- Churn Ratio: " << (churnRatio * 100.0) << "%\n";
        report << "- Volatile Ratio: " << (volatileRatio * 100.0) << "%\n";
        report << "- Stable Ratio: " << (stableRatio * 100.0) << "%\n\n";
        
        // Diagnosis
        report << "Cache Diagnosis:\n";
        
        if (contaminationRatio > 0.2) {
            report << "- HIGH CONTAMINATION: " << (contaminationRatio * 100.0) 
                   << "% of cache is rarely used but taking space.\n";
        }
        
        if (churnRatio > 0.1) {
            report << "- HIGH CHURN: Many materials with poor hit rates despite frequent access.\n";
            report << "  This suggests non-optimal equivalence settings or rapid property changes.\n";
        }
        
        if (volatileRatio > 0.3) {
            report << "- HIGH VOLATILITY: Materials are frequently created and evicted.\n";
            report << "  This suggests the cache size might be too small for the working set.\n";
        }
        
        if (stableRatio < 0.1 && cache.size() > 10) {
            report << "- LOW STABILITY: Few materials remain in cache for long periods.\n";
            report << "  This suggests high turnover and potential inefficiency.\n";
        }
        
        if (stats.getHitRate() < 40.0) {
            report << "- LOW HIT RATE: Cache is not effectively serving its purpose.\n";
        }
        
        report << "\n";
        
        // Recommendations
        report << "Optimization Recommendations:\n";
        
        if (contaminationRatio > 0.2) {
            report << "1. Call evictContaminators() to free up " 
                   << formatBytes(static_cast<size_t>(behaviorStats["Materials_Contaminator"]) * 1024)
                   << " of memory.\n";
        }
        
        if (stats.getHitRate() < 50.0 && cache.getMaxSize() > 0) {
            size_t recommendedSize = cache.getMaxSize() * 2;
            report << "2. Increase cache size from " << cache.getMaxSize() << " to approximately " 
                   << recommendedSize << " to improve hit rate.\n";
        }
        
        if (churnRatio > 0.1) {
            double currentTolerance = cache.getEquivalenceKey().getTolerance();
            double recommendedTolerance = currentTolerance * 2.0;
            report << "3. Consider increasing equivalence tolerance from " << currentTolerance 
                   << " to approximately " << recommendedTolerance << ".\n";
        }
        
        const auto& eqKey = cache.getEquivalenceKey();
        bool allPropertiesCompared = true;
        for (size_t i = 0; i < static_cast<size_t>(Material::MaterialProperty::COUNT); ++i) {
            auto prop = static_cast<Material::MaterialProperty>(i);
            if (!eqKey.isPropertyCompared(prop)) {
                allPropertiesCompared = false;
                break;
            }
        }
        
        if (allPropertiesCompared && churnRatio > 0.05) {
            report << "4. Consider disabling comparison of less important properties to improve hit rate.\n";
        }
        
        if (stats.getHitRate() > 98.0 && eqKey.getTolerance() > 1e-8) {
            report << "5. Your hit rate is very high. You might increase precision by decreasing\n";
            report << "   the tolerance without significant performance impact.\n";
        }
    } else {
        report << "Note: Enable contamination detection for more detailed analysis.\n";
    }
    
    return report.str();
}

std::string CacheContaminationReport::formatBytes(size_t bytes) {
    static const char* suffixes[] = {"B", "KB", "MB", "GB", "TB"};
    int suffixIndex = 0;
    double size = static_cast<double>(bytes);
    
    while (size >= 1024 && suffixIndex < 4) {
        size /= 1024;
        suffixIndex++;
    }
    
    std::stringstream ss;
    ss << std::fixed << std::setprecision(2) << size << " " << suffixes[suffixIndex];
    return ss.str();
}

