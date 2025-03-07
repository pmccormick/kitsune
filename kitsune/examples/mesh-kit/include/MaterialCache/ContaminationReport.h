/**
 * @file ContaminationReport.h
 * @brief Reporting utilities for cache contamination
 * @details
 *
 * This file provides the CacheContaminationReport class which generates
 * reports and analysis of cache contamination and performance.
 */

#pragma once

#include "MaterialCache.h"
#include <string>
#include <sstream>
#include <iomanip>
#include <chrono>

/**
 * @class CacheContaminationReport
 * @brief Utility for generating reports on cache contamination
 * @details
 *
 * This class provides methods to analyze a MaterialCache instance and generate
 * reports about cache contamination, usage patterns, and optimization suggestions.
 */
class CacheContaminationReport {
public:
    /**
     * @brief Generate a contamination report for a cache
     * @param cache The cache to analyze
     * @param verbose Whether to include detailed information
     * @return Report text
     */
    static std::string generateReport(const MaterialCache& cache, bool verbose = false);
    
    /**
     * @brief Generate a brief report focusing just on contamination
     * @param cache The cache to analyze
     * @return Report text
     */
    static std::string generateContaminationSummary(const MaterialCache& cache);
    
    /**
     * @brief Generate a more analytical report with performance recommendations
     * @param cache The cache to analyze
     * @return Report text with analysis and recommendations
     */
    static std::string generatePerformanceReport(const MaterialCache& cache);
    
private:
    /**
     * @brief Format bytes to human-readable string
     * @param bytes Number of bytes
     * @return Formatted string (e.g., "1.23 MB")
     */
    static std::string formatBytes(size_t bytes);
};

