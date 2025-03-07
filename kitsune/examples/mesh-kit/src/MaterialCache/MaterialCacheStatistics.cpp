/**
 * @file MaterialCacheStatistics.cpp
 * @brief Implementation of material cache statistics tracking
 */

#include "MaterialCacheStatistics.h"

void MaterialCacheStatistics::recordHit() {
    m_hits++;
    m_totalQueries++;
}

void MaterialCacheStatistics::recordMiss() {
    m_misses++;
    m_totalQueries++;
}

void MaterialCacheStatistics::recordEviction() {
    m_evictions++;
}

void MaterialCacheStatistics::recordQueryTime(std::chrono::nanoseconds duration) {
    m_totalQueryTimeNs += duration.count();
    if (m_totalQueries > 0) {
        m_avgQueryTimeNs = m_totalQueryTimeNs / m_totalQueries;
    }
}

void MaterialCacheStatistics::updateMemoryUsage(size_t bytes) {
    m_memoryUsageBytes = bytes;
}

size_t MaterialCacheStatistics::getHits() const {
    return m_hits;
}

size_t MaterialCacheStatistics::getMisses() const {
    return m_misses;
}

size_t MaterialCacheStatistics::getEvictions() const {
    return m_evictions;
}

size_t MaterialCacheStatistics::getTotalQueries() const {
    return m_totalQueries;
}

double MaterialCacheStatistics::getHitRate() const {
    size_t totalQueries = m_hits + m_misses;
    return totalQueries > 0 ? (static_cast<double>(m_hits) / totalQueries) * 100.0 : 0.0;
}

double MaterialCacheStatistics::getAvgQueryTimeNs() const {
    return m_avgQueryTimeNs;
}

size_t MaterialCacheStatistics::getMemoryUsageBytes() const {
    return m_memoryUsageBytes;
}

std::unordered_map<std::string, double> MaterialCacheStatistics::getStatistics() const {
    return {
        {"Hits", static_cast<double>(m_hits)},
        {"Misses", static_cast<double>(m_misses)},
        {"Total Queries", static_cast<double>(m_totalQueries)},
        {"Evictions", static_cast<double>(m_evictions)},
        {"Hit Rate", getHitRate()},
        {"Avg Query Time (ns)", m_avgQueryTimeNs},
        {"Memory Usage (bytes)", static_cast<double>(m_memoryUsageBytes)}
    };
}

void MaterialCacheStatistics::reset() {
    m_hits = 0;
    m_misses = 0;
    m_evictions = 0;
    m_totalQueries = 0;
    m_totalQueryTimeNs = 0;
    m_avgQueryTimeNs = 0;
    m_memoryUsageBytes = 0;
}


