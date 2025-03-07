/**
 * @file EvictionPolicies.cpp
 * @brief Implementation of cache eviction policy strategies
 */

#include "EvictionPolicies.h"
#include <algorithm>

//==============================================================================
// LRUEvictionPolicy Implementation
//==============================================================================

void LRUEvictionPolicy::onAccess(void* item) {
    auto it = m_itemMap.find(item);
    if (it != m_itemMap.end()) {
        // Move to front of list (most recently used)
        m_lruList.erase(it->second);
        m_lruList.push_front(item);
        it->second = m_lruList.begin();
        m_accessCount++;
    }
}

void* LRUEvictionPolicy::selectForEviction() {
    if (!m_lruList.empty()) {
        return m_lruList.back(); // Return least recently used
    }
    return nullptr;
}

void LRUEvictionPolicy::addItem(void* item) {
    if (m_itemMap.find(item) == m_itemMap.end()) {
        // Add to front of list (most recently used)
        m_lruList.push_front(item);
        m_itemMap[item] = m_lruList.begin();
    }
}

void LRUEvictionPolicy::removeItem(void* item) {
    auto it = m_itemMap.find(item);
    if (it != m_itemMap.end()) {
        m_lruList.erase(it->second);
        m_itemMap.erase(it);
    }
}

void LRUEvictionPolicy::clear() {
    m_lruList.clear();
    m_itemMap.clear();
    m_accessCount = 0;
}

size_t LRUEvictionPolicy::size() const {
    return m_lruList.size();
}

std::string LRUEvictionPolicy::getName() const {
    return "LRU (Least Recently Used)";
}

std::unordered_map<std::string, double> LRUEvictionPolicy::getStatistics() const {
    return {
        {"Access Count", static_cast<double>(m_accessCount)},
        {"Item Count", static_cast<double>(m_lruList.size())}
    };
}

//==============================================================================
// MRUEvictionPolicy Implementation
//==============================================================================

void MRUEvictionPolicy::onAccess(void* item) {
    auto it = m_itemMap.find(item);
    if (it != m_itemMap.end()) {
        // Move to front of list (most recently used)
        m_mruList.erase(it->second);
        m_mruList.push_front(item);
        it->second = m_mruList.begin();
        m_accessCount++;
    }
}

void* MRUEvictionPolicy::selectForEviction() {
    if (!m_mruList.empty()) {
        return m_mruList.front(); // Return most recently used
    }
    return nullptr;
}

void MRUEvictionPolicy::addItem(void* item) {
    if (m_itemMap.find(item) == m_itemMap.end()) {
        // Add to front of list (most recently used)
        m_mruList.push_front(item);
        m_itemMap[item] = m_mruList.begin();
    }
}

void MRUEvictionPolicy::removeItem(void* item) {
    auto it = m_itemMap.find(item);
    if (it != m_itemMap.end()) {
        m_mruList.erase(it->second);
        m_itemMap.erase(it);
    }
}

void MRUEvictionPolicy::clear() {
    m_mruList.clear();
    m_itemMap.clear();
    m_accessCount = 0;
}

size_t MRUEvictionPolicy::size() const {
    return m_mruList.size();
}

std::string MRUEvictionPolicy::getName() const {
    return "MRU (Most Recently Used)";
}

std::unordered_map<std::string, double> MRUEvictionPolicy::getStatistics() const {
    return {
        {"Access Count", static_cast<double>(m_accessCount)},
        {"Item Count", static_cast<double>(m_mruList.size())}
    };
}

//==============================================================================
// FIFOEvictionPolicy Implementation
//==============================================================================

void FIFOEvictionPolicy::onAccess(void* item) {
    // No change in order, just record the access
    if (m_itemSet.find(item) != m_itemSet.end()) {
        m_accessCount++;
    }
}

void* FIFOEvictionPolicy::selectForEviction() {
    if (!m_fifoQueue.empty()) {
        return m_fifoQueue.back(); // Return oldest inserted
    }
    return nullptr;
}

void FIFOEvictionPolicy::addItem(void* item) {
    if (m_itemSet.find(item) == m_itemSet.end()) {
        // Add to front of queue (newest)
        m_fifoQueue.push_front(item);
        m_itemSet.insert(item);
    }
}

void FIFOEvictionPolicy::removeItem(void* item) {
    auto it = std::find(m_fifoQueue.begin(), m_fifoQueue.end(), item);
    if (it != m_fifoQueue.end()) {
        m_fifoQueue.erase(it);
        m_itemSet.erase(item);
    }
}

void FIFOEvictionPolicy::clear() {
    m_fifoQueue.clear();
    m_itemSet.clear();
    m_accessCount = 0;
}

size_t FIFOEvictionPolicy::size() const {
    return m_fifoQueue.size();
}

std::string FIFOEvictionPolicy::getName() const {
    return "FIFO (First In First Out)";
}

std::unordered_map<std::string, double> FIFOEvictionPolicy::getStatistics() const {
    return {
        {"Access Count", static_cast<double>(m_accessCount)},
        {"Item Count", static_cast<double>(m_fifoQueue.size())}
    };
}

//==============================================================================
// FrequencyEvictionPolicy Implementation
//==============================================================================

void FrequencyEvictionPolicy::onAccess(void* item) {
    auto it = m_frequencyMap.find(item);
    if (it != m_frequencyMap.end()) {
        // Increase frequency counter
        it->second++;
        m_accessCount++;
        
        // Update the frequency-ordered list
        updateFrequencyList(item);
    }
}

void* FrequencyEvictionPolicy::selectForEviction() {
    if (!m_frequencyList.empty()) {
        // Return item with lowest frequency
        return m_frequencyList.front().first;
    }
    return nullptr;
}

void FrequencyEvictionPolicy::addItem(void* item) {
    if (m_frequencyMap.find(item) == m_frequencyMap.end()) {
        // Initialize with frequency 1
        m_frequencyMap[item] = 1;
        
        // Add to frequency list
        m_frequencyList.push_back(std::make_pair(item, 1));
        
        // Sort the list by frequency (least frequent first)
        sortFrequencyList();
    }
}

void FrequencyEvictionPolicy::removeItem(void* item) {
    auto it = m_frequencyMap.find(item);
    if (it != m_frequencyMap.end()) {
        m_frequencyMap.erase(it);
        
        // Remove from frequency list
        auto listIt = std::find_if(m_frequencyList.begin(), m_frequencyList.end(),
            [item](const auto& pair) { return pair.first == item; });
        if (listIt != m_frequencyList.end()) {
            m_frequencyList.erase(listIt);
        }
    }
}

void FrequencyEvictionPolicy::clear() {
    m_frequencyMap.clear();
    m_frequencyList.clear();
    m_accessCount = 0;
}

size_t FrequencyEvictionPolicy::size() const {
    return m_frequencyMap.size();
}

std::string FrequencyEvictionPolicy::getName() const {
    return "LFU (Least Frequently Used)";
}

std::unordered_map<std::string, double> FrequencyEvictionPolicy::getStatistics() const {
    double avgFrequency = 0.0;
    size_t maxFrequency = 0;
    
    if (!m_frequencyMap.empty()) {
        size_t totalFrequency = 0;
        for (const auto& pair : m_frequencyMap) {
            totalFrequency += pair.second;
            maxFrequency = std::max(maxFrequency, pair.second);
        }
        avgFrequency = static_cast<double>(totalFrequency) / m_frequencyMap.size();
    }
    
    return {
        {"Access Count", static_cast<double>(m_accessCount)},
        {"Item Count", static_cast<double>(m_frequencyMap.size())},
        {"Average Frequency", avgFrequency},
        {"Maximum Frequency", static_cast<double>(maxFrequency)}
    };
}

void FrequencyEvictionPolicy::updateFrequencyList(void* item) {
    // Find and update the item in the list
    auto it = std::find_if(m_frequencyList.begin(), m_frequencyList.end(),
        [item](const auto& pair) { return pair.first == item; });
    
    if (it != m_frequencyList.end()) {
        it->second = m_frequencyMap[item];
        sortFrequencyList();
    }
}

void FrequencyEvictionPolicy::sortFrequencyList() {
    // Sort by frequency (ascending)
    std::sort(m_frequencyList.begin(), m_frequencyList.end(),
        [](const auto& a, const auto& b) { return a.second < b.second; });
}

//==============================================================================
// TimeSensitiveEvictionPolicy Implementation
//==============================================================================

TimeSensitiveEvictionPolicy::TimeSensitiveEvictionPolicy(double recencyWeight)
    : m_recencyWeight(recencyWeight) {}

void TimeSensitiveEvictionPolicy::onAccess(void* item) {
    auto it = m_scoreMap.find(item);
    if (it != m_scoreMap.end()) {
        // Update last access time
        m_lastAccessMap[item] = m_accessCount;
        
        // Increase access count
        m_accessCountMap[item]++;
        
        // Update score
        updateScore(item);
        
        m_totalAccessCount++;
    }
}

void* TimeSensitiveEvictionPolicy::selectForEviction() {
    if (m_scoreList.empty()) {
        return nullptr;
    }
    
    // Return item with lowest score
    return m_scoreList.front().first;
}

void TimeSensitiveEvictionPolicy::addItem(void* item) {
    if (m_scoreMap.find(item) == m_scoreMap.end()) {
        // Initialize
        m_lastAccessMap[item] = m_accessCount;
        m_accessCountMap[item] = 1;
        
        // Calculate initial score
        double score = calculateScore(item);
        m_scoreMap[item] = score;
        
        // Add to score list
        m_scoreList.push_back(std::make_pair(item, score));
        
        // Sort the list by score (lowest first)
        sortScoreList();
        
        m_accessCount++;
    }
}

void TimeSensitiveEvictionPolicy::removeItem(void* item) {
    auto it = m_scoreMap.find(item);
    if (it != m_scoreMap.end()) {
        m_scoreMap.erase(it);
        m_lastAccessMap.erase(item);
        m_accessCountMap.erase(item);
        
        // Remove from score list
        auto listIt = std::find_if(m_scoreList.begin(), m_scoreList.end(),
            [item](const auto& pair) { return pair.first == item; });
        if (listIt != m_scoreList.end()) {
            m_scoreList.erase(listIt);
        }
    }
}

void TimeSensitiveEvictionPolicy::clear() {
    m_scoreMap.clear();
    m_lastAccessMap.clear();
    m_accessCountMap.clear();
    m_scoreList.clear();
    m_accessCount = 0;
    m_totalAccessCount = 0;
}

size_t TimeSensitiveEvictionPolicy::size() const {
    return m_scoreMap.size();
}

std::string TimeSensitiveEvictionPolicy::getName() const {
    return "Time-Sensitive (Recency+Frequency)";
}

std::unordered_map<std::string, double> TimeSensitiveEvictionPolicy::getStatistics() const {
    return {
        {"Access Count", static_cast<double>(m_totalAccessCount)},
        {"Item Count", static_cast<double>(m_scoreMap.size())},
        {"Recency Weight", m_recencyWeight}
    };
}

double TimeSensitiveEvictionPolicy::calculateScore(void* item) {
    // Normalize access counts to 0-1 range
    double maxAccessCount = 1.0;
    for (const auto& pair : m_accessCountMap) {
        maxAccessCount = std::max(maxAccessCount, static_cast<double>(pair.second));
    }
    
    // Calculate normalized frequency score (higher is better)
    double frequencyScore = static_cast<double>(m_accessCountMap[item]) / maxAccessCount;
    
    // Calculate normalized recency score (higher is better)
    double recencyScore = static_cast<double>(m_lastAccessMap[item]) / m_accessCount;
    
    // Combine scores (invert for eviction priority - lower is evicted first)
    return 1.0 - (m_recencyWeight * recencyScore + (1.0 - m_recencyWeight) * frequencyScore);
}

void TimeSensitiveEvictionPolicy::updateScore(void* item) {
    // Recalculate score
    double score = calculateScore(item);
    m_scoreMap[item] = score;
    
    // Update in score list
    auto listIt = std::find_if(m_scoreList.begin(), m_scoreList.end(),
        [item](const auto& pair) { return pair.first == item; });
    if (listIt != m_scoreList.end()) {
        listIt->second = score;
        sortScoreList();
    }
}

void TimeSensitiveEvictionPolicy::sortScoreList() {
    // Sort by score (ascending - lower scores evicted first)
    std::sort(m_scoreList.begin(), m_scoreList.end(),
        [](const auto& a, const auto& b) { return a.second < b.second; });
}

//==============================================================================
// AdaptiveEvictionPolicy Implementation
//==============================================================================

AdaptiveEvictionPolicy::AdaptiveEvictionPolicy(size_t adaptationInterval)
    : m_adaptationInterval(adaptationInterval) {
    // Create underlying policies
    m_lruPolicy = std::make_unique<LRUEvictionPolicy>();
    m_lfuPolicy = std::make_unique<FrequencyEvictionPolicy>();
    m_timePolicy = std::make_unique<TimeSensitiveEvictionPolicy>();
    
    // Start with LRU
    m_activePolicy = m_lruPolicy.get();
    m_activePolicyName = "LRU";
}

void AdaptiveEvictionPolicy::onAccess(void* item) {
    // Forward to all policies
    m_lruPolicy->onAccess(item);
    m_lfuPolicy->onAccess(item);
    m_timePolicy->onAccess(item);
    
    // Record access pattern
    recordAccess(item);
    
    // Check if adaptation is needed
    m_accessCount++;
    if (m_accessCount % m_adaptationInterval == 0) {
        adaptPolicy();
    }
}

void* AdaptiveEvictionPolicy::selectForEviction() {
    // Use active policy
    return m_activePolicy->selectForEviction();
}

void AdaptiveEvictionPolicy::addItem(void* item) {
    // Add to all policies
    m_lruPolicy->addItem(item);
    m_lfuPolicy->addItem(item);
    m_timePolicy->addItem(item);
}

void AdaptiveEvictionPolicy::removeItem(void* item) {
    // Remove from all policies
    m_lruPolicy->removeItem(item);
    m_lfuPolicy->removeItem(item);
    m_timePolicy->removeItem(item);
}

void AdaptiveEvictionPolicy::clear() {
    // Clear all policies
    m_lruPolicy->clear();
    m_lfuPolicy->clear();
    m_timePolicy->clear();
    m_accessCount = 0;
    m_accessInfo.recentAccesses.clear();
    m_accessInfo.accessCounts.clear();
}

size_t AdaptiveEvictionPolicy::size() const {
    // All policies should have the same size
    return m_lruPolicy->size();
}

std::string AdaptiveEvictionPolicy::getName() const {
    return "Adaptive (" + m_activePolicyName + ")";
}

std::unordered_map<std::string, double> AdaptiveEvictionPolicy::getStatistics() const {
    return {
        {"Access Count", static_cast<double>(m_accessCount)},
        {"Item Count", static_cast<double>(size())},
        {"Active Policy", 0.0}, // Just a placeholder, string in name
        {"Pattern", 0.0} // Just a placeholder, pattern is a string
    };
}

void AdaptiveEvictionPolicy::recordAccess(void* item) {
    // Keep track of recent accesses (limit to last 1000)
    m_accessInfo.recentAccesses.push_back(item);
    if (m_accessInfo.recentAccesses.size() > 1000) {
        m_accessInfo.recentAccesses.erase(m_accessInfo.recentAccesses.begin());
    }
    
    // Update access counts
    m_accessInfo.accessCounts[item]++;
}

void AdaptiveEvictionPolicy::adaptPolicy() {
    // Detect the dominant access pattern
    std::string pattern = detectPattern();
    
    // Switch policy based on pattern
    if (pattern == "SEQUENTIAL") {
        // Sequential access pattern favors MRU
        // but we don't have MRU as one of our policies
        // so use Time-Sensitive with high recency weight
        m_activePolicy = m_timePolicy.get();
        m_activePolicyName = "Time-Sensitive";
    }
    else if (pattern == "LOOP") {
        // Looping access pattern favors LRU
        m_activePolicy = m_lruPolicy.get();
        m_activePolicyName = "LRU";
    }
    else if (pattern == "RANDOM") {
        // Random access pattern with no clear structure
        // Time-sensitive is a good compromise
        m_activePolicy = m_timePolicy.get();
        m_activePolicyName = "Time-Sensitive";
    }
    else if (pattern == "FREQUENCY") {
        // Clear frequency-based pattern
        m_activePolicy = m_lfuPolicy.get();
        m_activePolicyName = "LFU";
    }
    else {
        // Default to LRU for unknown patterns
        m_activePolicy = m_lruPolicy.get();
        m_activePolicyName = "LRU";
    }
}

std::string AdaptiveEvictionPolicy::detectPattern() const {
    // This is a simplified pattern detection
    // In a real implementation, more sophisticated analysis would be done
    
    // Not enough data for analysis
    if (m_accessInfo.recentAccesses.size() < 100) {
        return "UNKNOWN";
    }
    
    // Check for frequency-dominated pattern
    // (Few items with very high access counts)
    bool highSkew = false;
    if (!m_accessInfo.accessCounts.empty()) {
        // Find max access count
        size_t maxCount = 0;
        for (const auto& pair : m_accessInfo.accessCounts) {
            maxCount = std::max(maxCount, pair.second);
        }
        
        // Check if max is significantly higher than average
        double avgCount = static_cast<double>(m_accessCount) / m_accessInfo.accessCounts.size();
        if (maxCount > avgCount * 5) {
            highSkew = true;
        }
    }
    
    if (highSkew) {
        return "FREQUENCY";
    }
    
    // Check for loop pattern (repeating sequence)
    // This is a simplistic check for repeating items
    bool hasLoop = false;
    if (m_accessInfo.recentAccesses.size() > 20) {
        size_t matchCount = 0;
        // Check if last 10 items match 10 items before them
        for (size_t i = 0; i < 10; i++) {
            size_t idx1 = m_accessInfo.recentAccesses.size() - 1 - i;
            size_t idx2 = m_accessInfo.recentAccesses.size() - 11 - i;
            if (idx2 < m_accessInfo.recentAccesses.size() && 
                m_accessInfo.recentAccesses[idx1] == m_accessInfo.recentAccesses[idx2]) {
                matchCount++;
            }
        }
        if (matchCount >= 7) { // 70% match
            hasLoop = true;
        }
    }
    
    if (hasLoop) {
        return "LOOP";
    }
    
    // Check for sequential access pattern
    // (Items accessed once and never again)
    bool sequential = true;
    for (size_t i = 0; i < std::min<size_t>(50, m_accessInfo.recentAccesses.size()); i++) {
        void* item = m_accessInfo.recentAccesses[m_accessInfo.recentAccesses.size() - 1 - i];
        if (m_accessInfo.accessCounts[item] > 1) {
            sequential = false;
            break;
        }
    }
    
    if (sequential) {
        return "SEQUENTIAL";
    }
    
    // Default: assume random access
    return "RANDOM";
}


