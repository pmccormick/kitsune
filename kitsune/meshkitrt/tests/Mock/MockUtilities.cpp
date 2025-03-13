#ifndef MOCK_UTILITIES_H
#define MOCK_UTILITIES_H

#include "mock/MockMeshBase.h"
#include "mock/MockField.h"
#include "mock/MockCell.h"
#include "mock/MockRegion.h"
#include "mock/MockBitArray.h"
#include <vector>
#include <random>
#include <cmath>

namespace mock {

/**
 * @brief Namespace for utility functions used in tests
 */
namespace utilities {

/**
 * @brief Generate a random binary pattern for testing
 * 
 * @param size Size of the pattern
 * @param density Probability of a bit being set (0.0-1.0)
 * @return std::vector<bool> Random binary pattern
 */
inline std::vector<bool> generateRandomPattern(size_t size, double density = 0.5) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);
    
    std::vector<bool> pattern(size, false);
    for (size_t i = 0; i < size; i++) {
        if (dis(gen) < density) {
            pattern[i] = true;
        }
    }
    
    return pattern;
}

/**
 * @brief Generate a sparse random pattern for testing
 * 
 * @param size Size of the pattern
 * @param numSetBits Number of bits to set to true
 * @return std::vector<bool> Sparse random pattern
 */
inline std::vector<bool> generateSparsePattern(size_t size, size_t numSetBits) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    
    std::vector<bool> pattern(size, false);
    
    if (numSetBits >= size) {
        // If we want more set bits than size, set all bits
        std::fill(pattern.begin(), pattern.end(), true);
        return pattern;
    }
    
    // Set numSetBits random bits
    std::vector<size_t> indices(size);
    for (size_t i = 0; i < size; i++) {
        indices[i] = i;
    }
    
    // Shuffle indices
    std::shuffle(indices.begin(), indices.end(), gen);
    
    // Set the first numSetBits indices to true
    for (size_t i = 0; i < numSetBits; i++) {
        pattern[indices[i]] = true;
    }
    
    return pattern;
}

/**
 * @brief Create a BitArray from a pattern
 * 
 * @param pattern Pattern of bits
 * @return MockBitArray BitArray with the specified pattern
 */
inline MockBitArray createBitArrayFromPattern(const std::vector<bool>& pattern) {
    MockBitArray array(pattern.size(), false);
    
    for (size_t i = 0; i < pattern.size(); i++) {
        if (pattern[i]) {
            array.set(i, true);
        }
    }
    
    return array;
}

/**
 * @brief Create a region from a pattern
 * 
 * @param id Region ID
 * @param name Region name
 * @param mesh Mock mesh for context
 * @param pattern Pattern of bits
 * @return MockRegion Region with the specified pattern
 */
inline MockRegion createRegionFromPattern(uint32_t id, const std::string& name, 
                                        const MockMeshBase& mesh, const std::vector<bool>& pattern) {
    MockRegion region(id, name, pattern.size());
    
    for (size_t i = 0; i < pattern.size(); i++) {
        if (pattern[i]) {
            region.addCellIndex(static_cast<int>(i));
        }
    }
    
    return region;
}

/**
 * @brief Compare expected vs actual results
 * 
 * @tparam T Value type
 * @param expected Expected values
 * @param actual Actual values
 * @param epsilon Tolerance for floating point comparisons
 * @return true if values match within tolerance
 */
template <typename T>
inline bool compareResults(const std::vector<T>& expected, const std::vector<T>& actual, 
                         double epsilon = 1e-6) {
    if (expected.size() != actual.size()) {
        return false;
    }
    
    for (size_t i = 0; i < expected.size(); i++) {
        if constexpr (std::is_floating_point_v<T>) {
            if (std::abs(expected[i] - actual[i]) > epsilon) {
                return false;
            }
        } else {
            if (expected[i] != actual[i]) {
                return false;
            }
        }
    }
    
    return true;
}

/**
 * @brief Fill a field with a function
 * 
 * @tparam T Field value type
 * @param field Field to fill
 * @param func Function that takes (i, j) and returns a value
 */
template <typename T, typename Func>
inline void fillFieldWithFunction(MockField<T>& field, Func func) {
    for (uint32_t j = 0; j < field.ny(); j++) {
        for (uint32_t i = 0; i < field.nx(); i++) {
            field(i, j) = func(i, j);
        }
    }
}

/**
 * @brief Log a bit array pattern (for debugging)
 * 
 * @param array BitArray to log
 * @param width Format width (0 = linear, >0 = 2D grid)
 */
inline void logBitArrayPattern(const BitArray& array, size_t width = 0) {
    std::string pattern;
    
    for (size_t i = 0; i < array.size(); i++) {
        pattern += array.get(i) ? '1' : '0';
        
        if (width > 0 && (i + 1) % width == 0) {
            pattern += '\n';
        }
    }
    
    printf("BitArray Pattern (size=%zu, count=%zu):\n%s\n", 
           array.size(), array.count(), pattern.c_str());
}

} // namespace utilities

} // namespace mock

#endif // MOCK_UTILITIES_H
