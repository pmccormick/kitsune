#ifndef MOCK_BIT_ARRAY_H
#define MOCK_BIT_ARRAY_H

#include "BitArray.h"
#include <vector>
#include <algorithm>
#include <cstdint>

namespace mock {

/**
 * @brief Extended BitArray class for testing purposes
 * 
 * This class extends the BitArray with additional testing-specific
 * methods and functionality not present in the production version.
 */
class MockBitArray : public BitArray {
public:
    /**
     * @brief Default constructor creates an empty BitArray
     */
    MockBitArray() : BitArray() {}
    
    /**
     * @brief Construct a new BitArray with a specified size
     * 
     * @param size Number of bits to allocate
     * @param initialValue Default value for all bits
     */
    explicit MockBitArray(size_t size, bool initialValue = false)
        : BitArray(size, initialValue) {}
    
    /**
     * @brief Copy constructor
     */
    MockBitArray(const BitArray& other) : BitArray(other) {}
    
    /**
     * @brief Create a BitArray with a specific pattern
     * 
     * @param size Number of bits to allocate
     * @param patternType Type of pattern (1=alternate, 2=even, 3=odd, 4=blocks)
     * @return MockBitArray with the specified pattern
     */
    static MockBitArray createPattern(size_t size, int patternType) {
        MockBitArray array(size, false);
        
        switch (patternType) {
            case 1: // Alternate (1,0,1,0,...)
                for (size_t i = 0; i < size; i += 2) {
                    array.set(i, true);
                }
                break;
                
            case 2: // Even indices only
                for (size_t i = 0; i < size; i++) {
                    if (i % 2 == 0) {
                        array.set(i, true);
                    }
                }
                break;
                
            case 3: // Odd indices only
                for (size_t i = 0; i < size; i++) {
                    if (i % 2 == 1) {
                        array.set(i, true);
                    }
                }
                break;
                
            case 4: // Block pattern (11110000111100001111...)
                for (size_t i = 0; i < size; i++) {
                    if ((i / 4) % 2 == 0) {
                        array.set(i, true);
                    }
                }
                break;
                
            default: // Default pattern (101010...)
                for (size_t i = 0; i < size; i += 2) {
                    array.set(i, true);
                }
        }
        
        return array;
    }
    
    /**
     * @brief Create a bit array from a string representation
     * 
     * @param bitString String of '0' and '1' characters
     * @return MockBitArray with the specified pattern
     */
    static MockBitArray fromString(const std::string& bitString) {
        MockBitArray array(bitString.length(), false);
        
        for (size_t i = 0; i < bitString.length(); i++) {
            if (bitString[i] == '1') {
                array.set(i, true);
            }
        }
        
        return array;
    }
    
    /**
     * @brief Convert the bit array to a string representation
     * 
     * @return std::string String of '0' and '1' characters
     */
    std::string toString() const {
        std::string result(size(), '0');
        
        for (size_t idx = findFirst(); idx < size(); idx = findNext(idx)) {
            result[idx] = '1';
        }
        
        return result;
    }
    
    /**
     * @brief Check if this bit array exactly matches another
     * 
     * @param other BitArray to compare with
     * @return true if both arrays have the same bits set
     */
    bool exactlyMatches(const BitArray& other) const {
        if (size() != other.size()) {
            return false;
        }
        
        const std::vector<WordType>& myWords = getWords();
        const std::vector<WordType>& otherWords = other.getWords();
        
        return myWords == otherWords;
    }
    
    /**
     * @brief Get counts of set and clear bits
     * 
     * @return std::pair<size_t, size_t> (set count, clear count)
     */
    std::pair<size_t, size_t> getCounts() const {
        size_t setCount = count();
        size_t clearCount = size() - setCount;
        return {setCount, clearCount};
    }
    
    /**
     * @brief Force the cached bit count to be recalculated
     * 
     * This method is used in tests to verify the bit count caching.
     * It simulates the m_bitCountDirty mechanism.
     */
    void forceBitCountRecalculation() {
        // Access the internal words to force the bit count to be marked dirty
        getWords();
        // Then read the count to force recalculation
        count();
    }
};

} // namespace mock

#endif // MOCK_BIT_ARRAY_H
