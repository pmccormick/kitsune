/**
 * @file BitArray.h
 * @brief High-performance bit array implementation for region storage
 * 
 * This class provides a memory-efficient, cache-friendly bit array implementation
 * with optimized operations for use in the Region class. It offers:
 * 
 * 1. Word-level bit operations for maximum performance
 * 2. Efficient population count with CPU intrinsics
 * 3. Cached size calculation to avoid repeated traversals
 * 4. Fast set operations (union, intersection, difference)
 */

#ifndef BIT_ARRAY_H
#define BIT_ARRAY_H

#include <vector>
#include <cstdint>
#include <cassert>
#include <algorithm>
#include <climits>

/**
 * @class BitArray
 * @brief High-performance bit array using word-level operations
 * 
 * This class uses 64-bit words to store bits efficiently, with optimized
 * implementations for common bit operations. It maintains a cache of the
 * number of set bits to avoid repeated traversals for size() calls.
 */
class BitArray {
public:
    using WordType = uint64_t;
    static constexpr size_t BITS_PER_WORD = sizeof(WordType) * CHAR_BIT;
    
    /**
     * @brief Construct a new empty bit array
     */
    BitArray() : m_size(0), m_bitCount(0), m_bitCountDirty(false) {}
    
    /**
     * @brief Construct a new bit array with specified size
     * 
     * @param size Number of bits to allocate
     * @param initialValue Default value for all bits
     */
    explicit BitArray(size_t size, bool initialValue = false) : m_size(size), m_bitCount(0), m_bitCountDirty(false) {
        m_words.resize(wordCount(size), initialValue ? ~WordType(0) : 0);
        
        // If the size is not a multiple of BITS_PER_WORD, clear the unused bits
        if (size % BITS_PER_WORD != 0 && initialValue) {
            // Clear bits outside the valid range
            size_t usedBitsInLastWord = size % BITS_PER_WORD;
            WordType mask = (WordType(1) << usedBitsInLastWord) - 1;
            m_words.back() &= mask;
        }
        
        // Calculate initial bit count if all bits are set
        if (initialValue) {
            m_bitCount = size;
        }
    }
    
    /**
     * @brief Copy constructor
     */
    BitArray(const BitArray& other) = default;
    
    /**
     * @brief Move constructor
     */
    BitArray(BitArray&& other) noexcept = default;
    
    /**
     * @brief Copy assignment operator
     */
    BitArray& operator=(const BitArray& other) = default;
    
    /**
     * @brief Move assignment operator
     */
    BitArray& operator=(BitArray&& other) noexcept = default;
    
    /**
     * @brief Get the value of a bit
     * 
     * @param index Bit index
     * @return true if the bit is set, false otherwise
     */
    bool get(size_t index) const {
        if (index >= m_size) return false;
        size_t wordIndex = index / BITS_PER_WORD;
        size_t bitIndex = index % BITS_PER_WORD;
        return (m_words[wordIndex] & (WordType(1) << bitIndex)) != 0;
    }
    
    /**
     * @brief Set the value of a bit
     * 
     * @param index Bit index
     * @param value New bit value
     */
    void set(size_t index, bool value) {
        if (index >= m_size) return;
        
        // Calculate word and bit position
        size_t wordIndex = index / BITS_PER_WORD;
        size_t bitIndex = index % BITS_PER_WORD;
        WordType bitMask = WordType(1) << bitIndex;
        
        // Get current value to track changes
        bool oldValue = (m_words[wordIndex] & bitMask) != 0;
        
        // Update the bit
        if (value) {
            m_words[wordIndex] |= bitMask;  // Set bit
        } else {
            m_words[wordIndex] &= ~bitMask; // Clear bit
        }
        
        // Update bit count if not dirty and value changed
        if (!m_bitCountDirty && oldValue != value) {
            m_bitCount += (value ? 1 : -1);
        }
    }
    
    /**
     * @brief Set all bits to the specified value
     * 
     * @param value Value to set for all bits
     */
    void setAll(bool value) {
        WordType fillPattern = value ? ~WordType(0) : 0;
        std::fill(m_words.begin(), m_words.end(), fillPattern);
        
        // If the size is not a multiple of BITS_PER_WORD, clear the unused bits
        if (m_size % BITS_PER_WORD != 0 && value) {
            // Clear bits outside the valid range
            size_t usedBitsInLastWord = m_size % BITS_PER_WORD;
            WordType mask = (WordType(1) << usedBitsInLastWord) - 1;
            m_words.back() &= mask;
        }
        
        // Update bit count
        m_bitCount = value ? m_size : 0;
        m_bitCountDirty = false;
    }
    
    /**
     * @brief Get size in bits
     * 
     * @return Number of bits in the array
     */
    size_t size() const {
        return m_size;
    }
    
    /**
     * @brief Resize the bit array
     * 
     * @param newSize New size in bits
     * @param value Value for new bits (if expanding)
     */
    void resize(size_t newSize, bool value = false) {
        // If not changing size, nothing to do
        if (newSize == m_size) return;
        
        // Save old size for later
        size_t oldSize = m_size;
        
        // Calculate new word count
        size_t newWordCount = wordCount(newSize);
        
        // Resizing to smaller
        if (newSize < m_size) {
            // Clear bits that will be beyond the new size
            if (newSize % BITS_PER_WORD != 0) {
                size_t usedBitsInLastWord = newSize % BITS_PER_WORD;
                WordType mask = (WordType(1) << usedBitsInLastWord) - 1;
                m_words[newWordCount - 1] &= mask;
            }
            
            // Resize the vector
            m_words.resize(newWordCount);
            m_size = newSize;
            
            // Mark bit count as dirty, will be recalculated on next count() call
            m_bitCountDirty = true;
        }
        // Resizing to larger
        else {
            // Get old word count
            size_t oldWordCount = m_words.size();
            
            // Resize the vector, new elements will be zero-initialized
            m_words.resize(newWordCount, value ? ~WordType(0) : 0);
            
            // If we need to set bits and have a partial last word in the new size
            if (value && newSize % BITS_PER_WORD != 0) {
                size_t usedBitsInLastWord = newSize % BITS_PER_WORD;
                WordType mask = (WordType(1) << usedBitsInLastWord) - 1;
                m_words.back() &= mask;
            }
            
            // Update bit count for the extended region if all ones
            if (value) {
                // Add the number of newly added bits that are set
                m_bitCount += (newSize - oldSize);
            }
            
            m_size = newSize;
        }
    }
    
    /**
     * @brief Count the number of set bits (population count)
     * 
     * @return Number of bits set to 1
     */
    size_t count() const {
        if (m_bitCountDirty) {
            // Recalculate bit count
            m_bitCount = 0;
            
            // Full words
            for (size_t i = 0; i < m_words.size(); ++i) {
                m_bitCount += popCount(m_words[i]);
            }
            
            m_bitCountDirty = false;
        }
        
        return m_bitCount;
    }
    
    /**
     * @brief Perform bitwise OR with another bit array (union)
     * 
     * @param other Bit array to OR with
     */
    void bitwiseOr(const BitArray& other) {
        // Ensure arrays are the same size
        assert(m_size == other.m_size && "Bit arrays must be the same size for bitwise OR");
        
        // Perform word-level OR operation
        for (size_t i = 0; i < m_words.size(); ++i) {
            m_words[i] |= other.m_words[i];
        }
        
        // Bit count is now dirty and needs recalculation
        m_bitCountDirty = true;
    }
    
    /**
     * @brief Perform bitwise AND with another bit array (intersection)
     * 
     * @param other Bit array to AND with
     */
    void bitwiseAnd(const BitArray& other) {
        // Ensure arrays are the same size
        assert(m_size == other.m_size && "Bit arrays must be the same size for bitwise AND");
        
        // Perform word-level AND operation
        for (size_t i = 0; i < m_words.size(); ++i) {
            m_words[i] &= other.m_words[i];
        }
        
        // Bit count is now dirty and needs recalculation
        m_bitCountDirty = true;
    }
    
    /**
     * @brief Perform bitwise AND-NOT with another bit array (difference)
     * 
     * This is equivalent to A & ~B (bits in A but not in B)
     * 
     * @param other Bit array to AND-NOT with
     */
    void bitwiseAndNot(const BitArray& other) {
        // Ensure arrays are the same size
        assert(m_size == other.m_size && "Bit arrays must be the same size for bitwise AND-NOT");
        
        // Perform word-level AND-NOT operation
        for (size_t i = 0; i < m_words.size(); ++i) {
            m_words[i] &= ~other.m_words[i];
        }
        
        // Bit count is now dirty and needs recalculation
        m_bitCountDirty = true;
    }
    
    /**
     * @brief Get the underlying words for direct manipulation
     * 
     * @note This will mark the bit count as dirty for later recalculation
     * 
     * @return Reference to the word vector
     */
    std::vector<WordType>& getWords() {
        m_bitCountDirty = true;
        return m_words;
    }
    
    /**
     * @brief Get the underlying words for read-only access
     * 
     * @return Const reference to the word vector
     */
    const std::vector<WordType>& getWords() const {
        return m_words;
    }
    
    /**
     * @brief Clear all bits in the array
     */
    void clear() {
        std::fill(m_words.begin(), m_words.end(), 0);
        m_bitCount = 0;
        m_bitCountDirty = false;
    }
    
    /**
     * @brief Check if all bits are clear (set to 0)
     * 
     * @return true if all bits are 0, false otherwise
     */
    bool isEmpty() const {
        // Fast path if we know the bit count
        if (!m_bitCountDirty && m_bitCount == 0) {
            return true;
        }
        
        // Check each word
        for (WordType word : m_words) {
            if (word != 0) {
                return false;
            }
        }
        
        // All words are zero
        if (m_bitCountDirty) {
            m_bitCount = 0;
            m_bitCountDirty = false;
        }
        
        return true;
    }
    
    /**
     * @brief Check if any bits are set
     * 
     * @return true if any bit is 1, false if all are 0
     */
    bool any() const {
        return !isEmpty();
    }
    
    /**
     * @brief Find the index of the first set bit
     * 
     * @return Index of the first set bit, or size() if none are set
     */
    size_t findFirst() const {
        for (size_t wordIndex = 0; wordIndex < m_words.size(); ++wordIndex) {
            WordType word = m_words[wordIndex];
            if (word != 0) {
                // Find the position of the least significant set bit
                unsigned int bitPos = countTrailingZeros(word);
                size_t index = wordIndex * BITS_PER_WORD + bitPos;
                return (index < m_size) ? index : m_size;
            }
        }
        return m_size; // No bits are set
    }
    
    /**
     * @brief Find the index of the next set bit after the given position
     * 
     * @param pos Position to start searching from (exclusive)
     * @return Index of the next set bit, or size() if none are set
     */
    size_t findNext(size_t pos) const {
        if (pos >= m_size) return m_size;
        
        // Start at the word containing the position
        size_t wordIndex = pos / BITS_PER_WORD;
        size_t bitIndex = pos % BITS_PER_WORD;
        
        // Check remaining bits in the current word
        WordType mask = ~((WordType(1) << (bitIndex + 1)) - 1);
        WordType remainingBits = m_words[wordIndex] & mask;
        
        if (remainingBits != 0) {
            // Found a set bit in the current word
            unsigned int bitPos = countTrailingZeros(remainingBits);
            size_t index = wordIndex * BITS_PER_WORD + bitPos;
            return (index < m_size) ? index : m_size;
        }
        
        // Check subsequent words
        for (wordIndex++; wordIndex < m_words.size(); ++wordIndex) {
            if (m_words[wordIndex] != 0) {
                unsigned int bitPos = countTrailingZeros(m_words[wordIndex]);
                size_t index = wordIndex * BITS_PER_WORD + bitPos;
                return (index < m_size) ? index : m_size;
            }
        }
        
        return m_size; // No more bits are set
    }
    
private:
    std::vector<WordType> m_words;  ///< Array of words storing the bits
    size_t m_size;                  ///< Number of bits in the array
    mutable size_t m_bitCount;      ///< Cached number of set bits
    mutable bool m_bitCountDirty;   ///< Whether the bit count cache is valid
    
    /**
     * @brief Calculate the number of words needed for a given number of bits
     * 
     * @param bitCount Number of bits
     * @return Number of words needed
     */
    static size_t wordCount(size_t bitCount) {
        return (bitCount + BITS_PER_WORD - 1) / BITS_PER_WORD;
    }
    
    /**
     * @brief Count the number of set bits in a word (population count)
     * 
     * Uses CPU intrinsics when available for maximum performance.
     * 
     * @param word Word to count bits in
     * @return Number of bits set to 1
     */
    static size_t popCount(WordType word) {
        #if defined(__GNUC__) || defined(__clang__)
            return __builtin_popcountll(word);
        #else
            // Fallback implementation for non-GCC/Clang compilers
            word = word - ((word >> 1) & 0x5555555555555555ULL);
            word = (word & 0x3333333333333333ULL) + ((word >> 2) & 0x3333333333333333ULL);
            word = (word + (word >> 4)) & 0x0F0F0F0F0F0F0F0FULL;
            return (word * 0x0101010101010101ULL) >> 56;
        #endif
    }
    
    /**
     * @brief Count trailing zeros in a word
     * 
     * Uses CPU intrinsics when available for maximum performance.
     * 
     * @param word Word to count trailing zeros in
     * @return Number of trailing zeros
     */
    static unsigned int countTrailingZeros(WordType word) {
        if (word == 0) return BITS_PER_WORD;
        
        #if defined(__GNUC__) || defined(__clang__)
            return __builtin_ctzll(word);
        #else
            // Fallback implementation for non-GCC/Clang compilers
            unsigned int count = 0;
            while ((word & 1) == 0) {
                word >>= 1;
                ++count;
            }
            return count;
        #endif
    }
};

#endif // BIT_ARRAY_H
       //
