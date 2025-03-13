/**
 * @file BitArray.h
 * @brief High-performance, thread-safe bit array implementation for region storage.
 */

#ifndef BIT_ARRAY_H
#define BIT_ARRAY_H

#include <vector>
#include <cstdint>
#include <cassert>
#include <algorithm>
#include <climits>
#include <atomic>
#include <mutex>
#include <type_traits>

#ifdef __AVX2__
#include <immintrin.h>
#endif

class BitArray {
public:
  /**
   * @brief The underlying word type used for bit storage
   */
  using WordType = uint64_t;
  
  /**
   * @brief Number of bits stored in each word
   */
  static constexpr size_t BITS_PER_WORD = sizeof(WordType) * CHAR_BIT;

  /**
   * @brief Default constructor creates an empty BitArray.
   */
  BitArray() : m_size(0), m_bitCount(0), m_bitCountDirty(false) {}

  /**
   * @brief Construct a new BitArray with a specified size.
   *
   * @param size Number of bits to allocate.
   * @param initialValue Default value for all bits (false = 0, true = 1).
   */
  explicit BitArray(size_t size, bool initialValue = false)
    : m_size(size), m_bitCount(0), m_bitCountDirty(false)
  {
    // Calculate how many words we need to store 'size' bits
    m_words.resize(wordCount(size), initialValue ? ~WordType(0) : 0);

    // If the size is not a multiple of BITS_PER_WORD and all bits were set,
    // we need to clear the extra bits in the last word.
    if (size % BITS_PER_WORD != 0 && initialValue) {
      size_t usedBitsInLastWord = size % BITS_PER_WORD;
      WordType mask = (WordType(1) << usedBitsInLastWord) - 1;
      m_words.back() &= mask;
    }

    // Initialize bit count: if all bits set, store 'size'
    if (initialValue) {
      m_bitCount = size;
    }
  }

  /**
   * @brief Copy constructor.
   */
  BitArray(const BitArray& other) 
    : m_words(other.m_words),
      m_size(other.m_size),
      m_bitCount(other.m_bitCount),
      m_bitCountDirty(other.m_bitCountDirty)
  {}

  /**
   * @brief Move constructor.
   */
  BitArray(BitArray&& other) noexcept
    : m_words(std::move(other.m_words)),
      m_size(other.m_size),
      m_bitCount(other.m_bitCount),
      m_bitCountDirty(other.m_bitCountDirty)
  {
    // Reset the moved-from object
    other.m_size = 0;
    other.m_bitCount = 0;
    other.m_bitCountDirty = false;
  }

  /**
   * @brief Copy assignment operator.
   */
  BitArray& operator=(const BitArray& other) {
    if (this != &other) {
      std::lock_guard<std::mutex> lock(m_writeMutex);
      std::lock_guard<std::mutex> otherLock(other.m_writeMutex);
      
      m_words = other.m_words;
      m_size = other.m_size;
      m_bitCount = other.m_bitCount;
      m_bitCountDirty = other.m_bitCountDirty;
    }
    return *this;
  }

  /**
   * @brief Move assignment operator.
   */
  BitArray& operator=(BitArray&& other) noexcept {
    if (this != &other) {
      std::lock_guard<std::mutex> lock(m_writeMutex);
      std::lock_guard<std::mutex> otherLock(other.m_writeMutex);
      
      m_words = std::move(other.m_words);
      m_size = other.m_size;
      m_bitCount = other.m_bitCount;
      m_bitCountDirty = other.m_bitCountDirty;
      
      // Reset the moved-from object
      other.m_size = 0;
      other.m_bitCount = 0;
      other.m_bitCountDirty = false;
    }
    return *this;
  }

  /**
   * @brief Get the value of a bit at the given index.
   *
   * @param index Bit index.
   * @return true if the bit is set, false otherwise.
   */
  bool get(size_t index) const {
    // Bounds check: return false for out-of-range indices
    if (index >= m_size) return false;
    
    // Calculate which word contains this bit
    size_t wordIndex = index / BITS_PER_WORD;
    
    // Calculate which bit within the word we need
    size_t bitIndex = index % BITS_PER_WORD;
    
    // Check if the bit is set
    return (m_words[wordIndex] & (WordType(1) << bitIndex)) != 0;
  }

  /**
   * @brief Set the value of a bit at the given index.
   *
   * @param index Bit index.
   * @param value New bit value.
   * @return true if the operation was successful, false if index was out of bounds.
   */
  bool set(size_t index, bool value) {
    // Bounds check: return false for out-of-range indices
    if (index >= m_size) return false;

    // Calculate which word contains this bit
    size_t wordIndex = index / BITS_PER_WORD;
    
    // Calculate which bit within the word we need
    size_t bitIndex = index % BITS_PER_WORD;
    
    // Create a mask with only the target bit set
    WordType bitMask = WordType(1) << bitIndex;

    // Lock the mutex to prevent concurrent modifications to the same word
    std::lock_guard<std::mutex> lock(m_writeMutex);
    
    // Determine the current bit value
    bool oldValue = (m_words[wordIndex] & bitMask) != 0;

    // Only perform the update if the value is changing
    if (oldValue != value) {
      // Update the bit: set or clear based on 'value'
      if (value) {
        m_words[wordIndex] |= bitMask;   // Set the bit using OR
      } else {
        m_words[wordIndex] &= ~bitMask;  // Clear the bit using AND with inverted mask
      }

      // Update the bit count if the cache is not marked dirty
      if (!m_bitCountDirty) {
        // If we're setting a bit, increment; if clearing, decrement
        m_bitCount += value ? 1 : -1;
      }
    }
    
    return true;
  }

  /**
   * @brief Set all bits to the specified value.
   *
   * @param value Value to set for all bits.
   */
  void setAll(bool value) {
    std::lock_guard<std::mutex> lock(m_writeMutex);
    
    // Different optimal approaches based on the value
    if (value) {
      // Setting all bits to 1: use a pattern of all bits set
      WordType fillPattern = ~WordType(0);  // All bits set to 1
      std::fill(m_words.begin(), m_words.end(), fillPattern);
      
      // If m_size is not a multiple of BITS_PER_WORD, clear the extra bits
      // in the last word to maintain consistency
      if (m_size % BITS_PER_WORD != 0) {
        size_t usedBitsInLastWord = m_size % BITS_PER_WORD;
        WordType mask = (WordType(1) << usedBitsInLastWord) - 1;
        m_words.back() &= mask;
      }
    } else {
      // Setting all bits to 0: more efficient to directly assign zeros
      m_words.assign(m_words.size(), 0);
    }

    // Update cached bit count
    m_bitCount = value ? m_size : 0;
    m_bitCountDirty = false;
  }

  /**
   * @brief Get the number of bits in the array.
   *
   * @return size_t Total number of bits in the array.
   */
  size_t size() const {
    return m_size;
  }

  /**
   * @brief Resize the bit array.
   *
   * @param newSize New size in bits.
   * @param value Value for new bits if the array is expanded.
   */
  void resize(size_t newSize, bool value = false) {
    std::lock_guard<std::mutex> lock(m_writeMutex);
    
    if (newSize == m_size) return;  // Nothing to do if size doesn't change

    size_t oldSize = m_size;
    
    if (newSize < oldSize) {
      // SHRINK CASE
        
      // Calculate new word count
      size_t newWordCount = wordCount(newSize);
        
      // If the last word will be partial, mask off the extra bits
      if (newSize % BITS_PER_WORD != 0) {
	size_t usedBitsInLastWord = newSize % BITS_PER_WORD;
	WordType mask = (WordType(1) << usedBitsInLastWord) - 1;
	m_words[newWordCount - 1] &= mask;
      }
        
      // Resize the vector to remove excess words
      m_words.resize(newWordCount);
        
      // Mark bit count as dirty since we've modified bits
      m_bitCountDirty = true;
    } else {
      // EXPAND CASE
        
      if (value) {
	// When expanding with value=true, new bits will be set to 1
            
	// Calculate word indices for more clarity
	size_t oldWordCount = wordCount(oldSize);
	size_t newWordCount = wordCount(newSize);
            
	// Remember the old bit count if not dirty
	bool wasDirty = m_bitCountDirty;
	size_t oldBitCount = wasDirty ? 0 : m_bitCount;
            
	// Calculate new bits to be added (for bit count update)
	size_t newBitsAdded = newSize - oldSize;
            
	// If last old word exists and is partial, we need special handling
	bool hasPartialOldWord = (oldSize % BITS_PER_WORD != 0);
	size_t lastOldWordIndex = hasPartialOldWord ? (oldSize / BITS_PER_WORD) : 0;
	WordType lastOldWord = hasPartialOldWord ? m_words[lastOldWordIndex] : 0;
            
	// Resize the vector, new words initialized to all 1s
	m_words.resize(newWordCount, ~WordType(0));
            
	// If the last old word was partial, we need to reconstruct it
	// to ensure we maintain the original bits and set only new bits to 1
	if (hasPartialOldWord) {
	  size_t usedBitsInLastOldWord = oldSize % BITS_PER_WORD;
	  WordType maskForOldBits = (WordType(1) << usedBitsInLastOldWord) - 1;
                
	  // Keep old bits as they were, set new bits in the word to 1
	  m_words[lastOldWordIndex] = (lastOldWord & maskForOldBits) | (~maskForOldBits);
	}
            
	// If the last new word is partial, clear the unused bits
	if (newSize % BITS_PER_WORD != 0) {
	  size_t lastNewWordIndex = (newSize - 1) / BITS_PER_WORD;
	  size_t usedBitsInLastNewWord = newSize % BITS_PER_WORD;
	  WordType maskForUsedBits = (WordType(1) << usedBitsInLastNewWord) - 1;
                
	  m_words[lastNewWordIndex] &= maskForUsedBits;
	}
            
	// Update bit count if it wasn't dirty before
	if (!wasDirty) {
	  m_bitCount = oldBitCount + newBitsAdded;
	} else {
	  m_bitCountDirty = true;
	}
      } else {
	// When expanding with value=false, simply extend with 0s
	size_t newWordCount = wordCount(newSize);
	m_words.resize(newWordCount, 0);
      }
    }
    
    m_size = newSize;
  }


  
#if 0
  void resize(size_t newSize, bool value = false) {
    std::lock_guard<std::mutex> lock(m_writeMutex);
    
    if (newSize == m_size) return;  // Nothing to do if size doesn't change

    size_t oldSize = m_size;
    size_t oldWordCount = m_words.size();
    size_t newWordCount = wordCount(newSize);

    if (newSize < oldSize) {
      // SHRINK CASE
        
      // If shrinking, clear any bits beyond the new boundary in the last word
      if (newSize % BITS_PER_WORD != 0) {
	size_t usedBitsInLastWord = newSize % BITS_PER_WORD;
	WordType mask = (WordType(1) << usedBitsInLastWord) - 1;
	m_words[newWordCount - 1] &= mask;
      }
        
      // Resize the vector to remove excess words
      m_words.resize(newWordCount);
        
      // Mark bit count as dirty since we've modified bits
      m_bitCountDirty = true;
    } else {
      // EXPAND CASE
        
      // Save the old word count for boundary handling
      size_t oldPartialWordBits = oldSize % BITS_PER_WORD;
      size_t lastOldWordIndex = oldSize / BITS_PER_WORD;
        
      // If setting new bits to 1, track the old bit count
      size_t addedOneBits = 0;
      if (value && !m_bitCountDirty) {
	addedOneBits = newSize - oldSize;
      }
        
      // Resize the vector with new words initialized to the appropriate value
      WordType fillValue = value ? ~WordType(0) : 0;
      m_words.resize(newWordCount, fillValue);
        
      // Handle the boundary word if it exists and we're adding 1s
      if (value && oldPartialWordBits > 0 && lastOldWordIndex < m_words.size()) {
	// Create a mask for the old bits in the boundary word
	WordType oldBitsMask = (WordType(1) << oldPartialWordBits) - 1;
            
	// Preserve the old bits, set new bits to the desired value
	WordType oldBits = m_words[lastOldWordIndex] & oldBitsMask;
	m_words[lastOldWordIndex] = oldBits | (fillValue & ~oldBitsMask);
      }
        
      // If the last new word is partial and we're setting to 1, clear unused bits
      if (value && newSize % BITS_PER_WORD != 0) {
	size_t usedBitsInLastWord = newSize % BITS_PER_WORD;
	WordType mask = (WordType(1) << usedBitsInLastWord) - 1;
	m_words.back() &= mask;
            
	// Adjust the added bit count if we're on the last word
	if (!m_bitCountDirty && lastOldWordIndex == m_words.size() - 1) {
	  // Count bits cleared in the last word
	  WordType clearedBits = ~mask & fillValue;
	  addedOneBits -= popCount(clearedBits);
	}
      }
        
      // Update bit count if not dirty and adding 1s
      if (value && !m_bitCountDirty) {
	m_bitCount += addedOneBits;
      } else if (value) {
	m_bitCountDirty = true;
      }
    }
    
    m_size = newSize;
  }

#endif 
  
  /**
   * @brief Count the number of bits set to 1 (population count).
   *
   * @return size_t Number of bits set to 1.
   */
  size_t count() const {
    // Check if our cached count is dirty
    if (m_bitCountDirty) {
      size_t count = 0;
      
      // Iterate through all words and count the set bits
      for (size_t i = 0; i < m_words.size(); ++i) {
        count += popCount(m_words[i]);
      }
      
      // Update the cache
      m_bitCount = count;
      m_bitCountDirty = false;
    }
    
    // Return the cached count
    return m_bitCount;
  }

  /**
   * @brief Perform bitwise OR with another BitArray (union operation).
   *
   * @param other BitArray to OR with.
   * @throw std::invalid_argument If bit arrays have different sizes.
   */
  void bitwiseOr(const BitArray& other) {
    // Size check: both arrays must be the same size
    if (m_size != other.m_size) {
      throw std::invalid_argument("Bit arrays must be the same size for bitwise OR");
    }
    
    std::lock_guard<std::mutex> lock(m_writeMutex);
    
#ifdef __AVX2__
    // AVX2 SIMD optimization path - process 256 bits (4 x 64-bit words) at a time
    const size_t avxWordCount = m_words.size() / 4 * 4;  // Round down to multiple of 4
    
    for (size_t i = 0; i < avxWordCount; i += 4) {
      // Load 4 words (256 bits) from each array
      __m256i a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&m_words[i]));
      __m256i b = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&other.m_words[i]));
      
      // Perform bitwise OR on all 256 bits in parallel
      __m256i result = _mm256_or_si256(a, b);
      
      // Store the result back
      _mm256_storeu_si256(reinterpret_cast<__m256i*>(&m_words[i]), result);
    }
    
    // Process remaining words (0-3) using scalar operations
    for (size_t i = avxWordCount; i < m_words.size(); ++i) {
      m_words[i] |= other.m_words[i];
    }
#else
    // Standard scalar path - process one word at a time
    for (size_t i = 0; i < m_words.size(); ++i) {
      m_words[i] |= other.m_words[i];
    }
#endif

    // Mark the bit count as dirty since we modified bits
    m_bitCountDirty = true;
  }

  /**
   * @brief Perform bitwise AND with another BitArray (intersection operation).
   *
   * @param other BitArray to AND with.
   * @throw std::invalid_argument If bit arrays have different sizes.
   */
  void bitwiseAnd(const BitArray& other) {
    // Size check: both arrays must be the same size
    if (m_size != other.m_size) {
      throw std::invalid_argument("Bit arrays must be the same size for bitwise AND");
    }

    std::lock_guard<std::mutex> lock(m_writeMutex);
    
#ifdef __AVX2__
    // AVX2 SIMD optimization path - process 256 bits at a time
    const size_t avxWordCount = m_words.size() / 4 * 4;
    
    for (size_t i = 0; i < avxWordCount; i += 4) {
      __m256i a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&m_words[i]));
      __m256i b = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&other.m_words[i]));
      __m256i result = _mm256_and_si256(a, b);
      _mm256_storeu_si256(reinterpret_cast<__m256i*>(&m_words[i]), result);
    }
    
    // Process remaining words
    for (size_t i = avxWordCount; i < m_words.size(); ++i) {
      m_words[i] &= other.m_words[i];
    }
#else
    // Standard scalar path
    for (size_t i = 0; i < m_words.size(); ++i) {
      m_words[i] &= other.m_words[i];
    }
#endif

    // Mark the bit count as dirty
    m_bitCountDirty = true;
  }

  /**
   * @brief Perform bitwise AND-NOT with another BitArray (difference operation).
   *
   * @param other BitArray to AND-NOT with.
   * @throw std::invalid_argument If bit arrays have different sizes.
   */
  void bitwiseAndNot(const BitArray& other) {
    // Size check: both arrays must be the same size
    if (m_size != other.m_size) {
      throw std::invalid_argument("Bit arrays must be the same size for bitwise AND-NOT");
    }

    std::lock_guard<std::mutex> lock(m_writeMutex);
    
#ifdef __AVX2__
    // AVX2 SIMD optimization path
    const size_t avxWordCount = m_words.size() / 4 * 4;
    
    for (size_t i = 0; i < avxWordCount; i += 4) {
      __m256i a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&m_words[i]));
      __m256i b = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&other.m_words[i]));
      
      // Compute ~b and then perform AND
      __m256i not_b = _mm256_xor_si256(b, _mm256_set1_epi32(-1));  // Complement all bits
      __m256i result = _mm256_and_si256(a, not_b);
      
      _mm256_storeu_si256(reinterpret_cast<__m256i*>(&m_words[i]), result);
    }
    
    // Process remaining words
    for (size_t i = avxWordCount; i < m_words.size(); ++i) {
      m_words[i] &= ~other.m_words[i];
    }
#else
    // Standard scalar path
    for (size_t i = 0; i < m_words.size(); ++i) {
      m_words[i] &= ~other.m_words[i];
    }
#endif

    // Mark the bit count as dirty
    m_bitCountDirty = true;
  }

  /**
   * @brief Perform bitwise XOR with another BitArray (symmetric difference).
   *
   * @param other BitArray to XOR with.
   * @throw std::invalid_argument If bit arrays have different sizes.
   */
  void bitwiseXor(const BitArray& other) {
    // Size check: both arrays must be the same size
    if (m_size != other.m_size) {
      throw std::invalid_argument("Bit arrays must be the same size for bitwise XOR");
    }

    std::lock_guard<std::mutex> lock(m_writeMutex);
    
#ifdef __AVX2__
    // AVX2 SIMD optimization path
    const size_t avxWordCount = m_words.size() / 4 * 4;
    
    for (size_t i = 0; i < avxWordCount; i += 4) {
      __m256i a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&m_words[i]));
      __m256i b = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&other.m_words[i]));
      __m256i result = _mm256_xor_si256(a, b);
      _mm256_storeu_si256(reinterpret_cast<__m256i*>(&m_words[i]), result);
    }
    
    // Process remaining words
    for (size_t i = avxWordCount; i < m_words.size(); ++i) {
      m_words[i] ^= other.m_words[i];
    }
#else
    // Standard scalar path
    for (size_t i = 0; i < m_words.size(); ++i) {
      m_words[i] ^= other.m_words[i];
    }
#endif

    // Mark the bit count as dirty
    m_bitCountDirty = true;
  }

  /**
   * @brief Get the underlying words for direct manipulation (non-const).
   *
   * WARNING: This function allows direct modification of the internal storage,
   * which bypasses thread-safety mechanisms and bit count tracking. Use with caution.
   *
   * @return Reference to the word vector.
   */
  std::vector<WordType>& getWords() {
    std::lock_guard<std::mutex> lock(m_writeMutex);
    
    // Mark bit count as dirty since the caller may modify the words directly
    m_bitCountDirty = true;
    return m_words;
  }

  /**
   * @brief Get the underlying words for read-only access.
   *
   * @return Const reference to the word vector.
   */
  const std::vector<WordType>& getWords() const {
    return m_words;
  }

  /**
   * @brief Clear all bits in the array.
   */
  void clear() {
    std::lock_guard<std::mutex> lock(m_writeMutex);
    
    // Zero out all words - using assign is more efficient than std::fill for zeroing
    m_words.assign(m_words.size(), 0);
    
    // Reset bit count
    m_bitCount = 0;
    m_bitCountDirty = false;
  }

  /**
   * @brief Check if all bits are clear (i.e., the array is empty).
   *
   * @return true if all bits are 0, false otherwise.
   */
  bool isEmpty() const {
    // Fast path using cached count if it's valid
    if (!m_bitCountDirty && m_bitCount == 0) {
      return true;
    }
    
    // Fallback: iterate through words
    for (WordType word : m_words) {
      if (word != 0) {
        return false;
      }
    }
    
    // If we get here, all bits are 0 - update cache
    m_bitCount = 0;
    m_bitCountDirty = false;
    return true;
  }

  /**
   * @brief Check if any bit is set in the array.
   *
   * @return true if at least one bit is 1; false if all are 0.
   */
  bool any() const {
    return !isEmpty();
  }

  /**
   * @brief Find the index of the first set bit.
   *
   * @return Index of the first set bit, or m_size if none are set.
   */
  size_t findFirst() const {
    // Iterate through words
    for (size_t wordIndex = 0; wordIndex < m_words.size(); ++wordIndex) {
      WordType word = m_words[wordIndex];
      if (word != 0) {
        // Found a word with at least one bit set
        unsigned int bitPos = countTrailingZeros(word);
        
        // Calculate the actual bit index
        size_t index = wordIndex * BITS_PER_WORD + bitPos;
        
        // Make sure we don't return an index beyond the array size
        return (index < m_size) ? index : m_size;
      }
    }
    
    // No bits are set
    return m_size;
  }

  /**
   * @brief Find the index of the next set bit after a given position.
   *
   * @param pos Position to start searching from (exclusive).
   * @return Index of the next set bit, or m_size if none are set.
   */
  size_t findNext(size_t pos) const {
    // Bounds check
    if (pos >= m_size) return m_size;

    // Find which word contains the starting position
    size_t wordIndex = pos / BITS_PER_WORD;
    size_t bitIndex = pos % BITS_PER_WORD;

    // Create a mask to ignore bits up to and including pos in the current word
    // For example, if bitIndex=3, mask will have bits 4-63 set (all bits after pos)
    WordType mask = (bitIndex == BITS_PER_WORD - 1) ? 0 : ~((WordType(1) << (bitIndex + 1)) - 1);
    
    // Apply mask to the current word to keep only bits after pos
    WordType remainingBits = m_words[wordIndex] & mask;

    if (remainingBits != 0) {
      // Found bits in the current word
      unsigned int bitPos = countTrailingZeros(remainingBits);
      size_t index = wordIndex * BITS_PER_WORD + bitPos;
      return (index < m_size) ? index : m_size;
    }

    // Check subsequent words
    for (++wordIndex; wordIndex < m_words.size(); ++wordIndex) {
      if (m_words[wordIndex] != 0) {
        unsigned int bitPos = countTrailingZeros(m_words[wordIndex]);
        size_t index = wordIndex * BITS_PER_WORD + bitPos;
        return (index < m_size) ? index : m_size;
      }
    }

    // No more bits set
    return m_size;
  }

private:
  std::vector<WordType> m_words;  ///< Array of words storing the bits
  size_t m_size;                  ///< Total number of bits in the array
  
  // Thread-safety variables
  mutable std::mutex m_writeMutex;  ///< Mutex for synchronized write operations
  mutable size_t m_bitCount;        ///< Cached count of bits set to 1
  mutable bool m_bitCountDirty;     ///< Indicates whether m_bitCount is valid

  /**
   * @brief Calculate the number of words required to store a given number of bits.
   *
   * @param bitCount Number of bits.
   * @return Number of words required.
   */
  static size_t wordCount(size_t bitCount) {
    // Divide by BITS_PER_WORD and round up
    return (bitCount + BITS_PER_WORD - 1) / BITS_PER_WORD;
  }

  /**
   * @brief Count the number of set bits in a 64-bit word (population count).
   *
   * @param word Word to count bits in.
   * @return Number of bits set to 1.
   */
  static size_t popCount(WordType word) {
#if defined(__GNUC__) || defined(__clang__)
    // Use GCC/Clang intrinsic for population count
    return __builtin_popcountll(word);
#else
    // Fallback implementation for other compilers
    word = word - ((word >> 1) & 0x5555555555555555ULL);
    word = (word & 0x3333333333333333ULL) + ((word >> 2) & 0x3333333333333333ULL);
    word = (word + (word >> 4)) & 0x0F0F0F0F0F0F0F0FULL;
    return (word * 0x0101010101010101ULL) >> 56;
#endif
  }

  /**
   * @brief Count the number of trailing zeros in a 64-bit word.
   *
   * @param word Word in which to count trailing zeros.
   * @return Number of trailing zeros.
   */
  static unsigned int countTrailingZeros(WordType word) {
    // If the word is zero, all bits are zeros
    if (word == 0) return BITS_PER_WORD;
    
#if defined(__GNUC__) || defined(__clang__)
    // Use GCC/Clang intrinsic for trailing zero count
    return __builtin_ctzll(word);
#else
    // Fallback implementation: count bits until we find a 1
    unsigned int count = 0;
    while ((word & 1) == 0) {
      word >>= 1;
      ++count;
    }
    return count;
#endif
  }
};

/**
 * @brief Helper class for efficiently iterating through set bits in a BitArray
 */
class BitArrayIterator {
public:
  /**
   * @brief Construct a new iterator for a BitArray
   * 
   * @param bitArray The BitArray to iterate through
   */
  explicit BitArrayIterator(const BitArray& bitArray)
    : m_bitArray(bitArray),
      m_currentIndex(bitArray.findFirst())
  {}
  
  /**
   * @brief Check if there are more set bits to visit
   * 
   * @return true if there are more set bits, false if iteration is complete
   */
  bool hasNext() const {
    return m_currentIndex < m_bitArray.size();
  }
  
  /**
   * @brief Move to the next set bit
   */
  void next() {
    if (hasNext()) {
      m_currentIndex = m_bitArray.findNext(m_currentIndex);
    }
  }
  
  /**
   * @brief Get the index of the current set bit
   * 
   * @return The index of the current bit
   */
  size_t index() const {
    return m_currentIndex;
  }
  
private:
  const BitArray& m_bitArray;  ///< Reference to the BitArray being iterated
  size_t m_currentIndex;       ///< Current bit index
};

#endif // BIT_ARRAY_H

