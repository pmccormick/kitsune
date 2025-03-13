#include <gtest/gtest.h>
#include "BitArray.h"

TEST(BitArrayEdgeCaseTest, EmptyArrayOperations) {
    BitArray array(0);
    EXPECT_EQ(0, array.size());
    EXPECT_TRUE(array.isEmpty());
    EXPECT_EQ(0, array.count());
    EXPECT_EQ(0, array.findFirst());
}

TEST(BitArrayEdgeCaseTest, SingleBitArray) {
    BitArray array(1, true);
    EXPECT_EQ(1, array.size());
    EXPECT_FALSE(array.isEmpty());
    EXPECT_EQ(1, array.count());
    EXPECT_EQ(0, array.findFirst());
}

TEST(BitArrayEdgeCaseTest, WordSizeBoundaryArray) {
    const size_t wordSize = BitArray::BITS_PER_WORD;  // BitArray uses uint64_t
    BitArray array(wordSize, false);
    array.set(0, true);
    array.set(wordSize - 1, true);
    EXPECT_EQ(2, array.count());
    EXPECT_EQ(0, array.findFirst());
    EXPECT_EQ(wordSize - 1, array.findNext(0));
}

TEST(BitArrayEdgeCaseTest, CrossWordBoundary) {
    const size_t wordSize = BitArray::BITS_PER_WORD;  // BitArray uses uint64_t
    BitArray array(wordSize * 2, false);
    array.set(wordSize - 1, true);  // Last bit in first word
    array.set(wordSize, true);      // First bit in second word
    
    EXPECT_EQ(wordSize - 1, array.findFirst());
    EXPECT_EQ(wordSize, array.findNext(wordSize - 1));
}

TEST(BitArrayEdgeCaseTest, LargestPossibleSize) {
    // Test with a large but not unreasonable size
    const size_t largeSize = 10'000'000;
    BitArray array(largeSize, false);
    array.set(0, true);
    array.set(largeSize - 1, true);
    
    EXPECT_EQ(2, array.count());
    EXPECT_EQ(0, array.findFirst());
    EXPECT_EQ(largeSize - 1, array.findNext(0));
}

TEST(BitArrayEdgeCaseTest, DirectWordAccess) {
    BitArray array(128, false);
    array.set(65, true);
    
    // Get words for direct manipulation
    std::vector<BitArray::WordType>& words = array.getWords();
    
    // Check that the correct word has the bit set
    EXPECT_NE(0, words[1]);  // Second word should have the bit set
    EXPECT_EQ(0, words[0]);  // First word should be zero
    
    // The bit count should be marked as dirty after direct word access
    // We can test this indirectly by checking the count before and after
    EXPECT_EQ(1, array.count());
}