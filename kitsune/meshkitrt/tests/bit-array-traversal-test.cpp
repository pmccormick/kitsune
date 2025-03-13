#include <gtest/gtest.h>
#include <vector>
#include "BitArray.h"

TEST(BitArrayTraversalTest, FindFirstInEmptyArray) {
    BitArray array(100, false);
    EXPECT_EQ(100, array.findFirst());
}

TEST(BitArrayTraversalTest, FindFirstWithSingleBit) {
    BitArray array(100, false);
    array.set(42, true);
    EXPECT_EQ(42, array.findFirst());
}

TEST(BitArrayTraversalTest, FindFirstWithMultipleBits) {
    BitArray array(100, false);
    array.set(42, true);
    array.set(24, true);
    EXPECT_EQ(24, array.findFirst());  // 24 comes before 42
}

TEST(BitArrayTraversalTest, FindNextAfterPosition) {
    BitArray array(100, false);
    array.set(25, true);
    array.set(50, true);
    array.set(75, true);
    
    EXPECT_EQ(50, array.findNext(25));
    EXPECT_EQ(75, array.findNext(50));
    EXPECT_EQ(100, array.findNext(75));  // No more bits, should return size
}

TEST(BitArrayTraversalTest, FindNextWithGaps) {
    BitArray array(100, false);
    array.set(20, true);
    array.set(60, true);
    
    EXPECT_EQ(60, array.findNext(20));
    EXPECT_EQ(100, array.findNext(60));
}

TEST(BitArrayTraversalTest, FindNextInNonSetBitRange) {
    BitArray array(100, false);
    array.set(50, true);
    
    // Should find the bit at 50 when searching from position 30
    EXPECT_EQ(50, array.findNext(30));
}

TEST(BitArrayTraversalTest, FullIterationPattern) {
    BitArray array(100, false);
    array.set(25, true);
    array.set(50, true);
    array.set(75, true);
    
    std::vector<size_t> foundIndices;
    for (size_t idx = array.findFirst(); idx < array.size(); idx = array.findNext(idx)) {
        foundIndices.push_back(idx);
    }
    
    ASSERT_EQ(3, foundIndices.size());
    EXPECT_EQ(25, foundIndices[0]);
    EXPECT_EQ(50, foundIndices[1]);
    EXPECT_EQ(75, foundIndices[2]);
}