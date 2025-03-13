#include <gtest/gtest.h>
#include "BitArray.h"

TEST(BitArrayResizeTest, GrowBitArray) {
    BitArray array(50, false);
    array.set(25, true);
    
    array.resize(100);
    EXPECT_EQ(100, array.size());
    EXPECT_TRUE(array.get(25));
    EXPECT_EQ(1, array.count());
}

TEST(BitArrayResizeTest, GrowBitArrayWithInitialValue) {
    BitArray array(50, false);
    array.set(25, true);
    
    array.resize(100, true);
    EXPECT_EQ(100, array.size());
    EXPECT_TRUE(array.get(25));
    for (size_t i = 50; i < 100; i++) {
        EXPECT_TRUE(array.get(i));
    }
    EXPECT_EQ(51, array.count());
}

TEST(BitArrayResizeTest, ShrinkBitArray) {
    BitArray array(100, false);
    array.set(25, true);
    array.set(75, true);
    
    array.resize(50);
    EXPECT_EQ(50, array.size());
    EXPECT_TRUE(array.get(25));
    EXPECT_EQ(1, array.count());
}

TEST(BitArrayResizeTest, ShrinkToExactWordBoundary) {
    const size_t wordSize = BitArray::BITS_PER_WORD;  // BitArray uses uint64_t (64 bits per word)
    BitArray array(wordSize * 2, false);
    array.set(wordSize - 1, true);  // Last bit in first word
    array.set(wordSize, true);      // First bit in second word
    
    array.resize(wordSize);
    EXPECT_EQ(wordSize, array.size());
    EXPECT_TRUE(array.get(wordSize - 1));
    EXPECT_EQ(1, array.count());
}

TEST(BitArrayResizeTest, ResizeToSameSizeMaintainsState) {
    BitArray array(100, false);
    array.set(25, true);
    array.set(75, true);
    
    array.resize(100);
    EXPECT_EQ(100, array.size());
    EXPECT_TRUE(array.get(25));
    EXPECT_TRUE(array.get(75));
    EXPECT_EQ(2, array.count());
}

TEST(BitArrayResizeTest, ResizeToZeroCreatesEmptyArray) {
    BitArray array(100, true);
    array.resize(0);
    EXPECT_EQ(0, array.size());
    EXPECT_TRUE(array.isEmpty());
    EXPECT_EQ(0, array.count());
}

TEST(BitArrayResizeTest, ResizeWithNonMultipleOfWordSize) {
    BitArray array(100, false);
    array.set(63, true);  // Last bit in first word
    array.set(64, true);  // First bit in second word
    
    array.resize(65);
    EXPECT_EQ(65, array.size());
    EXPECT_TRUE(array.get(63));
    EXPECT_TRUE(array.get(64));
    EXPECT_EQ(2, array.count());
}