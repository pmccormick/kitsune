#include <gtest/gtest.h>
#include "BitArray.h"

TEST(BitArrayManipulationTest, GetReturnsFalseForOutOfBoundsIndices) {
    BitArray array(10);
    EXPECT_FALSE(array.get(20));
}

TEST(BitArrayManipulationTest, SetBitToTrue) {
    BitArray array(100, false);
    array.set(50, true);
    EXPECT_TRUE(array.get(50));
    EXPECT_EQ(1, array.count());
}

TEST(BitArrayManipulationTest, SetBitToFalse) {
    BitArray array(100, true);
    array.set(50, false);
    EXPECT_FALSE(array.get(50));
    EXPECT_EQ(99, array.count());
}

TEST(BitArrayManipulationTest, SetIgnoresOutOfBoundsIndices) {
    BitArray array(10);
    array.set(20, true);  // Should be ignored
    for (size_t i = 0; i < 10; i++) {
        EXPECT_FALSE(array.get(i));
    }
    EXPECT_EQ(0, array.count());
}

TEST(BitArrayManipulationTest, SetAllToTrue) {
    BitArray array(100, false);
    array.setAll(true);
    EXPECT_EQ(100, array.count());
    for (size_t i = 0; i < 100; i++) {
        EXPECT_TRUE(array.get(i));
    }
}

TEST(BitArrayManipulationTest, SetAllToFalse) {
    BitArray array(100, true);
    array.setAll(false);
    EXPECT_EQ(0, array.count());
    for (size_t i = 0; i < 100; i++) {
        EXPECT_FALSE(array.get(i));
    }
}

TEST(BitArrayManipulationTest, SetAllWithNonMultipleOfBitsPerWord) {
    BitArray array(67, true);  // 67 is not a multiple of 64
    EXPECT_EQ(67, array.count());
    array.setAll(false);
    EXPECT_EQ(0, array.count());
}

TEST(BitArrayManipulationTest, ClearEmptiesArray) {
    BitArray array(100, true);
    array.clear();
    EXPECT_EQ(0, array.count());
    EXPECT_TRUE(array.isEmpty());
}