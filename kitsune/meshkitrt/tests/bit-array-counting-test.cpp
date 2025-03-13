#include <gtest/gtest.h>
#include "BitArray.h"

TEST(BitArrayCountingTest, CountOnEmptyArray) {
    BitArray array(100, false);
    EXPECT_EQ(0, array.count());
}

TEST(BitArrayCountingTest, CountOnFullArray) {
    BitArray array(100, true);
    EXPECT_EQ(100, array.count());
}

TEST(BitArrayCountingTest, CountAfterSettingBits) {
    BitArray array(100, false);
    array.set(25, true);
    array.set(50, true);
    array.set(75, true);
    EXPECT_EQ(3, array.count());
}

TEST(BitArrayCountingTest, CountAfterClearingBits) {
    BitArray array(100, true);
    array.set(25, false);
    array.set(50, false);
    EXPECT_EQ(98, array.count());
}

TEST(BitArrayCountingTest, IsEmptyOnEmptyArray) {
    BitArray array(100, false);
    EXPECT_TRUE(array.isEmpty());
}

TEST(BitArrayCountingTest, IsEmptyOnNonEmptyArray) {
    BitArray array(100, false);
    array.set(50, true);
    EXPECT_FALSE(array.isEmpty());
}

TEST(BitArrayCountingTest, AnyOnEmptyArray) {
    BitArray array(100, false);
    EXPECT_FALSE(array.any());
}

TEST(BitArrayCountingTest, AnyOnNonEmptyArray) {
    BitArray array(100, false);
    array.set(50, true);
    EXPECT_TRUE(array.any());
}

TEST(BitArrayCountingTest, CountAfterBitwiseOperations) {
    BitArray a(100, false);
    a.set(25, true);
    a.set(50, true);
    
    BitArray b(100, false);
    b.set(50, true);
    b.set(75, true);
    
    a.bitwiseOr(b);
    EXPECT_EQ(3, a.count());
}