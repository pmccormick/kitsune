#include <gtest/gtest.h>
#include "BitArray.h"

TEST(BitArrayConstructionTest, DefaultConstructorCreatesEmptyArray) {
    BitArray array;
    EXPECT_EQ(0, array.size());
    EXPECT_TRUE(array.isEmpty());
}

TEST(BitArrayConstructionTest, ConstructorWithSizeCreatesArrayOfSpecifiedSize) {
    BitArray array(100);
    EXPECT_EQ(100, array.size());
}

TEST(BitArrayConstructionTest, ConstructorWithInitialValueFalse) {
    BitArray array(100, false);
    EXPECT_EQ(100, array.size());
    EXPECT_TRUE(array.isEmpty());
    EXPECT_EQ(0, array.count());
}

TEST(BitArrayConstructionTest, ConstructorWithInitialValueTrue) {
    BitArray array(100, true);
    EXPECT_EQ(100, array.size());
    EXPECT_FALSE(array.isEmpty());
    EXPECT_EQ(100, array.count());
}

TEST(BitArrayConstructionTest, CopyConstructor) {
    BitArray original(100, false);
    original.set(50, true);
    
    BitArray copy(original);
    EXPECT_EQ(100, copy.size());
    EXPECT_TRUE(copy.get(50));
    EXPECT_FALSE(copy.get(49));
}

TEST(BitArrayConstructionTest, MoveConstructor) {
    BitArray original(100, false);
    original.set(50, true);
    
    BitArray moved(std::move(original));
    EXPECT_EQ(100, moved.size());
    EXPECT_TRUE(moved.get(50));
    EXPECT_FALSE(moved.get(49));
}

TEST(BitArrayConstructionTest, CopyAssignment) {
    BitArray original(100, false);
    original.set(50, true);
    
    BitArray copy;
    copy = original;
    EXPECT_EQ(100, copy.size());
    EXPECT_TRUE(copy.get(50));
    EXPECT_FALSE(copy.get(49));
}

TEST(BitArrayConstructionTest, MoveAssignment) {
    BitArray original(100, false);
    original.set(50, true);
    
    BitArray moved;
    moved = std::move(original);
    EXPECT_EQ(100, moved.size());
    EXPECT_TRUE(moved.get(50));
    EXPECT_FALSE(moved.get(49));
}