#include <gtest/gtest.h>
#include "BitArray.h"

TEST(BitArrayBitwiseTest, OrWithEmptyArrays) {
    BitArray a(100, false);
    BitArray b(100, false);
    a.bitwiseOr(b);
    EXPECT_EQ(0, a.count());
}

TEST(BitArrayBitwiseTest, OrWithDisjointBits) {
    BitArray a(100, false);
    a.set(25, true);
    
    BitArray b(100, false);
    b.set(75, true);
    
    a.bitwiseOr(b);
    EXPECT_TRUE(a.get(25));
    EXPECT_TRUE(a.get(75));
    EXPECT_EQ(2, a.count());
}

TEST(BitArrayBitwiseTest, OrWithOverlappingBits) {
    BitArray a(100, false);
    a.set(25, true);
    a.set(50, true);
    
    BitArray b(100, false);
    b.set(50, true);
    b.set(75, true);
    
    a.bitwiseOr(b);
    EXPECT_TRUE(a.get(25));
    EXPECT_TRUE(a.get(50));
    EXPECT_TRUE(a.get(75));
    EXPECT_EQ(3, a.count());
}

TEST(BitArrayBitwiseTest, AndWithEmptyFirstArray) {
    BitArray a(100, false);
    
    BitArray b(100, false);
    b.set(50, true);
    
    a.bitwiseAnd(b);
    EXPECT_EQ(0, a.count());
}

TEST(BitArrayBitwiseTest, AndWithEmptySecondArray) {
    BitArray a(100, false);
    a.set(50, true);
    
    BitArray b(100, false);
    
    a.bitwiseAnd(b);
    EXPECT_EQ(0, a.count());
}

TEST(BitArrayBitwiseTest, AndWithOverlappingBits) {
    BitArray a(100, false);
    a.set(25, true);
    a.set(50, true);
    
    BitArray b(100, false);
    b.set(50, true);
    b.set(75, true);
    
    a.bitwiseAnd(b);
    EXPECT_FALSE(a.get(25));
    EXPECT_TRUE(a.get(50));
    EXPECT_FALSE(a.get(75));
    EXPECT_EQ(1, a.count());
}

TEST(BitArrayBitwiseTest, AndNotOperation) {
    BitArray a(100, false);
    a.set(25, true);
    a.set(50, true);
    a.set(75, true);
    
    BitArray b(100, false);
    b.set(50, true);
    
    a.bitwiseAndNot(b);
    EXPECT_TRUE(a.get(25));
    EXPECT_FALSE(a.get(50));
    EXPECT_TRUE(a.get(75));
    EXPECT_EQ(2, a.count());
}

TEST(BitArrayBitwiseTest, XorOperation) {
    BitArray a(100, false);
    a.set(25, true);
    a.set(50, true);
    
    BitArray b(100, false);
    b.set(50, true);
    b.set(75, true);
    
    a.bitwiseXor(b);
    EXPECT_TRUE(a.get(25));
    EXPECT_FALSE(a.get(50));
    EXPECT_TRUE(a.get(75));
    EXPECT_EQ(2, a.count());
}