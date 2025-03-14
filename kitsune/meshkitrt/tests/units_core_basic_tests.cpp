#include <gtest/gtest.h>
#include "Units/core.h"

using namespace units;

TEST(UnitsCoreBasic, Construction) {
    // Default construction (zero value)
    Unit<length_dimension> length;
    EXPECT_DOUBLE_EQ(0.0, length.value());

    // Value construction
    Unit<mass_dimension> mass(42.5);
    EXPECT_DOUBLE_EQ(42.5, mass.value());

    // Different dimension types
    Unit<time_dimension> time(60.0);
    EXPECT_DOUBLE_EQ(60.0, time.value());
    
    // Dimensionless unit
    dimensionless ratio(0.75);
    EXPECT_DOUBLE_EQ(0.75, ratio.value());
}

TEST(UnitsCoreBasic, AdditionOperations) {
    Unit<length_dimension> a(10.0);
    Unit<length_dimension> b(5.0);
    
    // Basic addition
    auto result = a + b;
    EXPECT_DOUBLE_EQ(15.0, result.value());
    
    // Add zero
    auto resultZero = a + Unit<length_dimension>();
    EXPECT_DOUBLE_EQ(10.0, resultZero.value());
    
    // Commutativity
    EXPECT_DOUBLE_EQ((a + b).value(), (b + a).value());
    
    // Associativity
    Unit<length_dimension> c(7.0);
    EXPECT_DOUBLE_EQ(((a + b) + c).value(), (a + (b + c)).value());
}

TEST(UnitsCoreBasic, SubtractionOperations) {
    Unit<time_dimension> a(30.0);
    Unit<time_dimension> b(12.0);
    
    // Basic subtraction
    auto result = a - b;
    EXPECT_DOUBLE_EQ(18.0, result.value());
    
    // Subtract zero
    auto resultZero = a - Unit<time_dimension>();
    EXPECT_DOUBLE_EQ(30.0, resultZero.value());
    
    // Subtracting from itself
    auto resultSelf = a - a;
    EXPECT_DOUBLE_EQ(0.0, resultSelf.value());
    
    // Non-commutativity
    EXPECT_NE((a - b).value(), (b - a).value());
}

TEST(UnitsCoreBasic, MultiplicationByScalar) {
    Unit<length_dimension> length(10.0);
    
    // Multiply by a positive scalar
    auto result1 = length * 2.5;
    EXPECT_DOUBLE_EQ(25.0, result1.value());
    
    // Multiply by a negative scalar
    auto result2 = length * (-0.5);
    EXPECT_DOUBLE_EQ(-5.0, result2.value());
    
    // Multiply by zero
    auto result3 = length * 0.0;
    EXPECT_DOUBLE_EQ(0.0, result3.value());
    
    // Scalar multiplication from the left
    auto result4 = 3.0 * length;
    EXPECT_DOUBLE_EQ(30.0, result4.value());
    
    // Commutativity
    EXPECT_DOUBLE_EQ((length * 2.0).value(), (2.0 * length).value());
}

TEST(UnitsCoreBasic, DivisionByScalar) {
    Unit<mass_dimension> mass(100.0);
    
    // Divide by a positive scalar
    auto result1 = mass / 4.0;
    EXPECT_DOUBLE_EQ(25.0, result1.value());
    
    // Divide by a negative scalar
    auto result2 = mass / (-5.0);
    EXPECT_DOUBLE_EQ(-20.0, result2.value());
    
    // Division by very small scalar (watch for precision)
    auto result3 = mass / 1e-10;
    EXPECT_NEAR(1e12, result3.value(), 1e2);
    
    // Division by large scalar
    auto result4 = mass / 1e10;
    EXPECT_NEAR(1e-8, result4.value(), 1e-18);
}

TEST(UnitsCoreBasic, NegationOperator) {
    Unit<length_dimension> a(15.0);
    Unit<length_dimension> b(-7.5);
    
    // Negation of positive value
    auto result1 = -a;
    EXPECT_DOUBLE_EQ(-15.0, result1.value());
    
    // Negation of negative value
    auto result2 = -b;
    EXPECT_DOUBLE_EQ(7.5, result2.value());
    
    // Double negation
    auto result3 = -(-a);
    EXPECT_DOUBLE_EQ(15.0, result3.value());
    
    // Negation of zero
    auto result4 = -Unit<length_dimension>();
    EXPECT_DOUBLE_EQ(0.0, result4.value());
}

TEST(UnitsCoreBasic, ComparisonEqual) {
    Unit<time_dimension> a(60.0);
    Unit<time_dimension> b(60.0);
    Unit<time_dimension> c(120.0);
    
    // Equality with same value
    EXPECT_TRUE(a == b);
    EXPECT_FALSE(a == c);
    
    // Inequality with same value
    EXPECT_FALSE(a != b);
    EXPECT_TRUE(a != c);
    
    // Self-equality
    EXPECT_TRUE(a == a);
    EXPECT_FALSE(a != a);
    
    // Zero equality
    Unit<time_dimension> zero;
    Unit<time_dimension> zero2;
    EXPECT_TRUE(zero == zero2);
}

TEST(UnitsCoreBasic, ComparisonOrdering) {
    Unit<mass_dimension> small(10.0);
    Unit<mass_dimension> medium(50.0);
    Unit<mass_dimension> large(100.0);
    
    // Less than
    EXPECT_TRUE(small < medium);
    EXPECT_TRUE(medium < large);
    EXPECT_FALSE(medium < small);
    EXPECT_FALSE(large < medium);
    EXPECT_FALSE(small < small);
    
    // Greater than
    EXPECT_TRUE(medium > small);
    EXPECT_TRUE(large > medium);
    EXPECT_FALSE(small > medium);
    EXPECT_FALSE(medium > large);
    EXPECT_FALSE(small > small);
    
    // Less than or equal
    EXPECT_TRUE(small <= medium);
    EXPECT_TRUE(small <= small);
    EXPECT_FALSE(medium <= small);
    
    // Greater than or equal
    EXPECT_TRUE(medium >= small);
    EXPECT_TRUE(medium >= medium);
    EXPECT_FALSE(small >= medium);
}

TEST(UnitsCoreBasic, AssignmentOperations) {
    Unit<length_dimension> a(10.0);
    Unit<length_dimension> b(5.0);
    
    // Addition assignment
    auto c = a;
    c += b;
    EXPECT_DOUBLE_EQ(15.0, c.value());
    
    // Subtraction assignment
    c = a;
    c -= b;
    EXPECT_DOUBLE_EQ(5.0, c.value());
    
    // Multiplication assignment
    c = a;
    c *= 2.5;
    EXPECT_DOUBLE_EQ(25.0, c.value());
    
    // Division assignment
    c = a;
    c /= 2.0;
    EXPECT_DOUBLE_EQ(5.0, c.value());
}

TEST(UnitsCoreBasic, CopyAndAssignmentSemantics) {
    Unit<mass_dimension> original(42.0);
    
    // Copy construction
    Unit<mass_dimension> copy(original);
    EXPECT_DOUBLE_EQ(original.value(), copy.value());
    
    // Modification of copy should not affect original
    copy *= 2.0;
    EXPECT_DOUBLE_EQ(42.0, original.value());
    EXPECT_DOUBLE_EQ(84.0, copy.value());
    
    // Copy assignment
    Unit<mass_dimension> assigned;
    assigned = original;
    EXPECT_DOUBLE_EQ(original.value(), assigned.value());
    
    // Modification of assigned should not affect original
    assigned += Unit<mass_dimension>(8.0);
    EXPECT_DOUBLE_EQ(42.0, original.value());
    EXPECT_DOUBLE_EQ(50.0, assigned.value());
}

TEST(UnitsCoreBasic, DimensionlessUnit) {
    // Create dimensionless units
    dimensionless a(2.5);
    dimensionless b(0.5);
    
    // Operations on dimensionless units
    EXPECT_DOUBLE_EQ(3.0, (a + b).value());
    EXPECT_DOUBLE_EQ(2.0, (a - b).value());
    EXPECT_DOUBLE_EQ(1.25, (a * b).value());
    EXPECT_DOUBLE_EQ(5.0, (a / b).value());
    
    // Convert double to dimensionless
    auto dimless = make_dimensionless(3.14);
    EXPECT_DOUBLE_EQ(3.14, dimless.value());
}
