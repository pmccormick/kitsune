/**
 * @file FieldTest.cpp
 * @brief Unit tests for the Field class using Google Test framework
 */

#include "Field.h"
#include <gtest/gtest.h>
#include <limits>
#include <cmath>

namespace {

// Test fixture for testing Field class
class FieldTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create fields with default values
        centerField = std::make_unique<CellCenterField<double>>(5, 5);
        vertexField = std::make_unique<VertexField<double>>(5, 5);
        hEdgeField = std::make_unique<HorizontalEdgeField<double>>(5, 5);
        vEdgeField = std::make_unique<VerticalEdgeField<double>>(5, 5);
    }
    
    std::unique_ptr<CellCenterField<double>> centerField;
    std::unique_ptr<VertexField<double>> vertexField;
    std::unique_ptr<HorizontalEdgeField<double>> hEdgeField;
    std::unique_ptr<VerticalEdgeField<double>> vEdgeField;
};

// Test field construction and size calculation
TEST_F(FieldTest, ConstructionAndSize) {
    // Check sizes for cell-centered field
    EXPECT_EQ(5, centerField->nx());
    EXPECT_EQ(5, centerField->ny());
    EXPECT_EQ(25, centerField->size());
    
    // Check sizes for vertex-centered field (should have one more point in each direction)
    EXPECT_EQ(6, vertexField->nx());
    EXPECT_EQ(6, vertexField->ny());
    EXPECT_EQ(36, vertexField->size());
    
    // Check sizes for horizontal edge field
    EXPECT_EQ(5, hEdgeField->nx());
    EXPECT_EQ(6, hEdgeField->ny());
    EXPECT_EQ(30, hEdgeField->size());
    
    // Check sizes for vertical edge field
    EXPECT_EQ(6, vEdgeField->nx());
    EXPECT_EQ(5, vEdgeField->ny());
    EXPECT_EQ(30, vEdgeField->size());
}

// Test element access with operator()
TEST_F(FieldTest, ElementAccess) {
    // Initialize with some values
    centerField->fill(1.0);
    
    // Check 2D access
    EXPECT_DOUBLE_EQ(1.0, (*centerField)(2, 3));
    
    // Modify value
    (*centerField)(2, 3) = 5.0;
    EXPECT_DOUBLE_EQ(5.0, (*centerField)(2, 3));
    
    // Test 3D access for 2D field (k=0)
    EXPECT_DOUBLE_EQ(5.0, (*centerField)(2, 3, 0));
    
    // Test 3D vector field
    Vector3DField<double> vectorField(5, 5);
    vectorField.fill(1.0);
    
    // Set different components
    vectorField(1, 1, 0) = 2.0;
    vectorField(1, 1, 1) = 3.0;
    vectorField(1, 1, 2) = 4.0;
    
    // Check components
    EXPECT_DOUBLE_EQ(2.0, vectorField(1, 1, 0));
    EXPECT_DOUBLE_EQ(3.0, vectorField(1, 1, 1));
    EXPECT_DOUBLE_EQ(4.0, vectorField(1, 1, 2));
}

// Test filling field with values
TEST_F(FieldTest, FillAndReset) {
    // Fill with a constant
    centerField->fill(3.14);
    
    // Check a few points
    EXPECT_DOUBLE_EQ(3.14, (*centerField)(0, 0));
    EXPECT_DOUBLE_EQ(3.14, (*centerField)(2, 3));
    EXPECT_DOUBLE_EQ(3.14, (*centerField)(4, 4));
    
    // Fill with zero
    centerField->fillZero();
    
    // Check a few points
    EXPECT_DOUBLE_EQ(0.0, (*centerField)(0, 0));
    EXPECT_DOUBLE_EQ(0.0, (*centerField)(2, 3));
    EXPECT_DOUBLE_EQ(0.0, (*centerField)(4, 4));
}

// Test field arithmetic operations (field-field)
TEST_F(FieldTest, FieldArithmetic) {
    // Create two fields with different values
    CellCenterField<double> fieldA(3, 3);
    CellCenterField<double> fieldB(3, 3);
    
    fieldA.fill(2.0);
    fieldB.fill(3.0);
    
    // Test addition
    auto resultAdd = fieldA + fieldB;
    EXPECT_DOUBLE_EQ(5.0, resultAdd(1, 1));
    
    // Test subtraction
    auto resultSub = fieldA - fieldB;
    EXPECT_DOUBLE_EQ(-1.0, resultSub(1, 1));
    
    // Test multiplication
    auto resultMul = fieldA * fieldB;
    EXPECT_DOUBLE_EQ(6.0, resultMul(1, 1));
    
    // Test division
    auto resultDiv = fieldA / fieldB;
    EXPECT_DOUBLE_EQ(2.0/3.0, resultDiv(1, 1));
    
    // Test compound assignment
    fieldA += fieldB;
    EXPECT_DOUBLE_EQ(5.0, fieldA(1, 1));
    
    fieldA -= fieldB;
    EXPECT_DOUBLE_EQ(2.0, fieldA(1, 1));
    
    fieldA *= fieldB;
    EXPECT_DOUBLE_EQ(6.0, fieldA(1, 1));
    
    fieldA /= fieldB;
    EXPECT_DOUBLE_EQ(2.0, fieldA(1, 1));
}

// Test field-scalar arithmetic operations
TEST_F(FieldTest, ScalarArithmetic) {
    // Create a field
    CellCenterField<double> field(3, 3);
    field.fill(2.0);
    
    // Test addition
    auto resultAdd = field + 3.0;
    EXPECT_DOUBLE_EQ(5.0, resultAdd(1, 1));
    
    // Test subtraction
    auto resultSub = field - 1.0;
    EXPECT_DOUBLE_EQ(1.0, resultSub(1, 1));
    
    // Test multiplication
    auto resultMul = field * 3.0;
    EXPECT_DOUBLE_EQ(6.0, resultMul(1, 1));
    
    // Test division
    auto resultDiv = field / 2.0;
    EXPECT_DOUBLE_EQ(1.0, resultDiv(1, 1));
    
    // Test scalar on left side
    auto resultScalarMul = 4.0 * field;
    EXPECT_DOUBLE_EQ(8.0, resultScalarMul(1, 1));
    
    auto resultScalarAdd = 5.0 + field;
    EXPECT_DOUBLE_EQ(7.0, resultScalarAdd(1, 1));
    
    // Test compound assignment
    field += 3.0;
    EXPECT_DOUBLE_EQ(5.0, field(1, 1));
    
    field -= 1.0;
    EXPECT_DOUBLE_EQ(4.0, field(1, 1));
    
    field *= 2.0;
    EXPECT_DOUBLE_EQ(8.0, field(1, 1));
    
    field /= 4.0;
    EXPECT_DOUBLE_EQ(2.0, field(1, 1));
}

// Test field statistical operations
TEST_F(FieldTest, Statistics) {
    // Create a field with varying values
    CellCenterField<double> field(3, 3);
    
    // Fill with values from 1 to 9
    for (int j = 0; j < 3; ++j) {
        for (int i = 0; i < 3; ++i) {
            field(i, j) = i + j * 3 + 1;
        }
    }
    
    // Test min
    EXPECT_DOUBLE_EQ(1.0, field.min());
    
    // Test max
    EXPECT_DOUBLE_EQ(9.0, field.max());
    
    // Test sum (1+2+3+...+9 = 45)
    EXPECT_DOUBLE_EQ(45.0, field.sum());
    
    // Test average (45/9 = 5)
    EXPECT_DOUBLE_EQ(5.0, field.average());
}

// Test resizing fields
TEST_F(FieldTest, Resize) {
    // Create a field and fill it
    CellCenterField<double> field(3, 3);
    field.fill(2.0);
    
    // Resize larger
    field.resize(5, 5);
    EXPECT_EQ(5, field.nx());
    EXPECT_EQ(5, field.ny());
    EXPECT_EQ(25, field.size());
    
    // Original data should be preserved
    EXPECT_DOUBLE_EQ(2.0, field(1, 1));
    
    // Resize smaller
    field.resize(2, 2);
    EXPECT_EQ(2, field.nx());
    EXPECT_EQ(2, field.ny());
    EXPECT_EQ(4, field.size());
    
    // Original data should be preserved where it fits
    EXPECT_DOUBLE_EQ(2.0, field(1, 1));
}

// Test copying data between fields
TEST_F(FieldTest, CopyFrom) {
    // Create source and destination fields of different sizes
    CellCenterField<double> source(3, 3);
    CellCenterField<double> dest(2, 2);
    
    // Fill source with a value
    source.fill(3.14);
    
    // Copy to destination (should resize)
    dest.copyFrom(source);
    
    // Check sizes
    EXPECT_EQ(3, dest.nx());
    EXPECT_EQ(3, dest.ny());
    
    // Check values
    EXPECT_DOUBLE_EQ(3.14, dest(1, 1));
    EXPECT_DOUBLE_EQ(3.14, dest(2, 2));
}

// Test iterators
TEST_F(FieldTest, Iterators) {
    // Create a field and fill with a value
    CellCenterField<double> field(3, 3);
    field.fill(2.5);
    
    // Use iterator to sum values
    double sum = 0.0;
    for (const auto& val : field) {
        sum += val;
    }
    
    // Expected sum: 2.5 * 9 = 22.5
    EXPECT_DOUBLE_EQ(22.5, sum);
    
    // Use iterator to modify values
    for (auto& val : field) {
        val *= 2.0;
    }
    
    // Check modification
    EXPECT_DOUBLE_EQ(5.0, field(1, 1));
}

// Test forEach methods
TEST_F(FieldTest, ForEach) {
    // Create a field
    CellCenterField<double> field(3, 3);
    field.fill(2.0);
    
    // Apply a function to each element
    field.forEach([](double val) { return val * val; });
    
    // Check result (each value should be squared)
    EXPECT_DOUBLE_EQ(4.0, field(1, 1));
    
    // Test forEachIndexed
    field.forEachIndexed([](size_t i, size_t j, size_t /* k */, double& val) {
        val = static_cast<double>(i + j);
    });
    
    // Check result (each value should be i+j)
    EXPECT_DOUBLE_EQ(2.0, field(1, 1));
    EXPECT_DOUBLE_EQ(3.0, field(2, 1));
}

// Test swap operation
TEST_F(FieldTest, Swap) {
    // Create two fields
    CellCenterField<double> fieldA(3, 3);
    CellCenterField<double> fieldB(4, 4);
    
    fieldA.fill(1.0);
    fieldB.fill(2.0);
    
    // Swap the fields
    fieldA.swap(fieldB);
    
    // Check sizes
    EXPECT_EQ(4, fieldA.nx());
    EXPECT_EQ(4, fieldA.ny());
    EXPECT_EQ(3, fieldB.nx());
    EXPECT_EQ(3, fieldB.ny());
    
    // Check values
    EXPECT_DOUBLE_EQ(2.0, fieldA(1, 1));
    EXPECT_DOUBLE_EQ(1.0, fieldB(1, 1));
}

// Test 3D field functionality
TEST_F(FieldTest, ThreeDimensional) {
    // Create a 3D field
    Vector3DField<double> field(3, 3);
    
    // Check dimensions
    EXPECT_EQ(3, field.nx());
    EXPECT_EQ(3, field.ny());
    EXPECT_EQ(3, field.depth());
    EXPECT_EQ(27, field.size());
    
    // Fill with different values per component
    for (int j = 0; j < 3; ++j) {
        for (int i = 0; i < 3; ++i) {
            field(i, j, 0) = 1.0;
            field(i, j, 1) = 2.0;
            field(i, j, 2) = 3.0;
        }
    }
    
    // Test access to different components
    EXPECT_DOUBLE_EQ(1.0, field(1, 1, 0));
    EXPECT_DOUBLE_EQ(2.0, field(1, 1, 1));
    EXPECT_DOUBLE_EQ(3.0, field(1, 1, 2));
    
    // Test statistics on 3D field
    EXPECT_DOUBLE_EQ(1.0, field.min());
    EXPECT_DOUBLE_EQ(3.0, field.max());
    EXPECT_DOUBLE_EQ(2.0, field.average());
}

// Test direct array access
TEST_F(FieldTest, DirectAccess) {
    // Create a field
    CellCenterField<double> field(3, 3);
    field.fill(5.0);
    
    // Access raw data
    double* data = field.rawData();
    EXPECT_DOUBLE_EQ(5.0, data[0]);
    
    // Modify through raw pointer
    data[4] = 10.0; // Center element
    
    // Check modification through regular access
    EXPECT_DOUBLE_EQ(10.0, field(1, 1));
    
    // Test operator[] access
    EXPECT_DOUBLE_EQ(5.0, field[0]);
    EXPECT_DOUBLE_EQ(10.0, field[4]);
    
    // Modify through operator[]
    field[7] = 15.0;
    
    // Index 7 should correspond to (1,2) in 2D
    EXPECT_DOUBLE_EQ(15.0, field(1, 2));
}

} // namespace

