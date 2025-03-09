/**
 * @file FieldIteratorTests.cpp
 * @brief Unit tests for the Field iterator implementations
 * 
 * This file contains unit tests for the field iterator functionality,
 * focusing on the direct field access patterns. These tests serve as
 * the foundation for testing the higher-level accessor iterators.
 */

#include <gtest/gtest.h>
#include "Field.h"
#include "FieldIterators.h"

//------------------------------------------------------------------------------
// Test fixture for Field iterator tests
//------------------------------------------------------------------------------
class FieldIteratorTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create test fields of different types and locations
        scalarField = new Field<double, CellCenterTag>(10, 10);
        vectorField = new Field<double, CellCenterTag, 2>(10, 10);
        vertexField = new Field<double, VertexTag>(10, 10);
        edgeField = new Field<double, HorizontalEdgeTag>(10, 10);
        
        // Initialize fields with known patterns
        initializeField(*scalarField, 1.0);
        initializeVectorField(*vectorField);
        initializeField(*vertexField, 2.0);
        initializeField(*edgeField, 3.0);
    }
    
    void TearDown() override {
        delete scalarField;
        delete vectorField;
        delete vertexField;
        delete edgeField;
    }
    
    // Helper to initialize a field with a value
    template <typename T, typename LocationTag, size_t D>
    void initializeField(Field<T, LocationTag, D>& field, T value) {
        for (size_t i = 0; i < field.size(); ++i) {
            field[i] = value;
        }
    }
    
    // Helper to initialize a vector field with a pattern
    void initializeVectorField(Field<double, CellCenterTag, 2>& field) {
        for (size_t j = 0; j < field.ny(); ++j) {
            for (size_t i = 0; i < field.nx(); ++i) {
                field(i, j, 0) = static_cast<double>(i);  // x component
                field(i, j, 1) = static_cast<double>(j);  // y component
            }
        }
    }
    
    // Test fields
    Field<double, CellCenterTag>* scalarField;
    Field<double, CellCenterTag, 2>* vectorField;
    Field<double, VertexTag>* vertexField;
    Field<double, HorizontalEdgeTag>* edgeField;
};

//------------------------------------------------------------------------------
// Linear Iterator Tests
//------------------------------------------------------------------------------

TEST_F(FieldIteratorTest, LinearIteratorTraversal) {
    // Use the linear iterator to traverse a field
    auto it = scalarField->linearBegin();
    auto end = scalarField->linearEnd();
    
    size_t count = 0;
    while (it != end) {
        EXPECT_EQ(*it, 1.0) << "Value at index " << count << " should be 1.0";
        ++it;
        ++count;
    }
    
    EXPECT_EQ(count, scalarField->size()) << "Iterator should visit every element";
}

TEST_F(FieldIteratorTest, LinearIteratorModification) {
    // Modify values using the iterator
    auto it = scalarField->linearBegin();
    auto end = scalarField->linearEnd();
    
    size_t index = 0;
    while (it != end) {
        *it = static_cast<double>(index);
        ++it;
        ++index;
    }
    
    // Verify modification
    for (size_t i = 0; i < scalarField->size(); ++i) {
        EXPECT_EQ((*scalarField)[i], static_cast<double>(i)) 
            << "Value at index " << i << " should be " << i;
    }
}

TEST_F(FieldIteratorTest, LinearIteratorRandom) {
    // Test random access capabilities
    auto it = scalarField->linearBegin();
    
    // Jump to middle
    it += scalarField->size() / 2;
    EXPECT_EQ(it.index(), scalarField->size() / 2) 
        << "Iterator should be at middle position";
    
    // Jump to end
    it += scalarField->size() / 2;
    EXPECT_EQ(it, scalarField->linearEnd()) 
        << "Iterator should be at end position";
    
    // Jump back
    it -= 10;
    EXPECT_EQ(it.index(), scalarField->size() - 10) 
        << "Iterator should be 10 positions from end";
}

//------------------------------------------------------------------------------
// 2D Iterator Tests
//------------------------------------------------------------------------------

TEST_F(FieldIteratorTest, Iterator2DTraversal) {
    // Use 2D iterator for traversal
    auto it = scalarField->begin2D();
    auto end = scalarField->end2D();
    
    size_t count = 0;
    while (it != end) {
        EXPECT_EQ(*it, 1.0) << "Value at position (" << it.i() << "," << it.j() 
                           << ") should be 1.0";
        ++it;
        ++count;
    }
    
    EXPECT_EQ(count, scalarField->size()) << "2D iterator should visit every element";
}

TEST_F(FieldIteratorTest, Iterator2DPosition) {
    // Check position tracking in 2D iterator
    auto it = scalarField->begin2D();
    
    // First row validation
    for (size_t i = 0; i < scalarField->nx(); ++i) {
        EXPECT_EQ(it.i(), i) << "X-position should match iteration";
        EXPECT_EQ(it.j(), 0) << "Y-position should be 0 for first row";
        ++it;
    }
    
    // Second row first element
    EXPECT_EQ(it.i(), 0) << "X-position should wrap to 0 for new row";
    EXPECT_EQ(it.j(), 1) << "Y-position should increment to 1 for second row";
}

TEST_F(FieldIteratorTest, Iterator2DRange) {
    // Test range-based for loop with 2D range
    size_t count = 0;
    
    for (auto& value : scalarField->range2D()) {
        EXPECT_EQ(value, 1.0) << "Value should be 1.0";
        value = 2.0;  // Modify through reference
        ++count;
    }
    
    EXPECT_EQ(count, scalarField->size()) << "Range should cover all elements";
    
    // Verify modification
    for (size_t i = 0; i < scalarField->size(); ++i) {
        EXPECT_EQ((*scalarField)[i], 2.0) << "All values should be modified to 2.0";
    }
}

TEST_F(FieldIteratorTest, Vector2DIterator) {
    // Test 2D iterator with vector field
    auto it = vectorField->begin2D();
    
    for (size_t j = 0; j < vectorField->ny(); ++j) {
        for (size_t i = 0; i < vectorField->nx(); ++i) {
            EXPECT_EQ(it.i(), i) << "X-position should match iteration";
            EXPECT_EQ(it.j(), j) << "Y-position should match iteration";
            EXPECT_EQ(*it, static_cast<double>(i)) << "X-component should match i";
            ++it;
            
            EXPECT_EQ(it.i(), i) << "X-position should remain the same for y-component";
            EXPECT_EQ(it.j(), j) << "Y-position should remain the same for y-component";
            EXPECT_EQ(it.k(), 1) << "K-index should be 1 for y-component";
            EXPECT_EQ(*it, static_cast<double>(j)) << "Y-component should match j";
            ++it;
        }
    }
}

//------------------------------------------------------------------------------
// Block Iterator Tests
//------------------------------------------------------------------------------

TEST_F(FieldIteratorTest, BlockIteratorTraversal) {
    // Use block iterator for traversal with small block size for testing
    const size_t blockSizeX = 3;
    const size_t blockSizeY = 2;
    
    auto it = scalarField->beginBlock(blockSizeX, blockSizeY);
    auto end = scalarField->endBlock(blockSizeX, blockSizeY);
    
    size_t count = 0;
    while (it != end) {
        if (it.i() < scalarField->nx() && it.j() < scalarField->ny()) {
            EXPECT_EQ(*it, 1.0) << "Value at position (" << it.i() << "," << it.j() 
                               << ") should be 1.0";
            count++;
        }
        ++it;
    }
    
    EXPECT_EQ(count, scalarField->size()) << "Block iterator should visit every element";
}

TEST_F(FieldIteratorTest, BlockIteratorOrder) {
    // Check that block iterator traverses in a block-wise pattern
    const size_t blockSizeX = 3;
    const size_t blockSizeY = 2;
    
    auto it = scalarField->beginBlock(blockSizeX, blockSizeY);
    
    // First block
    EXPECT_EQ(it.i(), 0);
    EXPECT_EQ(it.j(), 0);
    ++it;
    
    EXPECT_EQ(it.i(), 1);
    EXPECT_EQ(it.j(), 0);
    ++it;
    
    EXPECT_EQ(it.i(), 2);
    EXPECT_EQ(it.j(), 0);
    ++it;
    
    EXPECT_EQ(it.i(), 0);
    EXPECT_EQ(it.j(), 1);
    ++it;
    
    EXPECT_EQ(it.i(), 1);
    EXPECT_EQ(it.j(), 1);
    ++it;
    
    EXPECT_EQ(it.i(), 2);
    EXPECT_EQ(it.j(), 1);
    ++it;
    
    // Should move to next block in x direction
    EXPECT_EQ(it.i(), 3);
    EXPECT_EQ(it.j(), 0);
}

TEST_F(FieldIteratorTest, BlockIteratorRange) {
    // Test range-based for loop with block range
    const size_t blockSizeX = 4;
    const size_t blockSizeY = 4;
    
    size_t count = 0;
    
    for (auto& value : scalarField->blockRange(blockSizeX, blockSizeY)) {
        value = 3.0;  // Modify through reference
        ++count;
    }
    
    EXPECT_EQ(count, scalarField->size()) << "Block range should cover all elements";
    
    // Verify modification
    for (size_t i = 0; i < scalarField->size(); ++i) {
        EXPECT_EQ((*scalarField)[i], 3.0) << "All values should be modified to 3.0";
    }
}

//------------------------------------------------------------------------------
// Strided Iterator Tests
//------------------------------------------------------------------------------

TEST_F(FieldIteratorTest, StridedRangePartitioning) {
    // Test partitioning a field for parallel processing
    const size_t numPartitions = 4;
    size_t totalCount = 0;
    
    for (size_t partitionId = 0; partitionId < numPartitions; ++partitionId) {
        auto partition = scalarField->getPartition(partitionId, numPartitions);
        
        size_t partitionCount = 0;
        for (auto& value : partition) {
            value = static_cast<double>(partitionId);  // Mark with partition ID
            ++partitionCount;
        }
        
        // Each partition should have approximately equal size
        EXPECT_GT(partitionCount, 0) << "Partition " << partitionId << " should have elements";
        totalCount += partitionCount;
    }
    
    EXPECT_EQ(totalCount, scalarField->size()) 
        << "All partitions combined should cover every element";
    
    // Verify each element was assigned to exactly one partition
    for (size_t i = 0; i < scalarField->size(); ++i) {
        double value = (*scalarField)[i];
        EXPECT_GE(value, 0.0) << "Element " << i << " should be assigned to a partition";
        EXPECT_LT(value, static_cast<double>(numPartitions)) 
            << "Element " << i << " should have valid partition ID";
    }
}

//------------------------------------------------------------------------------
// Edge Case Tests
//------------------------------------------------------------------------------

TEST_F(FieldIteratorTest, EmptyFieldIteration) {
    // Create an empty field (0x0)
    Field<double, CellCenterTag> emptyField(0, 0);
    
    // Linear iterator
    EXPECT_EQ(emptyField.linearBegin(), emptyField.linearEnd()) 
        << "Begin and end should be equal for empty field";
    
    // 2D iterator
    EXPECT_EQ(emptyField.begin2D(), emptyField.end2D()) 
        << "2D begin and end should be equal for empty field";
    
    // Range-based for loop
    size_t count = 0;
    for (auto& value : emptyField.range2D()) {
        ++count;
    }
    EXPECT_EQ(count, 0) << "Empty field should have no elements to iterate";
}

TEST_F(FieldIteratorTest, VertexFieldIterator) {
    // Check that vertex field has correct dimensions
    EXPECT_EQ(vertexField->nx(), 11) << "Vertex field should have nx+1 points in x direction";
    EXPECT_EQ(vertexField->ny(), 11) << "Vertex field should have ny+1 points in y direction";
    
    // Verify total size
    size_t count = 0;
    for (auto& value : vertexField->range2D()) {
        ++count;
    }
    EXPECT_EQ(count, 11 * 11) << "Vertex field should have (nx+1)*(ny+1) elements";
}

TEST_F(FieldIteratorTest, EdgeFieldIterator) {
    // Check that horizontal edge field has correct dimensions
    EXPECT_EQ(edgeField->nx(), 10) << "Horizontal edge field should have nx points in x direction";
    EXPECT_EQ(edgeField->ny(), 11) << "Horizontal edge field should have ny+1 points in y direction";
    
    // Verify total size
    size_t count = 0;
    for (auto& value : edgeField->range2D()) {
        ++count;
    }
    EXPECT_EQ(count, 10 * 11) << "Horizontal edge field should have nx*(ny+1) elements";
}

TEST_F(FieldIteratorTest, WhereIndicesFilter) {
    // Test filtering with a predicate
    auto filtered = scalarField->whereIndices([](size_t i, size_t j) {
        return (i % 2 == 0) && (j % 2 == 0);  // Only even indices
    });
    
    size_t count = 0;
    for (auto& value : filtered) {
        value = 4.0;  // Mark filtered elements
        ++count;
    }
    
    // Count should be 25 (5x5 for a 10x10 field with even indices only)
    EXPECT_EQ(count, 25) << "Should select 25 elements with even i,j indices";
    
    // Verify that only matching elements were modified
    for (size_t j = 0; j < scalarField->ny(); ++j) {
        for (size_t i = 0; i < scalarField->nx(); ++i) {
            if ((i % 2 == 0) && (j % 2 == 0)) {
                EXPECT_EQ((*scalarField)(i, j), 4.0) 
                    << "Matching element at (" << i << "," << j << ") should be modified";
            } else {
                EXPECT_EQ((*scalarField)(i, j), 1.0) 
                    << "Non-matching element at (" << i << "," << j << ") should be unchanged";
            }
        }
    }
}

//------------------------------------------------------------------------------
// Performance Comparison Tests (for sanity check, not timing)
//------------------------------------------------------------------------------

TEST_F(FieldIteratorTest, LinearVsSubscriptAccess) {
    // Baseline approach with direct subscripting
    Field<double, CellCenterTag> field1(100, 100);
    for (size_t j = 0; j < field1.ny(); ++j) {
        for (size_t i = 0; i < field1.nx(); ++i) {
            field1(i, j) = i + j;
        }
    }
    
    // LinearIterator approach
    Field<double, CellCenterTag> field2(100, 100);
    size_t index = 0;
    for (auto it = field2.linearBegin(); it != field2.linearEnd(); ++it, ++index) {
        size_t j = index / field2.nx();
        size_t i = index % field2.nx();
        *it = i + j;
    }
    
    // Verify both approaches produce identical results
    for (size_t j = 0; j < field1.ny(); ++j) {
        for (size_t i = 0; i < field1.nx(); ++i) {
            EXPECT_EQ(field1(i, j), field2(i, j)) 
                << "Results should match at position (" << i << "," << j << ")";
        }
    }
}

TEST_F(FieldIteratorTest, BlockVsLinearAccess) {
    // Linear traversal
    Field<double, CellCenterTag> field1(100, 100);
    for (auto& value : field1) {
        value = 1.0;
    }
    
    // Block traversal
    Field<double, CellCenterTag> field2(100, 100);
    for (auto& value : field2.blockRange(16, 16)) {
        value = 1.0;
    }
    
    // Verify both approaches produce identical results
    for (size_t i = 0; i < field1.size(); ++i) {
        EXPECT_EQ(field1[i], field2[i]) << "Results should match at index " << i;
    }
}

//------------------------------------------------------------------------------
// Range-based for loop tests
//------------------------------------------------------------------------------

TEST_F(FieldIteratorTest, RangeBasedForLoopSupport) {
    // Test traditional range-based for loop support
    size_t count = 0;
    for (auto& value : *scalarField) {  // Using STL iterators
        EXPECT_EQ(value, 1.0);
        value = 5.0;
        ++count;
    }
    EXPECT_EQ(count, scalarField->size());
    
    // Verify modification
    for (size_t i = 0; i < scalarField->size(); ++i) {
        EXPECT_EQ((*scalarField)[i], 5.0);
    }
}

