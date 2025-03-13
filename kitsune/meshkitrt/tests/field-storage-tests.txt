#include "gtest/gtest.h"
#include "Field.h"
#include "Mesh.h"
#include "FieldStorage.h"
#include <vector>
#include <cstdint>

// Test fixture for Field storage-specific tests
class FieldStorageTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create a mesh for testing different storage layouts
        mesh = new mesh::Mesh(16, 16);
    }

    void TearDown() override {
        delete mesh;
    }

    mesh::Mesh* mesh;
};

// Test 1: Row-major storage layout using field::storage::rowMajor function
TEST_F(FieldStorageTest, RowMajorStorageLayout) {
    mesh::Field<int> field(*mesh, 0);
    
    // Verify the linearIndex calculation matches field::storage::rowMajor
    for (uint32_t j = 0; j < field.ny(); ++j) {
        for (uint32_t i = 0; i < field.nx(); ++i) {
            // Get the linear index using mesh's linearIndex
            int meshIndex = mesh->linearIndex(i, j);
            
            // Calculate index using field::storage::rowMajor
            int rowMajorIndex = field::storage::rowMajor(*mesh, i, j);
            
            // Both should match
            EXPECT_EQ(meshIndex, rowMajorIndex);
            
            // Set the value using one index method
            field(i, j) = i * 100 + j;
            
            // Access the same element using the raw data and calculated index
            EXPECT_EQ(field.data()[rowMajorIndex], i * 100 + j);
        }
    }
}

// Test 2: Blocked storage layout using field::storage::blocked function
TEST_F(FieldStorageTest, BlockedStorageLayout) {
    // Create a new mesh with custom linearIndex that uses blocked layout
    class BlockedMesh : public mesh::Mesh {
    public:
        BlockedMesh(uint32_t nx, uint32_t ny, int blockSize) 
            : mesh::Mesh(nx, ny), m_blockSize(blockSize) {}
        
        uint32_t linearIndex(uint32_t i, uint32_t j) const override {
            return field::storage::blocked(*this, i, j, m_blockSize);
        }
        
    private:
        int m_blockSize;
    };
    
    // Create a mesh with 4x4 blocks
    BlockedMesh blockedMesh(16, 16, 4);
    mesh::Field<double> field(blockedMesh, 0.0);
    
    // Set values in the field
    for (uint32_t j = 0; j < field.ny(); ++j) {
        for (uint32_t i = 0; i < field.nx(); ++i) {
            field(i, j) = i + j * 0.1;
        }
    }
    
    // Verify values are stored and retrieved correctly
    for (uint32_t j = 0; j < field.ny(); ++j) {
        for (uint32_t i = 0; i < field.nx(); ++i) {
            EXPECT_DOUBLE_EQ(field(i, j), i + j * 0.1);
        }
    }
}

// Test 3: Z-order storage layout using field::storage::zOrder function
TEST_F(FieldStorageTest, ZOrderStorageLayout) {
    // Create a new mesh with custom linearIndex that uses z-order layout
    class ZOrderMesh : public mesh::Mesh {
    public:
        ZOrderMesh(uint32_t nx, uint32_t ny) : mesh::Mesh(nx, ny) {
            // Z-order works best with power-of-2 dimensions
            assert((nx & (nx - 1)) == 0 && "nx should be a power of 2");
            assert((ny & (ny - 1)) == 0 && "ny should be a power of 2");
        }
        
        uint32_t linearIndex(uint32_t i, uint32_t j) const override {
            return field::storage::zOrder(*this, i, j);
        }
    };
    
    // Create a mesh with power-of-2 dimensions for Z-order layout
    ZOrderMesh zOrderMesh(16, 16);
    mesh::Field<float> field(zOrderMesh, 0.0f);
    
    // Set values in the field
    for (uint32_t j = 0; j < field.ny(); ++j) {
        for (uint32_t i = 0; i < field.nx(); ++i) {
            field(i, j) = static_cast<float>(i * 10 + j);
        }
    }
    
    // Verify values are stored and retrieved correctly
    for (uint32_t j = 0; j < field.ny(); ++j) {
        for (uint32_t i = 0; i < field.nx(); ++i) {
            EXPECT_FLOAT_EQ(field(i, j), static_cast<float>(i * 10 + j));
        }
    }
}

// Test 4: Test default storage strategy
TEST_F(FieldStorageTest, DefaultStorageStrategy) {
    mesh::Field<int> field(*mesh, 0);
    
    // Set values using 2D indices
    for (uint32_t j = 0; j < field.ny(); ++j) {
        for (uint32_t i = 0; i < field.nx(); ++i) {
            field(i, j) = i + j * field.nx();
        }
    }
    
    // Verify that field::storage::index matches the default behavior
    for (uint32_t j = 0; j < field.ny(); ++j) {
        for (uint32_t i = 0; i < field.nx(); ++i) {
            int expectedIndex = field::storage::index(*mesh, i, j);
            int calculatedIndex = i + j * field.nx();
            EXPECT_EQ(expectedIndex, calculatedIndex);
            
            // The value stored should match the linear index
            EXPECT_EQ(field(i, j), calculatedIndex);
        }
    }
}

// Test 5: Test for index bounds checking
TEST_F(FieldStorageTest, IndexBoundsChecking) {
    mesh::Field<int> field(*mesh, 0);
    
    // Accessing within bounds should be fine
    EXPECT_NO_THROW(field(0, 0));
    EXPECT_NO_THROW(field(field.nx() - 1, field.ny() - 1));
    
    // Out-of-bounds access should trigger an assertion in debug builds
    // In release builds with NDEBUG defined, this might not fail,
    // so we don't test for exceptions here.
    
    // Test that meshes protect themselves against out-of-bounds linearIndex calls
    EXPECT_THROW(mesh->linearIndex(mesh->nx(), 0), std::out_of_range);
    EXPECT_THROW(mesh->linearIndex(0, mesh->ny()), std::out_of_range);
}

// Test 6: Test with vector types
TEST_F(FieldStorageTest, VectorTypeStorage) {
    using Vector2D = std::pair<double, double>;
    mesh::Field<Vector2D> field(*mesh, Vector2D(0.0, 0.0));
    
    // Set some vector values
    field(1, 2) = Vector2D(1.5, 2.5);
    field(3, 4) = Vector2D(3.5, 4.5);
    
    // Verify the vectors were stored correctly
    EXPECT_DOUBLE_EQ(field(1, 2).first, 1.5);
    EXPECT_DOUBLE_EQ(field(1, 2).second, 2.5);
    EXPECT_DOUBLE_EQ(field(3, 4).first, 3.5);
    EXPECT_DOUBLE_EQ(field(3, 4).second, 4.5);
    
    // Default values should be preserved elsewhere
    EXPECT_DOUBLE_EQ(field(0, 0).first, 0.0);
    EXPECT_DOUBLE_EQ(field(0, 0).second, 0.0);
}

// Test 7: Test field copy semantics
TEST_F(FieldStorageTest, FieldCopySemantics) {
    mesh::Field<int> originalField(*mesh, 42);
    
    // Change some values in the original
    originalField(1, 1) = 10;
    originalField(2, 2) = 20;
    
    // Create a new field as a copy (using copy constructor)
    mesh::Field<int> copiedField = originalField;
    
    // Verify the copied field has the same values
    EXPECT_EQ(copiedField(0, 0), 42);
    EXPECT_EQ(copiedField(1, 1), 10);
    EXPECT_EQ(copiedField(2, 2), 20);
    
    // Modify the original field
    originalField(1, 1) = 100;
    
    // Verify the copied field is independent
    EXPECT_EQ(copiedField(1, 1), 10); // Still has the old value
    EXPECT_EQ(originalField(1, 1), 100); // Has the new value
}

// Test 8: Memory layout and contiguous storage
TEST_F(FieldStorageTest, MemoryLayoutAndContiguousStorage) {
    mesh::Field<double> field(*mesh, 0.0);
    
    // Get raw pointer to the data
    double* data = field.data();
    
    // Fill with sequential values
    for (size_t i = 0; i < field.size(); ++i) {
        data[i] = static_cast<double>(i);
    }
    
    // Verify that operator() accesses the same memory
    for (uint32_t j = 0; j < field.ny(); ++j) {
        for (uint32_t i = 0; i < field.nx(); ++i) {
            int linearIndex = i + j * field.nx();
            EXPECT_DOUBLE_EQ(field(i, j), static_cast<double>(linearIndex));
        }
    }
    
    // Verify that memory is contiguous
    for (size_t i = 0; i < field.size() - 1; ++i) {
        // Pointers to adjacent elements should differ by exactly sizeof(double)
        EXPECT_EQ(reinterpret_cast<uintptr_t>(&data[i+1]) - 
                  reinterpret_cast<uintptr_t>(&data[i]), 
                  sizeof(double));
    }
}
