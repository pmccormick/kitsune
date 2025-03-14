#include "gtest/gtest.h"
#include "Field.h"
#include "Mesh.h"
#include "FieldStorage.h"
#include <vector>

// Test fixture for Field tests
class FieldTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create a 10x10 mesh for testing
        mesh = new mesh::Mesh(10, 10);
    }

    void TearDown() override {
        delete mesh;
    }

    mesh::Mesh* mesh;
};

// Test 1: Test creation of Field with default value
TEST_F(FieldTest, CreateFieldWithDefaultValue) {
    mesh::Field<double> field(*mesh, 3.14);
    
    // Check size matches mesh size
    EXPECT_EQ(field.size(), mesh->size());
    
    // Check nx and ny match mesh dimensions
    EXPECT_EQ(field.nx(), mesh->nx());
    EXPECT_EQ(field.ny(), mesh->ny());
    
    // Check all elements have the default value
    for (uint32_t j = 0; j < field.ny(); ++j) {
        for (uint32_t i = 0; i < field.nx(); ++i) {
            EXPECT_DOUBLE_EQ(field(i, j), 3.14);
        }
    }
}

// Test 2: Test element access and modification
TEST_F(FieldTest, ElementAccessAndModification) {
    mesh::Field<int> field(*mesh, 0);
    
    // Modify elements
    for (uint32_t j = 0; j < field.ny(); ++j) {
        for (uint32_t i = 0; i < field.nx(); ++i) {
            field(i, j) = i * 100 + j;
        }
    }
    
    // Verify modified values
    for (uint32_t j = 0; j < field.ny(); ++j) {
        for (uint32_t i = 0; i < field.nx(); ++i) {
            EXPECT_EQ(field(i, j), i * 100 + j);
        }
    }
}

// Test 3: Test const element access
TEST_F(FieldTest, ConstElementAccess) {
    mesh::Field<double> field(*mesh, 0.0);
    
    // Modify some elements
    field(1, 2) = 1.5;
    field(3, 4) = 2.5;
    
    // Create const reference
    const mesh::Field<double>& constField = field;
    
    // Test const access
    EXPECT_DOUBLE_EQ(constField(1, 2), 1.5);
    EXPECT_DOUBLE_EQ(constField(3, 4), 2.5);
}

// Test 4: Test raw data access
TEST_F(FieldTest, RawDataAccess) {
    mesh::Field<int> field(*mesh, 5);
    
    // Get raw pointer and modify data
    int* data = field.data();
    data[0] = 10;
    data[field.size() - 1] = 20;
    
    // Verify modifications through Field interface
    EXPECT_EQ(field(0, 0), 10);
    EXPECT_EQ(field(field.nx() - 1, field.ny() - 1), 20);
    
    // Test const data access
    const mesh::Field<int>& constField = field;
    const int* constData = constField.data();
    EXPECT_EQ(constData[0], 10);
    EXPECT_EQ(constData[field.size() - 1], 20);
}

// Test 5: Test boundary indices
TEST_F(FieldTest, BoundaryIndices) {
    mesh::Field<char> field(*mesh, 'x');
    
    // Modify boundary elements
    field(0, 0) = 'a';                 // Bottom-left corner
    field(field.nx() - 1, 0) = 'b';    // Bottom-right corner
    field(0, field.ny() - 1) = 'c';    // Top-left corner
    field(field.nx() - 1, field.ny() - 1) = 'd';  // Top-right corner
    
    // Verify boundary elements
    EXPECT_EQ(field(0, 0), 'a');
    EXPECT_EQ(field(field.nx() - 1, 0), 'b');
    EXPECT_EQ(field(0, field.ny() - 1), 'c');
    EXPECT_EQ(field(field.nx() - 1, field.ny() - 1), 'd');
}

// Test 6: Test with custom structure
struct Particle {
    double x, y;
    double mass;
    
    Particle(double x = 0.0, double y = 0.0, double m = 1.0) : x(x), y(y), mass(m) {}
    bool operator==(const Particle& other) const {
        return x == other.x && y == other.y && mass == other.mass;
    }
};

TEST_F(FieldTest, CustomStructureStorage) {
    Particle defaultParticle(0.0, 0.0, 1.0);
    mesh::Field<Particle> field(*mesh, defaultParticle);
    
    // Modify some particles
    field(1, 1) = Particle(1.5, 2.5, 3.5);
    field(2, 3) = Particle(4.5, 5.5, 6.5);
    
    // Verify modifications
    EXPECT_EQ(field(1, 1), Particle(1.5, 2.5, 3.5));
    EXPECT_EQ(field(2, 3), Particle(4.5, 5.5, 6.5));
    
    // All other elements should have the default value
    EXPECT_EQ(field(0, 0), defaultParticle);
    EXPECT_EQ(field(5, 5), defaultParticle);
}

// Test 7: Test with row-major storage explicitly
TEST_F(FieldTest, RowMajorStorage) {
    mesh::Field<int> field(*mesh, 0);
    
    // Linearize indices manually with row-major formula
    for (uint32_t j = 0; j < field.ny(); ++j) {
        for (uint32_t i = 0; i < field.nx(); ++i) {
            int linearIndex = i + j * field.nx();
            field.data()[linearIndex] = i + j;
        }
    }
    
    // Verify through 2D access
    for (uint32_t j = 0; j < field.ny(); ++j) {
        for (uint32_t i = 0; i < field.nx(); ++i) {
            EXPECT_EQ(field(i, j), i + j);
        }
    }
}

// Test 8: Test field resizing by rebinding to new mesh
TEST_F(FieldTest, MeshRebinding) {
    mesh::Field<double> field(*mesh, 1.0);
    
    // Create a new, smaller mesh
    mesh::Mesh smallerMesh(5, 5);
    
    // Create a new field with the smaller mesh
    mesh::Field<double> newField(smallerMesh, 2.0);
    
    // Check dimensions
    EXPECT_EQ(newField.nx(), 5u);
    EXPECT_EQ(newField.ny(), 5u);
    EXPECT_EQ(newField.size(), 25u);
    
    // Check all elements have the new default value
    for (uint32_t j = 0; j < newField.ny(); ++j) {
        for (uint32_t i = 0; i < newField.nx(); ++i) {
            EXPECT_DOUBLE_EQ(newField(i, j), 2.0);
        }
    }
}
