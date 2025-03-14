#include "gtest/gtest.h"
#include "Field.h"
#include "Mesh.h"
#include "Cell.h"
#include "CellIterators.h"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <vector>

// Test fixture for Field integration tests with other components
class FieldIntegrationTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create a mesh for testing
        mesh = new mesh::Mesh(10, 10);
    }

    void TearDown() override {
        delete mesh;
    }

    mesh::Mesh* mesh;
};

// Test 1: Using Field with Cell iterators
TEST_F(FieldIntegrationTest, FieldWithCellIterators) {
    mesh::Field<double> field(*mesh, 0.0);
    
    // Initialize field with a pattern
    for (uint32_t j = 0; j < field.ny(); ++j) {
        for (uint32_t i = 0; i < field.nx(); ++i) {
            field(i, j) = i * 10.0 + j;
        }
    }
    
    // Create a range for all cells in the mesh
    auto range = mesh->cells();
    
    // Use iterator to sum field values
    double sum = 0.0;
    for (auto it = range.begin(); it != range.end(); ++it) {
        mesh::Cell cell = *it;
        sum += field(cell.i(), cell.j());
    }
    
    // Calculate expected sum directly
    double expectedSum = 0.0;
    for (uint32_t j = 0; j < field.ny(); ++j) {
        for (uint32_t i = 0; i < field.nx(); ++i) {
            expectedSum += field(i, j);
        }
    }
    
    EXPECT_DOUBLE_EQ(sum, expectedSum);
}

// Test 2: Using range-based for loops with Field and Cells
TEST_F(FieldIntegrationTest, RangeBasedLoopsWithField) {
    mesh::Field<double> field(*mesh, 0.0);
    
    // Set field values using cell iterators
    auto cellRange = mesh->cells();
    for (const auto& cell : cellRange) {
        field(cell.i(), cell.j()) = cell.i() + cell.j();
    }
    
    // Verify with direct access
    for (uint32_t j = 0; j < field.ny(); ++j) {
        for (uint32_t i = 0; i < field.nx(); ++i) {
            EXPECT_DOUBLE_EQ(field(i, j), i + j);
        }
    }
}

// Test 3: Interior and boundary cell operations with Field
TEST_F(FieldIntegrationTest, InteriorAndBoundaryCells) {
    mesh::Field<double> field(*mesh, 0.0);
    
    // Set boundary cells to 1.0, interior cells to 2.0
    auto boundaryRange = mesh->boundaryCells();
    for (const auto& cell : boundaryRange) {
        field(cell.i(), cell.j()) = 1.0;
    }
    
    auto interiorRange = mesh->interiorCells();
    for (const auto& cell : interiorRange) {
        field(cell.i(), cell.j()) = 2.0;
    }
    
    // Verify with cell boundary check
    for (uint32_t j = 0; j < field.ny(); ++j) {
        for (uint32_t i = 0; i < field.nx(); ++i) {
            mesh::Cell cell = mesh->getCell(i, j);
            if (cell.isBoundary()) {
                EXPECT_DOUBLE_EQ(field(i, j), 1.0);
            } else {
                EXPECT_DOUBLE_EQ(field(i, j), 2.0);
            }
        }
    }
}

// Test 4: Implementing basic operations across fields
TEST_F(FieldIntegrationTest, BasicFieldOperations) {
    mesh::Field<double> fieldA(*mesh, 2.0);
    mesh::Field<double> fieldB(*mesh, 3.0);
    mesh::Field<double> resultField(*mesh, 0.0);
    
    // Implement field addition
    for (uint32_t j = 0; j < fieldA.ny(); ++j) {
        for (uint32_t i = 0; i < fieldA.nx(); ++i) {
            resultField(i, j) = fieldA(i, j) + fieldB(i, j);
        }
    }
    
    // Verify result
    for (uint32_t j = 0; j < resultField.ny(); ++j) {
        for (uint32_t i = 0; i < resultField.nx(); ++i) {
            EXPECT_DOUBLE_EQ(resultField(i, j), 5.0);
        }
    }
    
    // Implement field multiplication
    for (uint32_t j = 0; j < fieldA.ny(); ++j) {
        for (uint32_t i = 0; i < fieldA.nx(); ++i) {
            resultField(i, j) = fieldA(i, j) * fieldB(i, j);
        }
    }
    
    // Verify result
    for (uint32_t j = 0; j < resultField.ny(); ++j) {
        for (uint32_t i = 0; i < resultField.nx(); ++i) {
            EXPECT_DOUBLE_EQ(resultField(i, j), 6.0);
        }
    }
}

// Test 5: Implementing stencil operations with Field
TEST_F(FieldIntegrationTest, StencilOperations) {
    mesh::Field<double> inputField(*mesh, 0.0);
    mesh::Field<double> outputField(*mesh, 0.0);
    
    // Set input field to a pattern
    for (uint32_t j = 0; j < inputField.ny(); ++j) {
        for (uint32_t i = 0; i < inputField.nx(); ++i) {
            inputField(i, j) = i + j;
        }
    }
    
    // Apply a 5-point stencil for Laplacian approximation
    auto interiorRange = mesh->interiorCells();
    for (const auto& cell : interiorRange) {
        int i = cell.i();
        int j = cell.j();
        
        // Apply 5-point stencil: (u_{i+1,j} + u_{i-1,j} + u_{i,j+1} + u_{i,j-1} - 4*u_{i,j})
        outputField(i, j) = inputField(i+1, j) + inputField(i-1, j) +
                             inputField(i, j+1) + inputField(i, j-1) -
                             4 * inputField(i, j);
    }
    
    // Verify Laplacian of a linear function should be zero (within numerical precision)
    for (const auto& cell : interiorRange) {
        int i = cell.i();
        int j = cell.j();
        EXPECT_NEAR(outputField(i, j), 0.0, 1e-10);
    }
}

// Test 6: Using Cell neighbor functionality with Field
TEST_F(FieldIntegrationTest, CellNeighbors) {
    mesh::Field<int> field(*mesh, 0);
    
    // Set field values
    for (uint32_t j = 0; j < field.ny(); ++j) {
        for (uint32_t i = 0; i < field.nx(); ++i) {
            field(i, j) = i * 100 + j;
        }
    }
    
    // Get a cell in the middle of the mesh
    mesh::Cell centerCell = mesh->getCell(5, 5);
    
    // Check neighbor values
    mesh::Cell northNeighbor = centerCell.neighbor(mesh::NORTH);
    EXPECT_EQ(field(northNeighbor.i(), northNeighbor.j()), 5 * 100 + 6);
    
    mesh::Cell eastNeighbor = centerCell.neighbor(mesh::EAST);
    EXPECT_EQ(field(eastNeighbor.i(), eastNeighbor.j()), 6 * 100 + 5);
    
    mesh::Cell southNeighbor = centerCell.neighbor(mesh::SOUTH);
    EXPECT_EQ(field(southNeighbor.i(), southNeighbor.j()), 5 * 100 + 4);
    
    mesh::Cell westNeighbor = centerCell.neighbor(mesh::WEST);
    EXPECT_EQ(field(westNeighbor.i(), westNeighbor.j()), 4 * 100 + 5);
    
    // Test a diagonal neighbor
    mesh::Cell northEastNeighbor = centerCell.neighbor(mesh::NORTH | mesh::EAST);
    EXPECT_EQ(field(northEastNeighbor.i(), northEastNeighbor.j()), 6 * 100 + 6);
}

// Test 7: Using rectangular region iteration with Field
TEST_F(FieldIntegrationTest, RectangularRegion) {
    mesh::Field<double> field(*mesh, 0.0);
    
    // Set field values
    for (uint32_t j = 0; j < field.ny(); ++j) {
        for (uint32_t i = 0; i < field.nx(); ++i) {
            field(i, j) = i * j;
        }
    }
    
    // Define a rectangular region
    uint32_t startI = 2;
    uint32_t startJ = 3;
    uint32_t endI = 7;
    uint32_t endJ = 8;
    
    auto regionRange = mesh->cellsInRegion(startI, startJ, endI, endJ);
    
    // Calculate sum of values in the region
    double regionSum = 0.0;
    for (const auto& cell : regionRange) {
        regionSum += field(cell.i(), cell.j());
    }
    
    // Calculate expected sum directly
    double expectedSum = 0.0;
    for (uint32_t j = startJ; j < endJ; ++j) {
        for (uint32_t i = startI; i < endI; ++i) {
            expectedSum += field(i, j);
        }
    }
    
    EXPECT_DOUBLE_EQ(regionSum, expectedSum);
}

// Test 8: Building a complex simulation kernel with Field
TEST_F(FieldIntegrationTest, ComplexSimulationKernel) {
    // Define fields for a simple fluid simulation
    mesh::Field<double> pressure(*mesh, 0.0);
    mesh::Field<double> velocityX(*mesh, 0.0);
    mesh::Field<double> velocityY(*mesh, 0.0);
    mesh::Field<double> density(*mesh, 1.0);
    
    // Initialize with test values
    for (uint32_t j = 0; j < mesh->ny(); ++j) {
        for (uint32_t i = 0; i < mesh->nx(); ++i) {
            // Pressure gradient in x direction
            pressure(i, j) = static_cast<double>(i) / mesh->nx();
            
            // Initial velocity field
            velocityX(i, j) = 0.1 * std::sin(M_PI * static_cast<double>(j) / mesh->ny());
            velocityY(i, j) = 0.1 * std::cos(M_PI * static_cast<double>(i) / mesh->nx());
            
            // Density variation
            density(i, j) = 1.0 + 0.1 * std::sin(M_PI * static_cast<double>(i + j) / (mesh->nx() + mesh->ny()));
        }
    }
    
    // Create fields for the next time step
    mesh::Field<double> newVelocityX(*mesh, 0.0);
    mesh::Field<double> newVelocityY(*mesh, 0.0);
    
    // Simulate one time step with a simple explicit scheme (for interior cells)
    // This represents a simplified momentum equation without viscosity
    double dt = 0.01;    // Time step
    double dx = 1.0;     // Assumed grid spacing
    
    auto interiorRange = mesh->interiorCells();
    for (const auto& cell : interiorRange) {
        int i = cell.i();
        int j = cell.j();
        
        // Pressure gradient force
        double pressureGradX = (pressure(i+1, j) - pressure(i-1, j)) / (2 * dx);
        double pressureGradY = (pressure(i, j+1) - pressure(i, j-1)) / (2 * dx);
        
        // Advection term (simplified)
        double advectionX = velocityX(i, j) * (velocityX(i+1, j) - velocityX(i-1, j)) / (2 * dx) +
                           velocityY(i, j) * (velocityX(i, j+1) - velocityX(i, j-1)) / (2 * dx);
        
        double advectionY = velocityX(i, j) * (velocityY(i+1, j) - velocityY(i-1, j)) / (2 * dx) +
                           velocityY(i, j) * (velocityY(i, j+1) - velocityY(i, j-1)) / (2 * dx);
        
        // Update velocities
        newVelocityX(i, j) = velocityX(i, j) - dt * (pressureGradX / density(i, j) + advectionX);
        newVelocityY(i, j) = velocityY(i, j) - dt * (pressureGradY / density(i, j) + advectionY);
    }
    
    // Basic verification: Check conservation of kinetic energy (approximately)
    double initialEnergy = 0.0;
    double finalEnergy = 0.0;
    
    for (const auto& cell : interiorRange) {
        int i = cell.i();
        int j = cell.j();
        
        double vxSquared = velocityX(i, j) * velocityX(i, j);
        double vySquared = velocityY(i, j) * velocityY(i, j);
        initialEnergy += 0.5 * density(i, j) * (vxSquared + vySquared);
        
        vxSquared = newVelocityX(i, j) * newVelocityX(i, j);
        vySquared = newVelocityY(i, j) * newVelocityY(i, j);
        finalEnergy += 0.5 * density(i, j) * (vxSquared + vySquared);
    }
    
    // Energy should be approximately conserved for this simple scheme
    // The difference should be small for a short time step
    EXPECT_NEAR(finalEnergy / initialEnergy, 1.0, 0.1);
}
