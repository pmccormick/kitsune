#include "Cell.h"
#include "Material.h"
#include <gtest/gtest.h>
#include <memory>

class CellTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create a test material
        testMaterial = std::make_shared<Material>(Material::MaterialType::FLUID, "TestMaterial");
        testMaterial->setProperty(Material::MaterialProperty::DENSITY, 1000.0);
        testMaterial->setProperty(Material::MaterialProperty::DYNAMIC_VISCOSITY, 0.001);
    }

    std::shared_ptr<Material> testMaterial;
};

// Test basic cell properties
TEST_F(CellTest, BasicProperties) {
    Cell cell;
    
    // Test default values
    EXPECT_EQ(cell.getType(), Cell::CellType::FLUID);
    EXPECT_DOUBLE_EQ(cell.getTemperature(), 293.15);
    EXPECT_DOUBLE_EQ(cell.getPressure(), 101325.0);
    EXPECT_DOUBLE_EQ(cell.getDensity(), 1.0);
    EXPECT_FALSE(cell.isFixed());
    
    // Test setters and getters
    cell.setType(Cell::CellType::SOLID);
    EXPECT_EQ(cell.getType(), Cell::CellType::SOLID);
    
    cell.setTemperature(350.0);
    EXPECT_DOUBLE_EQ(cell.getTemperature(), 350.0);
    
    cell.setPressure(200000.0);
    EXPECT_DOUBLE_EQ(cell.getPressure(), 200000.0);
    
    cell.setDensity(2.5);
    EXPECT_DOUBLE_EQ(cell.getDensity(), 2.5);
    
    cell.setFixed(true);
    EXPECT_TRUE(cell.isFixed());
    
    // Test material association
    EXPECT_EQ(cell.getMaterial(), nullptr);
    cell.setMaterial(testMaterial);
    EXPECT_EQ(cell.getMaterial(), testMaterial);
}

// Test vertex velocity operations
TEST_F(CellTest, VertexVelocities) {
    Cell cell;
    
    // Test default velocity (should be zero)
    auto vel = cell.getVertexVelocity(Cell::VertexPosition::NORTHWEST);
    EXPECT_DOUBLE_EQ(vel.first, 0.0);
    EXPECT_DOUBLE_EQ(vel.second, 0.0);
    
    // Test setting and getting velocity
    cell.setVertexVelocity(Cell::VertexPosition::NORTHWEST, 10.0, 5.0);
    vel = cell.getVertexVelocity(Cell::VertexPosition::NORTHWEST);
    EXPECT_DOUBLE_EQ(vel.first, 10.0);
    EXPECT_DOUBLE_EQ(vel.second, 5.0);
    
    // Test all vertices have independent velocities
    cell.setVertexVelocity(Cell::VertexPosition::NORTHEAST, -10.0, 5.0);
    cell.setVertexVelocity(Cell::VertexPosition::SOUTHEAST, -10.0, -5.0);
    cell.setVertexVelocity(Cell::VertexPosition::SOUTHWEST, 10.0, -5.0);
    
    auto velNW = cell.getVertexVelocity(Cell::VertexPosition::NORTHWEST);
    auto velNE = cell.getVertexVelocity(Cell::VertexPosition::NORTHEAST);
    auto velSE = cell.getVertexVelocity(Cell::VertexPosition::SOUTHEAST);
    auto velSW = cell.getVertexVelocity(Cell::VertexPosition::SOUTHWEST);
    
    EXPECT_DOUBLE_EQ(velNW.first, 10.0);
    EXPECT_DOUBLE_EQ(velNW.second, 5.0);
    EXPECT_DOUBLE_EQ(velNE.first, -10.0);
    EXPECT_DOUBLE_EQ(velNE.second, 5.0);
    EXPECT_DOUBLE_EQ(velSE.first, -10.0);
    EXPECT_DOUBLE_EQ(velSE.second, -5.0);
    EXPECT_DOUBLE_EQ(velSW.first, 10.0);
    EXPECT_DOUBLE_EQ(velSW.second, -5.0);
    
    // Test direct vertex access
    Cell::Vertex& nwVertex = cell.getVertex(Cell::VertexPosition::NORTHWEST);
    EXPECT_DOUBLE_EQ(nwVertex.vx, 10.0);
    EXPECT_DOUBLE_EQ(nwVertex.vy, 5.0);
    
    // Modify through direct access
    nwVertex.vx = 15.0;
    nwVertex.vy = 7.5;
    
    // Verify changes reflected in get methods
    vel = cell.getVertexVelocity(Cell::VertexPosition::NORTHWEST);
    EXPECT_DOUBLE_EQ(vel.first, 15.0);
    EXPECT_DOUBLE_EQ(vel.second, 7.5);
}

// Test fixed properties
TEST_F(CellTest, FixedProperties) {
    Cell cell;
    
    // Test default values (should be zero)
    EXPECT_DOUBLE_EQ(cell.getProperty(Cell::PropertyType::VORTICITY), 0.0);
    EXPECT_DOUBLE_EQ(cell.getProperty(Cell::PropertyType::STREAM_FUNCTION), 0.0);
    
    // Test setting and getting properties
    cell.setProperty(Cell::PropertyType::VORTICITY, 0.25);
    EXPECT_DOUBLE_EQ(cell.getProperty(Cell::PropertyType::VORTICITY), 0.25);
    
    cell.setProperty(Cell::PropertyType::STREAM_FUNCTION, -1.5);
    EXPECT_DOUBLE_EQ(cell.getProperty(Cell::PropertyType::STREAM_FUNCTION), -1.5);
    
    // Test all properties are independent
    for (int i = 0; i < static_cast<int>(Cell::PropertyType::COUNT); ++i) {
        Cell::PropertyType propType = static_cast<Cell::PropertyType>(i);
        double value = i * 2.5;
        cell.setProperty(propType, value);
        EXPECT_DOUBLE_EQ(cell.getProperty(propType), value);
    }
}

// Test cell flags
TEST_F(CellTest, CellFlags) {
    Cell cell;
    
    // Test default flags (should be false)
    EXPECT_FALSE(cell.getFlag(Cell::CellFlag::IS_INLET));
    EXPECT_FALSE(cell.getFlag(Cell::CellFlag::NEEDS_UPDATE));
    EXPECT_FALSE(cell.getFlag(Cell::CellFlag::HAS_CONVERGED));
    
    // Test setting and getting flags
    cell.setFlag(Cell::CellFlag::IS_INLET, true);
    EXPECT_TRUE(cell.getFlag(Cell::CellFlag::IS_INLET));
    EXPECT_FALSE(cell.getFlag(Cell::CellFlag::IS_OUTLET)); // Other flags unchanged
    
    cell.setFlag(Cell::CellFlag::NEEDS_UPDATE, true);
    EXPECT_TRUE(cell.getFlag(Cell::CellFlag::NEEDS_UPDATE));
    
    // Test toggling flags
    cell.setFlag(Cell::CellFlag::IS_INLET, false);
    EXPECT_FALSE(cell.getFlag(Cell::CellFlag::IS_INLET));
    EXPECT_TRUE(cell.getFlag(Cell::CellFlag::NEEDS_UPDATE)); // Other flags unchanged
    
    // Test all flags independently
    for (int i = 0; i < static_cast<int>(Cell::CellFlag::COUNT); ++i) {
        Cell::CellFlag flag = static_cast<Cell::CellFlag>(i);
        
        // Set to true
        cell.setFlag(flag, true);
        EXPECT_TRUE(cell.getFlag(flag));
        
        // Set to false
        cell.setFlag(flag, false);
        EXPECT_FALSE(cell.getFlag(flag));
    }
}

// Test dynamic properties
TEST_F(CellTest, DynamicProperties) {
    Cell cell;
    
    // Test getting non-existent property returns default
    EXPECT_DOUBLE_EQ(cell.getDynamicProperty("nonexistent"), 0.0);
    EXPECT_DOUBLE_EQ(cell.getDynamicProperty("nonexistent", 3.14), 3.14);
    
    // Test setting and getting custom properties
    cell.setDynamicProperty("custom1", 123.45);
    EXPECT_DOUBLE_EQ(cell.getDynamicProperty("custom1"), 123.45);
    
    cell.setDynamicProperty("custom2", -67.89);
    EXPECT_DOUBLE_EQ(cell.getDynamicProperty("custom2"), -67.89);
    
    // Test updating existing property
    cell.setDynamicProperty("custom1", 999.99);
    EXPECT_DOUBLE_EQ(cell.getDynamicProperty("custom1"), 999.99);
    
    // Test special characters in property names
    cell.setDynamicProperty("property-with.special+chars", 42.0);
    EXPECT_DOUBLE_EQ(cell.getDynamicProperty("property-with.special+chars"), 42.0);
}

// Test cell reset
TEST_F(CellTest, Reset) {
    Cell cell;
    
    // Set some non-default values
    cell.setType(Cell::CellType::SOLID);
    cell.setTemperature(350.0);
    cell.setPressure(200000.0);
    cell.setDensity(2.5);
    cell.setMaterial(testMaterial);
    cell.setFixed(true);
    cell.setVertexVelocity(Cell::VertexPosition::NORTHWEST, 10.0, 5.0);
    cell.setProperty(Cell::PropertyType::VORTICITY, 0.25);
    cell.setFlag(Cell::CellFlag::NEEDS_UPDATE, true);
    cell.setDynamicProperty("custom", 123.45);
    
    // Reset the cell
    cell.reset();
    
    // Check default values are restored except type and material
    EXPECT_EQ(cell.getType(), Cell::CellType::SOLID); // Type should not be reset
    EXPECT_DOUBLE_EQ(cell.getTemperature(), 293.15);
    EXPECT_DOUBLE_EQ(cell.getPressure(), 101325.0);
    EXPECT_DOUBLE_EQ(cell.getDensity(), 1.0);
    EXPECT_EQ(cell.getMaterial(), testMaterial); // Material should not be reset
    
    // Check velocities are reset
    auto vel = cell.getVertexVelocity(Cell::VertexPosition::NORTHWEST);
    EXPECT_DOUBLE_EQ(vel.first, 0.0);
    EXPECT_DOUBLE_EQ(vel.second, 0.0);
    
    // Check properties are reset
    EXPECT_DOUBLE_EQ(cell.getProperty(Cell::PropertyType::VORTICITY), 0.0);
    
    // Check flags are reset
    EXPECT_FALSE(cell.getFlag(Cell::CellFlag::NEEDS_UPDATE));
    
    // Check dynamic properties are reset
    EXPECT_DOUBLE_EQ(cell.getDynamicProperty("custom"), 0.0);
}

// Test constructor with cell type
TEST_F(CellTest, TypeConstructor) {
    // Test fluid cell constructor
    Cell fluidCell(Cell::CellType::FLUID);
    EXPECT_EQ(fluidCell.getType(), Cell::CellType::FLUID);
    EXPECT_FALSE(fluidCell.isFixed());
    
    // Test solid cell constructor
    Cell solidCell(Cell::CellType::SOLID);
    EXPECT_EQ(solidCell.getType(), Cell::CellType::SOLID);
    EXPECT_TRUE(solidCell.isFixed()); // Solid cells are fixed by default
    
    // Test boundary cell constructor
    Cell boundaryCell(Cell::CellType::BOUNDARY);
    EXPECT_EQ(boundaryCell.getType(), Cell::CellType::BOUNDARY);
    EXPECT_TRUE(boundaryCell.isFixed()); // Boundary cells are fixed by default
    EXPECT_TRUE(boundaryCell.getFlag(Cell::CellFlag::IS_WALL)); // Default boundary is wall
}

