#include <gtest/gtest.h>
#include "units/core.h"
#include "units/si_units.h"
#include "units/bounds.h"

using namespace units;

TEST(UnitsBounds, Construction) {
    // Default constructor (zeros)
    PhysicalBounds<meter> defaultBounds;
    EXPECT_DOUBLE_EQ(0.0, defaultBounds.minX().value());
    EXPECT_DOUBLE_EQ(0.0, defaultBounds.minY().value());
    EXPECT_DOUBLE_EQ(0.0, defaultBounds.maxX().value());
    EXPECT_DOUBLE_EQ(0.0, defaultBounds.maxY().value());
    
    // Constructor with explicit values
    PhysicalBounds<meter> explicitBounds(
        meter(1.0), meter(2.0), meter(5.0), meter(6.0)
    );
    EXPECT_DOUBLE_EQ(1.0, explicitBounds.minX().value());
    EXPECT_DOUBLE_EQ(2.0, explicitBounds.minY().value());
    EXPECT_DOUBLE_EQ(5.0, explicitBounds.maxX().value());
    EXPECT_DOUBLE_EQ(6.0, explicitBounds.maxY().value());
    
    // Constructor with mixed unit types (should not compile)
    // PhysicalBounds<meter> mixedBounds(
    //     meter(1.0), centimeter(200.0), meter(5.0), meter(6.0)
    // );
    
    // Using the type alias
    Bounds2D meterBounds(meter(10.0), meter(20.0), meter(30.0), meter(40.0));
    EXPECT_DOUBLE_EQ(10.0, meterBounds.minX().value());
    EXPECT_DOUBLE_EQ(20.0, meterBounds.minY().value());
    EXPECT_DOUBLE_EQ(30.0, meterBounds.maxX().value());
    EXPECT_DOUBLE_EQ(40.0, meterBounds.maxY().value());
}

TEST(UnitsBounds, DimensionsCalculation) {
    // Test width and height calculations
    PhysicalBounds<meter> bounds(
        meter(10.0), meter(20.0), meter(30.0), meter(50.0)
    );
    
    // Width calculation (maxX - minX)
    EXPECT_DOUBLE_EQ(20.0, bounds.width().value());  // 30 - 10 = 20
    
    // Height calculation (maxY - minY)
    EXPECT_DOUBLE_EQ(30.0, bounds.height().value());  // 50 - 20 = 30
    
    // Area calculation (width * height)
    EXPECT_DOUBLE_EQ(600.0, bounds.area());  // 20 * 30 = 600
    
    // Zero dimension bounds
    PhysicalBounds<meter> zeroDimension(
        meter(10.0), meter(20.0), meter(10.0), meter(20.0)
    );
    EXPECT_DOUBLE_EQ(0.0, zeroDimension.width().value());
    EXPECT_DOUBLE_EQ(0.0, zeroDimension.height().value());
    EXPECT_DOUBLE_EQ(0.0, zeroDimension.area());
    
    // Negative dimension bounds (valid use case)
    PhysicalBounds<meter> negativeDimension(
        meter(30.0), meter(50.0), meter(10.0), meter(20.0)
    );
    EXPECT_DOUBLE_EQ(-20.0, negativeDimension.width().value());  // 10 - 30 = -20
    EXPECT_DOUBLE_EQ(-30.0, negativeDimension.height().value());  // 20 - 50 = -30
    EXPECT_DOUBLE_EQ(600.0, negativeDimension.area());  // abs(-20) * abs(-30) = 600
}

TEST(UnitsBounds, PointContainment) {
    // Test if a point is inside bounds
    PhysicalBounds<meter> bounds(
        meter(10.0), meter(20.0), meter(30.0), meter(50.0)
    );
    
    // Point inside the bounds
    EXPECT_TRUE(bounds.contains(meter(15.0), meter(30.0)));
    
    // Point on the boundary (considered inside)
    EXPECT_TRUE(bounds.contains(meter(10.0), meter(20.0)));  // min corner
    EXPECT_TRUE(bounds.contains(meter(30.0), meter(50.0)));  // max corner
    EXPECT_TRUE(bounds.contains(meter(10.0), meter(30.0)));  // left edge
    EXPECT_TRUE(bounds.contains(meter(20.0), meter(20.0)));  // bottom edge
    
    // Point outside the bounds
    EXPECT_FALSE(bounds.contains(meter(5.0), meter(30.0)));   // left
    EXPECT_FALSE(bounds.contains(meter(35.0), meter(30.0)));  // right
    EXPECT_FALSE(bounds.contains(meter(20.0), meter(15.0)));  // bottom
    EXPECT_FALSE(bounds.contains(meter(20.0), meter(55.0)));  // top
    EXPECT_FALSE(bounds.contains(meter(5.0), meter(15.0)));   // bottom-left
    
    // Edge case for zero dimension bounds
    PhysicalBounds<meter> zeroDimension(
        meter(10.0), meter(20.0), meter(10.0), meter(20.0)
    );
    EXPECT_TRUE(zeroDimension.contains(meter(10.0), meter(20.0)));  // exact point
    EXPECT_FALSE(zeroDimension.contains(meter(10.1), meter(20.0))); // slightly off
}

TEST(UnitsBounds, BoundsOverlap) {
    // Test if bounds overlap
    PhysicalBounds<meter> bounds1(
        meter(10.0), meter(20.0), meter(30.0), meter(50.0)
    );
    
    // Totally overlapping bounds (identical)
    PhysicalBounds<meter> bounds2(
        meter(10.0), meter(20.0), meter(30.0), meter(50.0)
    );
    EXPECT_TRUE(bounds1.overlaps(bounds2));
    EXPECT_TRUE(bounds2.overlaps(bounds1));  // should be commutative
    
    // Partially overlapping bounds
    PhysicalBounds<meter> bounds3(
        meter(20.0), meter(30.0), meter(40.0), meter(60.0)
    );
    EXPECT_TRUE(bounds1.overlaps(bounds3));
    EXPECT_TRUE(bounds3.overlaps(bounds1));
    
    // Bounds that touch at a point
    PhysicalBounds<meter> bounds4(
        meter(30.0), meter(50.0), meter(50.0), meter(70.0)
    );
    EXPECT_TRUE(bounds1.overlaps(bounds4));
    EXPECT_TRUE(bounds4.overlaps(bounds1));
    
    // Bounds that touch at an edge
    PhysicalBounds<meter> bounds5(
        meter(30.0), meter(20.0), meter(50.0), meter(50.0)
    );
    EXPECT_TRUE(bounds1.overlaps(bounds5));
    EXPECT_TRUE(bounds5.overlaps(bounds1));
    
    // Completely separate bounds
    PhysicalBounds<meter> bounds6(
        meter(40.0), meter(60.0), meter(50.0), meter(70.0)
    );
    EXPECT_FALSE(bounds1.overlaps(bounds6));
    EXPECT_FALSE(bounds6.overlaps(bounds1));
    
    // Zero dimension bounds (point)
    PhysicalBounds<meter> point(
        meter(15.0), meter(25.0), meter(15.0), meter(25.0)
    );
    EXPECT_TRUE(bounds1.overlaps(point));  // point within bounds1
    EXPECT_TRUE(point.overlaps(bounds1));
    
    // Zero dimension bounds (point) outside
    PhysicalBounds<meter> pointOutside(
        meter(5.0), meter(15.0), meter(5.0), meter(15.0)
    );
    EXPECT_FALSE(bounds1.overlaps(pointOutside));
    EXPECT_FALSE(pointOutside.overlaps(bounds1));
}

TEST(UnitsBounds, CenterCalculation) {
    // Test center point calculation
    PhysicalBounds<meter> bounds(
        meter(10.0), meter(20.0), meter(30.0), meter(60.0)
    );
    
    // Center X calculation
    EXPECT_DOUBLE_EQ(20.0, bounds.centerX().value());  // (10 + 30) / 2 = 20
    
    // Center Y calculation
    EXPECT_DOUBLE_EQ(40.0, bounds.centerY().value());  // (20 + 60) / 2 = 40
    
    // Zero dimension bounds (point)
    PhysicalBounds<meter> point(
        meter(15.0), meter(25.0), meter(15.0), meter(25.0)
    );
    EXPECT_DOUBLE_EQ(15.0, point.centerX().value());
    EXPECT_DOUBLE_EQ(25.0, point.centerY().value());
    
    // Negative dimension bounds
    PhysicalBounds<meter> negative(
        meter(30.0), meter(60.0), meter(10.0), meter(20.0)
    );
    EXPECT_DOUBLE_EQ(20.0, negative.centerX().value());  // (30 + 10) / 2 = 20
    EXPECT_DOUBLE_EQ(40.0, negative.centerY().value());  // (60 + 20) / 2 = 40
}

TEST(UnitsBounds, UnitConversion) {
    // Test unit conversion for bounds
    PhysicalBounds<meter> meterBounds(
        meter(1.0), meter(2.0), meter(5.0), meter(6.0)
    );
    
    // Convert to centimeters
    auto cmBounds = meterBounds.as<centimeter>();
    EXPECT_DOUBLE_EQ(100.0, cmBounds.minX().value());  // 1m = 100cm
    EXPECT_DOUBLE_EQ(200.0, cmBounds.minY().value());  // 2m = 200cm
    EXPECT_DOUBLE_EQ(500.0, cmBounds.maxX().value());  // 5m = 500cm
    EXPECT_DOUBLE_EQ(600.0, cmBounds.maxY().value());  // 6m = 600cm
    
    // Convert to kilometers
    auto kmBounds = meterBounds.as<kilometer>();
    EXPECT_DOUBLE_EQ(0.001, kmBounds.minX().value());  // 1m = 0.001km
    EXPECT_DOUBLE_EQ(0.002, kmBounds.minY().value());  // 2m = 0.002km
    EXPECT_DOUBLE_EQ(0.005, kmBounds.maxX().value());  // 5m = 0.005km
    EXPECT_DOUBLE_EQ(0.006, kmBounds.maxY().value());  // 6m = 0.006km
    
    // Verify bounds properties after conversion
    EXPECT_DOUBLE_EQ(400.0, cmBounds.width().value());  // 500 - 100 = 400cm
    EXPECT_DOUBLE_EQ(400.0, cmBounds.height().value());  // 600 - 200 = 400cm
    EXPECT_DOUBLE_EQ(0.004, kmBounds.width().value());  // 0.005 - 0.001 = 0.004km
    EXPECT_DOUBLE_EQ(0.004, kmBounds.height().value());  // 0.006 - 0.002 = 0.004km
    
    // Area calculation after conversion
    EXPECT_DOUBLE_EQ(160000.0, cmBounds.area());  // 400cm * 400cm = 160000 cm²
    EXPECT_DOUBLE_EQ(0.000016, kmBounds.area());  // 0.004km * 0.004km = 0.000016 km²
    
    // Round-trip conversion
    auto backToMeter = cmBounds.as<meter>();
    EXPECT_DOUBLE_EQ(1.0, backToMeter.minX().value());
    EXPECT_DOUBLE_EQ(2.0, backToMeter.minY().value());
    EXPECT_DOUBLE_EQ(5.0, backToMeter.maxX().value());
    EXPECT_DOUBLE_EQ(6.0, backToMeter.maxY().value());
}

TEST(UnitsBounds, DifferentUnitTypes) {
    // Test bounds with different unit types
    
    // Bounds in centimeters
    PhysicalBounds<centimeter> cmBounds(
        centimeter(100.0), centimeter(200.0), 
        centimeter(500.0), centimeter(600.0)
    );
    
    // Width and height
    EXPECT_DOUBLE_EQ(400.0, cmBounds.width().value());
    EXPECT_DOUBLE_EQ(400.0, cmBounds.height().value());
    
    // Area
    EXPECT_DOUBLE_EQ(160000.0, cmBounds.area());
    
    // Containment test
    EXPECT_TRUE(cmBounds.contains(centimeter(300.0), centimeter(400.0)));
    EXPECT_FALSE(cmBounds.contains(centimeter(50.0), centimeter(400.0)));
    
    // Overlap test
    PhysicalBounds<centimeter> anotherCmBounds(
        centimeter(300.0), centimeter(400.0),
        centimeter(700.0), centimeter(800.0)
    );
    EXPECT_TRUE(cmBounds.overlaps(anotherCmBounds));
    
    // Convert to meters and test overlap
    auto meterBounds = cmBounds.as<meter>();
    EXPECT_TRUE(meterBounds.overlaps(anotherCmBounds.as<meter>()));
    
    // Millimeter bounds
    PhysicalBounds<millimeter> mmBounds(
        millimeter(1000.0), millimeter(2000.0),
        millimeter(5000.0), millimeter(6000.0)
    );
    
    // Make sure these are the same physical size (1m = 1000mm = 100cm)
    EXPECT_DOUBLE_EQ(cmBounds.width().value() * 10.0, mmBounds.width().value());
    EXPECT_DOUBLE_EQ(cmBounds.height().value() * 10.0, mmBounds.height().value());
}

TEST(UnitsBounds, EdgeCases) {
    // Test edge cases for bounds
    
    // Inverted bounds (min > max) - should still function
    PhysicalBounds<meter> invertedBounds(
        meter(30.0), meter(50.0), meter(10.0), meter(20.0)
    );
    
    // Width and height are negative
    EXPECT_DOUBLE_EQ(-20.0, invertedBounds.width().value());
    EXPECT_DOUBLE_EQ(-30.0, invertedBounds.height().value());
    
    // Area is still positive (uses absolute values)
    EXPECT_DOUBLE_EQ(600.0, invertedBounds.area());
    
    // Center calculates correctly for inverted bounds
    EXPECT_DOUBLE_EQ(20.0, invertedBounds.centerX().value());
    EXPECT_DOUBLE_EQ(35.0, invertedBounds.centerY().value());
    
    // Containment is reversed for inverted bounds
    EXPECT_FALSE(invertedBounds.contains(meter(20.0), meter(30.0)));
    
    // Point bounds (single point)
    PhysicalBounds<meter> pointBounds(
        meter(10.0), meter(20.0), meter(10.0), meter(20.0)
    );
    
    EXPECT_DOUBLE_EQ(0.0, pointBounds.width().value());
    EXPECT_DOUBLE_EQ(0.0, pointBounds.height().value());
    EXPECT_DOUBLE_EQ(0.0, pointBounds.area());
    EXPECT_DOUBLE_EQ(10.0, pointBounds.centerX().value());
    EXPECT_DOUBLE_EQ(20.0, pointBounds.centerY().value());
    
    // Self-containment for point bounds
    EXPECT_TRUE(pointBounds.contains(meter(10.0), meter(20.0)));
    
    // Self-overlap for point bounds
    EXPECT_TRUE(pointBounds.overlaps(pointBounds));
    
    // Extremely large bounds (testing limits)
    PhysicalBounds<kilometer> largeBounds(
        kilometer(-1e6), kilometer(-1e6),
        kilometer(1e6), kilometer(1e6)
    );
    
    EXPECT_DOUBLE_EQ(2e6, largeBounds.width().value());
    EXPECT_DOUBLE_EQ(2e6, largeBounds.height().value());
    EXPECT_DOUBLE_EQ(4e12, largeBounds.area());
}

TEST(UnitsBounds, TypeAliases) {
    // Test the predefined type aliases
    
    // Bounds2D is PhysicalBounds<meter>
    Bounds2D bounds(
        meter(10.0), meter(20.0), meter(30.0), meter(40.0)
    );
    
    EXPECT_DOUBLE_EQ(10.0, bounds.minX().value());
    EXPECT_DOUBLE_EQ(20.0, bounds.minY().value());
    EXPECT_DOUBLE_EQ(30.0, bounds.maxX().value());
    EXPECT_DOUBLE_EQ(40.0, bounds.maxY().value());
    
    EXPECT_DOUBLE_EQ(20.0, bounds.width().value());
    EXPECT_DOUBLE_EQ(20.0, bounds.height().value());
    EXPECT_DOUBLE_EQ(400.0, bounds.area());
    
    // Converting Bounds2D to other units
    auto kmBounds = bounds.as<kilometer>();
    EXPECT_DOUBLE_EQ(0.01, kmBounds.minX().value());
    EXPECT_DOUBLE_EQ(0.02, kmBounds.minY().value());
    EXPECT_DOUBLE_EQ(0.03, kmBounds.maxX().value());
    EXPECT_DOUBLE_EQ(0.04, kmBounds.maxY().value());
}
