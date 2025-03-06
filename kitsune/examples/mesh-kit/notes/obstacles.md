# CFD Obstacle Creation Functions Documentation

This document provides comprehensive information about the obstacle creation functions available in the MeshUtils namespace, their usage patterns, and guidance on when to use each approach.

## Table of Contents

1. [Introduction](#introduction)
2. [Basic Geometric Obstacles](#basic-geometric-obstacles)
3. [Practical Flow Obstacles](#practical-flow-obstacles)
4. [Image-Based Obstacles](#image-based-obstacles)
5. [Function-Based Obstacles](#function-based-obstacles)
6. [Common Test Cases](#common-test-cases)
7. [Performance Considerations](#performance-considerations)
8. [References](#references)

## Introduction

Obstacles in CFD simulations define regions within the computational domain where fluid cannot flow. These are essential for modeling:
- Solid objects immersed in fluid flow
- Domain boundaries
- Complex geometries for realistic simulations
- Standard test cases for validation

The MeshUtils namespace provides functions for creating various types of obstacles, from simple geometric shapes to complex custom geometries defined by images or mathematical functions.

## Basic Geometric Obstacles

### Rectangular Obstacles

**Function:** `createRectangularObstacle`

**Purpose:** Creates a rectangular or square obstacle defined by two corner points.

**Usage:**
```cpp
MeshUtils::createRectangularObstacle(mesh, 
                                    0.2, 0.2,    // Bottom-left corner (x1,y1)
                                    0.8, 0.5,    // Top-right corner (x2,y2)
                                    material,    // Optional material
                                    true);       // Fill interior (true) or outline only (false)
```

**When to use:**
- Simple geometries like channels, ducts, and chambers
- Blockages or baffles in flow
- Domain boundaries
- As building blocks for more complex shapes

### Circular Obstacles

**Function:** `createCircularObstacle`

**Purpose:** Creates a circular obstacle defined by center point and radius.

**Usage:**
```cpp
MeshUtils::createCircularObstacle(mesh, 
                                 0.5, 0.5,    // Center (x,y)
                                 0.25,        // Radius
                                 material,    // Optional material
                                 true);       // Fill interior (true) or outline only (false)
```

**When to use:**
- Flow around cylinders or spheres (in 2D)
- Circular vents or openings
- Curved obstacles with constant radius
- Point sources or sinks

### Elliptical Obstacles

**Function:** `createEllipticalObstacle`

**Purpose:** Creates an elliptical obstacle with variable axes and rotation angle.

**Usage:**
```cpp
MeshUtils::createEllipticalObstacle(mesh, 
                                   0.5, 0.5,       // Center (x,y)
                                   0.3, 0.15,      // X and Y radii
                                   M_PI/4,         // Rotation angle (radians)
                                   material,       // Optional material
                                   true);          // Fill interior (true) or outline only (false)
```

**When to use:**
- Streamlined bodies with different aspect ratios
- Angled or rotated elliptical features
- Approximations for aerodynamic shapes
- Curved channels with varying width

### Triangular Obstacles

**Function:** `createTriangularObstacle`

**Purpose:** Creates a triangular obstacle defined by three vertices.

**Usage:**
```cpp
MeshUtils::createTriangularObstacle(mesh, 
                                   0.2, 0.2,    // First vertex (x1,y1)
                                   0.8, 0.2,    // Second vertex (x2,y2)
                                   0.5, 0.8,    // Third vertex (x3,y3)
                                   material,    // Optional material
                                   true);       // Fill interior (true) or outline only (false)
```

**When to use:**
- Wedges and triangular structures
- Prisms and delta wings
- Splitters and vanes
- Building blocks for complex polygonal shapes

### Polygonal Obstacles

**Function:** `createPolygonalObstacle`

**Purpose:** Creates an arbitrary polygonal obstacle defined by a sequence of vertices.

**Usage:**
```cpp
std::vector<std::pair<double, double>> vertices = {
    {0.2, 0.2}, {0.8, 0.2}, {0.8, 0.5}, {0.6, 0.8}, {0.2, 0.8}
};
MeshUtils::createPolygonalObstacle(mesh, vertices, material, true);
```

**When to use:**
- Complex shapes with straight edges
- Custom geometries that can't be represented by standard shapes
- Shapes with many vertices or irregular geometries
- Imported geometry data from CAD systems

## Practical Flow Obstacles

### Airfoil Obstacles

**Function:** `createAirfoilObstacle`

**Purpose:** Creates an airfoil based on NACA 4-digit profile definitions.

**Usage:**
```cpp
MeshUtils::createAirfoilObstacle(mesh, 
                                0.2, 0.5,      // Leading edge position (x,y)
                                0.6,           // Chord length
                                5.0 * M_PI/180, // Angle of attack (radians)
                                "NACA0012",    // Airfoil profile
                                material,      // Optional material
                                true);         // Fill interior (true) or outline only (false)
```

**When to use:**
- Aerodynamic simulations
- Wing section analysis
- Blade profiles for turbomachinery
- Teaching and demonstration of flow separation and lift generation

**NACA Profile Information:**
- First digit: Maximum camber as percentage of chord
- Second digit: Position of maximum camber in tenths of chord
- Last two digits: Maximum thickness as percentage of chord
- Example: NACA0012 = 0% camber, 12% thickness

More information: [NACA Airfoil Series](https://en.wikipedia.org/wiki/NACA_airfoil)

### Backward-Facing Step

**Function:** `createBackwardFacingStep`

**Purpose:** Creates a backward-facing step configuration, a classic CFD test case.

**Usage:**
```cpp
MeshUtils::createBackwardFacingStep(mesh, 
                                   0.3,     // Step x-position
                                   0.1,     // Step height
                                   0.2,     // Inlet height
                                   0.7,     // Outlet length
                                   material); // Optional material
```

**When to use:**
- Studying flow separation and reattachment
- Benchmark tests for turbulence models
- Recirculation zone analysis
- Code validation against experimental data

**Reference:**
Armaly, B. F., et al. (1983), "Experimental and theoretical investigation of backward-facing step flow", Journal of Fluid Mechanics, vol. 127, pp. 473-496.

### Forward-Facing Step

**Function:** `createForwardFacingStep`

**Purpose:** Creates a forward-facing step configuration for flow studies.

**Usage:**
```cpp
MeshUtils::createForwardFacingStep(mesh, 
                                  0.3,     // Step x-position
                                  0.1,     // Step height
                                  0.3,     // Inlet length
                                  0.2,     // Outlet height
                                  material); // Optional material
```

**When to use:**
- Studying adverse pressure gradients
- Flow separation over obstacles
- Comparison with backward-facing step results
- Validation against experimental data

### Cavity

**Function:** `createCavity`

**Purpose:** Creates a cavity configuration, often used for the lid-driven cavity test case.

**Usage:**
```cpp
MeshUtils::createCavity(mesh, 
                       0.2, 0.2,    // Bottom-left corner (x1,y1)
                       0.8, 0.8,    // Top-right corner (x2,y2)
                       material);   // Optional material
```

**When to use:**
- Lid-driven cavity flow simulations (classic CFD benchmark)
- Vortex studies
- Code validation
- Teaching CFD principles

**Reference:**
Ghia, U., et al. (1982), "High-Re solutions for incompressible flow using the Navier-Stokes equations and a multigrid method", Journal of Computational Physics, vol. 48, pp. 387-411.

### Cylinder Obstacle

**Function:** `createCylinderObstacle`

**Purpose:** Creates a circular cylinder for flow simulations.

**Usage:**
```cpp
MeshUtils::createCylinderObstacle(mesh, 
                                 0.5, 0.5,    // Center (x,y)
                                 0.1,         // Radius
                                 material,    // Optional material
                                 true);       // Fill interior (true) or outline only (false)
```

**When to use:**
- Flow around bluff bodies
- Vortex shedding studies
- Drag coefficient calculations
- Von Kármán vortex street simulations

**Reference:**
Zdravkovich, M. M. (1997), "Flow Around Circular Cylinders; Vol. 1 Fundamentals", Oxford University Press.

### Bluff Body Obstacle

**Function:** `createBluffBodyObstacle`

**Purpose:** Creates a rectangular bluff body that can be rotated to any angle.

**Usage:**
```cpp
MeshUtils::createBluffBodyObstacle(mesh, 
                                  0.5, 0.5,     // Center (x,y)
                                  0.2, 0.1,     // Width and height
                                  M_PI/6,       // Rotation angle (radians)
                                  material,     // Optional material
                                  true);        // Fill interior (true) or outline only (false)
```

**When to use:**
- Studying flow past non-streamlined objects
- Investigating drag and lift forces
- Wake formation analysis
- Comparison with streamlined bodies

## Image-Based Obstacles

### PNG and Other Standard Formats

**Function:** `createObstacleFromImage`

**Purpose:** Creates obstacles from standard image formats like PNG, BMP, JPG, etc.

**Usage:**
```cpp
MeshUtils::createObstacleFromImage(mesh, 
                                  "obstacle.png",  // Image filename
                                  0.2, 0.2,        // Bottom-left corner (x1,y1)
                                  0.8, 0.8,        // Top-right corner (x2,y2)
                                  material,        // Optional material
                                  128,             // Threshold (0-255)
                                  false);          // Invert (false = dark pixels are obstacles)
```

**When to use:**
- Complex geometries that are difficult to define programmatically
- Traced outlines from photographs or diagrams
- Imported designs from other software
- Detailed obstacle shapes created in image editors

**Format information:**
- [PNG Specification](http://www.libpng.org/pub/png/spec/)
- [BMP File Format](https://en.wikipedia.org/wiki/BMP_file_format)

**Dependencies:**
- Requires the stb_image library (included in header)

### PPM/PGM Formats

**Function:** `createObstacleFromPPM`

**Purpose:** Creates obstacles from simple, text-based PPM (color) or PGM (grayscale) image formats.

**Usage:**
```cpp
MeshUtils::createObstacleFromPPM(mesh, 
                                "obstacle.pgm",  // PPM/PGM filename
                                0.2, 0.2,        // Bottom-left corner (x1,y1)
                                0.8, 0.8,        // Top-right corner (x2,y2)
                                material,        // Optional material
                                128,             // Threshold
                                false);          // Invert (false = dark pixels are obstacles)
```

**When to use:**
- When you want a simple, human-readable image format
- For creating and editing obstacle templates in a text editor
- When avoiding external image library dependencies
- For programmatically generated shapes that need to be saved

**Format information:**
- [Netpbm Format Specification](http://netpbm.sourceforge.net/doc/pgm.html)

**Additional related functions:**
- `saveObstaclesToPGM`: Exports current obstacles to a PGM file
- `createPGMObstacleTemplate`: Creates a PGM template with a custom shape
- `createCommonObstacleTemplates`: Generates a set of common obstacle shapes as PGM files

### ASCII and Matrix-Based Formats

**Function:** `createObstacleFromASCII`

**Purpose:** Creates obstacles from ASCII art or text-based representations.

**Usage:**
```cpp
std::vector<std::string> asciiArt = {
    "...............",
    "....XXXXXXX....",
    "...XXXXXXXXX...",
    "..XXXXXXXXXXX..",
    "..XXXXXXXXXXX..",
    "..XXXXXXXXXXX..",
    "...XXXXXXXXX...",
    "....XXXXXXX....",
    "...............",
};
MeshUtils::createObstacleFromASCII(mesh, asciiArt, 0.2, 0.2, 0.8, 0.8, "X#*", material);
```

**Function:** `createObstacleFromMatrix`

**Purpose:** Creates obstacles from a binary matrix (2D array of 0s and 1s).

**Usage:**
```cpp
std::vector<std::vector<int>> matrix = {
    {0, 0, 0, 0, 0, 0, 0},
    {0, 0, 1, 1, 1, 0, 0},
    {0, 1, 1, 1, 1, 1, 0},
    {0, 1, 1, 1, 1, 1, 0},
    {0, 0, 1, 1, 1, 0, 0},
    {0, 0, 0, 0, 0, 0, 0}
};
MeshUtils::createObstacleFromMatrix(mesh, matrix, 0.2, 0.2, 0.8, 0.8, material);
```

**When to use:**
- For simple obstacle patterns easily expressed in text
- When working with programmatically generated patterns
- For quick prototyping without external dependencies
- In scripts or configuration files

## Function-Based Obstacles

### Parametric Functions

**Function:** `createParametricObstacle`

**Purpose:** Creates an obstacle defined by a mathematical function that determines if a point is inside the obstacle.

**Usage:**
```cpp
// Creating a heart shape
auto heartShape = [](double x, double y) -> bool {
    // Center and scale
    x = 3 * (x - 0.5);
    y = 3 * (y - 0.5);
    // Heart curve equation
    return pow(x*x + y*y - 1, 3) - x*x*y*y*y < 0;
};

MeshUtils::createParametricObstacle(mesh, heartShape, 0.2, 0.2, 0.8, 0.8, 50.0, material);
```

**When to use:**
- For mathematically defined shapes
- When shapes have complex analytical formulas
- For procedurally generated patterns
- When precise control is needed over obstacle geometry

### Distance Functions

**Function:** `createObstacleFromDistanceFunction`

**Purpose:** Creates an obstacle using a signed distance function (SDF), where negative values are inside the obstacle.

**Usage:**
```cpp
// Round rectangle with smooth corners using SDF
auto roundRectSDF = [](double x, double y) -> double {
    double rx = 0.3, ry = 0.2;   // Rectangle half-dimensions
    double r = 0.05;             // Corner radius
    
    // Center the coordinates
    x = x - 0.5;
    y = y - 0.5;
    
    // SDF for rounded rectangle
    double dx = std::abs(x) - (rx - r);
    double dy = std::abs(y) - (ry - r);
    
    // Outside components
    dx = std::max(dx, 0.0);
    dy = std::max(dy, 0.0);
    
    // Inside radius component
    double innerDist = std::max(std::min(dx, 0.0) + std::min(dy, 0.0), 0.0);
    
    // Total SDF
    return std::sqrt(dx*dx + dy*dy) - r + innerDist;
};

MeshUtils::createObstacleFromDistanceFunction(mesh, roundRectSDF, 0.2, 0.2, 0.8, 0.8, material);
```

**When to use:**
- For complex shapes with smooth boundaries
- When working with blended or metamorphic shapes
- For boolean operations on geometric primitives
- When distance information is needed for the simulation

**Resources:**
- [Signed Distance Field Reference](https://iquilezles.org/articles/distfunctions2d/)
- [SDF Functions Collection](https://www.shadertoy.com/view/Xds3zN)

## Common Test Cases

Below are some standard CFD test cases and the recommended obstacle creation functions to use for each:

### Flow Around a Cylinder

**Use:** `createCylinderObstacle`

**Example:**
```cpp
// Domain size: 2.2D x 16D
double D = 0.1; // Cylinder diameter
double centerX = 0.4;
double centerY = 0.5;

// Create the mesh with appropriate dimensions
auto mesh = std::make_shared<Mesh>(200, 80, 2.2, 0.8);

// Create the cylinder
MeshUtils::createCylinderObstacle(*mesh, centerX, centerY, D/2);
```

**References:**
- Schäfer, M., et al. (1996), "Benchmark Computations of Laminar Flow Around a Cylinder", Notes on Numerical Fluid Mechanics, Vol. 52, pp. 547-566.
- Example visualizations: [Flow past cylinder](https://commons.wikimedia.org/wiki/File:Cylinder_wake_animation.gif)

### Lid-Driven Cavity

**Use:** `createCavity`

**Example:**
```cpp
// Create the mesh with a square domain
auto mesh = std::make_shared<Mesh>(100, 100, 1.0, 1.0);

// Create the cavity (walls on three sides)
MeshUtils::createCavity(*mesh, 0.0, 0.0, 1.0, 1.0);

// Top lid is moving (not an obstacle) - this would be handled in the boundary conditions
```

**References:**
- Ghia, U., et al. (1982), "High-Re solutions for incompressible flow using the Navier-Stokes equations and a multigrid method", Journal of Computational Physics, vol. 48, pp. 387-411.
- Example visualizations: [Lid-driven cavity flow](https://commons.wikimedia.org/wiki/File:Lid-driven_cavity_animation.gif)

### Backward-Facing Step

**Use:** `createBackwardFacingStep`

**Example:**
```cpp
// Domain dimensions
double L = 40; // Total length
double H1 = 1; // Inlet height
double H2 = 2; // Outlet height
double S = H2-H1; // Step height
double xStep = 10; // Step position

// Create the mesh
auto mesh = std::make_shared<Mesh>(500, 100, L, H2);

// Create the backward-facing step
MeshUtils::createBackwardFacingStep(*mesh, xStep, S, H1, L-xStep);
```

**References:**
- Armaly, B. F., et al. (1983), "Experimental and theoretical investigation of backward-facing step flow", Journal of Fluid Mechanics, vol. 127, pp. 473-496.
- Example visualizations: [Backward-facing step flow](https://commons.wikimedia.org/wiki/File:Backward_facing_step_animation.gif)

### NACA Airfoil

**Use:** `createAirfoilObstacle`

**Example:**
```cpp
// Airfoil parameters
double chordLength = 1.0;
double leadingEdgeX = 0.5;
double leadingEdgeY = 1.0;
double angle = 5.0 * M_PI/180; // 5 degrees angle of attack

// Create the mesh (ensure domain is large enough)
auto mesh = std::make_shared<Mesh>(200, 100, 10.0, 5.0);

// Create the airfoil
MeshUtils::createAirfoilObstacle(*mesh, leadingEdgeX, leadingEdgeY, 
                              chordLength, angle, "NACA0012");
```

**References:**
- Abbott, I. H., and Von Doenhoff, A. E. (1959), "Theory of Wing Sections", Dover Publications.
- Example visualizations: [Airfoil flow](https://commons.wikimedia.org/wiki/File:Flow_around_an_airfoil.gif)

## Performance Considerations

When creating obstacles, consider the following performance aspects:

### Resolution and Grid Size

- **Highly detailed obstacles** may require finer grid resolution to capture features accurately
- **Complex curved boundaries** might benefit from local grid refinement near the obstacle
- **Performance vs. accuracy tradeoff**: Fine obstacles on coarse grids can lead to "staircase" boundaries

### Function vs. Image-Based Approaches

- **Function-based obstacles** 
  - Generally faster to create for simple shapes
  - More precise mathematical definition
  - Parameters can be easily changed

- **Image-based obstacles**
  - Better for extremely complex shapes
  - Can be slower to process for large images
  - Fixed resolution determined by the image

### Recommendations for Optimal Performance

1. Use the simplest obstacle representation that captures the required detail
2. For multiple identical obstacles, create a template and place it repeatedly
3. Consider generating obstacles at mesh creation time, not during simulation
4. Image-based obstacles with many tiny features may need mesh refinement
5. For function-based obstacles, keep the mathematical expressions efficient

## References

### Image Format Specifications

- **PNG**: [Portable Network Graphics Specification](http://www.libpng.org/pub/png/spec/)
- **BMP**: [BMP File Format](https://en.wikipedia.org/wiki/BMP_file_format)
- **PPM/PGM**: [Netpbm Format Specification](http://netpbm.sourceforge.net/doc/pgm.html)

### Classical CFD Test Cases

- Roache, P. J. (1998), "Verification and Validation in Computational Science and Engineering", Hermosa Publishers.
- Hirsch, C. (2007), "Numerical Computation of Internal and External Flows", Butterworth-Heinemann.
- ERCOFTAC Classic Collection Database: [ERCOFTAC](https://www.ercoftac.org/)

### Online Resources for Obstacle Examples

- NASA Turbulence Modeling Resource: [NASA TMR](https://turbmodels.larc.nasa.gov/)
- NPARC Alliance Validation Archive: [NPARC Archive](https://www.grc.nasa.gov/WWW/wind/valid/archive.html)
- CFD Online Test Cases: [CFD Online](https://www.cfd-online.com/Wiki/Main_Page)

### Function-Based Shape Definitions

- Signed Distance Functions: [Inigo Quilez - SDF](https://iquilezles.org/articles/distfunctions2d/)
- NACA Airfoil Equations: [NACA Airfoil](https://en.wikipedia.org/wiki/NACA_airfoil)

