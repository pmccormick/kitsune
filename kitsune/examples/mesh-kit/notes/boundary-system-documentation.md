# Boundary Condition System Design

## Introduction

Boundary conditions are essential components of any Computational Fluid Dynamics (CFD) framework, defining how the simulation domain interacts with its surroundings. They mathematically describe how variables behave at the edges of the computational domain and are critical for obtaining physically accurate solutions.

This document outlines our approach to implementing boundary conditions within the existing Field-based architecture, with a focus on maintaining high performance across various processor types (CPU and GPU), scalability, and extensibility.

## Boundary Conditions in CFD

### Physical Significance

Boundary conditions in CFD represent physical constraints that must be satisfied at the domain boundaries. They define how fluid behaves when encountering different types of boundaries such as:

1. **Inlet boundaries** - Where fluid enters the domain with specified properties
2. **Outlet boundaries** - Where fluid exits the domain
3. **Wall boundaries** - Solid surfaces where no-slip or slip conditions may apply
4. **Symmetry boundaries** - Where flow exhibits mirror-like behavior
5. **Periodic boundaries** - Where flow exiting one side re-enters the opposite side

### Mathematical Representation

Mathematically, boundary conditions can be categorized as:

1. **Dirichlet conditions** - Specify the value of a variable at the boundary (e.g., fixed velocity at an inlet)
2. **Neumann conditions** - Specify the gradient of a variable at the boundary (e.g., zero gradient at an outlet)
3. **Robin/Mixed conditions** - Specify a relationship between the value and gradient (e.g., convective heat transfer)

## Current Architecture Overview

Our simulation framework is built around a Field-based architecture that prioritizes memory layout and computational efficiency:

**Figure 1: Field-Based Architecture (Key elements to include in the diagram)**
- Field arrays storing data in contiguous memory
- Cell objects acting as facades to access Field data
- Mesh object managing the overall structure
- Arrows showing how Cells reference data in Fields
- Illustration of cache-friendly memory layout

Key components include:

- **Field class**: Provides contiguous storage for physical quantities, optimized for cache efficiency
- **Cell class**: A lightweight facade providing access to data stored in Fields
- **Mesh class**: Manages the overall grid structure and interconnections

This design separates data storage from data access, allowing for optimized memory layouts while maintaining an intuitive object-oriented interface.

## Boundary Condition System Design

Our boundary condition system extends this architecture without compromising its performance characteristics by using adapter patterns and clear separation of responsibilities.

### Design Objectives

1. **Performance**: Maintain the efficiency of the Field-based architecture
2. **Modularity**: Allow different boundary condition types without modifying core classes
3. **Extensibility**: Support new boundary condition types easily
4. **Processor-agnostic**: Work efficiently on both CPU and GPU architectures
5. **Usability**: Provide intuitive interfaces for setting up boundary conditions

### System Components

#### 1. Boundary Condition Hierarchy

A class hierarchy of boundary conditions, with a common base class:

```
BoundaryClass (abstract)
├── DirichletBoundary
├── NeumannBoundary
├── InflowBoundary
├── OutflowBoundary
├── NoSlipBoundary
├── SlipBoundary
└── PeriodicBoundary
```

Each boundary condition implements an `apply()` method that updates the relevant cells.

#### 2. MeshTopologyAdapter

This adapter bridges between the Field-based storage and the geometric/topological information needed for boundary handling:

**Figure 2: Topology Adapter (Key elements to include in the diagram)**
- Central MeshTopologyAdapter connecting Mesh object and boundary components
- Interface showing geometric queries (getCellVertices, getFaceNormal, etc.)
- Data flow from Mesh to geometric information
- Emphasis on computation over storage with optional caching

Key functions:
- Providing face normals, centroids, and other geometric properties
- Identifying neighboring cells
- Determining boundary cells

#### 3. BoundaryManager

Manages boundary zones and their application:

**Figure 3: Boundary Manager (Key elements to include in the diagram)**
- BoundaryManager coordinating multiple BoundaryZone objects
- Connections to Mesh and MeshTopologyAdapter
- Illustration of how boundary conditions apply to cells
- Assignment workflow from selection to zone creation
- Application workflow from zone to cell updates

Responsibilities:
- Grouping boundary cells into logical zones
- Assigning boundary conditions to zones
- Coordinating boundary condition application
- Validating boundary coverage

#### 4. BoundaryZone

A lightweight container that groups cells with the same boundary condition:

```
BoundaryZone
├── Name
├── Boundary Condition
├── Cell Indices
└── Active Flag
```

## Performance Considerations

### Memory Efficiency

1. **Index-Based Access**: We store cell indices rather than pointers, reducing memory overhead
2. **Computation vs. Storage**: Geometric properties are computed on-the-fly with optional caching for repeated access
3. **Contiguous Storage**: Boundary zones store cell indices in contiguous arrays for cache efficiency

### Parallelization Strategy

Our boundary system is designed for effective parallelization on both CPU and GPU:

**Figure 4: Parallelization Strategy (Key elements to include in the diagram)**
- Division of boundary zones across processing units
- Batched application to multiple cells
- Independent cell updates within a boundary type
- Data flow in CPU (multi-thread) and GPU (kernel) scenarios
- Performance comparison with sequential vs. parallel application

1. **Cell-Parallel Application**: Boundary conditions can be applied to cells independently
2. **Zone-Parallel Processing**: Different boundary zones can be processed in parallel
3. **Batch Processing**: Support for applying boundary conditions to multiple cells at once

### CPU-Specific Optimizations

1. **SIMD-Friendly Operations**: Simple arithmetic operations that can utilize vector instructions
2. **Cache-Conscious Design**: Contiguous arrays and predictable memory access patterns
3. **Thread-Pool Compatibility**: Boundary zone application can be distributed across thread pools

### GPU-Specific Optimizations

1. **Simple Data Structures**: Avoiding complex pointer-based structures that don't translate well to GPUs
2. **Coalesced Memory Access**: Organizing boundary cells for optimal GPU memory access patterns
3. **Minimal Branching**: Designing boundary condition application with minimal divergent execution paths

## Implementation Details

### Cell Class Integration

The Cell class serves as a facade over Field data. For boundary conditions, we:

1. Maintain the lightweight nature of the Cell class
2. Utilize existing methods for accessing and modifying Field data
3. Add minimal boundary-specific functionality through the BoundaryCondition reference

```cpp
// Cell already supports:
cell.setVelocityU(value);     // Sets u-velocity in the Field
cell.setBoundaryCondition(bc); // Sets boundary condition reference
```

### Mesh Class Integration

The Mesh class provides access to Fields and manages the overall domain. For boundaries, we:

1. Add methods to identify boundary cells
2. Store boundary condition references for cells
3. Provide access to physical coordinates needed by boundary conditions

### Field Integration

Fields remain unchanged, continuing to provide efficient data storage. Boundary conditions:

1. Read and write Field data through Cell methods
2. Don't require changes to the Field class itself
3. Benefit from the cache-efficient contiguous storage

## Example Workflow

Setting up and applying boundary conditions follows this workflow:

```cpp
// Create the mesh
Mesh mesh(100, 50); // 100x50 grid

// Create boundary manager
BoundaryManager boundaryManager(mesh);

// Create boundary conditions
auto inflow = std::make_shared<InflowBoundary>("inlet");
inflow->setVelocityU(1.0);

auto outflow = std::make_shared<NeumannBoundary>("outlet");
auto wall = std::make_shared<NoSlipBoundary>("wall");

// Assign boundaries to mesh regions
boundaryManager.assignToEdge(inflow, 3); // Left edge
boundaryManager.assignToEdge(outflow, 1); // Right edge
boundaryManager.assignToEdge(wall, 0); // Bottom edge
boundaryManager.assignToEdge(wall, 2); // Top edge

// In simulation loop:
while (simulating) {
    // Apply boundary conditions
    boundaryManager.applyAllBoundaryConditions(time, dt);
    
    // ... Solve equations ...
    
    time += dt;
}
```

## Validation and Quality Assurance

The boundary system includes validation tools to ensure:

1. All boundary cells have assigned conditions
2. Boundary conditions are physically compatible
3. Mesh quality is sufficient at boundaries

These validations are performed through the MeshTopologyAdapter and BoundaryManager.

## Extensibility

### Adding New Boundary Types

To add a new boundary condition type:

1. Create a new class inheriting from BoundaryClass
2. Implement the apply() method
3. Register with BoundaryFactory if using the factory pattern

```cpp
class CustomBoundary : public BoundaryClass {
public:
    void apply(Cell& cell, double x, double y, double dt,
               const std::vector<Cell*>* neighbors) override {
        // Custom implementation
    }
    
    std::string getType() const override { return "Custom"; }
};
```

### Supporting Complex Geometries

For more complex boundary geometries:

1. Use the coordinate-based selection in BoundaryManager
2. Create custom selector functions for specific shapes
3. Combine multiple boundary zones for compound boundaries

## Future Enhancements

1. **3D Support**: Extending geometry calculations and topology to 3D
2. **Unstructured Meshes**: Adapting the topology adapter for unstructured grids
3. **Dynamic Boundaries**: Supporting moving or deforming boundaries
4. **Multi-Physics Coupling**: Handling boundary conditions for coupled physics
5. **Adaptive Refinement**: Integrating with mesh adaptation near boundaries

## References

### Academic References

1. Ferziger, J.H., & Perić, M. (2002). *Computational Methods for Fluid Dynamics*. Springer. 
   - Chapter 8: "Boundary Conditions" provides comprehensive coverage of boundary condition implementation in finite volume methods

2. Versteeg, H.K., & Malalasekera, W. (2007). *An Introduction to Computational Fluid Dynamics: The Finite Volume Method*. Pearson Education.
   - Chapter 9: "Implementation of Boundary Conditions" details practical boundary condition implementation strategies

3. Moukalled, F., Mangani, L., & Darwish, M. (2016). *The Finite Volume Method in Computational Fluid Dynamics: An Advanced Introduction with OpenFOAM and Matlab*. Springer.
   - Chapter 7: "Boundary Conditions" covers boundary condition handling in modern CFD codes

4. Patankar, S.V. (1980). *Numerical Heat Transfer and Fluid Flow*. CRC Press.
   - Classic reference with foundational boundary condition treatment in section 4.3

### Web Resources

1. CFD Online: [Boundary Condition Implementation](https://www.cfd-online.com/Wiki/Boundary_conditions)
   - Community-maintained wiki with explanations and practical implementation advice

2. NASA Langley Research Center: [Turbulence Modeling Resource](https://turbmodels.larc.nasa.gov/boundary.html)
   - Reference implementations of boundary conditions for turbulence models

3. OpenFOAM Documentation: [Boundary Conditions](https://cfd.direct/openfoam/user-guide/boundaries/)
   - Open-source implementation patterns for various boundary condition types

### Performance Optimization

1. Meister, O., & Bader, M. (2018). *2D and 3D Simulations on GPUs*. In Optimization and Applications in Control and Data Sciences (pp. 205-234). Springer.
   - Section on boundary condition handling for GPU computations

2. Kirk, D.B., & Hwu, W.M.W. (2016). *Programming Massively Parallel Processors: A Hands-on Approach*. Morgan Kaufmann.
   - Chapter 10 covers optimized data layouts relevant to boundary condition implementation

3. NVIDIA Developer Blog: [Optimizing Parallel Reduction in CUDA](https://developer.nvidia.com/blog/faster-parallel-reductions-kepler/)
   - Techniques applicable to efficient boundary condition implementation on GPUs

### Software Design Patterns

1. Gamma, E., Helm, R., Johnson, R., & Vlissides, J. (1994). *Design Patterns: Elements of Reusable Object-Oriented Software*. Addison-Wesley.
   - Adapter and Facade patterns used in our boundary condition system

2. Stroustrup, B. (2013). *The C++ Programming Language (4th Edition)*. Addison-Wesley.
   - Chapter 22 discusses runtime polymorphism techniques used in our boundary class hierarchy
