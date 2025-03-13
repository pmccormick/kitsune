# Heat Transfer Example

This example demonstrates the use of MeshKit to solve a 2D heat transfer problem using the explicit finite difference method. It showcases how to use the core components of the framework (Mesh, Cell, Field, and Region) to build a simple but meaningful simulation.

## Physical Problem Description

The simulation solves the 2D heat equation:

$\frac{\partial T}{\partial t} = \alpha \nabla^2 T$

Where:
- $T$ is the temperature (°C)
- $t$ is time (seconds)
- $\alpha$ is the thermal diffusivity (m²/s)
- $\nabla^2$ is the Laplacian operator

Using a finite difference discretization on a uniform grid:

$T_{i,j}^{n+1} = T_{i,j}^n + \alpha \frac{\Delta t}{(\Delta x)^2} \left( T_{i+1,j}^n + T_{i-1,j}^n + T_{i,j+1}^n + T_{i,j-1}^n - 4 T_{i,j}^n \right)$

Where $\Delta t$ is the time step and $\Delta x$ is the grid spacing.

## Initial and Boundary Conditions

The simulation models heat diffusion from a hot spot in the center of a domain with either:
1. Fixed temperature (Dirichlet) boundaries
2. Zero-gradient (Neumann) boundaries

Initial conditions:
- Ambient temperature: 20°C throughout most of the domain
- Hot spot: 100°C at the center (covering ~20% of the domain)

## Implementation Details

### Core Components Used

- **Mesh**: Defines the spatial domain with grid cells
- **Cell**: Lightweight views for accessing and navigating the grid
- **Field**: Stores temperature data associated with each cell
- **Region**: Defines the interior subdomain where we solve the equations

### Simulation Flow

1. Setup the computational domain (mesh)
2. Create temperature fields (current and next step buffer)
3. Define interior region for computation
4. Initialize conditions with hot spot
5. For each time step:
   - Apply boundary conditions
   - Compute temperature updates for interior cells
   - Update the current temperature field
   - Output results at specified intervals

### Key Features

- **Flexible Boundary Conditions**: Supports both Dirichlet (fixed value) and Neumann (zero gradient) boundary types
- **Cell Navigation**: Uses Cell.neighbor() for accessing adjacent cells
- **Region-Based Iteration**: Solves equations only for interior cells using forEachCellInRegion
- **CSV Output**: Saves simulation results at regular intervals for visualization

## Running the Example

### Compilation

To compile the example, add it to your MeshKit build system or compile directly:

```bash
g++ -std=c++17 -I/path/to/meshkit/include HeatTransferExample.cpp -o heat_transfer
```

### Execution

Run the compiled executable:

```bash
./heat_transfer
```

The simulation will create a `heat_results` directory containing CSV files with temperature data at different time steps.

### Visualization

Use the provided Python script to visualize the results:

```bash
python visualize_heat_transfer.py
```

This will generate:
1. Heatmap images for each output time step
2. A 3D surface plot of the final temperature distribution
3. An animation showing the temperature evolution (requires ffmpeg)

## Example Results

A typical simulation with 1000 time steps shows heat gradually diffusing from the central hot spot toward the boundaries. With fixed temperature boundaries, the temperature eventually approaches the boundary values as the system reaches steady state.

## Modifying the Example

### Changing Physical Parameters

Modify the `setupSimulation()` function to adjust:
- `alpha`: Thermal diffusivity (material property)
- `dx`: Cell size
- `dt`: Time step (keep stability condition in mind)

### Changing Boundary Conditions

The simulation supports different boundary condition types for each edge:
```cpp
// Example: Set insulated boundaries on left and right, fixed temperature on top and bottom
params.leftBC = SimulationParameters::BoundaryType::NEUMANN;
params.rightBC = SimulationParameters::BoundaryType::NEUMANN;
params.bottomBC = SimulationParameters::BoundaryType::DIRICHLET;
params.topBC = SimulationParameters::BoundaryType::DIRICHLET;
```

### Stability Considerations

For the explicit scheme to be stable, the following condition must be met:
$\alpha \frac{\Delta t}{(\Delta x)^2} \leq 0.25$

The code checks this condition and warns if stability might be compromised.

## Extending the Example

Potential extensions to this example include:
1. Adding non-uniform material properties (spatially varying thermal diffusivity)
2. Implementing time-dependent boundary conditions
3. Adding heat sources/sinks
4. Parallelizing using the MeshKit Partition system
5. Using advanced visualization techniques

## Relation to MeshKit Design

This example demonstrates the core design principles of MeshKit:
- **Lightweight Cell Views**: Cells are created on-demand during iteration
- **Separation of Concerns**: Mesh structure, field data, and region definitions are clearly separated
- **Performance-Oriented Design**: Inline methods and efficient data structures
- **Extensibility**: Easy to add new boundary types or physical processes
