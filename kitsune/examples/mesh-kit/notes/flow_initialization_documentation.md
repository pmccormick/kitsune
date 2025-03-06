# Flow Initialization Reference Guide

This comprehensive guide documents the flow initialization utilities available in the CFD simulation framework. These functions provide standardized ways to set up initial velocity, pressure, and other flow field variables across the mesh for various simulation scenarios.

## Table of Contents

1. [Basic Testing Functions](#basic-testing-functions)
2. [Real Simulation Functions](#real-simulation-functions)
3. [Advanced Flow Functions](#advanced-flow-functions)
4. [Utility Functions](#utility-functions)
5. [Best Practices](#best-practices)
6. [Examples](#examples)

## Basic Testing Functions

### Uniform Flow
```cpp
inline FlowField initialize_uniform_flow(const Mesh& mesh, 
                                       const std::vector<double>& velocity = {1.0, 0.0}, 
                                       double pressure = 0.0);
```

Initializes the entire domain with a uniform flow field, useful for basic validation and as a building block for more complex flows.

**Parameters:**
- `mesh`: The computational mesh
- `velocity`: Vector representing uniform velocity (2D)
- `pressure`: Uniform pressure value

**Use Cases:**
- Standard test case for flow over objects
- Initial validation of solver behavior
- Baseline for comparison with more complex flows
- Wind tunnel simulations

**Example:**
```cpp
auto flow = MeshUtils::initialize_uniform_flow(mesh, {10.0, 0.0}, 0.0);
```

### Taylor-Green Vortex
```cpp
inline FlowField initialize_taylor_green(const Mesh& mesh, 
                                       double amplitude = 1.0, 
                                       const std::vector<double>& wavelength = {1.0, 1.0}, 
                                       double decay_factor = 0.0);
```

Creates a Taylor-Green vortex field - an exact solution of the incompressible Navier-Stokes equations commonly used to test numerical schemes.

**Parameters:**
- `mesh`: The computational mesh
- `amplitude`: Velocity amplitude
- `wavelength`: Spatial wavelength(s) of vortices
- `decay_factor`: Temporal decay factor for analytical solutions at time > 0

**Use Cases:**
- Testing numerical accuracy and convergence
- Validating energy conservation
- Examining vortex dynamics and decay
- Benchmark for comparison between different solvers

**Notes:**
For 2D, the velocity field is:
- u = A * sin(2π*x/λx) * cos(2π*y/λy) * exp(-decay_factor)
- v = -A * cos(2π*x/λx) * sin(2π*y/λy) * exp(-decay_factor)
- p = (A²/4) * (cos(4π*x/λx) + cos(4π*y/λy)) * exp(-2*decay_factor)

### Vortex Flow
```cpp
inline FlowField initialize_vortex(const Mesh& mesh, 
                                 const std::vector<std::vector<double>>& centers, 
                                 const std::vector<double>& strengths, 
                                 const std::vector<double>& radii,
                                 const std::vector<double>& base_flow = {0.0, 0.0},
                                 const std::string& pressure_model = "quadratic");

// Convenience overload for single vortex
inline FlowField initialize_vortex(const Mesh& mesh, 
                                 const std::vector<double>& center, 
                                 double strength, 
                                 double radius,
                                 const std::vector<double>& base_flow = {0.0, 0.0},
                                 const std::string& pressure_model = "quadratic");
```

Initializes a single vortex or multiple vortices in the flow field, using a Rankine vortex model with a forced rotation core and a free vortex in the outer region.

**Parameters:**
- `mesh`: The computational mesh
- `centers`: Vector of center coordinates for multiple vortices
- `strengths`: Vector of vortex strengths (positive for counterclockwise, negative for clockwise)
- `radii`: Vector of characteristic radii for vortices
- `base_flow`: Background flow velocity
- `pressure_model`: Model for pressure distribution ("quadratic", "linear", or "constant")

**Use Cases:**
- Vortex interaction studies
- Validation of vorticity transport
- Wake dynamics
- Aircraft trailing vortices simulation

**Example:**
```cpp
// Single vortex
auto flow = MeshUtils::initialize_vortex(mesh, {0.5, 0.5}, 1.0, 0.2);

// Multiple vortices
std::vector<std::vector<double>> centers = {{0.3, 0.3}, {0.7, 0.7}};
std::vector<double> strengths = {1.0, -1.0};
std::vector<double> radii = {0.1, 0.1};
auto flow = MeshUtils::initialize_vortex(mesh, centers, strengths, radii);
```

### Random Perturbation
```cpp
inline FlowField initialize_random_perturbation(const Mesh& mesh, 
                                              const FlowField* base_flow = nullptr,
                                              double perturbation_magnitude = 0.1, 
                                              int seed = -1,
                                              const std::string& spectrum = "white");
```

Adds random perturbations to a base flow field, useful for testing solver robustness and initializing turbulent simulations.

**Parameters:**
- `mesh`: The computational mesh
- `base_flow`: Base flow field to perturb (if nullptr, starts with zero flow)
- `perturbation_magnitude`: Maximum perturbation magnitude relative to base flow
- `seed`: Random seed for reproducibility
- `spectrum`: Energy spectrum for perturbation ("white", "red", or "von_karman")

**Use Cases:**
- Testing solver robustness under non-ideal conditions
- Initializing turbulence studies
- Transition to turbulence simulations
- Adding realistic noise to flow fields

**Spectrum Options:**
- `white`: Uniform energy across all scales
- `red`: Higher energy at larger scales (energy ~ 1/frequency)
- `von_karman`: Von Kármán spectrum for more realistic turbulence

## Real Simulation Functions

### Shear Flow
```cpp
inline FlowField initialize_shear_flow(const Mesh& mesh, 
                                     const std::vector<std::vector<double>>& velocity_gradient,
                                     const std::vector<double>& base_velocity = {0.0, 0.0},
                                     double pressure = 0.0);
```

Initializes a flow with linear velocity gradients, useful for studying mixing layers and boundary layer development.

**Parameters:**
- `mesh`: The computational mesh
- `velocity_gradient`: 2x2 matrix ((du/dx, du/dy), (dv/dx, dv/dy)) for 2D
- `base_velocity`: Vector (u0, v0) representing velocity at origin
- `pressure`: Initial pressure field value

**Use Cases:**
- Boundary layer development studies
- Mixing layer simulations
- Vorticity generation tests
- Wind shear models

**Example:**
```cpp
// Simple shear with du/dy = 1.0, all other gradients zero
std::vector<std::vector<double>> gradient = {{0.0, 1.0}, {0.0, 0.0}};
auto flow = MeshUtils::initialize_shear_flow(mesh, gradient);
```

### Channel Flow
```cpp
inline FlowField initialize_channel_flow(const Mesh& mesh, 
                                       double height,
                                       double max_velocity,
                                       const std::vector<double>& direction = {1.0, 0.0},
                                       const std::string& profile = "parabolic",
                                       double pressure_gradient = 0.0);
```

Initializes a channel flow with specified profile, ideal for internal flow simulations.

**Parameters:**
- `mesh`: The computational mesh
- `height`: Channel height
- `max_velocity`: Maximum velocity magnitude
- `direction`: Flow direction vector
- `profile`: Velocity profile type ("parabolic", "linear", or "plug")
- `pressure_gradient`: Optional pressure gradient along channel

**Use Cases:**
- Pipe and channel flow studies
- Heat exchanger simulations
- Entrance region development
- Wall-bounded turbulence research

**Profile Types:**
- `parabolic`: Fully developed laminar profile, u = u_max * (1 - (y/R)²)
- `linear`: Linear profile, u = u_max * (1 - |y|/R)
- `plug`: Uniform velocity within the channel

### Boundary Layer
```cpp
inline FlowField initialize_boundary_layer(const Mesh& mesh,
                                         const std::vector<double>& free_stream_velocity,
                                         const std::vector<std::vector<double>>& wall_points,
                                         double boundary_layer_thickness,
                                         const std::string& profile = "blasius");
```

Initializes a boundary layer profile over a specified wall.

**Parameters:**
- `mesh`: The computational mesh
- `free_stream_velocity`: Velocity outside boundary layer
- `wall_points`: Coordinates defining wall location
- `boundary_layer_thickness`: Thickness of boundary layer
- `profile`: Profile type ("blasius", "power_law", "logarithmic", or "constant")

**Use Cases:**
- Aerodynamic studies
- Boundary layer stability analysis
- Transition to turbulence research
- Flow separation prediction

**Profile Types:**
- `blasius`: Approximation of the Blasius solution for a flat plate
- `power_law`: 1/7th power law profile, typical for turbulent boundary layers
- `logarithmic`: Logarithmic law of the wall for turbulent boundary layers
- `constant`: Simple linear profile for quick approximations

### Stagnation Flow
```cpp
inline FlowField initialize_stagnation_flow(const Mesh& mesh,
                                          const std::vector<double>& stagnation_point,
                                          double strength,
                                          double pressure_amplitude);
```

Initializes a stagnation flow field with specified stagnation point and flow strength.

**Parameters:**
- `mesh`: The computational mesh
- `stagnation_point`: Coordinates of stagnation point
- `strength`: Flow strength parameter
- `pressure_amplitude`: Peak pressure value at stagnation point

**Use Cases:**
- Leading edge flow studies
- Impinging jet simulations
- Pressure distribution validation
- Flow around blunt bodies

### Jet Flow
```cpp
inline FlowField initialize_jet(const Mesh& mesh,
                              const std::vector<double>& origin,
                              const std::vector<double>& direction,
                              const std::string& velocity_profile = "gaussian",
                              double jet_diameter = 0.1,
                              double max_velocity = 1.0,
                              double spread_rate = 0.1);
```

Initializes a jet flow issuing from a specified origin with configurable profile.

**Parameters:**
- `mesh`: The computational mesh
- `origin`: Coordinates of jet origin
- `direction`: Jet direction vector
- `velocity_profile`: Velocity profile type ("gaussian" or "top_hat")
- `jet_diameter`: Diameter of the jet at origin
- `max_velocity`: Maximum jet velocity at centerline
- `spread_rate`: Rate of jet spreading

**Use Cases:**
- Jet mixing studies
- Combustion simulations
- HVAC system design
- Environmental dispersion modeling

**Profile Types:**
- `gaussian`: Gaussian profile, more realistic for developed jets
- `top_hat`: Uniform velocity within jet radius, simpler model

### Flow From File
```cpp
inline FlowField initialize_flow_from_file(const Mesh& mesh,
                                         const std::string& filename,
                                         const std::string& interpolation_method = "linear");
```

Initializes flow field by importing data from an external file, useful for continuing simulations or incorporating experimental data.

**Parameters:**
- `mesh`: The computational mesh
- `filename`: Path to flow field data file
- `interpolation_method`: Method for interpolating to mesh ("linear" or "nearest")

**Use Cases:**
- Continuation of previous simulations
- Multi-scale coupling
- Experimental data integration
- Benchmark comparisons

**File Format:**
- Expected format: space-separated values with (x, y, u, v, p) columns
- Header line should be present and indicate columns

## Advanced Flow Functions

### Potential Flow
```cpp
inline FlowField initialize_potential_flow(const Mesh& mesh,
                                         const std::vector<std::vector<double>>& sources = {},
                                         const std::vector<std::vector<double>>& vortices = {},
                                         const std::vector<std::vector<double>>& doublets = {},
                                         const std::vector<double>& uniform_flow = {0.0, 0.0});
```

Initializes a potential flow field around multiple sources, sinks, vortices, and doublets, for idealized inviscid flow modeling.

**Parameters:**
- `mesh`: The computational mesh
- `sources`: Vector of {x, y, strength} for sources (strength > 0) and sinks (strength < 0)
- `vortices`: Vector of {x, y, strength} for vortices
- `doublets`: Vector of {x, y, strength, angle} for doublets
- `uniform_flow`: Vector {u, v} for background uniform flow

**Use Cases:**
- Inviscid flow modeling
- Aerodynamic analysis of simple shapes
- Education and demonstration
- Quick approximation of flow fields

**Example:**
```cpp
// Flow around a cylinder using a uniform flow and a doublet
std::vector<std::vector<double>> doublets = {{0.0, 0.0, 1.0, 0.0}};
auto flow = MeshUtils::initialize_potential_flow(mesh, {}, {}, doublets, {1.0, 0.0});
```

### Synthetic Turbulence
```cpp
inline FlowField initialize_synthetic_turbulence(const Mesh& mesh,
                                              const std::vector<double>& mean_velocity = {1.0, 0.0},
                                              const std::vector<double>& reynolds_stresses = {0.1, 0.1, 0.0},
                                              double turbulent_length_scale = 0.1,
                                              int num_eddies = 100,
                                              int seed = -1);
```

Initializes a synthetic turbulent flow field using the Synthetic Eddy Method (SEM), useful for inflow conditions in turbulent simulations.

**Parameters:**
- `mesh`: The computational mesh
- `mean_velocity`: Background mean velocity field
- `reynolds_stresses`: Reynolds stress tensor components [u'u', v'v', u'v']
- `turbulent_length_scale`: Characteristic length scale of turbulent eddies
- `num_eddies`: Number of synthetic eddies to generate
- `seed`: Random seed for reproducibility

**Use Cases:**
- Turbulent inflow generation for LES/DNS
- Testing turbulence models
- Reproducing specific turbulence characteristics
- Studying turbulent structures

### Stratified Flow
```cpp
inline FlowField initialize_stratified_flow(const Mesh& mesh,
                                          const std::vector<double>& base_velocity = {0.0, 0.0},
                                          const std::vector<double>& stratification_direction = {0.0, 1.0},
                                          double density_gradient = -0.01,
                                          double reference_density = 1.0,
                                          double gravity_magnitude = 9.81);
```

Initializes a stratified flow field with density/temperature variations, useful for environmental and atmospheric flows.

**Parameters:**
- `mesh`: The computational mesh
- `base_velocity`: Base velocity field
- `stratification_direction`: Direction of stratification (typically vertical)
- `density_gradient`: Density gradient per unit distance
- `reference_density`: Reference density at origin
- `gravity_magnitude`: Magnitude of gravitational acceleration

**Use Cases:**
- Atmospheric boundary layer simulations
- Ocean circulation models
- Indoor air quality studies
- Thermal stratification in tanks

### Wave Flow
```cpp
inline FlowField initialize_wave_flow(const Mesh& mesh,
                                    const std::string& wave_type = "traveling",
                                    double amplitude = 0.1,
                                    double wavelength = 1.0,
                                    const std::vector<double>& direction = {1.0, 0.0},
                                    double phase_speed = 1.0);
```

Initializes a wave-like flow field with various wave types.

**Parameters:**
- `mesh`: The computational mesh
- `wave_type`: Type of wave ("standing", "traveling", "circular")
- `amplitude`: Wave amplitude
- `wavelength`: Wavelength
- `direction`: Direction of wave propagation
- `phase_speed`: Phase speed for traveling waves

**Use Cases:**
- Surface wave simulations
- Acoustic wave modeling
- Shock wave initialization
- Oscillatory flow studies

**Wave Types:**
- `standing`: Stationary wave pattern
- `traveling`: Wave propagating with phase speed
- `circular`: Radial wave propagating from a center point

### Swirling Flow
```cpp
inline FlowField initialize_swirling_flow(const Mesh& mesh,
                                        const std::vector<double>& center = {0.0, 0.0},
                                        const std::string& tangential_profile = "rankine",
                                        const std::string& radial_profile = "none",
                                        double max_tangential_velocity = 1.0,
                                        double max_radial_velocity = 0.0,
                                        double characteristic_radius = 1.0);
```

Initializes a swirling flow field with optional radial component, useful for modeling vortices, cyclones, and rotating flows.

**Parameters:**
- `mesh`: The computational mesh
- `center`: Center of swirling flow
- `tangential_profile`: Tangential velocity profile ("solid_body", "potential", "rankine")
- `radial_profile`: Radial velocity profile ("none", "source", "sink", "gaussian")
- `max_tangential_velocity`: Maximum tangential velocity
- `max_radial_velocity`: Maximum radial velocity (positive for outflow, negative for inflow)
- `characteristic_radius`: Radius at which tangential velocity reaches maximum

**Use Cases:**
- Cyclone separator simulations
- Tornado-like vortices
- Swirl combustors
- Rotating machinery

**Tangential Profiles:**
- `solid_body`: Rigid body rotation (v_θ ∝ r)
- `potential`: Potential vortex (v_θ ∝ 1/r)
- `rankine`: Combined model with solid body core and potential outer region

**Radial Profiles:**
- `none`: No radial velocity
- `source`: Outward flow (v_r > 0)
- `sink`: Inward flow (v_r < 0)
- `gaussian`: Gaussian profile decreasing with distance

## Utility Functions

### Validate Flow Field
```cpp
inline std::pair<bool, std::vector<std::string>> validate_flow_field(
    const Mesh& mesh, 
    const FlowField& flow_field, 
    const std::unordered_map<std::string, double>& constraints = {});
```

Verifies that a flow field satisfies physical and numerical constraints.

**Parameters:**
- `mesh`: The computational mesh
- `flow_field`: Flow field to validate
- `constraints`: Map of constraints to check

**Default Constraints:**
- `max_velocity`: Maximum allowable velocity magnitude
- `min_pressure`: Minimum allowable pressure
- `max_pressure`: Maximum allowable pressure
- `divergence_tolerance`: Maximum allowable velocity divergence

**Returns:**
- Boolean indicating validity
- List of constraint violations

### Combine Flow Fields
```cpp
inline FlowField combine_flow_fields(
    const Mesh& mesh, 
    const std::vector<FlowField>& field_list, 
    const std::string& method = "addition", 
    const std::vector<double>& weights = {});
```

Combines multiple flow fields into a single field using different methods.

**Parameters:**
- `mesh`: The computational mesh
- `field_list`: List of flow field objects to combine
- `method`: Combination method ("addition", "max", or "weighted")
- `weights`: Weights for weighted combination

**Combination Methods:**
- `addition`: Direct sum of fields
- `max`: Take maximum value at each point
- `weighted`: Weighted sum using provided weights

**Use Cases:**
- Building complex initial conditions from simpler components
- Combining results from different simulation runs
- Superposition of elementary solutions

## Best Practices

### Choosing the Right Initialization

1. **For Validation Testing:**
   - Start with `initialize_uniform_flow` for basic solver testing
   - Use `initialize_taylor_green` for verifying numerical accuracy 
   - Use analytical solutions when possible to enable error measurement

2. **For Realistic Simulations:**
   - Match initialization to your physical problem
   - Consider combining basic flows with `combine_flow_fields`
   - Add perturbations with `initialize_random_perturbation` for realistic turbulence

3. **When Continuing Previous Simulations:**
   - Use `initialize_flow_from_file` to import previous results
   - Ensure data format compatibility and sufficient resolution

### Performance Considerations

1. **For Large Meshes:**
   - Simple flows like `initialize_uniform_flow` are highly efficient
   - Complex flows with many elements (like SEM turbulence) can be computationally intensive
   - Consider initializing on a coarser auxiliary mesh and then interpolating

2. **Memory Usage:**
   - Most initialization functions have minimal memory overhead
   - `initialize_synthetic_turbulence` requires temporary storage proportional to number of eddies
   - `initialize_flow_from_file` may require significant memory for large datasets

### Numerical Stability

1. **Ensuring Valid Initial Conditions:**
   - Use `validate_flow_field` to check constraints before simulation
   - Avoid extreme velocity or pressure gradients that can cause numerical instability
   - Apply smoothing if needed for complex multi-component flows

2. **Managing Divergence:**
   - Ensure velocity fields are divergence-free for incompressible simulations
   - Some initializations (like `initialize_potential_flow`) guarantee divergence-free fields
   - For others, consider applying a projection step after initialization

## Examples

### Basic Flow Around Cylinder
```cpp
// Create uniform flow
auto base_flow = MeshUtils::initialize_uniform_flow(mesh, {1.0, 0.0});

// Add a doublet at origin for cylinder
std::vector<std::vector<double>> doublets = {{0.0, 0.0, 1.0, 0.0}};
auto potential_flow = MeshUtils::initialize_potential_flow(mesh, {}, {}, doublets, {1.0, 0.0});

// Validate the flow field
auto [valid, violations] = MeshUtils::validate_flow_field(mesh, potential_flow);
```

### Turbulent Channel Flow
```cpp
// Create parabolic channel flow
auto channel_flow = MeshUtils::initialize_channel_flow(mesh, 2.0, 1.0, {1.0, 0.0}, "parabolic");

// Add synthetic turbulence
auto turbulent_flow = MeshUtils::initialize_synthetic_turbulence(
    mesh, 
    {1.0, 0.0},                 // Mean flow
    {0.01, 0.005, 0.002},      // Reynolds stresses
    0.1,                        // Length scale
    200                         // Number of eddies
);

// Combine the flows with weights
auto combined_flow = MeshUtils::combine_flow_fields(
    mesh, 
    {channel_flow, turbulent_flow}, 
    "weighted", 
    {1.0, 0.1}    // 10% turbulence intensity
);
```

### Vortex Street Initialization
```cpp
std::vector<std::vector<double>> vortex_centers = {
    {0.3, 0.45}, {0.5, 0.55}, {0.7, 0.45}, {0.9, 0.55}
};
std::vector<double> vortex_strengths = {0.1, -0.1, 0.1, -0.1};
std::vector<double> vortex_radii = {0.05, 0.05, 0.05, 0.05};

// Create uniform background flow
auto base_flow = MeshUtils::initialize_uniform_flow(mesh, {0.5, 0.0});

// Add alternating vortices
auto vortex_flow = MeshUtils::initialize_vortex(
    mesh, 
    vortex_centers, 
    vortex_strengths, 
    vortex_radii, 
    {0.5, 0.0}    // Same background flow
);

// Validate combined flow
auto [valid, violations] = MeshUtils::validate_flow_field(mesh, vortex_flow);
```
