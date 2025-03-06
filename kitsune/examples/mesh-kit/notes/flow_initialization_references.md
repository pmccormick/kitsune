# Flow Initialization References and Examples

This document provides references to online resources, papers, and examples for implementing and understanding the real and advanced flow initialization functions in our framework.

## Real Simulation Functions

### Shear Flow

1. **NASA Turbulence Modeling Resource: Mixing Layer**  
   [https://turbmodels.larc.nasa.gov/mixlayer_val.html](https://turbmodels.larc.nasa.gov/mixlayer_val.html)  
   Provides validation cases and detailed velocity profiles for mixing layers, which are a classic application of shear flow initialization.

2. **OpenFOAM Shear Flow Example**  
   [https://www.openfoam.com/documentation/guides/latest/doc/verification-validation-shear-flow.html](https://www.openfoam.com/documentation/guides/latest/doc/verification-validation-shear-flow.html)  
   Implementation example showing how to set up a canonical shear flow test case in OpenFOAM.

3. **Blaisdell, G.A., Mansour, N.N., & Reynolds, W.C. (1991). "Numerical simulations of homogeneous turbulent shear flow"**  
   Stanford University Technical Report - [https://ntrs.nasa.gov/citations/19940019372](https://ntrs.nasa.gov/citations/19940019372)  
   Classic paper describing initialization strategies for turbulent shear flows.

### Channel Flow

1. **ERCOFTAC Classic Collection Database: Channel Flow**  
   [http://cfd.mace.manchester.ac.uk/ercoftac/](http://cfd.mace.manchester.ac.uk/ercoftac/)  
   Collection of experimental data for channel flow at various Reynolds numbers, useful for validation.

2. **DNS Database of Turbulent Channel Flow by MKM**  
   [https://turbulence.oden.utexas.edu/](https://turbulence.oden.utexas.edu/)  
   High-fidelity DNS data for fully-developed channel flow, including mean velocity profiles for initialization.

3. **Freitas, C.J. (1995). "Perspective: Selected Benchmarks From Commercial CFD Codes"**  
   Journal of Fluids Engineering - [https://doi.org/10.1115/1.2817139](https://doi.org/10.1115/1.2817139)  
   Includes standard channel flow initialization parameters used in benchmark studies.

4. **Poiseuille Flow Profile Generator (Online Tool)**  
   [https://www.simscale.com/blog/2017/08/poiseuille-flow/](https://www.simscale.com/blog/2017/08/poiseuille-flow/)  
   Interactive tool for generating parabolic velocity profiles for channel flows.

### Boundary Layer

1. **NASA Langley Research Center: 2D Zero Pressure Gradient Flat Plate**  
   [https://turbmodels.larc.nasa.gov/flatplate.html](https://turbmodels.larc.nasa.gov/flatplate.html)  
   Detailed description of boundary layer initialization for a standard flat plate test case.

2. **University of Manchester Boundary Layer Database**  
   [http://personalpages.manchester.ac.uk/staff/david.d.apsley/Research/research.html](http://personalpages.manchester.ac.uk/staff/david.d.apsley/Research/research.html)  
   Collection of experimental boundary layer profiles under various conditions.

3. **White, F.M. (2006). "Viscous Fluid Flow"**  
   McGraw-Hill, Section 4-4: Boundary Layer Similarity Solutions  
   Classical reference for analytical solutions of boundary layer profiles (Blasius solution, etc.).

4. **CFD Online: Boundary Layer Calculator**  
   [https://www.cfd-online.com/Tools/turbulent.php](https://www.cfd-online.com/Tools/turbulent.php)  
   Online tool for calculating turbulent boundary layer velocity profiles.

### Stagnation Flow

1. **Hiemenz Flow Solution**  
   [https://en.wikipedia.org/wiki/Hiemenz_flow](https://en.wikipedia.org/wiki/Hiemenz_flow)  
   Analytical solution for stagnation flow against a flat plate, used as reference.

2. **Tufts University: Potential Flow Around a Cylinder**  
   [http://sites.tufts.edu/andrewrosen/files/2014/04/twoD_flow_cylinder.pdf](http://sites.tufts.edu/andrewrosen/files/2014/04/twoD_flow_cylinder.pdf)  
   Includes stagnation flow initialization examples for flow around a cylinder.

3. **Schlichting, H., & Gersten, K. (2016). "Boundary-Layer Theory"**  
   Springer, Chapter 5: The Fundamentals of Boundary-Layer Theory  
   Describes exact solution of stagnation point flow and practical implementations.

### Jet Flow

1. **NASA Jet Flow Experiment Database**  
   [https://ntrs.nasa.gov/citations/20150002081](https://ntrs.nasa.gov/citations/20150002081)  
   Experimental data for axisymmetric jet flow, including velocity profiles for validation.

2. **OpenFOAM Jet Flow Tutorial**  
   [https://www.openfoam.com/documentation/tutorial-guide/tutorialse2.php](https://www.openfoam.com/documentation/tutorial-guide/tutorialse2.php)  
   Step-by-step tutorial for setting up and initializing a turbulent jet flow in OpenFOAM.

3. **Pope, S.B. (2000). "Turbulent Flows"**  
   Cambridge University Press, Section 5.2: Free Shear Flows  
   Provides analytical expressions for jet velocity profiles and spreading rates.

4. **Stanford Center for Turbulence Research: Jets and Plumes**  
   [https://ctr.stanford.edu/research-areas/combustion-modeling-simulation-and-theory](https://ctr.stanford.edu/research-areas/combustion-modeling-simulation-and-theory)  
   Resources for jets and plumes in various configurations, including initialization methods.

### Flow From File

1. **ERCOFTAC QNET Knowledge Base Wiki**  
   [http://www.ercoftac.org/kb_cfd_for_industrial_flows/](http://www.ercoftac.org/kb_cfd_for_industrial_flows/)  
   Collection of test cases with downloadable data files for initial conditions.

2. **Tecplot Data Format**  
   [https://tecplot.azureedge.net/doc/360/data_format_guide.pdf](https://tecplot.azureedge.net/doc/360/data_format_guide.pdf)  
   Standard format specification for CFD data files, useful for implementing file readers.

3. **ParaView State Files for CFD**  
   [https://www.paraview.org/Wiki/The_ParaView_Tutorial](https://www.paraview.org/Wiki/The_ParaView_Tutorial)  
   Examples of how to save and load flow field data using ParaView state files.

## Advanced Flow Functions

### Potential Flow

1. **MIT OpenCourseWare: Potential Flow**  
   [https://ocw.mit.edu/courses/mechanical-engineering/2-25-advanced-fluid-mechanics-fall-2013/potential-flows/](https://ocw.mit.edu/courses/mechanical-engineering/2-25-advanced-fluid-mechanics-fall-2013/potential-flows/)  
   Lecture materials with examples of source, sink, vortex, and doublet implementations.

2. **NASA Potential Flow Theory**  
   [https://www.grc.nasa.gov/www/k-12/airplane/potflow.html](https://www.grc.nasa.gov/www/k-12/airplane/potflow.html)  
   Practical examples of how to implement and combine potential flow elements.

3. **SU2 Potential Flow Example**  
   [https://su2code.github.io/tutorials/Potential_Flow/](https://su2code.github.io/tutorials/Potential_Flow/)  
   Step-by-step example of setting up a potential flow simulation with flow elements.

4. **Interactive Potential Flow Simulator**  
   [https://www.grc.nasa.gov/www/k-12/airplane/tuniv.html](https://www.grc.nasa.gov/www/k-12/airplane/tuniv.html)  
   Online tool for visualizing combinations of potential flow elements.

### Synthetic Turbulence

1. **Jarrin, N., et al. (2006). "A synthetic-eddy-method for generating inflow conditions for large-eddy simulations"**  
   International Journal of Heat and Fluid Flow - [https://doi.org/10.1016/j.ijheatfluidflow.2006.02.006](https://doi.org/10.1016/j.ijheatfluidflow.2006.02.006)  
   Original paper describing the Synthetic Eddy Method (SEM) with implementation details.

2. **Poletto, R., et al. (2013). "A new divergence free synthetic eddy method for the reproduction of inlet flow conditions for LES"**  
   Flow, Turbulence and Combustion - [https://doi.org/10.1007/s10494-013-9488-2](https://doi.org/10.1007/s10494-013-9488-2)  
   Improved version of SEM with divergence-free constraint.

3. **ANSYS Synthetic Turbulence Generator**  
   [https://ansyshelp.ansys.com/account/secured?returnurl=/Views/Secured/corp/v231/en/flu_th/flu_turb_syn_vortex_method.html](https://ansyshelp.ansys.com/account/secured?returnurl=/Views/Secured/corp/v231/en/flu_th/flu_turb_syn_vortex_method.html)  
   Commercial implementation of synthetic turbulence with practical examples.

4. **GitHub: channelflow-sm**  
   [https://github.com/ckjeong/channelflow-SEM](https://github.com/ckjeong/channelflow-SEM)  
   Open-source implementation of SEM for channel flow.

### Stratified Flow

1. **Woods Hole Oceanographic Institution: Stratified Flow Examples**  
   [https://www.whoi.edu/science/PO/people/pkumar/stratified.html](https://www.whoi.edu/science/PO/people/pkumar/stratified.html)  
   Collection of stratified flow examples with initialization parameters.

2. **Turner, J.S. (1979). "Buoyancy Effects in Fluids"**  
   Cambridge University Press  
   Classical reference for density stratification in various flow scenarios.

3. **GOTM (General Ocean Turbulence Model)**  
   [https://gotm.net/portfolio/scenarios/](https://gotm.net/portfolio/scenarios/)  
   Collection of stratified flow scenarios with initialization profiles.

4. **Stratified Flow Calculator (Environmental Fluid Mechanics)**  
   [https://www.waterqualityresearch.org/stratified-flow-calculator](https://www.waterqualityresearch.org/stratified-flow-calculator)  
   Online tool for calculating density profiles in stratified environments.

### Wave Flow

1. **MIT Wave Generation and Analysis**  
   [https://ocw.mit.edu/courses/mechanical-engineering/2-22-design-principles-for-ocean-vehicles-13-42-spring-2005/readings/r8_wavespectra.pdf](https://ocw.mit.edu/courses/mechanical-engineering/2-22-design-principles-for-ocean-vehicles-13-42-spring-2005/readings/r8_wavespectra.pdf)  
   Comprehensive guide to wave generation methods with velocity profiles.

2. **WEC-Sim (Wave Energy Converter Simulator)**  
   [https://wec-sim.github.io/WEC-Sim/](https://wec-sim.github.io/WEC-Sim/)  
   Open-source code for wave generation, including initialization functions.

3. **REEF3D Wave Generation Library**  
   [https://reef3d.wordpress.com/](https://reef3d.wordpress.com/)  
   Open-source implementation of various wave theories and generation methods.

4. **Interactive Wave Calculator**  
   [http://www.coastal.udel.edu/faculty/rad/wavetheory.html](http://www.coastal.udel.edu/faculty/rad/wavetheory.html)  
   Online tool for calculating wave properties and velocity profiles.

### Swirling Flow

1. **NASA Turbulence Modeling Resource: Swirling Flow**  
   [https://turbmodels.larc.nasa.gov/swirlingflow_val.html](https://turbmodels.larc.nasa.gov/swirlingflow_val.html)  
   Validation case for swirling flows with detailed initialization parameters.

2. **Escudier, M. (1988). "Confined vortices in flow machinery"**  
   Annual Review of Fluid Mechanics - [https://doi.org/10.1146/annurev.fl.20.010188.002225](https://doi.org/10.1146/annurev.fl.20.010188.002225)  
   Provides analytical expressions for various swirling flow profiles.

3. **Rankine Vortex Model Implementation**  
   [https://www.cfd-online.com/Wiki/Rankine_vortex](https://www.cfd-online.com/Wiki/Rankine_vortex)  
   Practical implementation details for the Rankine vortex model.

4. **Tornado Simulation Database**  
   [https://vortexlab.sites.ttu.edu/](https://vortexlab.sites.ttu.edu/)  
   Collection of tornado-like vortex data with velocity profiles at different radii.

## Examples from Open-Source CFD Projects

### OpenFOAM

1. **Channel Flow Tutorial**  
   Source: [https://github.com/OpenFOAM/OpenFOAM-dev/tree/master/tutorials/incompressible/simpleFoam/pitzDaily](https://github.com/OpenFOAM/OpenFOAM-dev/tree/master/tutorials/incompressible/simpleFoam/pitzDaily)  
   Demonstrates parabolic velocity profile initialization for channel flow.

2. **Synthetic Turbulence Implementation**  
   Source: [https://github.com/OpenFOAM/OpenFOAM-dev/blob/master/src/TurbulenceModels/turbulenceModels/LES/SmagorinskyZhang/SmagorinskyZhang.C](https://github.com/OpenFOAM/OpenFOAM-dev/blob/master/src/TurbulenceModels/turbulenceModels/LES/SmagorinskyZhang/SmagorinskyZhang.C)  
   OpenFOAM's implementation of synthetic turbulence for inflow conditions.

### Palabos

1. **Jet Flow Example**  
   Source: [https://github.com/palabos-code/palabos/tree/master/examples/showCases/jet](https://github.com/palabos-code/palabos/tree/master/examples/showCases/jet)  
   Implementation of jet flow initialization in the Palabos library.

2. **Stratified Flow Example**  
   Source: [https://github.com/palabos-code/palabos/tree/master/examples/showCases/boussinesq](https://github.com/palabos-code/palabos/tree/master/examples/showCases/boussinesq)  
   Example of stratified flow initialization using the Boussinesq approximation.

### SU2

1. **Potential Flow Implementation**  
   Source: [https://github.com/su2code/SU2/blob/master/SU2_CFD/src/solvers/CPotentialSolver.cpp](https://github.com/su2code/SU2/blob/master/SU2_CFD/src/solvers/CPotentialSolver.cpp)  
   Implementation of potential flow solver with initialization functions.

2. **Boundary Layer Initialization**  
   Source: [https://github.com/su2code/SU2/blob/master/SU2_CFD/src/drivers/CDriver.cpp](https://github.com/su2code/SU2/blob/master/SU2_CFD/src/drivers/CDriver.cpp)  
   Code showing how boundary layer profiles are initialized in SU2.

## Implementation Examples for Specific Functions

### Channel Flow Implementation Example

```cpp
// From: Stanford University CFD Group
// Source: https://github.com/stanfordhpccenter/cfd-lab

void initializeChannelFlow(double *u, double *v, double *p, int nx, int ny, double h, double u_max) {
    double y;
    for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++) {
            // Get y coordinate normalized to [0,1]
            y = (double)j / (ny - 1);
            
            // Parabolic profile: u = u_max * 4 * y * (1-y)
            u[i + j*nx] = u_max * 4.0 * y * (1.0 - y);
            
            // Zero vertical velocity
            v[i + j*nx] = 0.0;
            
            // Linear pressure drop
            p[i + j*nx] = (nx - i) * (8.0 * u_max * u_max) / (nx - 1);
        }
    }
}
```

### Synthetic Turbulence Implementation Example

```cpp
// From: Jarrin, N., et al. (2006)
// Simplified excerpt from: https://github.com/tgibson11/synthTurb

// Generate synthetic eddies
for (int i = 0; i < numEddies; i++) {
    // Random position
    eddy_pos[i][0] = randf() * (x_max - x_min) + x_min;
    eddy_pos[i][1] = randf() * (y_max - y_min) + y_min;
    eddy_pos[i][2] = randf() * (z_max - z_min) + z_min;
    
    // Random eddy intensity (normal distribution)
    for (int j = 0; j < 3; j++) {
        eddy_intensity[i][j] = randn();
    }
}

// Calculate velocity at each mesh point
for (int i = 0; i < numPoints; i++) {
    double u_fluct[3] = {0.0, 0.0, 0.0};
    
    // Sum contribution from all eddies
    for (int j = 0; j < numEddies; j++) {
        // Distance to eddy
        double dx = mesh_pos[i][0] - eddy_pos[j][0];
        double dy = mesh_pos[i][1] - eddy_pos[j][1];
        double dz = mesh_pos[i][2] - eddy_pos[j][2];
        
        // Normalized distance
        double rx = dx / length_scale;
        double ry = dy / length_scale;
        double rz = dz / length_scale;
        
        // Shape function
        double fx = (fabs(rx) < 1.0) ? sqrt(1.0 - fabs(rx)) : 0.0;
        double fy = (fabs(ry) < 1.0) ? sqrt(1.0 - fabs(ry)) : 0.0;
        double fz = (fabs(rz) < 1.0) ? sqrt(1.0 - fabs(rz)) : 0.0;
        
        // Add contribution
        for (int k = 0; k < 3; k++) {
            u_fluct[k] += eddy_intensity[j][k] * fx * fy * fz;
        }
    }
    
    // Apply Cholesky decomposition and scale
    double scale_factor = 1.0 / sqrt((double)numEddies);
    velocity[i][0] = mean_velocity[0] + (a11 * u_fluct[0]) * scale_factor;
    velocity[i][1] = mean_velocity[1] + (a21 * u_fluct[0] + a22 * u_fluct[1]) * scale_factor;
    velocity[i][2] = mean_velocity[2] + (a31 * u_fluct[0] + a32 * u_fluct[1] + a33 * u_fluct[2]) * scale_factor;
}
```

### Potential Flow Implementation Example

```cpp
// From: MIT OCW 2.25 Advanced Fluid Mechanics
// Source: https://ocw.mit.edu/courses/mechanical-engineering/2-25-advanced-fluid-mechanics-fall-2013/

// Example for computing velocity from complex potential
void computePotentialFlowVelocity(double x, double y, double *u, double *v,
                                 double *strength_source, double *x_source, double *y_source, int num_sources,
                                 double *strength_vortex, double *x_vortex, double *y_vortex, int num_vortices,
                                 double u_inf, double v_inf) {
    *u = u_inf;
    *v = v_inf;
    
    // Add source/sink contributions
    for (int i = 0; i < num_sources; i++) {
        double dx = x - x_source[i];
        double dy = y - y_source[i];
        double r_squared = dx*dx + dy*dy;
        
        if (r_squared > 1e-10) {
            double factor = strength_source[i] / (2.0 * M_PI * r_squared);
            *u += factor * dx;
            *v += factor * dy;
        }
    }
    
    // Add vortex contributions
    for (int i = 0; i < num_vortices; i++) {
        double dx = x - x_vortex[i];
        double dy = y - y_vortex[i];
        double r_squared = dx*dx + dy*dy;
        
        if (r_squared > 1e-10) {
            double factor = strength_vortex[i] / (2.0 * M_PI * r_squared);
            *u += -factor * dy;
            *v += factor * dx;
        }
    }
}
```
