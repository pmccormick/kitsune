#include "ConfigManager.h"
#include <iostream>

void setupDefaultConfig(ConfigManager& config) {
    // Simulation parameters
    config.setParameter<int>("Simulation", "maxIterations", 5000, "Maximum number of simulation iterations");
    config.setParameter<double>("Simulation", "timeStep", 0.001, "Time step size (s)");
    config.setParameter<double>("Simulation", "endTime", 10.0, "Simulation end time (s)");
    config.setParameter<bool>("Simulation", "adaptiveTimeStep", true, "Use adaptive time-stepping");
    config.setParameter<int>("Simulation", "saveInterval", 100, "Interval between saving results");
    
    // Grid parameters
    config.setParameter<int>("Grid", "width", 400, "Number of cells in x-direction");
    config.setParameter<int>("Grid", "height", 200, "Number of cells in y-direction");
    config.setParameter<double>("Grid", "physicalWidth", 2.0, "Physical width of domain (m)");
    config.setParameter<double>("Grid", "physicalHeight", 1.0, "Physical height of domain (m)");
    
    // Fluid parameters
    config.setParameter<double>("Fluid", "density", 1000.0, "Fluid density (kg/m³)");
    config.setParameter<double>("Fluid", "viscosity", 0.001, "Fluid dynamic viscosity (Pa·s)");
    config.setParameter<double>("Fluid", "inletVelocity", 1.0, "Inlet velocity (m/s)");
    config.setParameter<double>("Fluid", "inletTemperature", 300.0, "Inlet temperature (K)");
    config.setParameter<double>("Fluid", "ambientTemperature", 293.0, "Ambient temperature (K)");
    
    // Cylinder obstacle parameters
    config.setParameter<std::vector<double>>("Obstacle", "position", {0.5, 0.5}, "Cylinder center position (x,y)");
    config.setParameter<double>("Obstacle", "radius", 0.1, "Cylinder radius (m)");
    config.setParameter<double>("Obstacle", "temperature", 350.0, "Cylinder surface temperature (K)");
    
    // Boundary conditions
    config.setParameter<std::string>("Boundary", "leftType", "inlet", "Left boundary type (inlet/wall/etc)");
    config.setParameter<std::string>("Boundary", "rightType", "outlet", "Right boundary type");
    config.setParameter<std::string>("Boundary", "topType", "wall", "Top boundary type");
    config.setParameter<std::string>("Boundary", "bottomType", "wall", "Bottom boundary type");
    
    // Solver parameters
    config.setParameter<std::string>("Solver", "pressureMethod", "SIMPLE", "Pressure-velocity coupling method");
    config.setParameter<int>("Solver", "maxPressureIterations", 50, "Max iterations for pressure solver");
    config.setParameter<double>("Solver", "convergenceTolerance", 1e-6, "Convergence tolerance");
    config.setParameter<bool>("Solver", "useMultigrid", true, "Use multigrid acceleration");
    
    // Visualization parameters
    config.setParameter<bool>("Visualization", "saveVelocity", true, "Save velocity field");
    config.setParameter<bool>("Visualization", "savePressure", true, "Save pressure field");
    config.setParameter<bool>("Visualization", "saveTemperature", true, "Save temperature field");
    config.setParameter<std::string>("Visualization", "outputDirectory", "./results", "Directory for output files");
    config.setParameter<std::string>("Visualization", "outputFormat", "VTK", "Output file format");
}

int main() {
    ConfigManager config;
    
    // Setup default configuration
    setupDefaultConfig(config);
    
    // Save to file
    config.saveToFile("simulation_config.ini");
    
    // Print current configuration
    config.printConfig();
    
    // Modify some parameters
    config.setParameter<int>("Grid", "width", 800);
    config.setParameter<int>("Grid", "height", 400);
    config.setParameter<double>("Fluid", "inletVelocity", 2.0);
    
    // Save modified configuration
    config.saveToFile("simulation_config_modified.ini");
    
    // Load configuration from file
    ConfigManager loadedConfig;
    if (loadedConfig.loadFromFile("simulation_config.ini")) {
        std::cout << "\nLoaded configuration:\n";
        loadedConfig.printConfig();
    }
    
    // Access parameters
    double dt = config.getParameter<double>("Simulation", "timeStep");
    double density = config.getParameter<double>("Fluid", "density");
    std::string pressureMethod = config.getParameter<std::string>("Solver", "pressureMethod");
    
    std::cout << "\nAccessing specific parameters:\n";
    std::cout << "Time step: " << dt << " s\n";
    std::cout << "Fluid density: " << density << " kg/m³\n";
    std::cout << "Pressure method: " << pressureMethod << "\n";
    
    return 0;
}


