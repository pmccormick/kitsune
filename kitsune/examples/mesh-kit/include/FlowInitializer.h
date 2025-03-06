/**
 * ====================================================================
 * FlowInitializer - Module for creating flow field patterns in a CFD mesh
 * ====================================================================
 */
#pragma once

#include "Mesh.h"
#include <string>
#include <functional>

/**
 * @namespace FlowInitializer
 * @brief Provides functions for initializing different flow patterns in a mesh
 */
namespace FlowInitializer {

    /**
     * @brief Apply a uniform flow field
     * @param mesh Reference to the mesh
     * @param vx X-velocity in m/s
     * @param vy Y-velocity in m/s
     */
    void applyUniformFlow(Mesh& mesh, double vx, double vy = 0.0);
    
    /**
     * @brief Apply a uniform flow field with unit conversion
     * @param mesh Reference to the mesh
     * @param vx X-velocity in specified units
     * @param vy Y-velocity in specified units
     * @param velocityUnit Velocity unit (e.g., "m/s", "mph", "knot")
     */
    void applyUniformFlowWithUnits(
        Mesh& mesh,
        double vx, double vy = 0.0, 
        const std::string &velocityUnit = "m/s");
    
    /**
     * @brief Apply a parabolic flow profile
     * @param mesh Reference to the mesh
     * @param maxSpeed Maximum flow speed in m/s
     * @param direction Primary flow direction ('x' or 'y')
     */
    void applyParabolicFlow(
        Mesh& mesh,
        double maxSpeed, 
        char direction = 'x');
    
    /**
     * @brief Apply a parabolic flow profile with unit conversion
     * @param mesh Reference to the mesh
     * @param maxSpeed Maximum flow speed in specified units
     * @param direction Primary flow direction ('x' or 'y')
     * @param velocityUnit Velocity unit (e.g., "m/s", "mph", "knot")
     */
    void applyParabolicFlowWithUnits(
        Mesh& mesh,
        double maxSpeed, 
        char direction = 'x', 
        const std::string &velocityUnit = "m/s");
    
    /**
     * @brief Apply a shear flow
     * @param mesh Reference to the mesh
     * @param minSpeed Minimum flow speed in m/s
     * @param maxSpeed Maximum flow speed in m/s
     * @param direction Primary flow direction ('x' or 'y')
     */
    void applyShearFlow(
        Mesh& mesh,
        double minSpeed, double maxSpeed, 
        char direction = 'x');
    
    /**
     * @brief Apply a shear flow with unit conversion
     * @param mesh Reference to the mesh
     * @param minSpeed Minimum flow speed in specified units
     * @param maxSpeed Maximum flow speed in specified units
     * @param direction Primary flow direction ('x' or 'y')
     * @param velocityUnit Velocity unit (e.g., "m/s", "mph", "knot")
     */
    void applyShearFlowWithUnits(
        Mesh& mesh,
        double minSpeed, double maxSpeed, 
        char direction = 'x',
        const std::string &velocityUnit = "m/s");
    
    /**
     * @brief Apply a vortex flow
     * @param mesh Reference to the mesh
     * @param centerX X-coordinate of vortex center in meters
     * @param centerY Y-coordinate of vortex center in meters
     * @param maxSpeed Maximum tangential speed in m/s
     * @param radius Radius of maximum speed in meters
     */
    void applyVortexFlow(
        Mesh& mesh,
        double centerX, double centerY, 
        double maxSpeed, double radius);
    
    /**
     * @brief Apply a vortex flow with unit conversion
     * @param mesh Reference to the mesh
     * @param centerX X-coordinate of vortex center in specified units
     * @param centerY Y-coordinate of vortex center in specified units
     * @param maxSpeed Maximum tangential speed in specified units
     * @param radius Radius of maximum speed in specified units
     * @param lengthUnit Length unit (e.g., "m", "ft", "in")
     * @param velocityUnit Velocity unit (e.g., "m/s", "mph", "knot")
     */
    void applyVortexFlowWithUnits(
        Mesh& mesh,
        double centerX, double centerY, 
        double maxSpeed, double radius,
        const std::string &lengthUnit = "m", 
        const std::string &velocityUnit = "m/s");
    
    /**
     * @brief Apply a jet flow profile
     * @param mesh Reference to the mesh
     * @param entryX X-coordinate of jet entry point in meters
     * @param entryY Y-coordinate of jet entry point in meters
     * @param direction Direction of the jet in radians (0 = right, π/2 = up)
     * @param jetWidth Width of the jet in meters
     * @param jetSpeed Maximum speed of the jet in m/s
     */
    void applyJetFlow(
        Mesh& mesh,
        double entryX, double entryY, 
        double direction, 
        double jetWidth, double jetSpeed);
    
    /**
     * @brief Apply a custom flow field defined by a function
     * @param mesh Reference to the mesh
     * @param velocityFunction Function taking (x, y) coordinates and returning (vx, vy) velocity components
     */
    void applyCustomFlow(
        Mesh& mesh,
        std::function<std::pair<double, double>(double, double)> velocityFunction);
    
    /**
     * @brief Initialize the flow field based on potential flow theory
     * @param mesh Reference to the mesh
     * @param freeStreamSpeed Free-stream velocity in m/s
     * @param angle Angle of attack in radians
     */
    void applyPotentialFlow(
        Mesh& mesh,
        double freeStreamSpeed, 
        double angle = 0.0);
    
    /**
     * @brief Apply a temperature gradient to the flow field
     * @param mesh Reference to the mesh
     * @param startTemp Starting temperature in Kelvin
     * @param endTemp Ending temperature in Kelvin
     * @param direction Direction of gradient ('x', 'y', or 'r' for radial)
     */
    void applyTemperatureGradient(
        Mesh& mesh,
        double startTemp, double endTemp, 
        char direction = 'x');
    
    /**
     * @brief Apply a temperature gradient with unit conversion
     * @param mesh Reference to the mesh
     * @param startTemp Starting temperature in specified units
     * @param endTemp Ending temperature in specified units
     * @param direction Direction of gradient ('x', 'y', or 'r' for radial)
     * @param tempUnit Temperature unit (e.g., "K", "C", "F")
     */
    void applyTemperatureGradientWithUnits(
        Mesh& mesh,
        double startTemp, double endTemp, 
        char direction = 'x',
        const std::string &tempUnit = "K");
    
    /**
     * @brief Apply a pressure gradient to the flow field
     * @param mesh Reference to the mesh
     * @param startPressure Starting pressure in Pascal
     * @param endPressure Ending pressure in Pascal
     * @param direction Direction of gradient ('x', 'y', or 'r' for radial)
     */
    void applyPressureGradient(
        Mesh& mesh,
        double startPressure, double endPressure, 
        char direction = 'x');
    
    /**
     * @brief Apply a pressure gradient with unit conversion
     * @param mesh Reference to the mesh
     * @param startPressure Starting pressure in specified units
     * @param endPressure Ending pressure in specified units
     * @param direction Direction of gradient ('x', 'y', or 'r' for radial)
     * @param pressureUnit Pressure unit (e.g., "Pa", "bar", "atm")
     */
    void applyPressureGradientWithUnits(
        Mesh& mesh,
        double startPressure, double endPressure, 
        char direction = 'x',
        const std::string &pressureUnit = "Pa");
        
    /**
     * @brief Apply a boundary layer velocity profile near walls
     * @param mesh Reference to the mesh
     * @param freeStreamSpeed Free-stream velocity in m/s
     * @param boundaryLayerThickness Boundary layer thickness in meters
     * @param profileType Type of boundary layer profile ("laminar" or "turbulent")
     */
    void applyBoundaryLayerProfile(
        Mesh& mesh,
        double freeStreamSpeed, 
        double boundaryLayerThickness,
        const std::string &profileType = "laminar");
}

