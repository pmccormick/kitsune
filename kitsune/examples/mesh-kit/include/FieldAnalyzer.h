/**
 * ====================================================================
 * FieldAnalyzer - Module for analyzing fields in a CFD mesh
 * ====================================================================
 */
#pragma once

#include "Mesh.h"
#include <vector>
#include <string>

/**
 * @namespace FieldAnalyzer
 * @brief Provides functions for analyzing and computing derived fields from mesh data
 */
namespace FieldAnalyzer {

    /**
     * @brief Calculate vorticity at a specific cell
     * @param mesh Reference to the mesh
     * @param i Grid index in x-direction
     * @param j Grid index in y-direction
     * @return Vorticity value (1/s)
     */
    double calculateVorticity(const Mesh& mesh, size_t i, size_t j);
    
    /**
     * @brief Calculate and return the vorticity field
     * @param mesh Reference to the mesh
     * @return Vector of vorticity values
     */
    std::vector<double> computeVorticityField(const Mesh& mesh);
    
    /**
     * @brief Calculate divergence at a specific cell
     * @param mesh Reference to the mesh
     * @param i Grid index in x-direction
     * @param j Grid index in y-direction
     * @return Divergence value (1/s)
     */
    double calculateDivergence(const Mesh& mesh, size_t i, size_t j);
    
    /**
     * @brief Calculate and return the divergence field
     * @param mesh Reference to the mesh
     * @return Vector of divergence values
     */
    std::vector<double> computeDivergenceField(const Mesh& mesh);
    
    /**
     * @brief Calculate and return the stream function field
     * @param mesh Reference to the mesh
     * @return Vector of stream function values
     */
    std::vector<double> computeStreamFunctionField(const Mesh& mesh);
    
    /**
     * @brief Calculate and return the kinetic energy field
     * @param mesh Reference to the mesh
     * @return Vector of kinetic energy values (J/kg)
     */
    std::vector<double> computeKineticEnergyField(const Mesh& mesh);
    
    /**
     * @brief Calculate the pressure gradient field
     * @param mesh Reference to the mesh
     * @param gradX Output vector for x-component of pressure gradient
     * @param gradY Output vector for y-component of pressure gradient
     */
    void computePressureGradientField(
        const Mesh& mesh, 
        std::vector<double>& gradX, 
        std::vector<double>& gradY);
    
    /**
     * @brief Calculate the temperature gradient field
     * @param mesh Reference to the mesh
     * @param gradX Output vector for x-component of temperature gradient
     * @param gradY Output vector for y-component of temperature gradient
     */
    void computeTemperatureGradientField(
        const Mesh& mesh, 
        std::vector<double>& gradX, 
        std::vector<double>& gradY);
    
    /**
     * @brief Calculate heat flux field based on temperature gradients
     * @param mesh Reference to the mesh
     * @param fluxX Output vector for x-component of heat flux
     * @param fluxY Output vector for y-component of heat flux
     */
    void computeHeatFluxField(
        const Mesh& mesh, 
        std::vector<double>& fluxX, 
        std::vector<double>& fluxY);
    
    /**
     * @brief Calculate wall shear stress along obstacle boundaries
     * @param mesh Reference to the mesh
     * @return Map of coordinates to shear stress values along walls
     */
    std::vector<std::tuple<double, double, double>> computeWallShearStress(const Mesh& mesh);
    
    /**
     * @brief Calculate the strain rate tensor components
     * @param mesh Reference to the mesh
     * @param strainXX Output vector for xx-component
     * @param strainXY Output vector for xy-component
     * @param strainYY Output vector for yy-component
     */
    void computeStrainRateTensor(
        const Mesh& mesh, 
        std::vector<double>& strainXX, 
        std::vector<double>& strainXY, 
        std::vector<double>& strainYY);
    
    /**
     * @brief Calculate the vorticity-streamfunction relationship
     * @param mesh Reference to the mesh
     * @param vorticity Input vorticity field
     * @return Stream function field calculated from vorticity
     */
    std::vector<double> solveStreamFunctionFromVorticity(
        const Mesh& mesh, 
        const std::vector<double>& vorticity);
    
    /**
     * @brief Calculate statistics for a field
     * @param field The field to analyze
     * @return Tuple containing (min, max, mean, standard deviation)
     */
    std::tuple<double, double, double, double> calculateFieldStatistics(
        const std::vector<double>& field);
    
    /**
     * @brief Find cells where a field exceeds a threshold
     * @param mesh Reference to the mesh
     * @param field The field to analyze
     * @param threshold The threshold value
     * @param operator The comparison operator (">", "<", ">=", "<=", "==")
     * @return Vector of (i, j) indices where the condition is met
     */
    std::vector<std::pair<size_t, size_t>> findCellsExceedingThreshold(
        const Mesh& mesh,
        const std::vector<double>& field,
        double threshold,
        const std::string& op = ">");
    
    /**
     * @brief Interpolate a field value at a specific physical location
     * @param mesh Reference to the mesh
     * @param field The field to interpolate from
     * @param x Physical x-coordinate in meters
     * @param y Physical y-coordinate in meters
     * @return Interpolated field value
     */
    double interpolateField(
        const Mesh& mesh,
        const std::vector<double>& field,
        double x, double y);
    
    /**
     * @brief Calculate the circulation around a closed path
     * @param mesh Reference to the mesh
     * @param pathPoints Vector of (x,y) coordinates defining the closed path
     * @return Circulation value
     */
    double calculateCirculation(
        const Mesh& mesh,
        const std::vector<std::pair<double, double>>& pathPoints);
    
    /**
     * @brief Calculate the net flux through a closed path
     * @param mesh Reference to the mesh
     * @param pathPoints Vector of (x,y) coordinates defining the closed path
     * @return Net flux value
     */
    double calculateFlux(
        const Mesh& mesh,
        const std::vector<std::pair<double, double>>& pathPoints);
    
    /**
     * @brief Calculate the curl of a vector field
     * @param mesh Reference to the mesh
     * @param vectorFieldX X-component of the vector field
     * @param vectorFieldY Y-component of the vector field
     * @return Curl (vorticity) field
     */
    std::vector<double> calculateCurl(
        const Mesh& mesh,
        const std::vector<double>& vectorFieldX,
        const std::vector<double>& vectorFieldY);
    
    /**
     * @brief Calculate the divergence of a vector field
     * @param mesh Reference to the mesh
     * @param vectorFieldX X-component of the vector field
     * @param vectorFieldY Y-component of the vector field
     * @return Divergence field
     */
    std::vector<double> calculateDivergence(
        const Mesh& mesh,
        const std::vector<double>& vectorFieldX,
        const std::vector<double>& vectorFieldY);
    
    /**
     * @brief Calculate the gradient of a scalar field
     * @param mesh Reference to the mesh
     * @param scalarField The scalar field
     * @param gradientX Output vector for x-component of gradient
     * @param gradientY Output vector for y-component of gradient
     */
    void calculateGradient(
        const Mesh& mesh,
        const std::vector<double>& scalarField,
        std::vector<double>& gradientX,
        std::vector<double>& gradientY);
    
    /**
     * @brief Compute the Laplacian of a scalar field
     * @param mesh Reference to the mesh
     * @param scalarField The scalar field
     * @return Laplacian field
     */
    std::vector<double> calculateLaplacian(
        const Mesh& mesh,
        const std::vector<double>& scalarField);
    
    /**
     * @brief Calculate the average value of a field in a region
     * @param mesh Reference to the mesh
     * @param field The field to analyze
     * @param i_start Starting grid index in x-direction
     * @param i_end Ending grid index in x-direction
     * @param j_start Starting grid index in y-direction
     * @param j_end Ending grid index in y-direction
     * @return Average field value in the region
     */
    double calculateRegionAverage(
        const Mesh& mesh,
        const std::vector<double>& field,
        size_t i_start, size_t i_end,
        size_t j_start, size_t j_end);
    
    /**
     * @brief Calculate the average value of a field in a circular region
     * @param mesh Reference to the mesh
     * @param field The field to analyze
     * @param centerX X-coordinate of circle center in meters
     * @param centerY Y-coordinate of circle center in meters
     * @param radius Circle radius in meters
     * @return Average field value in the circular region
     */
    double calculateCircularRegionAverage(
        const Mesh& mesh,
        const std::vector<double>& field,
        double centerX, double centerY, double radius);
    
    /**
     * @brief Calculate flow rate through a line segment
     * @param mesh Reference to the mesh
     * @param x1 X-coordinate of first endpoint in meters
     * @param y1 Y-coordinate of first endpoint in meters
     * @param x2 X-coordinate of second endpoint in meters
     * @param y2 Y-coordinate of second endpoint in meters
     * @return Flow rate through the line segment
     */
    double calculateFlowRate(
        const Mesh& mesh,
        double x1, double y1, double x2, double y2);
    
    /**
     * @brief Calculate forces (pressure and viscous) on obstacle boundaries
     * @param mesh Reference to the mesh
     * @param forceX Output sum of x-component of forces
     * @param forceY Output sum of y-component of forces
     */
    void calculateForces(
        const Mesh& mesh,
        double& forceX, double& forceY);
    
    /**
     * @brief Identify vortex cores in the flow field
     * @param mesh Reference to the mesh
     * @return Vector of (x, y, strength) for identified vortex cores
     */
    std::vector<std::tuple<double, double, double>> identifyVortexCores(const Mesh& mesh);
}
