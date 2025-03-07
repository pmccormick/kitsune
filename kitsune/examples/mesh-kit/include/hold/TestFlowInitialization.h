#ifndef TEST_FLOW_INITIALIZATION_H
#define TEST_FLOW_INITIALIZATION_H

#include <vector>
#include <array>
#include <cmath>
#include <random>
#include <string>
#include <unordered_map>
#include <functional>
#include <algorithm>
#include <memory>
#include "mesh.h"

namespace MeshUtils {

/**
 * @brief Initialize the entire domain with a uniform flow field.
 * 
 * @param mesh The computational mesh containing node coordinates and cell information
 * @param velocity Vector representing uniform velocity (2D or 3D)
 * @param pressure Uniform pressure value
 * @return FlowField object containing initialized flow field variables
 */
inline FlowField initialize_uniform_flow(const Mesh& mesh, 
                                 const std::vector<double>& velocity = {1.0, 0.0}, 
                                 double pressure = 0.0) {
    FlowField flow_field;
    size_t num_nodes = mesh.get_node_count();
    size_t dim = velocity.size();
    
    // Initialize velocity field
    flow_field.velocity.resize(num_nodes, std::vector<double>(dim));
    flow_field.pressure.resize(num_nodes);
    
    // Set uniform values
    for (size_t i = 0; i < num_nodes; ++i) {
        for (size_t j = 0; j < dim; ++j) {
            flow_field.velocity[i][j] = velocity[j];
        }
        flow_field.pressure[i] = pressure;
    }
    
    return flow_field;
}

/**
 * @brief Initialize a Taylor-Green vortex field - a standard test case for turbulence.
 * 
 * The Taylor-Green vortex is an exact solution of the incompressible Navier-Stokes
 * equations and is commonly used to test numerical schemes.
 * 
 * For 2D, the velocity field is:
 * u = A * sin(2π*x/λx) * cos(2π*y/λy) * exp(-decay_factor)
 * v = -A * cos(2π*x/λx) * sin(2π*y/λy) * exp(-decay_factor)
 * p = (A²/4) * (cos(4π*x/λx) + cos(4π*y/λy)) * exp(-2*decay_factor)
 * 
 * @param mesh The computational mesh containing node coordinates
 * @param amplitude Velocity amplitude
 * @param wavelength Spatial wavelength(s) of vortices. If a single value, the same wavelength
 *                   is used for all directions. If a vector, individual wavelengths per dimension.
 * @param decay_factor Temporal decay factor (for analytical solutions at time > 0)
 * @return FlowField object containing initialized flow field variables
 */
inline FlowField initialize_taylor_green(const Mesh& mesh, 
                                 double amplitude = 1.0, 
                                 const std::vector<double>& wavelength = {1.0, 1.0}, 
                                 double decay_factor = 0.0) {
    FlowField flow_field;
    size_t num_nodes = mesh.get_node_count();
    
    // Handle wavelength specification
    double wavelength_x = wavelength[0];
    double wavelength_y = wavelength.size() > 1 ? wavelength[1] : wavelength[0];
    
    // Initialize flow field arrays
    flow_field.velocity.resize(num_nodes, std::vector<double>(2, 0.0));
    flow_field.pressure.resize(num_nodes, 0.0);
    
    // Decay term
    double decay_term = std::exp(-decay_factor);
    
    // Calculate flow field values at each node
    for (size_t i = 0; i < num_nodes; ++i) {
        const auto& node = mesh.get_node(i);
        double x = node[0];
        double y = node[1];
        
        // Calculate trigonometric terms
        double sin_x = std::sin(2 * M_PI * x / wavelength_x);
        double cos_x = std::cos(2 * M_PI * x / wavelength_x);
        double sin_y = std::sin(2 * M_PI * y / wavelength_y);
        double cos_y = std::cos(2 * M_PI * y / wavelength_y);
        
        // Velocity components
        flow_field.velocity[i][0] = amplitude * sin_x * cos_y * decay_term;  // u
        flow_field.velocity[i][1] = -amplitude * cos_x * sin_y * decay_term; // v
        
        // Pressure (proportional to the square of amplitude)
        flow_field.pressure[i] = (amplitude * amplitude / 4.0) * 
            (std::cos(4 * M_PI * x / wavelength_x) + 
             std::cos(4 * M_PI * y / wavelength_y)) * decay_term * decay_term;
    }
    
    return flow_field;
}

/**
 * @brief Initialize a single vortex or multiple vortices in the flow field.
 * 
 * This function creates a Rankine vortex or a collection of Rankine vortices,
 * which have a forced rotation core and a free vortex in the outer region.
 * 
 * @param mesh The computational mesh containing node coordinates
 * @param centers Vector of (x,y) centers for vortices
 * @param strengths Vector of vortex strengths (positive for counterclockwise, negative for clockwise)
 * @param radii Vector of characteristic radii for vortices
 * @param base_flow Background flow velocity
 * @param pressure_model Model for pressure distribution: "quadratic", "linear", or "constant"
 * @return FlowField object containing initialized flow field variables
 */
inline FlowField initialize_vortex(const Mesh& mesh, 
                           const std::vector<std::vector<double>>& centers, 
                           const std::vector<double>& strengths, 
                           const std::vector<double>& radii,
                           const std::vector<double>& base_flow = {0.0, 0.0},
                           const std::string& pressure_model = "quadratic") {
    
    FlowField flow_field;
    size_t num_nodes = mesh.get_node_count();
    size_t num_vortices = centers.size();
    
    // Validate input arrays
    if (strengths.size() != num_vortices || radii.size() != num_vortices) {
        throw std::invalid_argument("centers, strengths, and radii must have the same size");
    }
    
    // Initialize velocity with base flow
    flow_field.velocity.resize(num_nodes, std::vector<double>(2));
    flow_field.pressure.resize(num_nodes, 0.0);
    
    for (size_t i = 0; i < num_nodes; ++i) {
        flow_field.velocity[i][0] = base_flow[0];
        flow_field.velocity[i][1] = base_flow[1];
    }
    
    // Process each vortex
    for (size_t i = 0; i < num_vortices; ++i) {
        const auto& center = centers[i];
        double strength = strengths[i];
        double radius = radii[i];
        
        for (size_t j = 0; j < num_nodes; ++j) {
            const auto& node = mesh.get_node(j);
            
            // Calculate distance from vortex center
            double dx = node[0] - center[0];
            double dy = node[1] - center[1];
            double r = std::sqrt(dx*dx + dy*dy);
            
            // Skip if we're exactly at the center to avoid division by zero
            if (r < 1e-10) {
                continue;
            }
            
            // Calculate tangential velocity
            double v_tangential;
            if (r <= radius) {  // Forced vortex (solid body rotation)
                v_tangential = strength * r / radius;
            } else {  // Free vortex
                v_tangential = strength * radius / r;
            }
            
            // Calculate velocity components (perpendicular to radius)
            flow_field.velocity[j][0] -= v_tangential * dy / r;
            flow_field.velocity[j][1] += v_tangential * dx / r;
            
            // Calculate pressure contribution based on selected model
            double p_contribution = 0.0;
            if (pressure_model == "quadratic") {
                if (r <= radius) {
                    p_contribution = 0.5 * std::pow(strength * r / radius, 2);
                } else {
                    p_contribution = 0.5 * std::pow(strength * radius / r, 2);
                }
            } else if (pressure_model == "linear") {
                if (r <= radius) {
                    p_contribution = strength * r / radius;
                } else {
                    p_contribution = strength * radius / r;
                }
            }
            // "constant" model adds no pressure contribution
            
            flow_field.pressure[j] += p_contribution;
        }
    }
    
    return flow_field;
}

// Convenience overload for single vortex case
inline FlowField initialize_vortex(const Mesh& mesh, 
                           const std::vector<double>& center, 
                           double strength, 
                           double radius,
                           const std::vector<double>& base_flow = {0.0, 0.0},
                           const std::string& pressure_model = "quadratic") {
    
    return initialize_vortex(mesh, 
                            {center}, 
                            {strength}, 
                            {radius}, 
                            base_flow, 
                            pressure_model);
}

/**
 * @brief Add random perturbations to a base flow field.
 * 
 * This function is useful for testing solver robustness and initializing
 * turbulent simulations with stochastic perturbations.
 * 
 * @param mesh The computational mesh
 * @param base_flow Base flow field to perturb. If nullptr, starts with zero flow.
 * @param perturbation_magnitude Maximum perturbation magnitude relative to base flow
 * @param seed Random seed for reproducibility
 * @param spectrum Energy spectrum for perturbation: "white", "red", or "von_karman"
 * @return FlowField object containing perturbed flow field
 */
inline FlowField initialize_random_perturbation(const Mesh& mesh, 
                                        const FlowField* base_flow = nullptr,
                                        double perturbation_magnitude = 0.1, 
                                        int seed = -1,
                                        const std::string& spectrum = "white") {
    
    FlowField flow_field;
    size_t num_nodes = mesh.get_node_count();
    
    // Set random seed if provided
    std::mt19937 gen;
    if (seed >= 0) {
        gen.seed(seed);
    } else {
        std::random_device rd;
        gen.seed(rd());
    }
    
    // Initialize or copy base flow
    double base_magnitude;
    if (base_flow == nullptr) {
        flow_field.velocity.resize(num_nodes, std::vector<double>(2, 0.0));
        flow_field.pressure.resize(num_nodes, 0.0);
        base_magnitude = perturbation_magnitude;  // No base flow, use absolute perturbation
    } else {
        // Copy base flow
        flow_field.velocity = base_flow->velocity;
        flow_field.pressure = base_flow->pressure;
        
        // Calculate base flow magnitude for relative perturbation
        double sum_velocity_mag = 0.0;
        for (const auto& vel : flow_field.velocity) {
            sum_velocity_mag += std::sqrt(vel[0]*vel[0] + vel[1]*vel[1]);
        }
        
        base_magnitude = (sum_velocity_mag / num_nodes) * perturbation_magnitude;
        if (base_magnitude < 1e-10) {  // If base flow is essentially zero
            base_magnitude = perturbation_magnitude;  // Use absolute perturbation
        }
    }
    
    // Different spectrum implementations
    if (spectrum == "white") {
        // Simple white noise
        std::normal_distribution<double> dist(0.0, base_magnitude);
        
        for (size_t i = 0; i < num_nodes; ++i) {
            flow_field.velocity[i][0] += dist(gen);
            flow_field.velocity[i][1] += dist(gen);
            
            // Small pressure perturbations
            flow_field.pressure[i] += std::normal_distribution<double>(
                0.0, base_magnitude*base_magnitude/2.0)(gen);
        }
        
    } else if (spectrum == "red") {
        // Generate structured grid for FFT-based perturbation
        // This is a simplified approach for demonstration
        std::vector<double> min_coord = {
            std::numeric_limits<double>::max(),
            std::numeric_limits<double>::max()
        };
        std::vector<double> max_coord = {
            std::numeric_limits<double>::lowest(),
            std::numeric_limits<double>::lowest()
        };
        
        // Find domain extents
        for (size_t i = 0; i < num_nodes; ++i) {
            const auto& node = mesh.get_node(i);
            for (int j = 0; j < 2; ++j) {
                min_coord[j] = std::min(min_coord[j], node[j]);
                max_coord[j] = std::max(max_coord[j], node[j]);
            }
        }
        
        size_t grid_size = static_cast<size_t>(std::sqrt(num_nodes));  // Approximate grid size
        
        // Generate red noise on a structured grid
        std::vector<std::vector<double>> grid_u(grid_size, std::vector<double>(grid_size, 0.0));
        std::vector<std::vector<double>> grid_v(grid_size, std::vector<double>(grid_size, 0.0));
        
        std::uniform_real_distribution<double> phase_dist(0.0, 2.0 * M_PI);
        
        // Fill with random values
        for (size_t i = 0; i < grid_size; ++i) {
            for (size_t j = 0; j < grid_size; ++j) {
                double freq = std::sqrt(i*i + j*j) + 1e-6;  // Avoid division by zero
                double amplitude = base_magnitude / freq;
                double phase_u = phase_dist(gen);
                double phase_v = phase_dist(gen);
                grid_u[i][j] = amplitude * std::cos(phase_u);
                grid_v[i][j] = amplitude * std::cos(phase_v);
            }
        }
        
        // Interpolate from grid to mesh nodes
        double dx = (max_coord[0] - min_coord[0]) / (grid_size - 1);
        double dy = (max_coord[1] - min_coord[1]) / (grid_size - 1);
        
        for (size_t i = 0; i < num_nodes; ++i) {
            const auto& node = mesh.get_node(i);
            
            // Find grid indices (simplified nearest neighbor)
            int ix = static_cast<int>((node[0] - min_coord[0]) / dx);
            int iy = static_cast<int>((node[1] - min_coord[1]) / dy);
            ix = std::max(0, std::min(ix, static_cast<int>(grid_size) - 1));
            iy = std::max(0, std::min(iy, static_cast<int>(grid_size) - 1));
            
            flow_field.velocity[i][0] += grid_u[ix][iy];
            flow_field.velocity[i][1] += grid_v[ix][iy];
        }
        
    } else if (spectrum == "von_karman") {
        // Simplified von Kármán spectrum
        // This is a basic approximation
        
        // Calculate domain size as characteristic length
        double L = std::max(max_coord[0] - min_coord[0], max_coord[1] - min_coord[1]);
        
        std::uniform_real_distribution<double> phase_dist(0.0, 2.0 * M_PI);
        
        // Generate perturbations with von Kármán-like spectrum
        for (size_t i = 0; i < num_nodes; ++i) {
            const auto& node = mesh.get_node(i);
            
            // Generate multiple wave contributions
            double u_pert = 0.0;
            double v_pert = 0.0;
            
            // Sum contributions from different wavelengths
            for (int k = 1; k <= 10; ++k) {  // Use 10 wavelengths
                // von Kármán spectrum weighting
                double wavelength = L / k;
                double weight = base_magnitude * std::sqrt(
                    1.0 / std::pow(1.0 + std::pow(wavelength/L, 2), 5.0/6.0));
                
                // Random phase
                double phase_u = phase_dist(gen);
                double phase_v = phase_dist(gen);
                
                // Position dependent perturbation
                u_pert += weight * std::sin(2*M_PI*node[0]/wavelength + phase_u);
                v_pert += weight * std::sin(2*M_PI*node[1]/wavelength + phase_v);
            }
            
            flow_field.velocity[i][0] += u_pert;
            flow_field.velocity[i][1] += v_pert;
        }
    }
    
    return flow_field;
}

/**
 * @brief Verify that a flow field satisfies physical and numerical constraints.
 * 
 * @param mesh The computational mesh
 * @param flow_field Flow field to validate
 * @param constraints Map of constraints to check
 * @return std::pair<bool, std::vector<std::string>> Boolean indicating validity and list of violations
 */
inline std::pair<bool, std::vector<std::string>> validate_flow_field(
    const Mesh& mesh, 
    const FlowField& flow_field, 
    const std::unordered_map<std::string, double>& constraints = {}) {
    
    bool valid = true;
    std::vector<std::string> violations;
    
    // Set default constraints if none provided
    std::unordered_map<std::string, double> effective_constraints = constraints;
    if (constraints.empty()) {
        effective_constraints = {
            {"max_velocity", 1e3},
            {"divergence_tolerance", 1e-6},
            {"min_pressure", -1e6},
            {"max_pressure", 1e6}
        };
    }
    
    // Check velocity magnitude
    auto max_vel_iter = effective_constraints.find("max_velocity");
    if (max_vel_iter != effective_constraints.end()) {
        double max_vel = 0.0;
        for (const auto& vel : flow_field.velocity) {
            double vel_mag = std::sqrt(vel[0]*vel[0] + vel[1]*vel[1]);
            max_vel = std::max(max_vel, vel_mag);
        }
        
        if (max_vel > max_vel_iter->second) {
            valid = false;
            violations.push_back("Maximum velocity (" + std::to_string(max_vel) + 
                               ") exceeds limit (" + std::to_string(max_vel_iter->second) + ")");
        }
    }
    
    // Check pressure bounds
    auto min_pressure_iter = effective_constraints.find("min_pressure");
    if (min_pressure_iter != effective_constraints.end()) {
        double min_pressure = *std::min_element(flow_field.pressure.begin(), flow_field.pressure.end());
        if (min_pressure < min_pressure_iter->second) {
            valid = false;
            violations.push_back("Minimum pressure below limit (" + 
                               std::to_string(min_pressure_iter->second) + ")");
        }
    }
    
    auto max_pressure_iter = effective_constraints.find("max_pressure");
    if (max_pressure_iter != effective_constraints.end()) {
        double max_pressure = *std::max_element(flow_field.pressure.begin(), flow_field.pressure.end());
        if (max_pressure > max_pressure_iter->second) {
            valid = false;
            violations.push_back("Maximum pressure exceeds limit (" + 
                               std::to_string(max_pressure_iter->second) + ")");
        }
    }
    
    // Check for NaN/inf values
    bool has_nan = false;
    bool has_inf = false;
    
    for (const auto& vel : flow_field.velocity) {
        for (double component : vel) {
            if (std::isnan(component)) has_nan = true;
            if (std::isinf(component)) has_inf = true;
        }
    }
    
    for (double p : flow_field.pressure) {
        if (std::isnan(p)) has_nan = true;
        if (std::isinf(p)) has_inf = true;
    }
    
    if (has_nan) {
        valid = false;
        violations.push_back("NaN values detected in flow field");
    }
    
    if (has_inf) {
        valid = false;
        violations.push_back("Infinite values detected in flow field");
    }
    
    // Check approximate divergence (needs cell connectivity information)
    // This is a simplified placeholder - real implementation would need mesh topology
    auto div_tol_iter = effective_constraints.find("divergence_tolerance");
    if (div_tol_iter != effective_constraints.end()) {
        // Placeholder for divergence calculation
        // In a real implementation, would calculate ∇·v for each cell
    }
    
    return {valid, violations};
}

/**
 * @brief Combine multiple flow fields into a single field.
 * 
 * @param mesh The computational mesh
 * @param field_list List of flow field objects to combine
 * @param method Combination method: "addition", "max", or "weighted"
 * @param weights Weights for weighted combination
 * @return FlowField object containing combined flow field
 */
inline FlowField combine_flow_fields(
    const Mesh& mesh, 
    const std::vector<FlowField>& field_list, 
    const std::string& method = "addition", 
    const std::vector<double>& weights = {}) {
    
    if (field_list.empty()) {
        throw std::invalid_argument("Empty field list provided");
    }
    
    size_t num_fields = field_list.size();
    size_t num_nodes = mesh.get_node_count();
    
    // Initialize combined field with zeros
    FlowField combined_field;
    combined_field.velocity.resize(num_nodes, std::vector<double>(2, 0.0));
    combined_field.pressure.resize(num_nodes, 0.0);
    
    // Handle weights for weighted method
    std::vector<double> effective_weights;
    if (method == "weighted") {
        if (weights.empty()) {
            effective_weights.resize(num_fields, 1.0 / num_fields);
        } else if (weights.size() != num_fields) {
            throw std::invalid_argument("Expected " + std::to_string(num_fields) + 
                                      " weights, got " + std::to_string(weights.size()));
        } else {
            effective_weights = weights;
        }
    }
    
    // Combine fields according to specified method
    if (method == "addition" || method == "weighted") {
        for (size_t i = 0; i < num_fields; ++i) {
            double weight = (method == "weighted") ? effective_weights[i] : 1.0;
            
            for (size_t j = 0; j < num_nodes; ++j) {
                for (size_t k = 0; k < 2; ++k) {
                    combined_field.velocity[j][k] += field_list[i].velocity[j][k] * weight;
                }
                combined_field.pressure[j] += field_list[i].pressure[j] * weight;
            }
        }
    } else if (method == "max") {
        // Start with the first field
        combined_field = field_list[0];
        
        // Find maximum values
        for (size_t i = 1; i < num_fields; ++i) {
            for (size_t j = 0; j < num_nodes; ++j) {
                // For velocity, use the one with larger magnitude at each point
                double vel_mag_combined = std::sqrt(
                    combined_field.velocity[j][0]*combined_field.velocity[j][0] + 
                    combined_field.velocity[j][1]*combined_field.velocity[j][1]);
                
                double vel_mag_current = std::sqrt(
                    field_list[i].velocity[j][0]*field_list[i].velocity[j][0] + 
                    field_list[i].velocity[j][1]*field_list[i].velocity[j][1]);
                
                // If current field has larger magnitude, use it
                if (vel_mag_current > vel_mag_combined) {
                    combined_field.velocity[j][0] = field_list[i].velocity[j][0];
                    combined_field.velocity[j][1] = field_list[i].velocity[j][1];
                }
                
                // For pressure, simply take the maximum
                combined_field.pressure[j] = std::max(
                    combined_field.pressure[j], field_list[i].pressure[j]);
            }
        }
    } else {
        throw std::invalid_argument("Unknown combination method: " + method);
    }
    
    return combined_field;
}

} // namespace MeshUtils

#endif // TEST_FLOW_INITIALIZATION_H
  
