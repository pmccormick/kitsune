#ifndef ADVANCED_FLOW_INITIALIZATION_H
#define ADVANCED_FLOW_INITIALIZATION_H

#include <vector>
#include <array>
#include <cmath>
#include <string>
#include <unordered_map>
#include <functional>
#include <algorithm>
#include <memory>
#include <complex>
#include <random>
#include "mesh.h"

namespace MeshUtils {

/**
 * @brief Initialize a potential flow field around multiple sources, sinks, and vortices.
 * 
 * @param mesh The computational mesh
 * @param sources Vector of {x, y, strength} for sources (strength > 0) and sinks (strength < 0)
 * @param vortices Vector of {x, y, strength} for vortices
 * @param doublets Vector of {x, y, strength, angle} for doublets
 * @param uniform_flow Vector {u, v} for background uniform flow
 * @return FlowField object containing initialized flow field variables
 */
inline FlowField initialize_potential_flow(const Mesh& mesh,
                                          const std::vector<std::vector<double>>& sources = {},
                                          const std::vector<std::vector<double>>& vortices = {},
                                          const std::vector<std::vector<double>>& doublets = {},
                                          const std::vector<double>& uniform_flow = {0.0, 0.0}) {
    FlowField flow_field;
    size_t num_nodes = mesh.get_node_count();
    
    // Initialize flow field
    flow_field.velocity.resize(num_nodes, std::vector<double>(2, 0.0));
    flow_field.pressure.resize(num_nodes, 0.0);
    
    // Add uniform flow contribution
    if (uniform_flow.size() >= 2) {
        for (size_t i = 0; i < num_nodes; ++i) {
            flow_field.velocity[i][0] = uniform_flow[0];
            flow_field.velocity[i][1] = uniform_flow[1];
        }
    }
    
    // Process each node
    for (size_t i = 0; i < num_nodes; ++i) {
        const auto& node = mesh.get_node(i);
        double x = node[0];
        double y = node[1];
        
        // Complex point
        std::complex<double> z(x, y);
        std::complex<double> w(0.0, 0.0); // Complex potential
        
        // Add source/sink contributions
        for (const auto& source : sources) {
            if (source.size() < 3) continue;
            
            double sx = source[0];
            double sy = source[1];
            double strength = source[2];
            
            std::complex<double> zs(sx, sy);
            std::complex<double> dz = z - zs;
            
            // Avoid singularity at source/sink location
            if (std::abs(dz) < 1e-10) continue;
            
            // Source/sink potential: w = (m/2π) * log(z - zs)
            w += (strength / (2.0 * M_PI)) * std::log(dz);
        }
        
        // Add vortex contributions
        for (const auto& vortex : vortices) {
            if (vortex.size() < 3) continue;
            
            double vx = vortex[0];
            double vy = vortex[1];
            double strength = vortex[2];
            
            std::complex<double> zv(vx, vy);
            std::complex<double> dz = z - zv;
            
            // Avoid singularity at vortex location
            if (std::abs(dz) < 1e-10) continue;
            
            // Vortex potential: w = -i * (Γ/2π) * log(z - zv)
            w += -std::complex<double>(0.0, 1.0) * (strength / (2.0 * M_PI)) * std::log(dz);
        }
        
        // Add doublet contributions
        for (const auto& doublet : doublets) {
            if (doublet.size() < 4) continue;
            
            double dx = doublet[0];
            double dy = doublet[1];
            double strength = doublet[2];
            double angle = doublet.size() > 3 ? doublet[3] : 0.0;
            
            std::complex<double> zd(dx, dy);
            std::complex<double> dz = z - zd;
            
            // Avoid singularity at doublet location
            if (std::abs(dz) < 1e-10) continue;
            
            // Rotation factor for doublet orientation
            std::complex<double> orientation = std::exp(std::complex<double>(0.0, angle));
            
            // Doublet potential: w = -(μ/2π) * 1/(z - zd)
            w += -(strength / (2.0 * M_PI)) * orientation / dz;
        }
        
        // Calculate velocity from complex potential
        // v = dw/dz = u - i*v  (conjugate of velocity)
        // We compute this by numerical differentiation
        
        const double eps = 1e-6;
        std::complex<double> zx(x + eps, y);
        std::complex<double> zy(x, y + eps);
        
        std::complex<double> wx(0.0, 0.0);
        std::complex<double> wy(0.0, 0.0);
        
        // Recompute potential at zx and zy
        for (const auto& source : sources) {
            if (source.size() < 3) continue;
            
            double sx = source[0];
            double sy = source[1];
            double strength = source[2];
            
            std::complex<double> zs(sx, sy);
            std::complex<double> dzx = zx - zs;
            std::complex<double> dzy = zy - zs;
            
            if (std::abs(dzx) > 1e-10) {
                wx += (strength / (2.0 * M_PI)) * std::log(dzx);
            }
            
            if (std::abs(dzy) > 1e-10) {
                wy += (strength / (2.0 * M_PI)) * std::log(dzy);
            }
        }
        
        for (const auto& vortex : vortices) {
            if (vortex.size() < 3) continue;
            
            double vx = vortex[0];
            double vy = vortex[1];
            double strength = vortex[2];
            
            std::complex<double> zv(vx, vy);
            std::complex<double> dzx = zx - zv;
            std::complex<double> dzy = zy - zv;
            
            if (std::abs(dzx) > 1e-10) {
                wx += -std::complex<double>(0.0, 1.0) * (strength / (2.0 * M_PI)) * std::log(dzx);
            }
            
            if (std::abs(dzy) > 1e-10) {
                wy += -std::complex<double>(0.0, 1.0) * (strength / (2.0 * M_PI)) * std::log(dzy);
            }
        }
        
        for (const auto& doublet : doublets) {
            if (doublet.size() < 4) continue;
            
            double dx = doublet[0];
            double dy = doublet[1];
            double strength = doublet[2];
            double angle = doublet.size() > 3 ? doublet[3] : 0.0;
            
            std::complex<double> zd(dx, dy);
            std::complex<double> dzx = zx - zd;
            std::complex<double> dzy = zy - zd;
            
            std::complex<double> orientation = std::exp(std::complex<double>(0.0, angle));
            
            if (std::abs(dzx) > 1e-10) {
                wx += -(strength / (2.0 * M_PI)) * orientation / dzx;
            }
            
            if (std::abs(dzy) > 1e-10) {
                wy += -(strength / (2.0 * M_PI)) * orientation / dzy;
            }
        }
        
        // Compute velocity components by numerical differentiation
        std::complex<double> dw_dx = (wx - w) / eps;
        std::complex<double> dw_dy = (wy - w) / eps;
        
        // Velocity components (u = Re(dw/dz), v = -Im(dw/dz))
        flow_field.velocity[i][0] += dw_dx.real();
        flow_field.velocity[i][1] += -dw_dx.imag();
        
        // Compute pressure using Bernoulli's equation
        double v_squared = flow_field.velocity[i][0] * flow_field.velocity[i][0] + 
                         flow_field.velocity[i][1] * flow_field.velocity[i][1];
        flow_field.pressure[i] = 0.5 * (1.0 - v_squared); // Assuming density=1, free-stream pressure=0.5
    }
    
    return flow_field;
}

/**
 * @brief Initialize a synthetic turbulent flow field using the Synthetic Eddy Method (SEM).
 * 
 * @param mesh The computational mesh
 * @param mean_velocity Background mean velocity field
 * @param reynolds_stresses Reynolds stress tensor components [u'u', v'v', u'v']
 * @param turbulent_length_scale Characteristic length scale of turbulent eddies
 * @param num_eddies Number of synthetic eddies to generate
 * @param seed Random seed for reproducibility
 * @return FlowField object containing initialized flow field with turbulent fluctuations
 */
inline FlowField initialize_synthetic_turbulence(const Mesh& mesh,
                                               const std::vector<double>& mean_velocity = {1.0, 0.0},
                                               const std::vector<double>& reynolds_stresses = {0.1, 0.1, 0.0},
                                               double turbulent_length_scale = 0.1,
                                               int num_eddies = 100,
                                               int seed = -1) {
    FlowField flow_field;
    size_t num_nodes = mesh.get_node_count();
    
    // Validate inputs
    if (mean_velocity.size() != 2) {
        throw std::invalid_argument("Mean velocity must be a 2D vector");
    }
    
    if (reynolds_stresses.size() != 3) {
        throw std::invalid_argument("Reynolds stresses must contain [u'u', v'v', u'v']");
    }
    
    // Initialize random number generator
    std::mt19937 gen;
    if (seed >= 0) {
        gen.seed(seed);
    } else {
        std::random_device rd;
        gen.seed(rd());
    }
    
    // Initialize flow field with mean velocity
    flow_field.velocity.resize(num_nodes, std::vector<double>(2, 0.0));
    flow_field.pressure.resize(num_nodes, 0.0);
    
    for (size_t i = 0; i < num_nodes; ++i) {
        flow_field.velocity[i][0] = mean_velocity[0];
        flow_field.velocity[i][1] = mean_velocity[1];
    }
    
    // Determine domain bounds
    std::vector<double> min_coord = {
        std::numeric_limits<double>::max(),
        std::numeric_limits<double>::max()
    };
    std::vector<double> max_coord = {
        std::numeric_limits<double>::lowest(),
        std::numeric_limits<double>::lowest()
    };
    
    for (size_t i = 0; i < num_nodes; ++i) {
        const auto& node = mesh.get_node(i);
        for (int j = 0; j < 2; ++j) {
            min_coord[j] = std::min(min_coord[j], node[j]);
            max_coord[j] = std::max(max_coord[j], node[j]);
        }
    }
    
    // Extend domain for eddy box (typical SEM approach)
    double domain_extension = 2.0 * turbulent_length_scale;
    for (int i = 0; i < 2; ++i) {
        min_coord[i] -= domain_extension;
        max_coord[i] += domain_extension;
    }
    
    // Compute Cholesky decomposition of Reynolds stress tensor
    // For 2D: [a11 0; a21 a22] * [a11 a21; 0 a22]^T = [u'u' u'v'; u'v' v'v']
    double uu = reynolds_stresses[0];
    double vv = reynolds_stresses[1];
    double uv = reynolds_stresses[2];
    
    double a11 = std::sqrt(uu);
    double a21 = uv / a11;
    double a22 = std::sqrt(vv - a21 * a21);
    
    // Generate synthetic eddies
    std::vector<std::vector<double>> eddy_positions(num_eddies, std::vector<double>(2));
    std::vector<std::vector<double>> eddy_intensities(num_eddies, std::vector<double>(2));
    
    std::uniform_real_distribution<double> pos_dist_x(min_coord[0], max_coord[0]);
    std::uniform_real_distribution<double> pos_dist_y(min_coord[1], max_coord[1]);
    std::normal_distribution<double> intensity_dist(0.0, 1.0);
    
    for (int i = 0; i < num_eddies; ++i) {
        // Random position within extended domain
        eddy_positions[i][0] = pos_dist_x(gen);
        eddy_positions[i][1] = pos_dist_y(gen);
        
        // Random intensity with normal distribution
        eddy_intensities[i][0] = intensity_dist(gen);
        eddy_intensities[i][1] = intensity_dist(gen);
    }
    
    // Apply synthetic eddies to flow field
    for (size_t i = 0; i < num_nodes; ++i) {
        const auto& node = mesh.get_node(i);
        double x = node[0];
        double y = node[1];
        
        // Initialize fluctuation velocity
        double u_fluct = 0.0;
        double v_fluct = 0.0;
        
        // Sum contributions from all eddies
        for (int j = 0; j < num_eddies; ++j) {
            // Distance to eddy
            double dx = x - eddy_positions[j][0];
            double dy = y - eddy_positions[j][1];
            
            // Normalized distance
            double rx = dx / turbulent_length_scale;
            double ry = dy / turbulent_length_scale;
            
            // Shape function (compact support within turbulent_length_scale)
            auto shape_function = [](double r) {
                if (std::abs(r) > 1.0) return 0.0;
                return std::sqrt(1.0 - std::abs(r)) * (1.0 - std::abs(r));
            };
            
            double fx = shape_function(rx);
            double fy = shape_function(ry);
            
            // Add eddy contribution
            u_fluct += eddy_intensities[j][0] * fx * fy;
            v_fluct += eddy_intensities[j][1] * fx * fy;
        }
        
        // Scale fluctuations by Cholesky factors and normalize by sqrt(num_eddies)
        double scale_factor = 1.0 / std::sqrt(static_cast<double>(num_eddies));
        double u_prime = (a11 * u_fluct) * scale_factor;
        double v_prime = (a21 * u_fluct + a22 * v_fluct) * scale_factor;
        
        // Add fluctuations to mean flow
        flow_field.velocity[i][0] += u_prime;
        flow_field.velocity[i][1] += v_prime;
    }
    
    return flow_field;
}

/**
 * @brief Initialize a stratified flow field with density/temperature variations.
 * 
 * @param mesh The computational mesh
 * @param base_velocity Base velocity field
 * @param stratification_direction Direction of stratification (typically vertical)
 * @param density_gradient Density gradient per unit distance
 * @param reference_density Reference density at origin
 * @param gravity_magnitude Magnitude of gravitational acceleration
 * @return FlowField object containing initialized flow field with stratification effects
 */
inline FlowField initialize_stratified_flow(const Mesh& mesh,
                                           const std::vector<double>& base_velocity = {0.0, 0.0},
                                           const std::vector<double>& stratification_direction = {0.0, 1.0},
                                           double density_gradient = -0.01,
                                           double reference_density = 1.0,
                                           double gravity_magnitude = 9.81) {
    FlowField flow_field;
    size_t num_nodes = mesh.get_node_count();
    
    // Validate inputs
    if (base_velocity.size() != 2 || stratification_direction.size() != 2) {
        throw std::invalid_argument("Velocity and stratification direction must be 2D vectors");
    }
    
    // Normalize stratification direction
    double dir_mag = std::sqrt(stratification_direction[0]*stratification_direction[0] + 
                             stratification_direction[1]*stratification_direction[1]);
    
    if (dir_mag < 1e-10) {
        throw std::invalid_argument("Stratification direction cannot be zero");
    }
    
    std::vector<double> strat_dir = {
        stratification_direction[0] / dir_mag,
        stratification_direction[1] / dir_mag
    };
    
    // Initialize flow field
    flow_field.velocity.resize(num_nodes, std::vector<double>(2));
    flow_field.pressure.resize(num_nodes, 0.0);
    flow_field.density.resize(num_nodes, 0.0);  // Add density field
    
    // Compute stratified flow
    for (size_t i = 0; i < num_nodes; ++i) {
        const auto& node = mesh.get_node(i);
        
        // Project node onto stratification direction
        double height = node[0]*strat_dir[0] + node[1]*strat_dir[1];
        
        // Calculate density based on linear stratification
        double density = reference_density + density_gradient * height;
        flow_field.density[i] = std::max(0.01, density);  // Ensure positive density
        
        // Set base velocity
        flow_field.velocity[i][0] = base_velocity[0];
        flow_field.velocity[i][1] = base_velocity[1];
        
        // Compute hydrostatic pressure contribution
        // Simple model: p = p_ref - ρ*g*h
        flow_field.pressure[i] = -flow_field.density[i] * gravity_magnitude * height;
    }
    
    return flow_field;
}

/**
 * @brief Initialize a wave-like flow field.
 * 
 * @param mesh The computational mesh
 * @param wave_type Type of wave ("standing", "traveling", "circular")
 * @param amplitude Wave amplitude
 * @param wavelength Wavelength
 * @param direction Direction of wave propagation
 * @param phase_speed Phase speed for traveling waves
 * @return FlowField object containing initialized wave-like flow field
 */
inline FlowField initialize_wave_flow(const Mesh& mesh,
                                     const std::string& wave_type = "traveling",
                                     double amplitude = 0.1,
                                     double wavelength = 1.0,
                                     const std::vector<double>& direction = {1.0, 0.0},
                                     double phase_speed = 1.0) {
    FlowField flow_field;
    size_t num_nodes = mesh.get_node_count();
    
    // Validate inputs
    if (direction.size() != 2) {
        throw std::invalid_argument("Direction must be a 2D vector");
    }
    
    // Normalize direction
    double dir_mag = std::sqrt(direction[0]*direction[0] + direction[1]*direction[1]);
    if (dir_mag < 1e-10) {
        throw std::invalid_argument("Direction vector cannot be zero");
    }
    
    std::vector<double> dir_normalized = {
        direction[0] / dir_mag,
        direction[1] / dir_mag
    };
    
    // Perpendicular direction (for divergence-free condition)
    std::vector<double> perp_dir = {-dir_normalized[1], dir_normalized[0]};
    
    // Wave number
    double k = 2.0 * M_PI / wavelength;
    
    // Initialize time for traveling waves
    double t = 0.0; // Initial time
    
    // Initialize flow field
    flow_field.velocity.resize(num_nodes, std::vector<double>(2, 0.0));
    flow_field.pressure.resize(num_nodes, 0.0);
    
    // Wave center for circular waves
    std::vector<double> wave_center = {0.0, 0.0};
    if (wave_type == "circular") {
        // Use domain center as wave center
        double x_sum = 0.0, y_sum = 0.0;
        for (size_t i = 0; i < num_nodes; ++i) {
            const auto& node = mesh.get_node(i);
            x_sum += node[0];
            y_sum += node[1];
        }
        wave_center[0] = x_sum / num_nodes;
        wave_center[1] = y_sum / num_nodes;
    }
    
    // Compute wave flow
    for (size_t i = 0; i < num_nodes; ++i) {
        const auto& node = mesh.get_node(i);
        
        double phase = 0.0;
        
        if (wave_type == "standing") {
            // Standing wave: ψ = A*sin(k*x)*cos(ω*t), but t=0 initially
            // Project node onto wave direction
            double x_proj = node[0]*dir_normalized[0] + node[1]*dir_normalized[1];
            phase = k * x_proj;
            
            // Stream function gradient gives velocity
            flow_field.velocity[i][0] = amplitude * k * std::sin(phase) * perp_dir[0];
            flow_field.velocity[i][1] = amplitude * k * std::sin(phase) * perp_dir[1];
            
        } else if (wave_type == "traveling") {
            // Traveling wave: ψ = A*sin(k*x - ω*t)
            double x_proj = node[0]*dir_normalized[0] + node[1]*dir_normalized[1];
            double omega = k * phase_speed;
            phase = k * x_proj - omega * t;
            
            // Stream function gradient gives velocity
            flow_field.velocity[i][0] = amplitude * k * std::cos(phase) * perp_dir[0];
            flow_field.velocity[i][1] = amplitude * k * std::cos(phase) * perp_dir[1];
            
        } else if (wave_type == "circular") {
            // Circular/radial wave: ψ = A*sin(k*r - ω*t)
            double dx = node[0] - wave_center[0];
            double dy = node[1] - wave_center[1];
            double r = std::sqrt(dx*dx + dy*dy);
            
            if (r < 1e-10) {
                // Avoid singularity at center
                continue;
            }
            
            double omega = k * phase_speed;
            phase = k * r - omega * t;
            
            // For circular waves, velocity is perpendicular to radius
            // v_θ = (1/r) * ∂ψ/∂r
            double v_theta = amplitude * k * std::cos(phase) / std::max(r, 1e-6);
            
            // Convert to Cartesian coordinates
            flow_field.velocity[i][0] = -v_theta * dy / r;  // v_x = -v_θ * sin(θ)
            flow_field.velocity[i][1] = v_theta * dx / r;   // v_y = v_θ * cos(θ)
        }
        
        // Compute pressure using simplified Bernoulli equation
        double v_squared = flow_field.velocity[i][0] * flow_field.velocity[i][0] + 
                         flow_field.velocity[i][1] * flow_field.velocity[i][1];
        flow_field.pressure[i] = 0.5 * (1.0 - v_squared);
    }
    
    return flow_field;
}

/**
 * @brief Initialize a swirling flow field with optional radial component.
 * 
 * @param mesh The computational mesh
 * @param center Center of swirling flow
 * @param tangential_profile Tangential velocity profile ("solid_body", "potential", "rankine")
 * @param radial_profile Radial velocity profile ("none", "source", "sink", "gaussian")
 * @param max_tangential_velocity Maximum tangential velocity
 * @param max_radial_velocity Maximum radial velocity (positive for outflow, negative for inflow)
 * @param characteristic_radius Radius at which tangential velocity reaches maximum
 * @return FlowField object containing initialized swirling flow field
 */
inline FlowField initialize_swirling_flow(const Mesh& mesh,
                                         const std::vector<double>& center = {0.0, 0.0},
                                         const std::string& tangential_profile = "rankine",
                                         const std::string& radial_profile = "none",
                                         double max_tangential_velocity = 1.0,
                                         double max_radial_velocity = 0.0,
                                         double characteristic_radius = 1.0) {
    FlowField flow_field;
    size_t num_nodes = mesh.get_node_count();
    
    // Validate inputs
    if (center.size() != 2) {
        throw std::invalid_argument("Center must be a 2D point");
    }
    
    // Initialize flow field
    flow_field.velocity.resize(num_nodes, std::vector<double>(2, 0.0));
    flow_field.pressure.resize(num_nodes, 0.0);
    
    // Process each node
    for (size_t i = 0; i < num_nodes; ++i) {
        const auto& node = mesh.get_node(i);
        
        // Vector from center to node
        double dx = node[0] - center[0];
        double dy = node[1] - center[1];
        double r = std::sqrt(dx*dx + dy*dy);
        
        // Avoid singularity at center
        if (r < 1e-10) {
            continue;
        }
        
        // Compute tangential velocity
        double v_theta = 0.0;
        
        if (tangential_profile == "solid_body") {
            // Solid body rotation: v_θ = ω*r
            v_theta = max_tangential_velocity * r / characteristic_radius;
            if (r > characteristic_radius) {
                v_theta = max_tangential_velocity;
            }
        } else if (tangential_profile == "potential") {
            // Potential vortex: v_θ = Γ/(2πr)
            v_theta = max_tangential_velocity * characteristic_radius / r;
        } else if (tangential_profile == "rankine") {
            // Rankine vortex: combined solid body and potential vortex
            if (r <= characteristic_radius) {
                // Solid body region
                v_theta = max_tangential_velocity * r / characteristic_radius;
            } else {
                // Potential vortex region
                v_theta = max_tangential_velocity * characteristic_radius / r;
            }
        } else {
            throw std::invalid_argument("Unknown tangential profile: " + tangential_profile);
        }
        
        // Compute radial velocity
        double v_r = 0.0;
        
        if (radial_profile == "source" || radial_profile == "sink") {
            // Source/sink flow: v_r = Q/(2πr)
            v_r = max_radial_velocity;
        } else if (radial_profile == "gaussian") {
            // Gaussian profile: v_r = v_max * exp(-r²/R²)
            v_r = max_radial_velocity * std::exp(-std::pow(r/characteristic_radius, 2));
        }
        // "none" profile keeps v_r = 0
        
        // Convert to Cartesian coordinates
        // Tangential component: perpendicular to radius
        flow_field.velocity[i][0] = -v_theta * dy / r;  // v_x = -v_θ * sin(θ)
        flow_field.velocity[i][1] = v_theta * dx / r;   // v_y = v_θ * cos(θ)
        
        // Radial component: along radius
        if (v_r != 0.0) {
            flow_field.velocity[i][0] += v_r * dx / r;  // v_x += v_r * cos(θ)
            flow_field.velocity[i][1] += v_r * dy / r;  // v_y += v_r * sin(θ)
        }
        
        // Compute pressure using simplified Bernoulli equation with centrifugal contribution
        // p = p_∞ - ρ*v²/2 - ρ*v_θ²/r (for swirling flows)
        double v_squared = flow_field.velocity[i][0] * flow_field.velocity[i][0] + 
                         flow_field.velocity[i][1] * flow_field.velocity[i][1];
        double centrifugal = std::pow(v_theta, 2) / std::max(r, 1e-6);
        
        flow_field.pressure[i] = 0.5 - 0.5 * v_squared - centrifugal;
    }
    
    return flow_field;
}

} // namespace MeshUtils

#endif // ADVANCED_FLOW_INITIALIZATION_H
 
