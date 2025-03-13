/**
 * @file ScientificMaterialEquivalence.h
 * @brief Scientific Material Equivalence Class
 */
#pragma once

#include "Material.h"
#include <cstdint>
#include <memory>
#include <map>
#include <string>
#include <vector>

/**
 * @class ScientificMaterialEquivalence
 * @brief Class for determining scientific equivalence between materials
 * 
 * This class implements a scientific approach to material equivalence
 * checking, using parametric analysis of material properties to determine
 * if two materials are functionally equivalent for simulation purposes.
 */
class ScientificMaterialEquivalence {
public:
    /**
     * @enum ScaleType
     * @brief Type of scaling to use for property range mapping
     */
    enum class ScaleType {
        LINEAR,     ///< Linear scaling
        LOGARITHMIC ///< Logarithmic scaling
    };
    
    /**
     * @enum SimulationType
     * @brief Predefined simulation types for configuration
     */
    enum class SimulationType {
        GENERIC,         ///< Generic simulation
        CFD_MULTIPHASE,  ///< Computational fluid dynamics with multiple phases
        THERMAL,         ///< Thermal simulation
        STRUCTURAL       ///< Structural simulation
    };
    
    /**
     * @brief Default constructor
     */
    ScientificMaterialEquivalence();
    
    /**
     * @brief Constructor with simulation type
     * @param type Simulation type to configure for
     * @param hashBits Number of bits to use for hash (default: 64)
     */
    ScientificMaterialEquivalence(SimulationType type, size_t hashBits = 64);
    
    /**
     * @brief Destructor
     */
    ~ScientificMaterialEquivalence();
    
    /**
     * @brief Check if two materials are equivalent
     * @param a First material
     * @param b Second material
     * @return True if materials are equivalent
     */
    bool areEquivalent(const std::shared_ptr<Material>& a, 
                      const std::shared_ptr<Material>& b) const;
    
    /**
     * @brief Calculate hash for a material
     * @param material Material to hash
     * @return Hash value
     */
    uint64_t hash(const std::shared_ptr<Material>& material) const;
    
    /**
     * @brief Calculate hash for a single property value
     * @param prop Property type
     * @param value Property value
     * @return Hash value for this property
     */
    uint64_t hashProperty(Material::MaterialProperty prop, double value) const;
    
    /**
     * @brief Apply avalanche function to mix bits
     * @param hash Initial hash value
     * @return Mixed hash value
     */
    uint64_t avalanche(uint64_t hash) const;
    
    /**
     * @brief Get name of this equivalence key
     * @return Name of the key
     */
    std::string getName() const;
    
    /**
     * @brief Get bit allocation for each property
     * @return Map of property to bit count
     */
    std::map<Material::MaterialProperty, size_t> getBitAllocation() const;
    
    /**
     * @brief Configure for CFD simulation
     * @param isMultiphase Whether simulation involves multiple phases
     * @param hashBits Number of bits to use for hash
     * @param includeHeatTransfer Whether to include heat transfer in simulation
     */
    void configureForCFD(bool isMultiphase, size_t hashBits, bool includeHeatTransfer = false);
    
    /**
     * @brief Configure for thermal simulation
     * @param hashBits Number of bits to use for hash
     * @param includeFluidFlow Whether to include fluid flow in simulation
     */
    void configureForThermal(size_t hashBits, bool includeFluidFlow = false);
    
    /**
     * @brief Configure for structural simulation
     * @param hashBits Number of bits to use for hash
     * @param includeThermalStress Whether to include thermal stress in simulation
     */
    void configureForStructural(size_t hashBits, bool includeThermalStress = false);
    
    /**
     * @brief Set number of bits to use for a property
     * @param prop Property to set bits for
     * @param bits Number of bits
     */
    void setPropertyBits(Material::MaterialProperty prop, size_t bits);
    
    /**
     * @brief Set range for a property
     * @param prop Property to set range for
     * @param min Minimum value
     * @param max Maximum value
     * @param scaleType Type of scaling to use
     */
    void setPropertyRange(Material::MaterialProperty prop, 
                         double min, double max, 
                         ScaleType scaleType = ScaleType::LINEAR);
    
    /**
     * @brief Set hash resolution in bits
     * @param bits Number of bits
     */
    void setHashResolution(size_t bits);
    
    /**
     * @brief Get version compatibility mode
     * @return True if compatibility mode is supported
     */
    bool supportsVersionCompatibilityMode() const;
    
    /**
     * @brief Set version compatibility mode
     * @param version Version to be compatible with (0 = auto)
     */
    void setVersionCompatibilityMode(int version);
    
    /**
     * @brief Set equivalence tolerance
     * @param tolerance Tolerance value (default: 0.01 for 1%)
     */
    void setEquivalenceTolerance(double tolerance);
    
    /**
     * @brief Get equivalence tolerance
     * @return Current tolerance value
     */
    double getEquivalenceTolerance() const;
    
private:
    /**
     * @brief Initialize default configuration
     */
    void initialize();
    
    /**
     * @brief Calculate normalized property value for hashing
     * @param prop Property type
     * @param value Property value
     * @return Normalized value (0.0-1.0)
     */
    double normalizeProperty(Material::MaterialProperty prop, double value) const;
    
    /**
     * @brief Check if two property values are equivalent within tolerance
     * @param prop Property type
     * @param valueA First value
     * @param valueB Second value
     * @return True if values are equivalent
     */
    bool arePropertyValuesEquivalent(Material::MaterialProperty prop, 
                                    double valueA, double valueB) const;
    
    // Member variables
    std::map<Material::MaterialProperty, size_t> m_propertyBits;
    std::map<Material::MaterialProperty, std::pair<double, double>> m_propertyRanges;
    std::map<Material::MaterialProperty, ScaleType> m_propertyScales;
    size_t m_hashBits;
    int m_compatibilityVersion;
    std::string m_name;
    double m_equivalenceTolerance;
};


