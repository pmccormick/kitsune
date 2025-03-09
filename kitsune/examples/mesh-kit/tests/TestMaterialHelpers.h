/**
 * @file TestMaterialHelpers.h
 * @brief Helper functions for creating test materials
 */
#pragma once

#include "Material.h"
#include <memory>
#include <cmath>
#include <limits>
#include <string>

/**
 * A collection of helper functions for creating material instances for testing.
 */
class TestMaterialHelpers {
public:
    /**
     * Creates a water material with standard properties.
     * @return Shared pointer to a water material
     */
    static std::shared_ptr<Material> createWaterMaterial() {
        auto water = std::make_shared<Material>(Material::MaterialType::FLUID, "Water");
        
        // Set standard water properties at 20°C
        water->setProperty(Material::MaterialProperty::DENSITY, 998.2);                // kg/m³
        water->setProperty(Material::MaterialProperty::DYNAMIC_VISCOSITY, 0.001002);   // Pa·s
        water->setProperty(Material::MaterialProperty::THERMAL_CONDUCTIVITY, 0.5984);  // W/(m·K)
        water->setProperty(Material::MaterialProperty::SPECIFIC_HEAT, 4182.0);         // J/(kg·K)
        water->setProperty(Material::MaterialProperty::THERMAL_EXPANSION, 0.000207);   // 1/K
        water->setProperty(Material::MaterialProperty::SURFACE_TENSION, 0.0728);       // N/m
        water->setProperty(Material::MaterialProperty::ELECTRICAL_CONDUCTIVITY, 5.5e-6); // S/m
        
        return water;
    }
    
    /**
     * Creates an air material with standard properties.
     * @return Shared pointer to an air material
     */
    static std::shared_ptr<Material> createAirMaterial() {
        auto air = std::make_shared<Material>(Material::MaterialType::FLUID, "Air");
        
        // Set standard air properties at 20°C, 1 atm
        air->setProperty(Material::MaterialProperty::DENSITY, 1.204);                // kg/m³
        air->setProperty(Material::MaterialProperty::DYNAMIC_VISCOSITY, 1.825e-5);   // Pa·s
        air->setProperty(Material::MaterialProperty::THERMAL_CONDUCTIVITY, 0.0257);  // W/(m·K)
        air->setProperty(Material::MaterialProperty::SPECIFIC_HEAT, 1005.0);         // J/(kg·K)
        air->setProperty(Material::MaterialProperty::THERMAL_EXPANSION, 0.00343);    // 1/K
        air->setProperty(Material::MaterialProperty::SURFACE_TENSION, 0.0);          // N/m (N/A for gas)
        air->setProperty(Material::MaterialProperty::ELECTRICAL_CONDUCTIVITY, 0.0);  // S/m (near zero)
        
        return air;
    }
    
    /**
     * Creates an aluminum material with standard properties.
     * @return Shared pointer to an aluminum material
     */
    static std::shared_ptr<Material> createAluminumMaterial() {
        auto aluminum = std::make_shared<Material>(Material::MaterialType::SOLID, "Aluminum");
        
        // Set standard aluminum properties
        aluminum->setProperty(Material::MaterialProperty::DENSITY, 2700.0);              // kg/m³
        aluminum->setProperty(Material::MaterialProperty::DYNAMIC_VISCOSITY, 0.0);       // Pa·s (N/A for solid)
        aluminum->setProperty(Material::MaterialProperty::THERMAL_CONDUCTIVITY, 237.0);  // W/(m·K)
        aluminum->setProperty(Material::MaterialProperty::SPECIFIC_HEAT, 897.0);         // J/(kg·K)
        aluminum->setProperty(Material::MaterialProperty::THERMAL_EXPANSION, 23.1e-6);   // 1/K
        aluminum->setProperty(Material::MaterialProperty::SURFACE_TENSION, 0.914);       // N/m (molten)
        aluminum->setProperty(Material::MaterialProperty::ELECTRICAL_CONDUCTIVITY, 3.5e7); // S/m
        
        return aluminum;
    }
    
    /**
     * Creates a steel material with standard properties.
     * @return Shared pointer to a steel material
     */
    static std::shared_ptr<Material> createSteelMaterial() {
        auto steel = std::make_shared<Material>(Material::MaterialType::SOLID, "Steel");
        
        // Set standard steel properties (average carbon steel)
        steel->setProperty(Material::MaterialProperty::DENSITY, 7850.0);               // kg/m³
        steel->setProperty(Material::MaterialProperty::DYNAMIC_VISCOSITY, 0.0);        // Pa·s (N/A for solid)
        steel->setProperty(Material::MaterialProperty::THERMAL_CONDUCTIVITY, 50.0);    // W/(m·K)
        steel->setProperty(Material::MaterialProperty::SPECIFIC_HEAT, 490.0);          // J/(kg·K)
        steel->setProperty(Material::MaterialProperty::THERMAL_EXPANSION, 12.0e-6);    // 1/K
        steel->setProperty(Material::MaterialProperty::SURFACE_TENSION, 1.7);          // N/m (molten)
        steel->setProperty(Material::MaterialProperty::ELECTRICAL_CONDUCTIVITY, 6.99e6); // S/m
        
        return steel;
    }
    
    /**
     * Creates an oil material with standard properties.
     * @return Shared pointer to an oil material
     */
    static std::shared_ptr<Material> createOilMaterial() {
        auto oil = std::make_shared<Material>(Material::MaterialType::FLUID, "Engine Oil");
        
        // Set standard engine oil properties
        oil->setProperty(Material::MaterialProperty::DENSITY, 870.0);                // kg/m³
        oil->setProperty(Material::MaterialProperty::DYNAMIC_VISCOSITY, 0.25);       // Pa·s
        oil->setProperty(Material::MaterialProperty::THERMAL_CONDUCTIVITY, 0.145);   // W/(m·K)
        oil->setProperty(Material::MaterialProperty::SPECIFIC_HEAT, 1900.0);         // J/(kg·K)
        oil->setProperty(Material::MaterialProperty::THERMAL_EXPANSION, 0.00070);    // 1/K
        oil->setProperty(Material::MaterialProperty::SURFACE_TENSION, 0.03);         // N/m
        oil->setProperty(Material::MaterialProperty::ELECTRICAL_CONDUCTIVITY, 1e-12); // S/m
        
        return oil;
    }
    
    /**
     * Creates a mercury material with standard properties.
     * @return Shared pointer to a mercury material
     */
    static std::shared_ptr<Material> createMercuryMaterial() {
        auto mercury = std::make_shared<Material>(Material::MaterialType::FLUID, "Mercury");
        
        // Set standard mercury properties
        mercury->setProperty(Material::MaterialProperty::DENSITY, 13534.0);             // kg/m³
        mercury->setProperty(Material::MaterialProperty::DYNAMIC_VISCOSITY, 0.00155);   // Pa·s
        mercury->setProperty(Material::MaterialProperty::THERMAL_CONDUCTIVITY, 8.3);    // W/(m·K)
        mercury->setProperty(Material::MaterialProperty::SPECIFIC_HEAT, 140.0);         // J/(kg·K)
        mercury->setProperty(Material::MaterialProperty::THERMAL_EXPANSION, 0.000181);  // 1/K
        mercury->setProperty(Material::MaterialProperty::SURFACE_TENSION, 0.487);       // N/m
        mercury->setProperty(Material::MaterialProperty::ELECTRICAL_CONDUCTIVITY, 1.0e6); // S/m
        
        return mercury;
    }
    
    /**
     * Creates a concrete material with standard properties.
     * @return Shared pointer to a concrete material
     */
    static std::shared_ptr<Material> createConcreteMaterial() {
        auto concrete = std::make_shared<Material>(Material::MaterialType::SOLID, "Concrete");
        
        // Set standard concrete properties
        concrete->setProperty(Material::MaterialProperty::DENSITY, 2400.0);             // kg/m³
        concrete->setProperty(Material::MaterialProperty::DYNAMIC_VISCOSITY, 0.0);      // Pa·s (N/A for solid)
        concrete->setProperty(Material::MaterialProperty::THERMAL_CONDUCTIVITY, 1.8);   // W/(m·K)
        concrete->setProperty(Material::MaterialProperty::SPECIFIC_HEAT, 880.0);        // J/(kg·K)
        concrete->setProperty(Material::MaterialProperty::THERMAL_EXPANSION, 10.0e-6);  // 1/K
        concrete->setProperty(Material::MaterialProperty::SURFACE_TENSION, 0.0);        // N/m (N/A for solid)
        concrete->setProperty(Material::MaterialProperty::ELECTRICAL_CONDUCTIVITY, 1.0e-5); // S/m
        
        return concrete;
    }
    
    /**
     * Creates a copper material with standard properties.
     * @return Shared pointer to a copper material
     */
    static std::shared_ptr<Material> createCopperMaterial() {
        auto copper = std::make_shared<Material>(Material::MaterialType::SOLID, "Copper");
        
        // Set standard copper properties
        copper->setProperty(Material::MaterialProperty::DENSITY, 8960.0);              // kg/m³
        copper->setProperty(Material::MaterialProperty::DYNAMIC_VISCOSITY, 0.0);       // Pa·s (N/A for solid)
        copper->setProperty(Material::MaterialProperty::THERMAL_CONDUCTIVITY, 401.0);  // W/(m·K)
        copper->setProperty(Material::MaterialProperty::SPECIFIC_HEAT, 385.0);         // J/(kg·K)
        copper->setProperty(Material::MaterialProperty::THERMAL_EXPANSION, 17.0e-6);   // 1/K
        copper->setProperty(Material::MaterialProperty::SURFACE_TENSION, 1.3);         // N/m (molten)
        copper->setProperty(Material::MaterialProperty::ELECTRICAL_CONDUCTIVITY, 5.8e7); // S/m
        
        return copper;
    }
    
    /**
     * Creates a glycerin material with standard properties.
     * @return Shared pointer to a glycerin material
     */
    static std::shared_ptr<Material> createGlycerinMaterial() {
        auto glycerin = std::make_shared<Material>(Material::MaterialType::FLUID, "Glycerin");
        
        // Set standard glycerin properties
        glycerin->setProperty(Material::MaterialProperty::DENSITY, 1260.0);             // kg/m³
        glycerin->setProperty(Material::MaterialProperty::DYNAMIC_VISCOSITY, 1.41);     // Pa·s
        glycerin->setProperty(Material::MaterialProperty::THERMAL_CONDUCTIVITY, 0.285); // W/(m·K)
        glycerin->setProperty(Material::MaterialProperty::SPECIFIC_HEAT, 2430.0);       // J/(kg·K)
        glycerin->setProperty(Material::MaterialProperty::THERMAL_EXPANSION, 0.00050);  // 1/K
        glycerin->setProperty(Material::MaterialProperty::SURFACE_TENSION, 0.064);      // N/m
        glycerin->setProperty(Material::MaterialProperty::ELECTRICAL_CONDUCTIVITY, 6.4e-8); // S/m
        
        return glycerin;
    }
    
    /**
     * Creates a salt water material with standard properties.
     * @return Shared pointer to a salt water material
     */
    static std::shared_ptr<Material> createSaltWaterMaterial(double salinity = 35.0) {
        auto saltWater = std::make_shared<Material>(Material::MaterialType::FLUID, "Salt Water");
        
        // Adjust properties based on salinity (default: average ocean salinity of 35 g/kg)
        double densityFactor = 1.0 + (salinity / 1000.0);
        
        // Set salt water properties
        saltWater->setProperty(Material::MaterialProperty::DENSITY, 998.2 * densityFactor); // kg/m³
        saltWater->setProperty(Material::MaterialProperty::DYNAMIC_VISCOSITY, 0.00108);    // Pa·s
        saltWater->setProperty(Material::MaterialProperty::THERMAL_CONDUCTIVITY, 0.563);   // W/(m·K)
        saltWater->setProperty(Material::MaterialProperty::SPECIFIC_HEAT, 3993.0);         // J/(kg·K)
        saltWater->setProperty(Material::MaterialProperty::THERMAL_EXPANSION, 0.000214);   // 1/K
        saltWater->setProperty(Material::MaterialProperty::SURFACE_TENSION, 0.0728);       // N/m
        saltWater->setProperty(Material::MaterialProperty::ELECTRICAL_CONDUCTIVITY, 4.8);  // S/m
        
        return saltWater;
    }
    
    /**
     * Creates a gasoline material with standard properties.
     * @return Shared pointer to a gasoline material
     */
    static std::shared_ptr<Material> createGasolineMaterial() {
        auto gasoline = std::make_shared<Material>(Material::MaterialType::FLUID, "Gasoline");
        
        // Set standard gasoline properties
        gasoline->setProperty(Material::MaterialProperty::DENSITY, 750.0);               // kg/m³
        gasoline->setProperty(Material::MaterialProperty::DYNAMIC_VISCOSITY, 0.0006);    // Pa·s
        gasoline->setProperty(Material::MaterialProperty::THERMAL_CONDUCTIVITY, 0.116);  // W/(m·K)
        gasoline->setProperty(Material::MaterialProperty::SPECIFIC_HEAT, 2200.0);        // J/(kg·K)
        gasoline->setProperty(Material::MaterialProperty::THERMAL_EXPANSION, 0.00095);   // 1/K
        gasoline->setProperty(Material::MaterialProperty::SURFACE_TENSION, 0.022);       // N/m
        gasoline->setProperty(Material::MaterialProperty::ELECTRICAL_CONDUCTIVITY, 1e-14); // S/m
        
        return gasoline;
    }
    
    /**
     * Creates a material with all properties set to NaN.
     * @return Shared pointer to a material with NaN properties
     */
    static std::shared_ptr<Material> createNaNMaterial() {
        auto nanMaterial = std::make_shared<Material>(Material::MaterialType::FLUID, "NaN Material");
        
        // Set all properties to NaN
        for (int i = 0; i < static_cast<int>(Material::MaterialProperty::COUNT); i++) {
            Material::MaterialProperty prop = static_cast<Material::MaterialProperty>(i);
            nanMaterial->setProperty(prop, std::numeric_limits<double>::quiet_NaN());
        }
        
        return nanMaterial;
    }
    
    /**
     * Creates a material with all properties set to infinity.
     * @return Shared pointer to a material with infinity properties
     */
    static std::shared_ptr<Material> createInfinityMaterial() {
        auto infMaterial = std::make_shared<Material>(Material::MaterialType::FLUID, "Infinity Material");
        
        // Set all properties to infinity
        for (int i = 0; i < static_cast<int>(Material::MaterialProperty::COUNT); i++) {
            Material::MaterialProperty prop = static_cast<Material::MaterialProperty>(i);
            infMaterial->setProperty(prop, std::numeric_limits<double>::infinity());
        }
        
        return infMaterial;
    }
    
    /**
     * Creates a custom material with specified properties.
     * @param name The name of the material
     * @param type The material type
     * @param density Density value (kg/m³)
     * @param viscosity Dynamic viscosity value (Pa·s)
     * @param thermalConductivity Thermal conductivity value (W/(m·K))
     * @param specificHeat Specific heat value (J/(kg·K))
     * @param thermalExpansion Thermal expansion coefficient (1/K)
     * @param surfaceTension Surface tension value (N/m)
     * @param electricalConductivity Electrical conductivity value (S/m)
     * @return Shared pointer to a custom material
     */
    static std::shared_ptr<Material> createCustomMaterial(
        const std::string& name,
        Material::MaterialType type,
        double density,
        double viscosity,
        double thermalConductivity,
        double specificHeat,
        double thermalExpansion,
        double surfaceTension,
        double electricalConductivity
    ) {
        auto material = std::make_shared<Material>(type, name);
        
        material->setProperty(Material::MaterialProperty::DENSITY, density);
        material->setProperty(Material::MaterialProperty::DYNAMIC_VISCOSITY, viscosity);
        material->setProperty(Material::MaterialProperty::THERMAL_CONDUCTIVITY, thermalConductivity);
        material->setProperty(Material::MaterialProperty::SPECIFIC_HEAT, specificHeat);
        material->setProperty(Material::MaterialProperty::THERMAL_EXPANSION, thermalExpansion);
        material->setProperty(Material::MaterialProperty::SURFACE_TENSION, surfaceTension);
        material->setProperty(Material::MaterialProperty::ELECTRICAL_CONDUCTIVITY, electricalConductivity);
        
        return material;
    }
};

