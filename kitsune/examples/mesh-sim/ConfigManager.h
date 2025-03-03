#pragma once

#include <string>
#include <map>
#include <vector>
#include <fstream>
#include <sstream>
#include <iostream>
#include <memory>
#include <any>
#include <typeinfo>
#include <typeindex>
#include <filesystem>
#include <variant>

/**
 * @class ConfigManager
 * @brief Manages configuration parameters for the CFD simulation
 * 
 * The ConfigManager provides a flexible system for storing, retrieving,
 * and persisting simulation parameters. It supports various parameter
 * types and organized parameter groups.
 */
class ConfigManager {
public:
  // Constructor
  ConfigManager();

  // Parameter type definitions for type-safe storage
  using ParamValue = std::variant<int, double, bool, std::string, 
				  std::vector<int>, std::vector<double>, std::vector<std::string>>;

  /**
   * @struct ParameterGroup
   * @brief Organizes related parameters into logical groups
   */
  struct ParameterGroup {
    std::string name;
    std::string description;
    std::map<std::string, ParamValue> parameters;
  };

  // Methods to add/set parameter groups and parameters
  void addGroup(const std::string& name, const std::string& description = "");
  bool hasGroup(const std::string& name) const;
    
  // Template methods for setting parameters with type safety
  template<typename T>
  void setParameter(const std::string& groupName, const std::string& paramName, 
		    const T& value, const std::string& description = "");
    
  // Template methods for getting parameters with type safety
  template<typename T>
  T getParameter(const std::string& groupName, const std::string& paramName, 
		 const T& defaultValue = T()) const;
    
  // Load configuration from file
  bool loadFromFile(const std::string& filename);
    
  // Save configuration to file
  bool saveToFile(const std::string& filename) const;
    
  // Print configuration (useful for debugging)
  void printConfig(std::ostream& os = std::cout) const;
    
  // Get all parameter groups
  const std::map<std::string, ParameterGroup>& getAllGroups() const;

  // Validate the configuration
  bool validate() const;

private:
  std::map<std::string, ParameterGroup> m_groups;
  std::map<std::string, std::map<std::string, std::string>> m_paramDescriptions;
    
  // Helper methods for serialization
  std::string serializeValue(const ParamValue& value) const;
  ParamValue deserializeValue(const std::string& type, const std::string& value) const;
    
  // Helper method to get the type string of a variant
  std::string getTypeString(const ParamValue& value) const;
};

// Template method implementations
template<typename T>
void ConfigManager::setParameter(const std::string& groupName, const std::string& paramName, 
				 const T& value, const std::string& description) {
  // Create group if it doesn't exist
  if (!hasGroup(groupName)) {
    addGroup(groupName);
  }
    
  // Store the parameter
  m_groups[groupName].parameters[paramName] = value;
    
  // Store description if provided
  if (!description.empty()) {
    m_paramDescriptions[groupName][paramName] = description;
  }
}

template<typename T>
T ConfigManager::getParameter(const std::string& groupName, const std::string& paramName, 
			      const T& defaultValue) const {
  // Check if group exists
  if (!hasGroup(groupName)) {
    return defaultValue;
  }
    
  const auto& group = m_groups.at(groupName);
    
  // Check if parameter exists
  if (group.parameters.find(paramName) == group.parameters.end()) {
    return defaultValue;
  }
    
  // Try to get the value with the right type
  try {
    return std::get<T>(group.parameters.at(paramName));
  } catch (const std::bad_variant_access&) {
    std::cerr << "Type mismatch for parameter: " << groupName << "." << paramName << std::endl;
    return defaultValue;
  }
}


