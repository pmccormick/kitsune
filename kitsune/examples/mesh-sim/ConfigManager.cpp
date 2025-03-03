#include "ConfigManager.h"
#include <regex>
#include <iomanip>

ConfigManager::ConfigManager() {
  // Initialize with default groups
  addGroup("Simulation", "General simulation parameters");
  addGroup("Grid", "Computational grid parameters");
  addGroup("Fluid", "Fluid properties and parameters");
  addGroup("Boundary", "Boundary condition parameters");
  addGroup("Solver", "Numerical solver parameters");
  addGroup("Visualization", "Visualization and output parameters");
}

void ConfigManager::addGroup(const std::string& name, const std::string& description) {
  ParameterGroup group;
  group.name = name;
  group.description = description;
  m_groups[name] = group;
}

bool ConfigManager::hasGroup(const std::string& name) const {
    return m_groups.find(name) != m_groups.end();
}

std::string ConfigManager::getTypeString(const ConfigManager::ParamValue& value) const {
  if (std::holds_alternative<int>(value)) return "int";
  if (std::holds_alternative<double>(value)) return "double";
  if (std::holds_alternative<bool>(value)) return "bool";
  if (std::holds_alternative<std::string>(value)) return "string";
  if (std::holds_alternative<std::vector<int>>(value)) return "vector<int>";
  if (std::holds_alternative<std::vector<double>>(value)) return "vector<double>";
  if (std::holds_alternative<std::vector<std::string>>(value)) return "vector<string>";
  return "unknown";
}

std::string ConfigManager::serializeValue(const ConfigManager::ParamValue& value) const {
  std::ostringstream oss;
    
  if (std::holds_alternative<int>(value)) {
    oss << std::get<int>(value);
  }
  else if (std::holds_alternative<double>(value)) {
    oss << std::fixed << std::setprecision(10) << std::get<double>(value);
  }
  else if (std::holds_alternative<bool>(value)) {
    oss << (std::get<bool>(value) ? "true" : "false");
  }
  else if (std::holds_alternative<std::string>(value)) {
    oss << "\"" << std::get<std::string>(value) << "\"";
  }
  else if (std::holds_alternative<std::vector<int>>(value)) {
    const auto& vec = std::get<std::vector<int>>(value);
    oss << "[";
    for (size_t i = 0; i < vec.size(); ++i) {
      oss << vec[i];
      if (i < vec.size() - 1) oss << ", ";
    }
    oss << "]";
  }
  else if (std::holds_alternative<std::vector<double>>(value)) {
    const auto& vec = std::get<std::vector<double>>(value);
    oss << "[";
    for (size_t i = 0; i < vec.size(); ++i) {
      oss << std::fixed << std::setprecision(10) << vec[i];
      if (i < vec.size() - 1) oss << ", ";
    }
    oss << "]";
  }
  else if (std::holds_alternative<std::vector<std::string>>(value)) {
    const auto& vec = std::get<std::vector<std::string>>(value);
    oss << "[";
    for (size_t i = 0; i < vec.size(); ++i) {
      oss << "\"" << vec[i] << "\"";
      if (i < vec.size() - 1) oss << ", ";
    }
    oss << "]";
  }
    
  return oss.str();
}

ConfigManager::ParamValue ConfigManager::deserializeValue(const std::string& type, const std::string& value) const {
  if (type == "int") {
    return std::stoi(value);
  }
  else if (type == "double") {
    return std::stod(value);
  }
  else if (type == "bool") {
    return (value == "true" || value == "1");
  }
  else if (type == "string") {
    // Remove surrounding quotes if present
    std::string str = value;
    if (str.size() >= 2 && str.front() == '\"' && str.back() == '\"') {
      str = str.substr(1, str.size() - 2);
    }
    return str;
  }
  else if (type == "vector<int>") {
    std::vector<int> result;
    // Parse array format [val1, val2, ...]
    std::string content = value.substr(1, value.size() - 2); // Remove []
    std::istringstream iss(content);
    std::string token;
    while (std::getline(iss, token, ',')) {
      // Trim whitespace
      token.erase(0, token.find_first_not_of(" \t"));
      token.erase(token.find_last_not_of(" \t") + 1);
      if (!token.empty()) {
	result.push_back(std::stoi(token));
      }
    }
    return result;
  }
  else if (type == "vector<double>") {
    std::vector<double> result;
    // Parse array format [val1, val2, ...]
    std::string content = value.substr(1, value.size() - 2); // Remove []
    std::istringstream iss(content);
    std::string token;
    while (std::getline(iss, token, ',')) {
      // Trim whitespace
      token.erase(0, token.find_first_not_of(" \t"));
      token.erase(token.find_last_not_of(" \t") + 1);
      if (!token.empty()) {
	result.push_back(std::stod(token));
      }
    }
    return result;
  }
  else if (type == "vector<string>") {
    std::vector<std::string> result;
    // Parse array format ["val1", "val2", ...]
    std::string content = value.substr(1, value.size() - 2); // Remove []
        
    // More complex parsing needed for strings with commas
    size_t pos = 0;
    while (pos < content.size()) {
      // Find next quoted string
      size_t startQuote = content.find('\"', pos);
      if (startQuote == std::string::npos) break;
            
      size_t endQuote = content.find('\"', startQuote + 1);
      if (endQuote == std::string::npos) break;
            
      // Extract the string without quotes
      std::string str = content.substr(startQuote + 1, endQuote - startQuote - 1);
      result.push_back(str);
            
      // Move past this string and the comma
      pos = endQuote + 1;
      pos = content.find(',', pos);
      if (pos == std::string::npos) break;
      pos++;
    }
        
    return result;
  }
    
  // Default to string if type is unknown
  return value;
}

bool ConfigManager::loadFromFile(const std::string& filename) {
  std::ifstream file(filename);
  if (!file.is_open()) {
    std::cerr << "Failed to open config file: " << filename << std::endl;
    return false;
  }
    
  // Clear existing configuration
  m_groups.clear();
    
  std::string currentGroup;
  std::string line;
  std::regex groupRegex(R"(\[(.+)\])");
  std::regex paramRegex(R"(([^=]+)=\s*([^#]+)(?:#\s*(.+))?)");
  std::regex typeRegex(R"((.+)::(.+))");
    
  while (std::getline(file, line)) {
    // Skip empty lines and comments
    if (line.empty() || line[0] == '#') continue;
        
    // Check for group header [Group]
    std::smatch groupMatch;
    if (std::regex_search(line, groupMatch, groupRegex)) {
      currentGroup = groupMatch[1].str();
            
      // Extract description if present after #
      size_t hashPos = line.find('#');
      std::string description;
      if (hashPos != std::string::npos) {
	description = line.substr(hashPos + 1);
	// Trim whitespace
	description.erase(0, description.find_first_not_of(" \t"));
      }
            
      addGroup(currentGroup, description);
      continue;
    }
        
    // Parse parameter
    std::smatch paramMatch;
    if (std::regex_search(line, paramMatch, paramRegex) && !currentGroup.empty()) {
      std::string paramName = paramMatch[1].str();
      std::string paramValue = paramMatch[2].str();
      std::string paramDescription;
            
      // Trim whitespace
      paramName.erase(0, paramName.find_first_not_of(" \t"));
      paramName.erase(paramName.find_last_not_of(" \t") + 1);
            
      paramValue.erase(0, paramValue.find_first_not_of(" \t"));
      paramValue.erase(paramValue.find_last_not_of(" \t") + 1);
            
      // Check for type information in parameter name
      std::smatch typeMatch;
      std::string type = "string"; // Default type
      if (std::regex_match(paramName, typeMatch, typeRegex)) {
	paramName = typeMatch[1].str();
	type = typeMatch[2].str();
      }
            
      // Extract description if present
      if (paramMatch.size() > 3 && paramMatch[3].matched) {
	paramDescription = paramMatch[3].str();
	// Trim whitespace
	paramDescription.erase(0, paramDescription.find_first_not_of(" \t"));
      }
            
      // Store parameter
      ConfigManager::ParamValue value = deserializeValue(type, paramValue);
      m_groups[currentGroup].parameters[paramName] = value;
            
      // Store description if available
      if (!paramDescription.empty()) {
	m_paramDescriptions[currentGroup][paramName] = paramDescription;
      }
    }
  }
    
  return true;
}

bool ConfigManager::saveToFile(const std::string& filename) const {
  std::ofstream file(filename);
  if (!file.is_open()) {
    std::cerr << "Failed to create config file: " << filename << std::endl;
    return false;
  }
    
  file << "# CFD Simulation Configuration File\n";
  file << "# Generated on: " << __DATE__ << " " << __TIME__ << "\n\n";
    
  for (const auto& [groupName, group] : m_groups) {
    file << "[" << groupName << "]";
        
    // Add group description if available
    if (!group.description.empty()) {
      file << " # " << group.description;
    }
    file << "\n";
        
    for (const auto& [paramName, value] : group.parameters) {
      std::string typeStr = getTypeString(value);
      file << paramName << "::" << typeStr << " = " << serializeValue(value);
            
      // Add parameter description if available
      if (m_paramDescriptions.count(groupName) && 
	  m_paramDescriptions.at(groupName).count(paramName)) {
	file << " # " << m_paramDescriptions.at(groupName).at(paramName);
      }
      file << "\n";
    }
    file << "\n";
  }
    
  return true;
}

void ConfigManager::printConfig(std::ostream& os) const {
  os << "======== CFD Simulation Configuration ========\n";
    
  for (const auto& [groupName, group] : m_groups) {
    os << "[" << groupName << "]";
    if (!group.description.empty()) {
      os << " - " << group.description;
    }
    os << "\n";
        
    for (const auto& [paramName, value] : group.parameters) {
      os << "  " << paramName << " = " << serializeValue(value);
            
      // Add parameter description if available
      if (m_paramDescriptions.count(groupName) && 
	  m_paramDescriptions.at(groupName).count(paramName)) {
	os << " (" << m_paramDescriptions.at(groupName).at(paramName) << ")";
      }
      os << "\n";
    }
    os << "\n";
  }
    
  os << "==============================================\n";
}

const std::map<std::string, ConfigManager::ParameterGroup>& ConfigManager::getAllGroups() const {
  return m_groups;
}

bool ConfigManager::validate() const {
  // In a real implementation, you would check for required parameters
  // and validate that parameters are within acceptable ranges.
    
  // For now, just check that we have some basic required groups
  std::vector<std::string> requiredGroups = {
    "Simulation", "Grid", "Fluid", "Boundary", "Solver"
  };
    
  for (const auto& group : requiredGroups) {
    if (!hasGroup(group)) {
      std::cerr << "Validation failed: Missing required group '" << group << "'" << std::endl;
      return false;
    }
  }
    
  // Check for specific required parameters in each group
  // This would be expanded with proper validation in a real implementation
    
  return true;
}


