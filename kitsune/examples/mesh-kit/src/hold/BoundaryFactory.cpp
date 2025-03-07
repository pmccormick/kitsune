#include "BoundaryFactory.h"
#include "DirichletBoundary.h"
#include "InflowBoundary.h"
// Include all other boundary classes here

// Initialize static member variables
std::unordered_map<std::string, BoundaryFactory::BoundaryCreator>
    BoundaryFactory::s_boundaryCreators;

std::unordered_map<BoundaryType, BoundaryFactory::BoundaryCreator>
    BoundaryFactory::s_boundaryEnumCreators;

// Register standard boundary types
void BoundaryFactory::registerBoundaryTypes() {
  // Register each boundary type with both string and enum identifiers

  // Dirichlet boundary
  registerBoundaryType(
      "Dirichlet",
      [](const std::string &name) -> std::shared_ptr<BoundaryClass> {
        return std::make_shared<DirichletBoundary>(name);
      });
  registerBoundaryTypeEnum(
      BoundaryType::DIRICHLET,
      [](const std::string &name) -> std::shared_ptr<BoundaryClass> {
        return std::make_shared<DirichletBoundary>(name);
      });

  // Inflow boundary
  registerBoundaryType(
      "Inflow", [](const std::string &name) -> std::shared_ptr<BoundaryClass> {
        return std::make_shared<InflowBoundary>(name);
      });
  registerBoundaryTypeEnum(
      BoundaryType::INFLOW,
      [](const std::string &name) -> std::shared_ptr<BoundaryClass> {
        return std::make_shared<InflowBoundary>(name);
      });

  // Register other standard boundary types here with both string and enum
  // identifiers For example: registerBoundaryType("NoSlip", ...);
  // registerBoundaryTypeEnum(BoundaryType::NO_SLIP, ...);
}

// Create boundary by string type identifier
std::shared_ptr<BoundaryClass>
BoundaryFactory::createBoundary(const std::string &type,
                                const std::string &name) {

  auto it = s_boundaryCreators.find(type);
  if (it != s_boundaryCreators.end()) {
    return it->second(name);
  }
  return nullptr;
}

// Create boundary by enum type
std::shared_ptr<BoundaryClass>
BoundaryFactory::createBoundaryByEnum(BoundaryType type,
                                      const std::string &name) {

  auto it = s_boundaryEnumCreators.find(type);
  if (it != s_boundaryEnumCreators.end()) {
    return it->second(name);
  }
  return nullptr;
}

// Deserialize a boundary from string representation using string type
// identifier
std::shared_ptr<BoundaryClass>
BoundaryFactory::deserializeBoundary(const std::string &type,
                                     const std::string &data) {

  // Create the boundary instance
  auto boundary = createBoundary(type, "");
  if (!boundary) {
    return nullptr;
  }

  // Deserialize the data
  if (!boundary->deserialize(data)) {
    return nullptr;
  }

  return boundary;
}

// Deserialize a boundary from string representation using enum type identifier
std::shared_ptr<BoundaryClass>
BoundaryFactory::deserializeBoundaryByEnum(BoundaryType type,
                                           const std::string &data) {

  // Create the boundary instance
  auto boundary = createBoundaryByEnum(type, "");
  if (!boundary) {
    return nullptr;
  }

  // Deserialize the data
  if (!boundary->deserialize(data)) {
    return nullptr;
  }

  return boundary;
}

// Register a custom boundary type with string identifier
void BoundaryFactory::registerBoundaryType(
    const std::string &type,
    std::function<std::shared_ptr<BoundaryClass>(const std::string &)>
        creator) {

  s_boundaryCreators[type] = creator;
}

// Register a custom boundary type with enum identifier
void BoundaryFactory::registerBoundaryTypeEnum(
    BoundaryType type,
    std::function<std::shared_ptr<BoundaryClass>(const std::string &)>
        creator) {

  s_boundaryEnumCreators[type] = creator;
}

// Check if boundary type is registered by string
bool BoundaryFactory::isTypeRegistered(const std::string &type) {
  return s_boundaryCreators.find(type) != s_boundaryCreators.end();
}

// Check if boundary type is registered by enum
bool BoundaryFactory::isTypeRegistered(BoundaryType type) {
  return s_boundaryEnumCreators.find(type) != s_boundaryEnumCreators.end();
}

// Get list of all registered boundary types
std::vector<std::string> BoundaryFactory::getRegisteredTypes() {
  std::vector<std::string> types;
  types.reserve(s_boundaryCreators.size());

  for (const auto &pair : s_boundaryCreators) {
    types.push_back(pair.first);
  }

  return types;
}
