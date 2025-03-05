#include "BoundaryFactory.h"

// Include all boundary condition classes
#include "DirichletBoundary.h"
#include "InflowBoundary.h"
#include "NeumannBoundary.h"
#include "NoSlipBoundary.h"
#include "PeriodicBoundary.h"
#include "SlipBoundary.h"

#include <algorithm>
#include <vector>

// Initialize static registry
std::unordered_map<std::string, BoundaryFactory::BoundaryCreator>
    BoundaryFactory::s_boundaryCreators;

std::shared_ptr<BoundaryClass>
BoundaryFactory::createBoundary(const std::string &type,
                                const std::string &name) {
  // Look for the creator function in the registry
  auto it = s_boundaryCreators.find(type);
  if (it != s_boundaryCreators.end()) {
    // Call the creator function with the provided name
    return it->second(name);
  }

  // Type not found
  return nullptr;
}

void BoundaryFactory::registerBoundaryTypes() {
  // Register all standard boundary types with their creation functions

  // DirichletBoundary
  registerBoundaryType("Dirichlet", [](const std::string &name) {
    return std::make_shared<DirichletBoundary>(
        name.empty() ? "Dirichlet Boundary" : name);
  });

  // InflowBoundary
  registerBoundaryType("Inflow", [](const std::string &) {
    return std::make_shared<InflowBoundary>();
  });

  // NeumannBoundary
  registerBoundaryType("Neumann", [](const std::string &) {
    return std::make_shared<NeumannBoundary>();
  });

  // NoSlipBoundary
  registerBoundaryType("NoSlip", [](const std::string &name) {
    return std::make_shared<NoSlipBoundary>(name.empty() ? "NoSlip Boundary"
                                                         : name);
  });

  // PeriodicBoundary
  registerBoundaryType("Periodic", [](const std::string &name) {
    // Note: PeriodicBoundary requires a paired boundary name, which would need
    // to be set later
    return std::make_shared<PeriodicBoundary>(name.empty() ? "Periodic Boundary"
                                                           : name,
                                              "" // Empty paired boundary name
    );
  });

  // SlipBoundary
  registerBoundaryType("Slip", [](const std::string &) {
    return std::make_shared<SlipBoundary>();
  });
}

std::shared_ptr<BoundaryClass>
BoundaryFactory::deserializeBoundary(const std::string &type,
                                     const std::string &data) {
  // Create a new boundary instance of the specified type
  auto boundary = createBoundary(type);

  // If creation was successful, deserialize the data
  if (boundary) {
    if (boundary->deserialize(data)) {
      return boundary;
    }
    // If deserialization failed, return nullptr
  }

  return nullptr;
}

void BoundaryFactory::registerBoundaryType(
    const std::string &type,
    std::function<std::shared_ptr<BoundaryClass>(const std::string &)>
        creator) {
  // Register the creator function for this boundary type
  s_boundaryCreators[type] = creator;
}

bool BoundaryFactory::isTypeRegistered(const std::string &type) {
  return s_boundaryCreators.find(type) != s_boundaryCreators.end();
}

std::vector<std::string> BoundaryFactory::getRegisteredTypes() {
  std::vector<std::string> types;
  types.reserve(s_boundaryCreators.size());

  // Extract all keys from the registry
  for (const auto &pair : s_boundaryCreators) {
    types.push_back(pair.first);
  }

  // Sort for consistent ordering
  std::sort(types.begin(), types.end());

  return types;
}
