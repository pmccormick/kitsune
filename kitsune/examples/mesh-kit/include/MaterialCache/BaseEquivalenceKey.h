/**
 * @class BaseEquivalenceKey
 * @brief Base class for equivalence key implementations with shared functionality
 */
class BaseEquivalenceKey : public IMaterialEquivalenceKey {
public:
  explicit BaseEquivalenceKey(double tolerance = 1e-6);
  
  // Core implementation of hash that can be reused
  size_t hashImpl(const std::shared_ptr<Material>& material,
                  double tolerance,
                  bool compareType,
                  bool compareName,
                  bool compareRefTemperature,
                  bool compareMixtureComponents,
                  const std::array<bool, static_cast<size_t>(Material::MaterialProperty::COUNT)>& propertyFlags,
                  const std::array<double, static_cast<size_t>(Material::MaterialProperty::COUNT)>& propertyTolerances) const;
  
  // Default implementations of IMaterialEquivalenceKey methods
  double getTolerance() const override { return m_tolerance; }
  void setTolerance(double tolerance) override { m_tolerance = tolerance; }
  
protected:
  double m_tolerance;
};

// Implementation
BaseEquivalenceKey::BaseEquivalenceKey(double tolerance)
  : m_tolerance(tolerance) {
}

size_t BaseEquivalenceKey::hashImpl(
    const std::shared_ptr<Material>& material,
    double tolerance,
    bool compareType,
    bool compareName,
    bool compareRefTemperature,
    bool compareMixtureComponents,
    const std::array<bool, static_cast<size_t>(Material::MaterialProperty::COUNT)>& propertyFlags,
    const std::array<double, static_cast<size_t>(Material::MaterialProperty::COUNT)>& propertyTolerances) const {
  
  // Use a prime number for initial hash
  size_t h = 17;
  
  // Hash the type if comparison is enabled
  if (compareType) {
    h = h * 31 + std::hash<int>{}(static_cast<int>(material->getType()));
  }
  
  // Hash the name if comparison is enabled
  if (compareName) {
    h = h * 31 + std::hash<std::string>{}(material->getName());
  }
  
  // Hash the reference temperature if comparison is enabled
  if (compareRefTemperature) {
    h = h * 31 + hashDouble(material->getReferenceTemperature(), tolerance);
  }
  
  // Hash each compared property using the bit pattern approach
  for (size_t i = 0; i < static_cast<size_t>(Material::MaterialProperty::COUNT); ++i) {
    if (propertyFlags[i]) {
      auto prop = static_cast<Material::MaterialProperty>(i);
      double value = material->getProperty(prop);
      
      // Use property-specific tolerance if provided
      double propTolerance = propertyTolerances[i];
      
      // Use our bit-pattern hash function for consistent floating-point hashing
      h = h * 31 + hashDouble(value, propTolerance);
    }
  }
  
  // Hash mixture components if enabled
  if (compareMixtureComponents && material->isMixture()) {
    auto components = material->getMixtureComponents();
    
    // Hash component count
    h = h * 31 + components.size();
    
    // Hash each component
    for (size_t i = 0; i < components.size(); ++i) {
      // This is a simplification - in actual code, need to implement recursive hashing
      // according to the specific key implementation
      h = h * 31 + hashDouble(components[i].second, tolerance); // Hash fraction
    }
  }
  
  return h;
}

