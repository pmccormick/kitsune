#pragma once

#include "BoundaryClass.h"
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>

/**
 * @class BoundaryFactory
 * @brief Factory class for creating and managing boundary condition objects
 *
 * This factory centralizes the creation and deserialization of boundary
 * conditions, providing a single point of access for all boundary-related
 * construction operations. It implements the Factory design pattern to decouple
 * the creation of boundary objects from the code that uses them.
 *
 * Features:
 * - Creation of boundary objects by type name
 * - Deserialization of boundary objects from string data
 * - Registration system for boundary types
 * - Support for custom boundary creation logic
 *
 * Usage:
 * - Call registerBoundaryTypes() once at program initialization
 * - Use createBoundary() to create new boundary instances
 * - Use deserializeBoundary() to recreate boundaries from serialized data
 */
class BoundaryFactory {
public:
  /**
   * @brief Create a boundary by type name
   * @param type The type identifier of the boundary (e.g., "Inflow", "Slip")
   * @param name Optional name for the boundary
   * @return Shared pointer to the created boundary, or nullptr if type is
   * unknown
   */
  static std::shared_ptr<BoundaryClass>
  createBoundary(const std::string &type, const std::string &name = "");

  /**
   * @brief Register all standard boundary types
   *
   * This method must be called once at program initialization before
   * using any other factory methods.
   */
  static void registerBoundaryTypes();

  /**
   * @brief Deserialize a boundary from its string representation
   * @param type The type identifier of the boundary
   * @param data Serialized boundary data
   * @return Shared pointer to the deserialized boundary, or nullptr on failure
   */
  static std::shared_ptr<BoundaryClass>
  deserializeBoundary(const std::string &type, const std::string &data);

  /**
   * @brief Register a custom boundary type
   * @param type The type identifier for the boundary
   * @param creator Function to create an instance of this boundary type
   */
  static void registerBoundaryType(
      const std::string &type,
      std::function<std::shared_ptr<BoundaryClass>(const std::string &)>
          creator);

  /**
   * @brief Check if a boundary type is registered
   * @param type The type identifier to check
   * @return True if the type is registered, false otherwise
   */
  static bool isTypeRegistered(const std::string &type);

  /**
   * @brief Get a list of all registered boundary types
   * @return Vector of type names
   */
  static std::vector<std::string> getRegisteredTypes();

private:
  // Function signature for boundary creation
  using BoundaryCreator =
      std::function<std::shared_ptr<BoundaryClass>(const std::string &)>;

  // Registry of boundary creators
  static std::unordered_map<std::string, BoundaryCreator> s_boundaryCreators;

  // Private constructor to prevent instantiation
  BoundaryFactory() = default;
};