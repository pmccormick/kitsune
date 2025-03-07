#pragma once

#include "Mesh.h"
#include "Vector2D.h"
#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

/**
 * @class MeshValidator
 * @brief Provides validation functionality for computational meshes
 *
 * This class performs various checks on a mesh to ensure it meets
 * quality requirements for stable and accurate CFD simulations.
 */
class MeshValidator {
public:
  /**
   * @struct ValidationIssue
   * @brief Represents a single validation issue detected in the mesh
   */
  struct ValidationIssue {
    enum class Severity {
      INFO,    // Informational only
      WARNING, // May affect solution quality
      ERROR    // Will likely cause simulation failure
    };

    Severity severity;
    std::string message;
    std::string location; // Could be cell ID, face ID, etc.
    double value;         // Relevant metric value
    double threshold;     // Threshold that was exceeded/not met

    ValidationIssue(Severity sev, const std::string &msg,
                    const std::string &loc = "", double val = 0.0,
                    double thresh = 0.0)
        : severity(sev), message(msg), location(loc), value(val),
          threshold(thresh) {}
  };

  /**
   * @brief Constructor
   * @param mesh Reference to the mesh to validate
   */
  MeshValidator(const Mesh &mesh);

  /**
   * @brief Validate mesh orthogonality
   * @param maxNonOrthogonality Maximum acceptable non-orthogonality angle
   * (degrees)
   * @return List of validation issues
   */
  std::vector<ValidationIssue>
  checkOrthogonality(double maxNonOrthogonality = 45.0);

  /**
   * @brief Validate cell aspect ratios
   * @param maxAspectRatio Maximum acceptable aspect ratio
   * @return List of validation issues
   */
  std::vector<ValidationIssue> checkAspectRatio(double maxAspectRatio = 100.0);

  /**
   * @brief Validate cell skewness
   * @param maxSkewness Maximum acceptable skewness (0-1 scale)
   * @return List of validation issues
   */
  std::vector<ValidationIssue> checkSkewness(double maxSkewness = 0.85);

  /**
   * @brief Validate mesh boundary faces
   * @return List of validation issues
   */
  std::vector<ValidationIssue> checkBoundaryFaces();

  /**
   * @brief Validate mesh connectivity
   * @return List of validation issues
   */
  std::vector<ValidationIssue> checkMeshConnectivity();

  /**
   * @brief Validate volume ratio between adjacent cells
   * @param maxRatio Maximum acceptable volume ratio
   * @return List of validation issues
   */
  std::vector<ValidationIssue> checkVolumeRatio(double maxRatio = 2.0);

  /**
   * @brief Run all validation checks with default parameters
   * @return Map of check name to list of validation issues
   */
  std::unordered_map<std::string, std::vector<ValidationIssue>> validateAll();

  /**
   * @brief Run a specific set of validation checks
   * @param checks Names of checks to run
   * @return Map of check name to list of validation issues
   */
  std::unordered_map<std::string, std::vector<ValidationIssue>>
  validate(const std::vector<std::string> &checks);

  /**
   * @brief Get a summary of validation results
   * @param results Map of validation results from validate() or validateAll()
   * @return Summary string with error and warning counts
   */
  std::string
  getSummary(const std::unordered_map<std::string, std::vector<ValidationIssue>>
                 &results);

  /**
   * @brief Get a detailed report of validation results
   * @param results Map of validation results from validate() or validateAll()
   * @param includeInfos Whether to include INFO level issues in the report
   * @return Detailed report string
   */
  std::string getDetailedReport(
      const std::unordered_map<std::string, std::vector<ValidationIssue>>
          &results,
      bool includeInfos = false);

private:
  const Mesh &m_mesh;

  // Helper functions for calculating mesh metrics
  double calculateNonOrthogonality(const Face &face, const Cell &cell1,
                                   const Cell &cell2);
  double calculateAspectRatio(const Cell &cell);
  double calculateSkewness(const Face &face);
  double calculateVolumeRatio(const Cell &cell1, const Cell &cell2);

  // Map of available validation checks
  std::unordered_map<std::string, std::function<std::vector<ValidationIssue>()>>
      m_validationChecks;
};