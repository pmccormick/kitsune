#include "MeshValidation.h"
#include "Cell.h"
#include "Face.h"
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <sstream>

MeshValidator::MeshValidator(const Mesh &mesh) : m_mesh(mesh) {
  // Initialize the map of validation checks
  m_validationChecks["orthogonality"] = [this]() {
    return this->checkOrthogonality();
  };
  m_validationChecks["aspect_ratio"] = [this]() {
    return this->checkAspectRatio();
  };
  m_validationChecks["skewness"] = [this]() { return this->checkSkewness(); };
  m_validationChecks["boundary_faces"] = [this]() {
    return this->checkBoundaryFaces();
  };
  m_validationChecks["connectivity"] = [this]() {
    return this->checkMeshConnectivity();
  };
  m_validationChecks["volume_ratio"] = [this]() {
    return this->checkVolumeRatio();
  };
}

std::vector<MeshValidator::ValidationIssue>
MeshValidator::checkOrthogonality(double maxNonOrthogonality) {
  std::vector<ValidationIssue> issues;

  // Convert degrees to radians for internal calculations
  double maxRadians = maxNonOrthogonality * M_PI / 180.0;

  // Get internal faces (faces between two cells)
  std::vector<Face *> internalFaces = m_mesh.getInternalFaces();

  for (const auto &face : internalFaces) {
    // Get the cells on either side of the face
    Cell *cell1 = m_mesh.getOwnerCell(face->getID());
    Cell *cell2 = m_mesh.getNeighborCell(face->getID());

    if (!cell1 || !cell2) {
      // This shouldn't happen for internal faces
      issues.push_back(
          ValidationIssue(ValidationIssue::Severity::ERROR,
                          "Internal face missing owner or neighbor cell",
                          std::to_string(face->getID())));
      continue;
    }

    // Calculate non-orthogonality
    double nonOrtho = calculateNonOrthogonality(*face, *cell1, *cell2);

    // Convert back to degrees for reporting
    double nonOrthoDegrees = nonOrtho * 180.0 / M_PI;

    if (nonOrtho > maxRadians) {
      ValidationIssue::Severity severity = ValidationIssue::Severity::WARNING;

      // If severely non-orthogonal, mark as error
      if (nonOrtho > 1.5 * maxRadians) {
        severity = ValidationIssue::Severity::ERROR;
      }

      std::string location = "Face " + std::to_string(face->getID()) +
                             " between cells " +
                             std::to_string(cell1->getID()) + " and " +
                             std::to_string(cell2->getID());

      issues.push_back(ValidationIssue(
          severity, "Non-orthogonality exceeds maximum threshold", location,
          nonOrthoDegrees, maxNonOrthogonality));
    }
  }

  return issues;
}

std::vector<MeshValidator::ValidationIssue>
MeshValidator::checkAspectRatio(double maxAspectRatio) {
  std::vector<ValidationIssue> issues;

  // Get all cells in the mesh
  std::vector<Cell *> cells = m_mesh.getCells();

  for (const auto &cell : cells) {
    double aspectRatio = calculateAspectRatio(*cell);

    if (aspectRatio > maxAspectRatio) {
      ValidationIssue::Severity severity = ValidationIssue::Severity::WARNING;

      // If severely stretched, mark as error
      if (aspectRatio > 2.0 * maxAspectRatio) {
        severity = ValidationIssue::Severity::ERROR;
      }

      std::string location = "Cell " + std::to_string(cell->getID());

      issues.push_back(ValidationIssue(
          severity, "Cell aspect ratio exceeds maximum threshold", location,
          aspectRatio, maxAspectRatio));
    }
  }

  return issues;
}

std::vector<MeshValidator::ValidationIssue>
MeshValidator::checkSkewness(double maxSkewness) {
  std::vector<ValidationIssue> issues;

  // Get all faces in the mesh
  std::vector<Face *> faces = m_mesh.getFaces();

  for (const auto &face : faces) {
    double skewness = calculateSkewness(*face);

    if (skewness > maxSkewness) {
      ValidationIssue::Severity severity = ValidationIssue::Severity::WARNING;

      // If severely skewed, mark as error
      if (skewness > 0.95) {
        severity = ValidationIssue::Severity::ERROR;
      }

      std::string location = "Face " + std::to_string(face->getID());

      issues.push_back(
          ValidationIssue(severity, "Face skewness exceeds maximum threshold",
                          location, skewness, maxSkewness));
    }
  }

  return issues;
}

std::vector<MeshValidator::ValidationIssue>
MeshValidator::checkBoundaryFaces() {
  std::vector<ValidationIssue> issues;

  // Get all boundary faces
  std::vector<MeshFaceID> boundaryFaceIDs = m_mesh.getBoundaryFaces();

  // Check if all boundary faces have an assigned boundary condition
  for (auto faceID : boundaryFaceIDs) {
    bool hasAssignedBC = false;

    // Check if this face is included in any boundary zone
    for (const auto &zone : m_mesh.getBoundaryZones()) {
      const auto &zoneFaces = zone->getBoundaryFaces();
      if (std::find(zoneFaces.begin(), zoneFaces.end(), faceID) !=
          zoneFaces.end()) {
        hasAssignedBC = true;
        break;
      }
    }

    if (!hasAssignedBC) {
      issues.push_back(
          ValidationIssue(ValidationIssue::Severity::ERROR,
                          "Boundary face has no assigned boundary condition",
                          "Face " + std::to_string(faceID)));
    }
  }

  return issues;
}

std::vector<MeshValidator::ValidationIssue>
MeshValidator::checkMeshConnectivity() {
  std::vector<ValidationIssue> issues;

  // This is a simplified version - a real implementation would perform more
  // thorough checks

  // Get all cells and faces
  std::vector<Cell *> cells = m_mesh.getCells();
  std::vector<Face *> faces = m_mesh.getFaces();

  // Check if each internal face connects exactly two cells
  for (const auto &face : faces) {
    if (!face->isBoundary()) {
      Cell *owner = m_mesh.getOwnerCell(face->getID());
      Cell *neighbor = m_mesh.getNeighborCell(face->getID());

      if (!owner || !neighbor) {
        issues.push_back(
            ValidationIssue(ValidationIssue::Severity::ERROR,
                            "Internal face does not connect two cells",
                            "Face " + std::to_string(face->getID())));
      }
    } else {
      // Boundary face should connect to exactly one cell
      Cell *owner = m_mesh.getOwnerCell(face->getID());

      if (!owner) {
        issues.push_back(
            ValidationIssue(ValidationIssue::Severity::ERROR,
                            "Boundary face does not connect to any cell",
                            "Face " + std::to_string(face->getID())));
      }
    }
  }

  return issues;
}

std::vector<MeshValidator::ValidationIssue>
MeshValidator::checkVolumeRatio(double maxRatio) {
  std::vector<ValidationIssue> issues;

  // Get internal faces (faces between two cells)
  std::vector<Face *> internalFaces = m_mesh.getInternalFaces();

  for (const auto &face : internalFaces) {
    // Get the cells on either side of the face
    Cell *cell1 = m_mesh.getOwnerCell(face->getID());
    Cell *cell2 = m_mesh.getNeighborCell(face->getID());

    if (!cell1 || !cell2) {
      // Already handled in connectivity check
      continue;
    }

    // Calculate volume ratio
    double volumeRatio = calculateVolumeRatio(*cell1, *cell2);

    if (volumeRatio > maxRatio) {
      ValidationIssue::Severity severity = ValidationIssue::Severity::WARNING;

      // If severely imbalanced, mark as error
      if (volumeRatio > 2.0 * maxRatio) {
        severity = ValidationIssue::Severity::ERROR;
      }

      std::string location = "Face " + std::to_string(face->getID()) +
                             " between cells " +
                             std::to_string(cell1->getID()) + " and " +
                             std::to_string(cell2->getID());

      issues.push_back(ValidationIssue(
          severity,
          "Volume ratio between adjacent cells exceeds maximum threshold",
          location, volumeRatio, maxRatio));
    }
  }

  return issues;
}

std::unordered_map<std::string, std::vector<MeshValidator::ValidationIssue>>
MeshValidator::validateAll() {
  std::unordered_map<std::string, std::vector<ValidationIssue>> results;

  // Run all registered validation checks
  for (const auto &[name, check] : m_validationChecks) {
    results[name] = check();
  }

  return results;
}

std::unordered_map<std::string, std::vector<MeshValidator::ValidationIssue>>
MeshValidator::validate(const std::vector<std::string> &checks) {
  std::unordered_map<std::string, std::vector<ValidationIssue>> results;

  // Run only the specified validation checks
  for (const auto &name : checks) {
    auto it = m_validationChecks.find(name);
    if (it != m_validationChecks.end()) {
      results[name] = it->second();
    }
  }

  return results;
}

std::string MeshValidator::getSummary(
    const std::unordered_map<std::string, std::vector<ValidationIssue>>
        &results) {

  int errorCount = 0;
  int warningCount = 0;
  int infoCount = 0;
  int totalIssues = 0;

  // Count different severity levels
  for (const auto &[name, issues] : results) {
    for (const auto &issue : issues) {
      totalIssues++;
      switch (issue.severity) {
      case ValidationIssue::Severity::ERROR:
        errorCount++;
        break;
      case ValidationIssue::Severity::WARNING:
        warningCount++;
        break;
      case ValidationIssue::Severity::INFO:
        infoCount++;
        break;
      }
    }
  }

  // Format the summary
  std::ostringstream oss;
  oss << "Mesh Validation Summary:\n";
  oss << "------------------------\n";
  oss << "Total Issues: " << totalIssues << "\n";
  oss << "Errors:       " << errorCount << "\n";
  oss << "Warnings:     " << warningCount << "\n";
  oss << "Info:         " << infoCount << "\n\n";

  if (errorCount > 0) {
    oss << "VALIDATION FAILED: Mesh has " << errorCount
        << " critical issues.\n";
  } else if (warningCount > 0) {
    oss << "VALIDATION WARNING: Mesh has " << warningCount
        << " potential issues.\n";
  } else {
    oss << "VALIDATION PASSED: Mesh appears to be suitable for simulation.\n";
  }

  return oss.str();
}

std::string MeshValidator::getDetailedReport(
    const std::unordered_map<std::string, std::vector<ValidationIssue>>
        &results,
    bool includeInfos) {

  std::ostringstream oss;
  oss << getSummary(results) << "\n";
  oss << "Detailed Issues Report:\n";
  oss << "-----------------------\n\n";

  // Report issues by category
  for (const auto &[name, issues] : results) {
    if (issues.empty()) {
      continue;
    }

    oss << name << " (" << issues.size() << " issues):\n";
    oss << std::string(name.length() + 12, '-') << "\n";

    // Group by severity
    for (auto severity :
         {ValidationIssue::Severity::ERROR, ValidationIssue::Severity::WARNING,
          ValidationIssue::Severity::INFO}) {

      // Skip INFO level if not requested
      if (severity == ValidationIssue::Severity::INFO && !includeInfos) {
        continue;
      }

      // Get severity name
      std::string severityName;
      switch (severity) {
      case ValidationIssue::Severity::ERROR:
        severityName = "ERROR";
        break;
      case ValidationIssue::Severity::WARNING:
        severityName = "WARNING";
        break;
      case ValidationIssue::Severity::INFO:
        severityName = "INFO";
        break;
      }

      // Filter issues by severity
      bool hasSeverity = false;
      for (const auto &issue : issues) {
        if (issue.severity == severity) {
          hasSeverity = true;
          break;
        }
      }

      if (!hasSeverity) {
        continue;
      }

      oss << severityName << ":\n";

      // Format each issue
      for (const auto &issue : issues) {
        if (issue.severity == severity) {
          oss << "  - " << issue.message << "\n";
          if (!issue.location.empty()) {
            oss << "    Location: " << issue.location << "\n";
          }
          if (issue.value != 0.0 || issue.threshold != 0.0) {
            oss << "    Value: " << std::fixed << std::setprecision(3)
                << issue.value << " (threshold: " << issue.threshold << ")\n";
          }
          oss << "\n";
        }
      }
    }

    oss << "\n";
  }

  return oss.str();
}

// Private helper methods for mesh metric calculations

double MeshValidator::calculateNonOrthogonality(const Face &face,
                                                const Cell &cell1,
                                                const Cell &cell2) {
  // In 2D, non-orthogonality is the angle between:
  // 1. The face normal vector
  // 2. The vector connecting the centers of the two cells

  // Get face normal
  Vector2D faceNormal = face.getNormal();
  faceNormal.normalize();

  // Get vector between cell centers
  Vector2D cellVector = cell2.getCenter() - cell1.getCenter();
  cellVector.normalize();

  // Calculate dot product
  double dotProduct = faceNormal.dot(cellVector);

  // Clamp to valid range due to potential floating point errors
  dotProduct = std::max(-1.0, std::min(1.0, dotProduct));

  // Calculate angle
  return std::acos(std::abs(dotProduct));
}

double MeshValidator::calculateAspectRatio(const Cell &cell) {
  // For a 2D cell, aspect ratio is the ratio of the
  // longest to shortest distance between any two vertices

  // Get cell vertices
  const auto &vertices = cell.getVertices();
  if (vertices.size() < 2) {
    return 1.0; // Default for degenerate case
  }

  double maxDist = 0.0;
  double minDist = std::numeric_limits<double>::max();

  // Calculate distances between all pairs of vertices
  for (size_t i = 0; i < vertices.size(); ++i) {
    for (size_t j = i + 1; j < vertices.size(); ++j) {
      double dist = (vertices[j] - vertices[i]).magnitude();
      maxDist = std::max(maxDist, dist);
      minDist = std::min(minDist, dist);
    }
  }

  if (minDist < 1e-10) {
    return std::numeric_limits<double>::max(); // Near-degenerate cell
  }

  return maxDist / minDist;
}

double MeshValidator::calculateSkewness(const Face &face) {
  // Skewness for a 2D face (line segment) is defined as the deviation
  // of the face center from the ideal position halfway between its vertices

  // Get face vertices
  const auto &vertices = face.getVertices();
  if (vertices.size() != 2) {
    return 0.0; // Default for non-line faces
  }

  // Calculate ideal center (midpoint of vertices)
  Vector2D idealCenter = (vertices[0] + vertices[1]) * 0.5;

  // Get actual center
  Vector2D actualCenter = face.getCentroid();

  // Calculate distance between ideal and actual centers
  double centerDeviation = (actualCenter - idealCenter).magnitude();

  // Calculate face length
  double faceLength = (vertices[1] - vertices[0]).magnitude();

  // Normalize by face length
  return centerDeviation / faceLength;
}

double MeshValidator::calculateVolumeRatio(const Cell &cell1,
                                           const Cell &cell2) {
  // Calculate volumes of both cells
  double volume1 = cell1.getVolume();
  double volume2 = cell2.getVolume();

  // Ensure positive volumes
  volume1 = std::abs(volume1);
  volume2 = std::abs(volume2);

  // Avoid division by zero
  if (volume2 < 1e-10) {
    return std::numeric_limits<double>::max();
  }

  // Calculate ratio - ensure it's always >= 1.0
  return std::max(volume1 / volume2, volume2 / volume1);
}