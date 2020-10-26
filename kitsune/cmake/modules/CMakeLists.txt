# Generate a list of CMake library targets so that other CMake projects can
# link against them. LLVM calls its version of this file LLVMExports.cmake, but
# the usual CMake convention seems to be ${Project}Targets.cmake.
set(KITSUNE_INSTALL_PACKAGE_DIR lib${LLVM_LIBDIR_SUFFIX}/cmake/kitsune)
set(kitsune_cmake_builddir "${CMAKE_BINARY_DIR}/${KITSUNE_INSTALL_PACKAGE_DIR}")

# Keep this in sync with llvm/cmake/CMakeLists.txt!
set(LLVM_INSTALL_PACKAGE_DIR lib${LLVM_LIBDIR_SUFFIX}/cmake/llvm)
set(llvm_cmake_builddir "${LLVM_BINARY_DIR}/${LLVM_INSTALL_PACKAGE_DIR}")

get_property(KITSUNE_EXPORTS GLOBAL PROPERTY KITSUNE_EXPORTS)
export(TARGETS ${KITSUNE_EXPORTS} FILE ${kitsune_cmake_builddir}/KitsuneTargets.cmake)

# Generate KitsuneConfig.cmake for the build tree.
set(KITSUNE_CONFIG_CMAKE_DIR "${kitsune_cmake_builddir}")
set(KITSUNE_CONFIG_LLVM_CMAKE_DIR "${llvm_cmake_builddir}")
set(KITSUNE_CONFIG_EXPORTS_FILE "${kitsune_cmake_builddir}/KitsuneTargets.cmake")
set(KITSUNE_CONFIG_INCLUDE_DIRS
  "${KITSUNE_SOURCE_DIR}/include"
  "${KITSUNE_BINARY_DIR}/include"
  )
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/KitsuneConfig.cmake.in
  ${kitsune_cmake_builddir}/KitsuneConfig.cmake
  @ONLY)
set(KITSUNE_CONFIG_CMAKE_DIR)
set(KITSUNE_CONFIG_LLVM_CMAKE_DIR)
set(KITSUNE_CONFIG_EXPORTS_FILE)

# Generate KitsuneConfig.cmake for the install tree.
set(KITSUNE_CONFIG_CODE "
  # Compute the installation prefix from this LLVMConfig.cmake file location.
  get_filename_component(KITSUNE_INSTALL_PREFIX \"\${CMAKE_CURRENT_LIST_FILE}\" PATH)")
# Construct the proper number of get_filename_component(... PATH)
# calls to compute the installation prefix.
string(REGEX REPLACE "/" ";" _count "${KITSUNE_INSTALL_PACKAGE_DIR}")
foreach(p ${_count})
  set(KITSUNE_CONFIG_CODE "${KITSUNE_CONFIG_CODE}
      get_filename_component(KITSUNE_INSTALL_PREFIX \"\${KITSUNE_INSTALL_PREFIX}\" PATH)")
endforeach(p)

set(KITSUNE_CONFIG_CMAKE_DIR "\${KITSUNE_INSTALL_PREFIX}/${KITSUNE_INSTALL_PACKAGE_DIR}")
set(KITSUNE_CONFIG_LLVM_CMAKE_DIR "\${KITSUNE_INSTALL_PREFIX}/${LLVM_INSTALL_PACKAGE_DIR}")
set(KITSUNE_CONFIG_EXPORTS_FILE "\${KITSUNE_CMAKE_DIR}/KitsuneTargets.cmake")
set(KITSUNE_CONFIG_INCLUDE_DIRS "\${KITSUNE_INSTALL_PREFIX}/include")

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/KitsuneConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/KitsuneConfig.cmake
  @ONLY)

set(KITSUNE_CONFIG_CODE)
set(KITSUNE_CONFIG_CMAKE_DIR)
set(KITSUNE_CONFIG_EXPORTS_FILE)

if (NOT LLVM_INSTALL_TOOLCHAIN_ONLY)
  get_property(kitsune_has_exports GLOBAL PROPERTY KITSUNE_HAS_EXPORTS)
  if(kitsune_has_exports)
    install(EXPORT KitsuneTargets DESTINATION ${KITSUNE_INSTALL_PACKAGE_DIR}
            COMPONENT kitsune-cmake-exports)
  endif()

  install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/KitsuneConfig.cmake
    DESTINATION ${KITSUNE_INSTALL_PACKAGE_DIR}
    COMPONENT kitsune-cmake-exports)

  if(NOT LLVM_ENABLE_IDE)
    # Add a dummy target so this can be used with LLVM_DISTRIBUTION_COMPONENTS
    add_custom_target(kitsune-cmake-exports)
    add_llvm_install_targets(install-kitsune-cmake-exports
                             COMPONENT kitsune-cmake-exports)
  endif()
endif()

