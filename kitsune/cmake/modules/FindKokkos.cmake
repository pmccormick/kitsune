# This module will set the following variables in your project:
#
# Kokkos_FOUND: Was a Kokkos installation found?
# Kokkos_INCLUDE_DIR : where to find Kokkos header files.
# Kokkos_LIBRARY_DIR : where to find the Kokkos library files.
# Kokkos_LINK_LIB    : the libraries to link against to use Kokkos.

message(STATUS "Looking for kokkos...")

find_path(Kokkos_INCLUDE_DIR Kokkos_Core.hpp
  PATHS /usr/local/include
        /opt/include
	      /opt/local/include
        $ENV{KOKKOS_PATH}/include
)

find_library(Kokkos_LIBRARY kokkoscore
  PATHS /usr/local/lib64
        /usr/local/lib
        /opt/lib64
        /opt/lib
	      /opt/local/lib64
	      /opt/local/lib
        $ENV{KOKKOS_PATH}/lib
        $ENV{KOKKOS_PATH}/lib64
)

include(FindPackageHandleStandardArgs)

find_package_handle_standard_args(Kokkos DEFAULT_MSG
				  Kokkos_INCLUDE_DIR
				  Kokkos_LIBRARY)

if (Kokkos_FOUND)

  message(STATUS "Kokkos found.")

  get_filename_component(Kokkos_LIBRARY_DIR
                         ${Kokkos_LIBRARY}
                         DIRECTORY
                         CACHE)

  # Basic set of libraries -- will potenitally need additional
  # libraries based on how Kokkos is built.
  set(Kokkos_LINK_LIBS "-lkokkoscore"
    CACHE
    STRING "List of libraries needed for Kokkos linking.")

  message(STATUS " Kokkos include directories: ${Kokkos_INCLUDE_DIR}")
  message(STATUS " Kokkos library directories: ${Kokkos_LIBRARY_DIR}")
  message(STATUS " Kokkos link libraries     : ${Kokkos_LINK_LIBS}")

else()

  message(STATUS "Could not find Kokkos.")

endif()
