# This module will set the following variables in your project:
#
#  Realm_FOUND       -- Realm was found.
#  Realm_INCLUDE_DIR -- directory with Realm header files.
#  Realm_LIBRARY     -- full path to the Realm library.
#  Realm_LIBRARY_DIR -- path to where the Realm library is installed.
#  Realm_LINK_LIBS   -- set of link libraries (e.g. -lrealm)

message(STATUS "Looking for Realm runtime...")

find_path(Realm_INCLUDE_DIR realm.h
  PATHS /usr/local/include
        /opt/include
        /opt/local/include
        $ENV{REALM_PATH}/include
)

find_library(Realm_LIBRARY realm
  PATHS /usr/local/lib64
        /usr/local/lib
	      /opt/lib64
	      /opt/lib
	      /opt/local/lib64
	      /opt/local/lib
        $ENV{REALM_PATH}/lib
        $ENV{REALM_PATH}/lib64
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Realm DEFAULT_MSG
  Realm_INCLUDE_DIR
	Realm_LIBRARY
)

if (Realm_FOUND)

  message(STATUS "Found Realm runtime.")

  get_filename_component(Realm_LIBRARY_DIR
                         ${Realm_LIBRARY}
                         DIRECTORY
                         CACHE)
  set(Realm_LINK_LIBS -lrealm CACHE STRING "List of libraries need to link with for Realm.")

  # Basic set of libraries -- will potenitally need additional
  # libraries based on Realm is built.
  set(Realm_LINK_LIBS "-lrealm -lpthread -ldl -lrt"
    CACHE
    STRING "List of libraries needed for Realm linking.")

  message(STATUS "Realm runtime include directory: ${Realm_INCLUDE_DIR}")
  message(STATUS "Realm runtime library directory: ${Realm_LIBRARY_DIR}")
  message(STATUS "Realm runtime link libraries   : ${Realm_LINK_LIBS}")

else()
  message(STATUS "kitsune: looking for realm... NOT FOUND")
  set(KITSUNE_ENABLE_REALM FALSE CACHE BOOL "Enable automatic include and library flags for Realm.")
  set(Realm_LINK_LIBS "" CACHE STRING "List of libraries need to link with for Realm.")
endif()

  message(WARNING "Could not find Realm runtime.")

endif()
