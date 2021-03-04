message(STATUS "kitsune-rt: configuring...")

#find_path(REALM_INCLUDE_DIR  realm.h)
#find_library(REALM_LIBRARY realm)

#include(FindPackageHandleStandardArgs)
#find_package_handle_standard_args(Realm DEFAULT_MSG
#				  REALM_INCLUDE_DIR 
#				  REALM_LIBRARY)
if (KITSUNE_ENABLE_REALM_TARGET)
  message(STATUS "kitsune-rt: looking for realm... FOUND")
  #get_filename_component(REALM_LIBRARY_DIR
#		         ${REALM_LIBRARY}
#                         DIRECTORY)

  include_directories(
    $ENV{REALM_INCLUDE}
    )
  link_directories(
    $ENV{REALM_LIBDIR}	
    )
  
  file(GLOB realm_srcs Realm/*.cc)
  file(GLOB realm_incs Realm/*.h)

  add_library(kitsunerealm STATIC ${realm_srcs} ${realm_incs})

  target_link_libraries(kitsunerealm PUBLIC realm pthread dl rt)

  # The following is intended to enable the use of find_package(kitsunerealm REQUIRED)
  # primarily in the case that clang is being built with the kitsune realm backend enabled
  target_include_directories(kitsunerealm PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/Realm>
    $<INSTALL_INTERFACE:include> )

  set_target_properties(kitsunerealm PROPERTIES
    PUBLIC_HEADER "${realm_incs}"
    )
  
  install(TARGETS kitsunerealm
    EXPORT kitsunerealm-targets
    PUBLIC_HEADER DESTINATION include
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
    )

  install(EXPORT kitsunerealm-targets
    NAMESPACE kitsunerealm::
    FILE kitsunerealm-config.cmake
    DESTINATION lib/cmake/kitsunerealm
    )

else()
  message(SEND_ERROR "kitsune-rt: looking for realm... NOT FOUND")
endif()
