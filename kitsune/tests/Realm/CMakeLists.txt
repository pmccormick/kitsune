#
#

message(STATUS "Including realm tests...")

include_directories(${KITSUNE_SOURCE_DIR}/lib/Realm)
include_directories(${Realm_INCLUDE_DIR})

set(LLVM_OPTIONAL_SOURCES
  realm_init_test.cpp
  )

foreach(src_file IN ITEMS ${LLVM_OPTIONAL_SOURCES})
  message(STATUS "  realm test: ${src_file}")
  get_filename_component(base ${src_file} NAME_WLE)
  set(target_name ${base})
  add_executable(${target_name} ${src_file})
  target_link_directories(${target_name}
    PUBLIC ${Realm_LIBRARY_DIR} ${LLVM_BINARY_DIR}/lib)
  target_link_libraries(${target_name} kitrealm realm pthread dl rt)

endforeach()
