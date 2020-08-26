message(STATUS "kitsune opencl runtime: configuring...")

add_library(kitsuneOpenCL SHARED cl.cc)

target_link_libraries(kitsuneOpenCL PUBLIC OpenCL)

install(TARGETS kitsuneOpenCL
  LIBRARY DESTINATION lib
)
