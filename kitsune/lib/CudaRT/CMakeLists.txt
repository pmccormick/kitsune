project (kitcudaart)

find_package(CUDA REQUIRED)

include_directories(
  ${KITSUNE_INCLUDE_DIR}
  ${CUDA_INCLUDE_DIRS}
)

file(GLOB kit_cudart_srcs
  *.cpp
)

add_library(kitcudart SHARED ${kit_cudart_srcs})

set_target_properties(kitcudart
  PROPERTIES
  POSITION_INDEPENDENT_CODE   1
  COMPILE_FLAGS  "-fPIC"
)

link_directories(${CUDA_LIBRARY_DIR} ${CUDA_RTLIBRARY_DIR})

install(TARGETS kitcudart
        DESTINATION lib
)


