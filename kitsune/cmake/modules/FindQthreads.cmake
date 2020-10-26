# This module will set the following variables in your project:
# 
#  Qthreads_FOUND        -- qthreads headers and libraries were found.
#  Qthreads_INCLUDE_DIR  -- directory with qthreads header file(s). 
#  Qthreads_LIBRARY      -- full path to qthreads library. 
#  Qthreads_LIBRARY_DIR  -- path to where the qthreads library is installed. 
#  Qthreads_LINK_LIBS    -- set of link libraries (e.g. -lqthreads) 

message(STATUS "Looking for qthreads...")

find_path(Qthreads_INCLUDE_DIR qthread.h
  PATHS /usr/local/include
        /opt/include
	/opt/local/include
)

find_library(Qthreads_LIBRARY qthread
  PATHS /usr/local/lib64
        /usr/local/lib
        /opt/lib64
        /opt/lib	
	/opt/local/lib64
	/opt/local/lib
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Qthreads 
  DEFAULT_MSG
	Qthreads_INCLUDE_DIR 
)

if (Qthreads_FOUND) 
  message(STATUS "Found Qthreads runtime.")
  get_filename_component(Qthreads_LIBRARY_DIR
                         ${Qthreads_LIBRARY}
                         DIRECTORY
                         CACHE
  )

  set(Qthreads_LINK_LIBS "-lqthread"
     CACHE
     STRING "List of libraries needed for Qthreads runtime."
  )
     
  message(STATUS "Qthreads runtime include directory : ${Qthreads_INCLUDE_DIR}")
  message(STATUS "Qthreads runtime library directory : ${Qthreads_LIBRARY_DIR}")
  message(STATUS "Qthreads runtime link libraries    : ${Qthreads_LINK_LIBS}")
else()

  message(WARNING "Qthreads runtime NOT found.")
  
endif()
