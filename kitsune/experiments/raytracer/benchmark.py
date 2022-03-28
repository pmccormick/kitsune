#
# NOTE: You will need at least Python 3.7 for this script.
#
from audioop import add
import sys 
import subprocess
import platform
import csv
from datetime import datetime
import matplotlib.pyplot as plt
import numpy  as np


# Grab the local machine's architecture details and the date+time to
# create a unique filename for this benchmark run.
march = platform.machine()
date_str = str(datetime.now())
date_str = date_str.replace(' ', '_')
date_str = date_str.replace(':', '.')


# NOTE: For now it is best not to tweak the coarsening factor -- it makes
# controlling the benchmark characteristics a bit too difficult...
coarsen_factor = 1


# Set up the benchmark parameters.  These currently include:
#
#    1. A set of fixed grainsize parameters for the strip mining
#       transformation.  This is the number of items each thread
#       will work on (think of it as an inner loop within the
#       thread's execution).  For GPU benchmarking we need to
#       limit this in comparison to what we would do on  CPU to
#       avoid thrashing the cache...
#
#    2. Problem size (the loop trip count).
#
#    3. Run the kernel in a mode where data is prefetched or "prime"
#       the memory by running the kernel twice, timing only the
#       second executable.

# Set up some values to control the code generation grainsize.
grainsizes = []
for cf in (2**p for p in range(0, 3)):
  grainsizes.append(cf)

sample_counts = []
for sc in (2**p for p in range(2, 8)):
  sample_counts.append(sc)

image_dims = [ [640, 480], [1280, 720], [1920, 1080], [2560, 1080] ]

# The various executables we will generate for the experiment identify
# the grainsize used.  The executable name is controlled by the makefile 
# in this directory and follows the following convention:
#
#     executable.ABI.[GRAINSIZE].MARCH
#
# we build a set of executables (via make) for each grainsize parameter 
# created above. 
additional_executables = []
for gs in grainsizes:
  make_arg0 = "make"
  make_arg1 = "GRAINSIZE=" + str(gs)
  # make_arg2 = "BLOCKS_PER_GRID=" + str(blocks_per_grid)
  # make_arg3 = "THREADS_PER_BLOCK=" + str(tbp)
  print("running: ", make_arg0, make_arg1)
  result = subprocess.run([make_arg0, make_arg1], capture_output=True, text=True)
  exe_name = "raytrace-forall.cuda."+ str(gs) + "." + march
  additional_executables.append(exe_name)

# The benchmarking expects two other executables to have been created.  Both
# are CUDA versions of the benchmark and one is compiled by Clang and the other
# by nvcc.
executables = ["raytrace.clang."+march,
               "raytrace.nvcc."+march, 
               "raytrace.kokkos."+march]
executables = executables + additional_executables

# The benchmark run will have a CSV file that is automatically created with the
# captured performance of the various parameters set by either the compilation
# of the executables above or by command line arguments to each executable.
csv_filename = str("raytrace-benchmark-") + march + "-" + date_str + ".csv";
header = ["Dimensions"]
benchmark_data = {}

with open(csv_filename, 'w', newline='') as csvfile:

  writer = csv.writer(csvfile)
  writer.writerow([header])
  row=[]
  for dims in image_dims: 
    for s in sample_counts:
      row.append(dims)
      for exe in executables:
        print(exe)
        arg0 = "./" + exe
        print("-----------")
        print(arg0, str(s), str(dims[0]), str(dims[1]))
        kernel_runtime = 0.0
        for rc in range(5):
          result = subprocess.run([arg0, str(s), str(dims[0]), str(dims[1])], capture_output=True, text=True)
          kernel_runtime = kernel_runtime + float(result.stdout)
          print('.', end='', flush=True)
        kernel_runtime = kernel_runtime / 5.0
        print (" ")
        print("  kernel runtime:", kernel_runtime)
        print("-----------");
        row.append(kernel_runtime)
      print("***")
      writer.writerow(row)
      row.clear()

print("benchmark results saved to: ", csv_filename)
