## Using Kitsune+Tapir 

There are two primary controls in levearing Kitsune+Tapir.   The first set of 
details are controlled by the options provided during compilation.  Specifically, a 
target for the runtime system and architecture should be specified via the 
`-ftapir=rt-target`.   Without this flag, the toolchain will default ot the standard 
behaviors of the LLVM/Clang release that Kitsune+Tapir is built upon. 


Presently, the following runtime targets (`rt-target`) are supported for CPUs: 

* `opencilk`: The OpenCilk RTS. 
* `realm`: The Realm runtime system. 
* `qthreads`: The Qthreads runtime system. 

While other runtimes are listed, they should be considered as being unstable or soon 
to be deprecated.  See below for a quick summary of controlling the behavior of 
these targets via the use of enviornment variables.


### OpenCilk Runtime Target 

The OpenCilk runtime target supports one primary enviornment variable that controls the number of worker threads that will be used to execute supported language constructs (e.g., `forall`).  This enviornment variable is `CILK_NWORKERS`:

```bash
$ clang++ -ftapir=opencilk ... file.cpp 
$ export CILK_NWORKERS=16   # use 16 worker threads during execution. 
$ a.out 
``` 

 ### QThreads Runtime Target 

 The Qthreads runtime has [several settings](https://cs.sandia.gov/qthreads/man/
 qthread_init.html#toc3) via the environment that can impact behavior and 
 performance.  At a minimum setting `QTHREAD_NUM_SHEPHERDS` will allow 
 you to control the number of threads assigned to the execution of an executable. 

 ```bash 
 $ clang++ -ftapir=qthreads ... file.cpp 
 $ export QTHREAD_NUMBER_SHEPHERDS=16 # use 16 threads during execution. 
 $ a.out
 ``` 

### Realm Runtime Target 

Realm is the underlying (low-level) runtime system that is part of the Legion Programming System.  When running a Realm target you can provide a full command 
line via the `REALM_DEFAULT_ARGS` enviornment variable.  At present you must 
use a full command line style value (i.e., include `argv[0]`) as shown below.  The 
value of this portion of the command line is ignored.   More details on the various 
command line arguments can be found [here](https://legion.stanford.edu/starting/); look for the Command-Line Flags** section. 

```bash 
$ clang++ -ftapir=realm ... file.cpp 
$ export REALM_DEFAULT_ARGS="dummy -ll:cpu 1 -ll:force_kthreads \
     -level task=2,taskreg=2"
$ a.out 
``` 

