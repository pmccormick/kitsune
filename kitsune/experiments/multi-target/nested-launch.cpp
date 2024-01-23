#include <iostream>
#include <iomanip>
#include <chrono>
#include <kitsune.h>

using namespace std;

const size_t ROW_SIZE = 1024 * 1024 * 400;
const size_t COL_SIZE = 8;
const 

// FIXME: If allowed to inline this generates bad code that
// will fail within the runtime's assertion checks... 
__attribute__ ((noinline))
void do_row_work(float *data_ptr, int col_id) {
  [[tapir::target("cuda")]]
  forall(size_t i = 0; i < ROW_SIZE; i++) {
    data_ptr[i] = col_id + float(i)/2.0f;
  }
}

int main (int argc, char* argv[]) {
  using namespace std;
  cout << "---- multi-target nested parallel launch (forall) ----\n"
       << "  Row size: " << ROW_SIZE << "\n"
       << "  Column size: " << COL_SIZE << "\n\n";

  cout << "allocating data...\n";
  float *data[COL_SIZE];
  cout << "\t column: ";
  for(unsigned i = 0; i < COL_SIZE; i++) {
    cout << i << " " << std::flush;
    data[i] = alloc<float>(ROW_SIZE);
  }
  cout << "\n" << std::flush;

  cout << "running..." << std::flush;
  auto start_time = chrono::steady_clock::now();  
  forall(unsigned ci = 0; ci < COL_SIZE; ci++) {
    // launch a kernel to do the work on each row.  
    do_row_work(data[ci], ci);
    // there is a stream but not a device synchrnoize here...
  }
  auto end_time = chrono::steady_clock::now();
  auto elapsed_time = chrono::duration<double>(end_time-start_time).count();  
      
  cout << "  done.\n" << std::flush;
  cout << "checking results.\n";
  int error_count = 0;
  for(size_t ci = 0; ci < COL_SIZE; ci++) {
    float *row = data[ci];    
    for(int i = 0; i < ROW_SIZE; i++) {
      float val = ci + float(i) / 2.0f;
      if (row[i] != val)
	error_count++;
    }
  }

  if (error_count > 0)
    cout << "  FAILED: " <<  error_count << " incorrect positions.\n\n";
  else {
    cout << "  pass (answers match).\n\n"
	 << "  Total time: " << elapsed_time << " seconds.\n"
         << "----\n\n";    
  }
    
  return error_count;
}
