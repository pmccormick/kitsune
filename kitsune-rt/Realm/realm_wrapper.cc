// Written by Alexis Perry-Holby for use with Tapir-LLVM

#include "kitsune_realm_c.h"
#include "realm.h"
#include <set>
#include <vector>
#include <atomic>

extern "C" {
  
  typedef struct context {
    Realm::Runtime rt;
    std::set<Realm::Event> events;
    std::vector<Realm::Processor> procs;
    Realm::Processor procgroup;
    size_t numprocs;
    std::atomic<unsigned> cur_task;
  } context;

  static context *_globalCTX;  //global variable
  static bool initCalled = false; //global variable

  context * getRealmCTX() {
    std::cout << "  getRealmCTX" << std::endl;
    if ( _globalCTX) 
      return _globalCTX;
    else 
      return NULL;
  }
  
  int realmInitRuntime(int argc, char** argv) {
    std::cout << "realmInitRuntime" << std::endl;
    if (initCalled)
      return 0;

    _globalCTX = new context();
    assert(_globalCTX);

    _globalCTX->rt.init(&argc, &argv); 

    //get CPU processors only, GPUs might be TOC_PROC instead
    Realm::Machine::ProcessorQuery procquery(Realm::Machine::get_machine());
    Realm::Machine::ProcessorQuery locprocquery = procquery.only_kind(Realm::Processor::LOC_PROC); 

    _globalCTX->numprocs = locprocquery.count();
    assert (_globalCTX->numprocs > 0); //assert that at least one processor exists
    //assert ( procquery.random() != Realm::Processor::NO_PROC); //another possible way to do this

    for(auto it = locprocquery.begin(); it != locprocquery.end(); it++)
      _globalCTX->procs.push_back(*it);

    Realm::Processor first_proc = _globalCTX->procs[0];
    _globalCTX->procgroup = first_proc.create_group(_globalCTX->procs);

    //_globalCTX->cur_task = Realm::Processor::TASK_ID_FIRST_AVAILABLE;
    (_globalCTX->cur_task).store(Realm::Processor::TASK_ID_FIRST_AVAILABLE);

    initCalled = true;
    return 0;
  }

  size_t realmGetNumProcs() {
    std::cout << "realmGetNumProcs" << std::endl;
    if ( _globalCTX)
      return _globalCTX->numprocs;
    else
      return 0;
  }
  
  void realmSpawn(Realm::Processor::TaskFuncPtr func, const void* args, size_t arglen, void* user_data, size_t user_data_len);

  void realmSpawn(Realm::Processor::TaskFuncPtr func, 
		  const void* args, 
		  size_t arglen, 
		  void* user_data, 
		  size_t user_data_len){ 
    /* take a function pointer to the task you want to run, 
       creates a CodeDescriptor from it directly
       needs pointer to user data and arguments (NULL for void?)
       needs size_t for len (0 for void?)
    */
    std::cout << " realmSpawn" << std::endl;

    context *ctx = getRealmCTX();
    assert(ctx);

    Realm::Processor::TaskFuncID taskID = (ctx->cur_task).load();
    std::cout << "   accessed cur_task" << std::endl;
    ctx->cur_task++;
    std::cout << "    incremented cur_task" << std::endl;

    //get a processor to run on
    Realm::Processor p = ctx->procgroup; //spawn on the group to enable Realm's magic load-balancing
    //Realm::Processor p = (ctx->procs)[i]; //do round-robin spawning on the vector of procs (needs i calculated)

    // Create a CodeDescriptor from the TaskFuncPtr   
    Realm::CodeDescriptor cd = Realm::CodeDescriptor(func);

    const Realm::ProfilingRequestSet prs;  //We don't care what it is for now, the default is fine

    //register the task with the runtime
    Realm::Event e1 = p.register_task(taskID, cd, prs, user_data, user_data_len);
    std::cout << "     registered task: " << taskID << std::endl;
    ctx->events.insert(e1); //might not actually need to keep track of this one
    std::cout << "      inserted e1" << std::endl;

    //spawn the task
    Realm::Event e2 = p.spawn(taskID, args, arglen, e1, 0); //predicated on the completion of the task's registration
    std::cout << "       spawned task: " << taskID << std::endl;
    ctx->events.insert(e2);
    std::cout << "        inserted e2" << std::endl;
    return;
  }
  
  int realmSync() {
    std::cout << " realmSync" << std::endl;
    context *ctx = getRealmCTX();
    assert(ctx);
    //create an event that does not trigger until all previous events have triggered
    Realm::Event e = Realm::Event::NO_EVENT;
    if (!(ctx->events).empty()) {
      e = e.merge_events(ctx->events); 
      //can clear the events in the list now and insert only the sync event
      ctx->events.clear();
      //ctx->events.insert(e);
    }
    else
      std::cout << "ctx->events is empty" << std::endl;

    // Do not return until sync is complete
    e.wait();
    std::cout << "e.wait() has completed" << std::endl;
    //while (!e.has_triggered()) {
    //std::cout << "not done yet" << std::endl;
    //continue;
    //}
    //std:: cout << "done waiting" << std::endl;
    
    return 0;
  }

  void realmFinalize() {
    if ( _globalCTX) {
      delete _globalCTX;
      return;
    }
    else
      return;
  }
}
