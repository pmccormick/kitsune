// Written by Alexis Perry for use with Tapir-LLVM

#include "kitsune_realm_c.h"
#include "realm.h"
#include <set>
#include <vector>

extern "C" {
  
  typedef struct context {
    Realm::Runtime rt;
    std::set<Realm::Event> events;
    std::vector<Realm::Processor> procs;
    Realm::Processor procgroup;
    size_t numprocs;
    unsigned cur_task;
  } context;

  static context *_globalCTX;  //global variable

  context * getRealmCTX() {
    if ( _globalCTX) 
      return _globalCTX;
    else 
      return NULL;
  }
  
  int realmInitRuntime(int argc, char** argv) {
    _globalCTX = new context();

    _globalCTX->rt.init(&argc, &argv); 

    //get CPU processors only, GPUs might be TOC_PROC instead
    Realm::Machine::ProcessorQuery procquery(Realm::Machine::get_machine());
    Realm::Machine::ProcessorQuery locprocquery = procquery.only_kind(Realm::Processor::LOC_PROC); 

    _globalCTX->numprocs = locprocquery.count();
    assert (_globalCTX->numprocs > 0); //assert that at least one processor exists
    //assert ( procquery.random() != Realm::Processor::NO_PROC); //another possible way to do this

    for(auto it = locprocquery.begin(); it != locprocquery.end(); it++)
      _globalCTX->procs.push_back(*it);

    (_globalCTX->procgroup).create_group(_globalCTX->procs);

    _globalCTX->cur_task = Realm::Processor::TASK_ID_FIRST_AVAILABLE;

    return 0;
  }

  size_t realmGetNumProcs() {
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
    
    context *ctx = getRealmCTX();

    Realm::Processor::TaskFuncID taskID = ctx->cur_task;

    //get a processor to run on
    Realm::Processor p = ctx->procgroup; //spawn on the group to enable Realm's magic load-balancing
    //Realm::Processor p = (ctx->procs)[i]; //do round-robin spawning on the vector of procs (needs i calculated)

    //get a memory associated with that processor to copy to
    //Realm::Machine::MemoryQuery memquery(Realm::Machine::get_machine());
    //Realm::Memory m = memquery.local_address_space().best_affinity_to(p).random();
    //assert ( m != Realm::Memory::NO_MEMORY); //assert that the memory exists

    // Create a CodeDescriptor from the TaskFuncPtr   
    Realm::CodeDescriptor cd = Realm::CodeDescriptor(func);

    const Realm::ProfilingRequestSet prs;  //We don't care what it is for now, the default is fine

    //register the task with the runtime
    Realm::Event e1 = p.register_task(taskID, cd, prs, user_data, user_data_len);
    ctx->events.insert(e1); //might not actually need to keep track of this one

    //spawn the task
    Realm::Event e2 = p.spawn(taskID, args, arglen, e1, 0); //predicated on the completion of the task's registration
    ctx->events.insert(e2);

    return;
  }
  
  int realm_sync() {
    context *ctx = getRealmCTX();

    //create an event that does not trigger until all previous events have triggered
    Realm::Event e;
    e = e.merge_events(ctx->events); 

    //can clear the events in the list now and insert only the sync event
    ctx->events.clear();

    // Do not return until sync is complete
    e.wait();

    return 0;
  }
}

/// BEGIN EXTRANEOUS STUFF THAT I'M NOT READY TO THROW AWAY YET
#if 0
  //realmCreateRegion
  void* realmCreateRegion_int(int* data) {

    context * ctx = getRealmCTX();

    const Realm::ProfilingRequestSet prs;  //We don't care what it is for now, the default is fine
    std::cout << "Created default prs" << std::endl;

    //get a processor to run on
    Realm::Machine::ProcessorQuery procquery(Realm::Machine::get_machine());
    Realm::Processor p = procquery.local_address_space().random();
    assert ( p != Realm::Processor::NO_PROC); //assert that the processor exists
    std::cout << "Got a processor" << std::endl;

    //get a memory associated with that processor to copy to
    Realm::Machine::MemoryQuery memquery(Realm::Machine::get_machine());
    Realm::Memory m = memquery.local_address_space().best_affinity_to(p).random();
    assert ( m != Realm::Memory::NO_MEMORY); //assert that the memory exists
    std::cout << "Got a memory" << std::endl;

    //create a physical region for the copy
    //Realm::RegionInstance * R = new Realm::RegionInstance();
    Realm::RegionInstance * R = m.get_instance(id(Realm::Event::NO_EVENT))
    std::cout << "Created default RegionInstance" << std::endl;

    //create a point object out of the data being passed
    const size_t length = sizeof(data)/sizeof(data[0]);
    Realm::Point<length,int> pt = Realm::Point<length,int>(data);
    std::cout << "Created Point" << std::endl;

    //create an indexspace out of the point
    std::vector<Realm::Point<length, int> > myPointVec;
    myPointVec.push_back(pt);
    const Realm::IndexSpace<length,int> is = Realm::IndexSpace<length,int>(myPointVec);
    std::cout << "Created IndexSpace" << std::endl;

    //create a vector of field sizes
    std::vector<size_t> field_sizes = {sizeof(data[0])}; //data is an array of ints, so there is only one field
    std::cout << "Created field_sizes" << std::endl;
    //constexpr auto user_data_type = std::type_index(DTYPE);
    //constexpr auto user_data_type = (constexpr)DTYPE.name();
    //Realm::InstanceLayout<user_data_len,typeid(user_data[0]).name()> il;
    //Realm::InstanceLayout<1,user_data_type> il = Realm::InstanceLayoutOpaque(user_data_len,alignof(user_data)); //alignment is what?
    //Realm::InstanceLayout<1,typeid(user_element).name()> il = Realm::InstanceLayoutOpaque(user_data_len,alignof(user_data)); //alignment is what?
    //const Realm::InstanceLayoutGeneric * il = ((Realm::RegionInstance *)data_region)->get_layout(); //copy the layout of the source region

    //Realm::Event regEvt = Realm::RegionInstance::create_instance(R,m,(Realm::InstanceLayoutGeneric *)il,prs, Realm::Event::NO_EVENT);
    Realm::Event regEvt = Realm::RegionInstance::create_instance(*R, m, is, field_sizes, 0, prs, Realm::Event::NO_EVENT); //the 0 denotes use SOA layout
    std::cout << "Executed create_instance" << std::endl;
    ctx->mem_events.insert(regEvt);
    std::cout << "Inserted creation instance into ctx->mem_events" << std::endl;

    return (void*) R;
  }

  void realmDestroyRegion(void *region) {
    //region->destroy(*event);
    ((Realm::RegionInstance *)region)->destroy(Realm::Event::NO_EVENT); //destroys immediately
    return;
  }

  //only use this internally
  Realm::Event mem_sync();
  Realm::Event mem_sync() {
    context * ctx = getRealmCTX();
    Realm::Event e;
    e = e.merge_events(ctx->mem_events);
    std::cout << "merged memory events" << std::endl;

    ctx->mem_events.clear();
    std::cout << "Cleared the context's set of memory events" << std::endl;
    ctx->mem_events.insert(e);
    std::cout << "Added mem_sync event to context's set of memory events" << std::endl;

    return e;
  }

  //only use internally
  //Note: borrowed this routine from https://github.com/StanfordLegion/legion/blob/stable/examples/realm_stencil/realm_stencil.cc
  Realm::Event realmCopy_int(Realm::RegionInstance src_inst, 
			 Realm::RegionInstance dst_inst, 
			 Realm::FieldID fid, //int
			 Realm::Event wait_for) {
    Realm::CopySrcDstField src_field;
    src_field.inst = src_inst;
    src_field.field_id = fid;
    src_field.size = sizeof(int);

    std::vector<Realm::CopySrcDstField> src_fields;
    src_fields.push_back(src_field);

    Realm::CopySrcDstField dst_field;
    dst_field.inst = dst_inst;
    dst_field.field_id = fid;
    dst_field.size = sizeof(int);

    std::vector<Realm::CopySrcDstField> dst_fields;
    dst_fields.push_back(dst_field);

    return dst_inst.get_indexspace<2, long long int>().copy(src_fields, dst_fields,
						      Realm::ProfilingRequestSet(),
						      wait_for);
  }

  //only use internally
  //Note: borrowed this routine from https://github.com/StanfordLegion/legion/blob/stable/examples/realm_stencil/realm_stencil.cc
  Realm::Event realmCopy_double(Realm::RegionInstance src_inst, 
			 Realm::RegionInstance dst_inst, 
			 Realm::FieldID fid, //int
			 Realm::Event wait_for) {
    Realm::CopySrcDstField src_field;
    src_field.inst = src_inst;
    src_field.field_id = fid;
    src_field.size = sizeof(double);

    std::vector<Realm::CopySrcDstField> src_fields;
    src_fields.push_back(src_field);

    Realm::CopySrcDstField dst_field;
    dst_field.inst = dst_inst;
    dst_field.field_id = fid;
    dst_field.size = sizeof(double);

    std::vector<Realm::CopySrcDstField> dst_fields;
    dst_fields.push_back(dst_field);

    return dst_inst.get_indexspace<2, long long int>().copy(src_fields, dst_fields,
						      Realm::ProfilingRequestSet(),
						      wait_for);
  }

  //helper - makes a Realm::Processor::TaskFuncPtr out of available things
  void canonicalTask(const void *args, size_t arglen, void *user_data, size_t user_data_len, Realm::Processor p);
  void canonicalTask(const void *args, size_t arglen, void *user_data, size_t user_data_len, Realm::Processor p){}
  Realm::Processor::TaskFuncPtr makeCanonicalTask(const void *args, size_t arglen, void *user_data, size_t user_data_len, Realm::Processor p);
  makeCanonicalTask(const void *args, size_t arglen, void *user_data, size_t user_data_len, Realm::Processor p){
    Realm::Processor::TaskFuncPtr task = 
      return task;
  }
#endif
