## 05.01

Par. Prog Paradigms:
- dist memory
- shared memory; all devices see same chunk of memory
- heterogeneous; diff devices with their own memory


- not only express paralellism & correctness, but also evaluating performance

Domains:
- numerical sim: car industry, chemistry
- data manip: bio informatics

### State of the Art
 - benchmark to get metrics to compare machines
 - most well known: top500.org
 - Linpack: solves dense linear systems, lots of matrix mult w double precision
   - speed in # Tflops per second
 - top 500 nov 2021: Japan/Fugaku. most powerful has 7M cores...
   - to get the most performance out of it, need to give 7M things to do in parallel, and do it millions of times per second

- Rpeak: upper bound, theoretical, just calculataed from vendor specs. 
- Rmax: actual performance achieved solving linpack
- ratio of rmax and rpeak; give some notion of efficiency

Homogenous machines: all cores are the same
- when programming and expressing parallelism, you know all run at almost same speed

Heterogeneus: CPUS and GPUs; GPUS cant work alone, no OS. More complex, maybe better performance with lower consumption.

We will program for homogeneus and heterogeneous clusters.

- supercomputers change Every 3-6 years
- parallel applications; complex to program and to validate; 10-30 years; 4-5-6 generations of supercomputers. During its lifetime it will have to work on different hardware
  - challenge is to be able to express parallelism right for the architecture.
  - Do it in the least intrusive way possible.

- Green 500
- Graph 500

Q:

Can you please talk a bit more about that "parallism layer" that is between the actual application (e.g. with the physics) and the hardware?

how are things from a software architecture perspective? do we have to create adapters, APIs, are there some common design patterns?  

A:


### HIstory
- Cray 1: 1 cpu
- Cray 2: 4 cpus
- ASCI Red: already Tflops, built 1997
- IBM roadrunner: hybrid (processor for system I/O, another for numerical computations). Achieved Pflops

Roadmap: exascale in 2022+
- EuroHPC / EPI
- US: ECP
- designing exascale machine, not only 1 company can do it. 

- Co-design: create product specific for a customer? 
  - many years of R&D together with customer.

FUGAKU supercomputer:
- each core is simple, if code generated or source code is not optimal enough to use all cores; performance is really bad

EPI: european processor initiative
- EU exascale machine based on EU processor by 2023
- trend to regain european independence in designing and building processors...

- French Ecosystem: Teratec


### Architecture Levels

- architecture: assemlby instruction

- micro architecture; mechanism to help you but you have to drive it yourself?
  - example: vector units; SIMD instruction; to do same op on mulitple data at the same time
- the more arch. features you have, the more stress you put on the software to make optimal use

Course Outline

- MPI: dist memory 
- OpenMP: shared memory
- Heterogeneous: CUDA model on NVIDIA GPGPU
- project: paralelize projects mixing all these together

## 12.

Task: work to do

Thread: sequential!   total order in the instructions

Process: common address space, one or more threads

Parallel computing: multiple processes

Dependencies: For testing correctness; run and get same result several times (scheduling may be different in every run...)

### Parallelism Type
- Control (task based)
- Flow (pipeline): when you need diff operations on the same data (steps in a row...); example when you are done with 1st step of data, start 2nd step of 2st data but also 2st step of 2nd data
- Data parallelism: simple example; vector addition, take 1st cell and add 2nd cell of the other. then same with 2nd cells, etc. Same op on multiple data. You can exploit data parallelism

- mixing them: example in img processing; apply filter to 2 pixels at the same time...

### Shared Memory Systems
- everybody access the same part of memory
  - advantage, easily share things
  - disadvantage: deal with synchronization

### Dist. Memory System
- everybody access their own private view of the memory
  - a way to communicate (e.g. network, inter socket, inter-core, inter-node) but not accessing each other's memory
  
### Mixed Systems
- Cluster: one big dist. memory system, with multiple share mem. systems within


### Parallel Progra. Paradigm
- dist. memory model, shared mem. model
- acctually independent from system & hardware!

### SHared Mem. Model
- parallel tasks: same view of memory
- deal concurrent memory accesses

- difficult on dist. memory systems
- easy on shared memory systems
- Examples: Java Threads, pthread (posix API), openMP (C, C++, FORTRAN) - hides some complexity; compiler directives

### Dist. Memory model
- parallel tasks , own memory space
- data split among parallel tasks
- communication: message passing
- good instruction & memory locality
- easy on dist. memory system
- easy on shared-memory system (use different memory parts). Implemented with processes

- Examples PVM (old), MPI - standard (no ISE or IEEE)  

### Message Exchange
- high level protocol: sender and receiver agree, receiver must be there to receive

- data sent: piece of memroy & length. receiver copies data from sender to a block of its own. needs to be pure data, if there were pointers, they will not point to valid memory after being copied.  To copy more complex data structures, need to serialize
(sender can then modify the passed block without affecting data at the receiver side)

- every send needs a receive (otherwise program blocks forever)

### Intro to MPI

- High level API (abstracts what we would have to do with sockets, etc...)

- function calls; it's a library

- include mpi.h (signatures)
- when linking: indicate libraries of MPI 
`gcc -I<path to header> -o hello hello.c -L<path to lib> -lmpi`

- all functions begin with MPI_
- no MPI calls before MPI_Init(&argc, &argv), before dealing with argc, argv yourself! (sometimes MPI adds own stuff to that, after the call you get the clean versions)
- no MPI calls after MPI_Finalize

#### Compilation

- like with any other library
  - simple way: mpicc script
  - complex way: regular compiler with options specifying paths to the library

- Simple way: 
  `mpicc -o hello hello.c`

- in MPI_Init it creates environment of MPI, creates routes for communication, code will already be in parallel, 

#### Execution
- w/ Job Manager
  - Slurm can help you with MPI
  `salloc -n 4 mpirun ./hello`

  - each process will run the command we tell it it, everyone will run MPI_Init, everyone must call MPI_Finalize

  - MPI is in charge of setting up communication between the 4 processes

- W. script
  - .batch (check slide 43)

#### Communicator
- Group of processes from a communicator MPI_COMM_WIRLD
- to send or receive a message, every process has a "rank" (simple integer), within communicator, identifies each uniquely
- MPI_Comm_size writes result to a provided int address, does not return it directly. Every MPI function uses the return value for communicating error or success

#### Process Rank
MPI_Comm_rank(MPI_Comm comm, int * rank)

- MPI assigns to all n processes ranks 0 to n-1

- by default slurm assigns 1 core per MPI rank

- Multiple cores per rank?  you  need only 1 for regular processes that are sequential. If inside each MPI process we use some other threading model, we need to use `-c` to explicitly assign more cores for that

### Sending Messages

- MPI_Datatype: to ensure same size across diff architectures, etc. (MPI_CHAR, etc)

- tag to identify each message

- when MPI_Send finishes, there is no guarantee that the receiver has the data.
- it only means the part of the memory you wanted to send, you can safely modify it.
- basically blocks until it is safe to use the buffer!

- MPI_Recv, blocks until it is safe to use the buffer!

### Message Protocols
- msg: envelope & data
- eager: faster from sender point of view (requires more memory, internal copy?), sends assuming destination can store
- if you want to send something too large: Rendezvous: waits for receiver to be ready

- Blocking vs Non BLocking
  - MPI_Isend, MPI_Irecv (inmediately returns, no guarantee)
  - block with MPI_Wait (wait on a "request" that is associated with a Isend, or Irecv call)
  
- MPI_Ssend (synchronous send)
  - you are guaranteed that it returns when the message is received (is like forcing Rendezvous); for debugging it helps, to assure worst case scenario. If it works, you can replace with normal send that can sometimes use eager protocol


## 19.01

### Collective communication

- MPI Barrier: collective sync (applies to everything in the COMM)

- Broadcast: one to all collective comm

- Scatter: send diff data to other processses
  - sent data: same size, same type
  - useful for projects: to assign each process a sub-part of the available data
  - useful for img filt, send 1 block to each MPI process
  - for APM: split the text in multiple pieces

- Gather: reverse of scatter.
  - all-to-one

- Allgather: almost like gather + broadcast (everyone receives all the results)

- Reduction (mpi allows multiple reductions too?)
- Reduce to all: reduction but every process has the final result
  - example: propagate the timestep from one iteration to the others?

- Per Rank data size: MPI_Scatterv

### Data Parallelism
- ghost cells: values that a rank needs to do its calc. but it's another rank's responsibility to update,
- some point to point comms are needed once in a while

- MPICH, Open MPI : official open source MPI implementations

## 26.01
Open MP: open multi-processing

- indepentent from underlying architecture; as long as system abstracts it to you with a shared memory view
- easier to share data
- data races possible (segfaults, nondeterminism) 

Threads vs Processes:
- launching a binary: OS creates a process (stack, heap, globals, code, virtual memory view - OS then maps it to physical memory view)
- within 1 process you can create additional threads (each own stack, heap can be accessed by all)

- malloc is thread-safe

- openMP worksharing directives; distribute loop iterations

- parallel loop: we ned to know how many iterations we will do
- scheduling policy: who will do what
- by default: barrier at the end of loop
- pragma omp for   - does not create threads, just distributes for across available threads

- large chunks: less sch. overhad but less load balancing
- small chunks: good LB byt large scheduling overhead



