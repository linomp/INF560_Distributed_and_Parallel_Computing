# 01.03
Tip: "Whatever the document you are referring to, you should focus on the reference manual for CUDA Runtime API"

## GPGPU Device Properties
- **Machine**: allemagne.polytechnique.fr
- **Device detected**: Quadro P2000

[full Device Query dump](./device-query-dump.md) | [full Bandwidth dump](./bandwidth-dump.md)

### Device Query summary

```
- Architecture: Pascal
- Nearest GeForce: GeForce GTX 1060
- Core: GP106-875-K1
- CUDA Driver Version / Runtime Version          11.6 / 11.4
- CUDA Capability Major/Minor version number:    6.1
- Total amount of global memory:                 5037 MBytes (5281218560 bytes)
- ( 8) Multiprocessors, (128) CUDA Cores/MP:     1024 CUDA Cores
- Total amount of shared memory per block:       49152 bytes
- Total number of registers available per block: 65536
- Warp size:                                     32
- Maximum number of threads per multiprocessor:  2048
- Maximum number of threads per block:           1024
- Max dimension size of a thread block (x,y,z): (1024, 1024, 64)
- Max dimension size of a grid size    (x,y,z): (2147483647, 65535, 65535)
```

### Bandwidth Test summary

```
- Host to Device:   13.1 (GB/s)
- Device to Host:   12.7 (GB/s)
- Device to Device: 121.5 (GB/s)
```

_Which result is the best?_ 
  
Device to Device

_What would be the impact on programming a GPU?_ 
  
We must program in a way that Host to Device / Device to Host communications are minimized; they are relatively very slow.


