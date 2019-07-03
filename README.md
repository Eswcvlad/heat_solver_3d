# heat_solver_3d
MPI library for solving 3D heat/diffusion equations.

Check the `examples` folder on how the library can be used.

# Build
Requires CMake and a MPI library (for example OpenMPI).
```
git clone https://github.com/Eswcvlad/heat_solver_3d.git
cd heat_solver_3d
mkdir build
cd build
cmake ..
make
```

To build examples add `-DBUILD_EXAMPLES=1` to the `cmake` command.
