# Runtime policy

LAMMPS is not built inside the Docker image.

The Docker image provides:

- CUDA
- Python
- PyTorch
- MACE
- cuEquivariance-related packages
- scientific Python tools
- build tools for later LAMMPS compilation

LAMMPS will be built directly on the cloud GPU machine.
