# Contexto - AIs

# Deucalion Supercomputer: System Context & Usage Guidelines

## 1. Hardware Architecture & Compute Nodes

Deucalion is a heterogeneous cluster incorporating ARM and x86 microprocessors.

| Partition Type | Nodes | CPU | GPU | Memory | Storage | Network |
| --- | --- | --- | --- | --- | --- | --- |
| **arm** | 1632 | Fujitsu A64FX (48-core 2.0 GHz) | None | 32GB | 512GB NVMe PCIe | 100 Gb/s ConnectX-6 |
| **x86** | 500 | 2x AMD EPYC 7742 (64-core 2.25 GHz) | None | 256GB DDR4 | 480GB SSD | 100 Gb/s ConnectX-6 |
| **a100-40** | 17 | 2x AMD EPYC 7742 (64-core 2.25 GHz) | 4x NVIDIA A100 40GB | 512GB DDR4 | 480GB SATA | 2x 200 Gb/s ConnectX-6 |
| **a100-80** | 16 | 2x AMD EPYC 7742 (64-core 2.25 GHz) | 4x NVIDIA A100 80GB | 512GB DDR4 | 480GB SATA | 2x 200 Gb/s ConnectX-6 |

### 1.1. Slurm Partitions

- **ARM (`aarch64`):** `dev-arm` (24h max), `normal-arm` (48h), `large-arm` (72h)
- **x86 (`x86_64`):** `dev-x86` (24h), `normal-x86` (48h), `large-x86` (72h)
- **GPU 40GB:** `dev-a100-40` (4h), `normal-a100-40` (48h)
- **GPU 80GB:** `dev-a100-80` (4h), `normal-a100-80` (48h)

---

## 2. File Systems & Storage Policies

- **`/home` (NAS):** 25GB quota, max 20,000 files. Strictly for configurations and source code.
- **`/projects/$project` (Lustre Parallel File System):** Heavy I/O. **All computational jobs MUST run here.**
- **`/apps`:** System-wide software installations.

---

## 3. Software Environment & Compilation

OS: Rocky Linux 8 | Job Scheduler: Slurm 23.11.4

### 3.1. Environment Modules (Lmod)

- Search/List: `module avail`, `module spider <name>`, `module list`
- Load/Unload: `module load <name>`, `module purge`, `module unload <name>`
- **EESSI (European Environment for Scientific Software Installations):** Available for ARM and GPU partitions.
    - Initialization: `unset MODULEPATH` followed by `source /cvmfs/software.eessi.io/versions/2023.06/init/bash`

### 3.2. Compilation Strategies

- **Rule of Thumb:** Compile inside `/projects/` to avoid I/O bottlenecks.
- **ARM Compilation:** * *Cross-compilation* (Login nodes): Use `frtpx`, `fccpx`, `FCCpx`.
    - *Native* (Compute nodes via `dev-arm` allocation): Use `frt`, `fcc`, `FCC`, or GNU `gfortran`, `gcc`, `g++`.
- **x86/GPU Compilation:**
    - Use Native compilers directly: GCC, Intel oneAPI (`ifort`, `icc`), NVIDIA HPC SDK (`nvcc`, `nvc`).

---

## 4. Slurm Job Scheduling

All batch jobs are submitted via `sbatch`. Interactive sessions use `salloc` or `srun`.

### 4.1. Standard CPU Batch Script Example

```bash
#!/bin/bash
#SBATCH --job-name=exampleJob
#SBATCH --account=<slurm_account> # Check via: sacctmgr show Association
#SBATCH --partition=normal-x86
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G

module load OpenMPI/4.1.5-GCC-12.3.0
srun myapp -i input -o output
```

### 4.2. GPU Batch Script Requirements

- GPU nodes (`gnx[501-533]`) are non-exclusive.
- **Constraint:** Requesting 1 GPU strictly requires requesting 32 CPUs (`-cpus-per-task=32`).

```python
#!/bin/bash
#SBATCH --partition=normal-a100-40
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --account=<slurm_account> # Usually ends in 'g' for GPU allocations
```

## 5. Containerization & Python Workflows

Containers are highly recommended for Python (due to strict inode limits on `/home`).

- **Singularity (3.11):** Available cluster-wide.
    - Run: `singularity exec --nv container.sif python script.py`
    - Build constraint: Build images (`singularity build`) on the target architecture node (e.g., build on ARM to run on ARM).
- **Enroot & Pyxis:** Unprivileged sandboxing available **only** on GPU accelerated nodes (`gnx`).
    - Run via Pyxis: `srun --container-image=/path/to/image.sqsh ...`
- **Python/Conda:**
    - If using `venv`, initialize and install directly inside `/projects/`.
    - If using `conda`, explicitly modify `~/.condarc` to point `envs_dirs` and `pkgs_dirs` to `/projects/<your_project>/.conda/`.

## 6. Energy Profiling (`get_energy`)

A CLI tool to measure energy consumption and CO2 emissions of jobs. Output is in JSON format.

- Total user energy: `get_energy -u $USER --all`
- Project energy: `get_energy -a <account_name>`
- Specific job: `get_energy -j <job_id>`
- Detailed breakdown (per node/component): Add `-format list`