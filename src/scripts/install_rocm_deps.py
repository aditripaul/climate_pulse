import glob
import os
import shutil
import subprocess
import sys
import tarfile


def patch_file(filepath, replacements):
    if not os.path.exists(filepath):
        print(f"Warning: {filepath} not found.", flush=True)
        return False
    with open(filepath) as f:
        content = f.read()

    modified = content
    for old, new in replacements.items():
        modified = modified.replace(old, new)

    if modified != content:
        with open(filepath, "w") as f:
            f.write(modified)
        print(f"Patched {filepath}", flush=True)
        return True
    else:
        print(f"No changes made to {filepath}", flush=True)
        return False


def main():
    build_dir = ".venv/tmp_build"
    # Ensure a fresh directory
    if os.path.exists(build_dir):
        shutil.rmtree(build_dir)
    os.makedirs(build_dir, exist_ok=True)

    # Download source distributions
    print("Downloading torch-scatter and torch-sparse sources...", flush=True)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "download",
            "--no-deps",
            "--no-binary",
            ":all:",
            "--no-build-isolation",
            "torch-scatter",
            "torch-sparse",
        ],
        cwd=build_dir,
        check=True,
    )

    # Extract packages
    archives = glob.glob(os.path.join(build_dir, "*.tar.gz"))
    for archive in archives:
        print(f"Extracting {archive}...", flush=True)
        with tarfile.open(archive, "r:gz") as tar:
            tar.extractall(path=build_dir)

    # Find extracted directories
    scatter_dirs = glob.glob(os.path.join(build_dir, "torch_scatter-*")) + glob.glob(
        os.path.join(build_dir, "torch-scatter-*")
    )
    sparse_dirs = glob.glob(os.path.join(build_dir, "torch_sparse-*")) + glob.glob(
        os.path.join(build_dir, "torch-sparse-*")
    )

    # Filter out directories only
    scatter_dirs = [d for d in scatter_dirs if os.path.isdir(d)]
    sparse_dirs = [d for d in sparse_dirs if os.path.isdir(d)]

    if not scatter_dirs or not sparse_dirs:
        print("Error: Could not locate extracted source directories.", flush=True)
        sys.exit(1)

    scatter_dir = scatter_dirs[0]
    sparse_dir = sparse_dirs[0]

    # Patch torch-scatter (patch both CUDA source and HIP output if pre-generated)
    for cuh in ["csrc/cuda/utils.cuh", "csrc/hip/utils.cuh"]:
        scatter_cuh = os.path.join(scatter_dir, cuh)
        patch_file(
            scatter_cuh,
            {
                "const unsigned mask": "const unsigned long long mask",
                "#define SHFL_UP_SYNC(mask, var, delta) __shfl_up(var, delta)": "#define SHFL_UP_SYNC(mask, var, delta) __shfl_up_sync((unsigned long long)(mask), var, delta)",
                "#define SHFL_DOWN_SYNC(mask, var, delta) __shfl_down(var, delta)": "#define SHFL_DOWN_SYNC(mask, var, delta) __shfl_down_sync((unsigned long long)(mask), var, delta)",
            },
        )

    # Patch torch-sparse (patch both CUDA source and HIP output if pre-generated)
    for cuh in ["csrc/cuda/utils.cuh", "csrc/hip/utils.cuh"]:
        sparse_cuh = os.path.join(sparse_dir, cuh)
        patch_file(
            sparse_cuh,
            {
                "const unsigned mask": "const unsigned long long mask",
                "#define SHFL_UP_SYNC(mask, var, delta) __shfl_up(var, delta)": "#define SHFL_UP_SYNC(mask, var, delta) __shfl_up_sync((unsigned long long)(mask), var, delta)",
                "#define SHFL_DOWN_SYNC(mask, var, delta) __shfl_down(var, delta)": "#define SHFL_DOWN_SYNC(mask, var, delta) __shfl_down_sync((unsigned long long)(mask), var, delta)",
                "#define SHFL_SYNC(mask, var, delta) __shfl(var, delta)": "#define SHFL_SYNC(mask, var, delta) __shfl_sync((unsigned long long)(mask), var, delta)",
            },
        )

    # Compile and install
    print("Installing patched torch-scatter...", flush=True)
    subprocess.run(
        [sys.executable, "-m", "pip", "install", scatter_dir, "--no-build-isolation"], check=True
    )

    print("Installing patched torch-sparse...", flush=True)
    subprocess.run(
        [sys.executable, "-m", "pip", "install", sparse_dir, "--no-build-isolation"], check=True
    )

    print("Successfully built and installed torch-scatter and torch-sparse.", flush=True)

    # Cleanup build directory
    print("Cleaning up build directory...", flush=True)
    shutil.rmtree(build_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
