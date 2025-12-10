### LUMI Environment Summary

- LUMI modules:
  - `module load LUMI/22.12`
  - `module load partition/G`
  - (add others if needed)

- Container:
  - Built using `bash_scripts/build_container.sh`
  - Based on ROCm-enabled PyTorch image: `ghcr.io/...` (example)

- Python requirements:
  - Inside the container, install:
    ```bash
    pip install -r requirements-lumi.txt
    ```
  - `requirements-lumi.txt` reflects the exact Python versions used in the experiments.