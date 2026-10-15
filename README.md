# Hamiltonian-Informed Quantum Machine Learning

This repository contains the code and data for the paper:
"Electronic-Structure-Informed Many-Body Encoding for Quantum Machine Learning"


## Overview


## Methodology



## Installation

Run commands from the repository root, in Bash or Zsh. The reference environment
uses standard CPython 3.13.2 and the dependency versions below:

```sh
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install \
  numpy==2.5.1 scipy==1.18.0 torch==2.13.0 PennyLane==0.45.1 \
  pandas==3.0.5 matplotlib==3.11.1 pyscf==2.14.0
```
Training uses the stored descriptor archives and does not require
rerunning electronic-structure calculations. No editable package installation
is required when invoking the modules from the repository root.

## Citation

Publication details and a BibTeX entry will be added when finalized.

## License

## Contact
