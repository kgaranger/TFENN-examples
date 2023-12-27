# TFENN-examples

Code and training scripts used to reproduced the results presented in:
> Garanger, Kévin, Julie Kraus, and Julian J. Rimoli. "Symmetry-enforcing neural networks with applications to constitutive modeling." arXiv preprint [arXiv:2312.13511 (2023)](https://arxiv.org/abs/2312.13511).

This repository code depends on the package
[TFENN](https://github.com/kgaranger/TFENN).

## Training script

To run a training script, the following command can be used from the root of the repository:
`python -m scripts.train ...`
Use this command followed by `-h` to see its possible arguments.

## Datasets

Datasets used in the paper can be downloaded [here](https://drive.google.com/file/d/1EKgJUhYkdxEzeXMiMD9I5cZxdR9mbwy3/view?usp=sharing).
Please extract the content of the archive in a `datasets` directory placed at the root of the repository.

## Running multiple scripts at once

Some `zsh` training scripts are provided for convenience to run multiple training scripts on a given dataset with various models and training set sizes.
It relies on `tmux` to start multiple detached sessions.
Please ensure that the the machine can support running all these sessions at the same time before launching one of these zsh scripts.
If not, it is advised to modify these scripts to reduced the number of sessions that will be started.
