# Code for running RASSCoL without Random Forest

*This README is intended for Tombstone2 users in the Woolfson group only.*

## Installation

This code requires a small number of external libraries (see `env/RASSCoL_env.yml`) as was tested using Python 3.9.18.

To set up the environment, run:

```bash
conda env create -f env/RASSCol_env.yml
```

Then to make the environment visible in the Jupyter Notebook install `ipykernel`:

```bash
pip install ipykernel==6.19.2
```

At this point you should be good to interact with the example notebooks (found in `notebooks` directory).
