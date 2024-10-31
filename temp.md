# KSH-LSPI: Kernel Sieve Hybrid Least-Squares Policy Iteration

[Previous content remains the same until Installation section]

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ksh-lspi.git
cd ksh-lspi
```

2. Create and activate a conda environment:
```bash
conda create -n ksh_env python=3.10.9
conda activate ksh_env
```

3. Install required packages:
```bash
pip install numpy pandas torch patsy formulaic csaps matplotlib seaborn wandb d3rlpy
```

## Running the Code

All scripts should be run from the project root directory:

```bash
# Run simulation experiments
python simulation/run_simulation.py config.yaml --nonlinear_states

# Train models
python simulation/train_simulated_ksh_model.py config.yaml --nonlinear_states

# Evaluate models
python simulation/evaluate_model.py config.yaml
```

[Rest of README remains the same]