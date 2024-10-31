"""
Update the config.yaml file, e.g.,
    python update_config.py config_job1.yaml --STATE_DIM 1000
"""

import argparse
import re


def get_arguments():
    parser = argparse.ArgumentParser(description='Update configuration.')
    parser.add_argument('new_config_file', type=str, help='New config file name')
    parser.add_argument('--STATE_DIM', type=int, help='New STATE_DIM value')
    parser.add_argument('--NUM_ACTIONS', type=int, help='New NUM_ACTIONS value')
    parser.add_argument('--DISCOUNT', type=float, help='New DISCOUNT value')
    parser.add_argument('--SIGMA', type=float, help='New SIGMA value')
    parser.add_argument('--B', type=float, help='New B value')
    parser.add_argument('--DELTA', type=float, help='New DELTA value')
    parser.add_argument('--DF', type=int, help='New DF value')
    parser.add_argument('--DEGREE', type=int, help='New DEGREE value')
    parser.add_argument('--MU', type=float, help='New MU value')
    parser.add_argument('--H', type=float, help='New H value')
    parser.add_argument('--LAMBDA_REG', type=float, help='New LAMBDA_REG value')
    parser.add_argument('--NUM_Z', type=int, help='New NUM_Z value')
    parser.add_argument('--CANDIDATE_STATE_IDX', type=int, help='New CANDIDATE_STATE_IDX value')
    parser.add_argument('--MAX_ITER', type=int, help='New MAX_ITER value')
    parser.add_argument('--MAX_POLICY_ITER', type=int, help='New MAX_POLICY_ITER value')
    parser.add_argument('--BASIS_TYPE', type=int, help='New BASIS_TYPE value')
    parser.add_argument('--NUM_EPISODES', type=int, help='New NUM_EPISODES value')
    parser.add_argument('--MAX_STEPS', type=int, help='New MAX_STEPS value')
    parser.add_argument('--GAMMA', type=float, help='New GAMMA value')
    parser.add_argument('--MC_NUM_EPISODES', type=int, help='New MC_NUM_EPISODES value')
    parser.add_argument('--MC_MAX_STEPS', type=int, help='New MC_MAX_STEPS value')
    parser.add_argument('--N_EPOCHS', type=int, help='New N_EPOCHS value')
    parser.add_argument('--N_STEPS_PER_EPOCH', type=int, help='New N_STEPS_PER_EPOCH value')
    parser.add_argument('--N_FOLDS', type=int, help='New N_FOLDS value')
    return parser.parse_args()

def main():
    args = get_arguments()

    # Define your template configuration file
    template_config_file = 'config.yaml'

    # Generate a configuration file for the job
    new_config_file = args.new_config_file

    with open(template_config_file, 'r') as file:
        lines = file.readlines()

    for i, line in enumerate(lines):
        for arg in vars(args):
            if arg != 'new_config_file' and getattr(args, arg) is not None:
                # If the line starts with the parameter name, replace it
                if re.match(arg + r':', line):
                    lines[i] = f'{arg}: {getattr(args, arg)}\n'

    with open(new_config_file, 'w') as file:
        file.writelines(lines)

if __name__ == '__main__':
    main()