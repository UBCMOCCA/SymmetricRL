# SymmetricRL

## Running Locally

To run an experiment named `test_experiment` with the PyBullet humanoid environment you can run:

```bash
./scripts/local_run_playground_train.sh  test_experiment  env_name='Symmetric_HumanoidBulletEnv-v0'
```

This will create a new experiment directory inside the `runs` directory that contains the following files:

- `pid`: the process ID of the task running the training algorithm
- `progress.csv`: a CSV file containing the data about the the training progress
- `slurm.out`: the output of the process. You can use `tail -f` to view the contents
- `configs.json`: a JSON file containing all the hyper-parameter values used in this run
- `run.json`: extra useful stuff about the run including the host information and the git commit ID (only works if GitPython is installed)
- `models`: a directory containing the saved models

## Plotting Results

The `plot_from_csv.py` script can be helpful for plotting the learning curves:

```bash
python -m playground.plot_from_csv --load_paths runs/*/*/  --columns mean_rew max_rew  --smooth 2
```

- The `load_paths` argument specifies which directories the script should look into
- It opens the `progress.csv` file and plots the `columns` as the y-axis and uses the `row` for the x-axis (defaults to `total_num_steps`)
- You can also provide a `name_regex` to make the figure legends simpler and more readable, e.g. `--name_regex 'walker-(.*)mirror\/'` would turn `runs/2019_07_08__23_53_20__walker-lossmirror/1` to simply `loss`.


## Running Learned Policy

The `enjoy.py` script can be used to run a learned policy and render the results:

```bash
python -m playground.enjoy with experiment_dir=runs/<EXPERIMENT_DIRECTORY>

# plot the joint positions over time
python -m playground.evaluate with experiment_dir=runs/<EXPERIMENT_DIRECTORY> plot=True
```

## Evaluating Results

The `evaluate.py` script is used for evaluating learned policies. The results will be saved to a new file called `evaluate.json` inside the `experiment_dir` where the policy was loaded from.

```bash
# evaluate with 10000 steps
python -m playground.evaluate with experiment_dir=runs/<EXPERIMENT_DIRECTORY> max_steps=10000
# with rendering
python -m playground.evaluate with experiment_dir=runs/<EXPERIMENT_DIRECTORY> render=True

# evaluate all experiments
for d in runs/*/*/; do python -m playground.evaluate with experiment_dir=$d; done
```

Metrics (will be expanded):
 - Symmetric Index (joint angle and torque)
   * Uses the `MetricsEnv` environment wrapper from `symmetric/metric_utils.py`

**NOTE:** the current script assumes that the environment has a similar interface to PyBullet locomotion environments. It assumes the following parameters are present in the unwrapped environment:
  - `robot.feet_contact`
  - `robot.ordered_joints`
    - The joint names that have the word `left` in them are assumed to be on the left side. Any other joint that does not contain the word `abdomen` is assumed to be on the right side.

The environments that are currently tested and supported:
 - `Walker2DBulletEnv`
 - `HumanoidBulletEnv`
 - `Walker3DCustomEnv`
 - `Walker3DStepperEnv`
