# SymmetricRL

## Running Locally
To run an experiment named `test_experiment` with the PyBullet humanoid environment you can run:

```bash
./cc_run_scripts/local_run_playground_train.sh  test_experiment  env_name='pybullet_envs:HumanoidBulletEnv-v0'
```

This will create a new experiment directory inside the `runs` directory that contains the following files:
 - `pid`: the process ID of the task running the training algorithm
 - `progress.csv`: a CSV file containing the data about the the training progress
 - `slurm.out`: the output of the process. You can use `tail -f` to view the contents
 - `configs.json`: a JSON file containing all the hyper-parameter values used in this run
 - `run.json`: extra useful stuff about the run including the host information and the git commit ID (only works if GitPython is installed)
 - `models`: a directory containing the saved models