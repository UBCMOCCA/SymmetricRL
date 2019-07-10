#!/bin/bash
set -e

num_replicates=1
experiment_path="$HOME/projects/def-vandepan/symmetric"
project_path="$HOME/projects/def-vandepan/$USER/SymmetricRL"
today=`date '+%Y_%m_%d__%H_%M_%S'`

name=$1
if [ $# -eq 0 ]
then
    echo "No arguments supplied: experiment name required"
    exit 1
fi
shift;

log_path=$experiment_path/runs/${today}__${name}
mkdir -p $log_path
cat > $log_path/run_script.sh <<EOF
#!/bin/bash
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --exclusive
#SBATCH --cpus-per-task=24
#SBATCH --mem-per-cpu=400M
#SBATCH --array=1-$num_replicates
. $project_path/../venv/bin/activate
cd $project_path
python playground/train.py with experiment_dir="$log_path/\$SLURM_ARRAY_TASK_ID" replicate_num=\$SLURM_ARRAY_TASK_ID $@
EOF

cd $log_path

for ((i=1;i<=$num_replicates;i++)) do
    mkdir $i
done

sbatch run_script.sh
echo "Logging at: $log_path"