#!/bin/bash
full_name=$1
last_two=$(echo "$full_name" | rev | cut -d'/' -f1,2 | rev)
b=$(echo "$last_two" | rev | cut -d'/' -f1 | rev)
a=$(echo "$last_two" | rev | cut -d'/' -f2 | rev)
xpref=${a}_${b}

job_name=$xpref
echo job_name=$job_name

sbatch --requeue <<EOT
#!/bin/bash
#SBATCH --partition=vision-torralba
#SBATCH --qos=vision-torralba-main
#SBATCH --nodes=1
#SBATCH --time=3-00:00:00
#SBATCH --gpus-per-node=v100:8
#SBATCH --exclude=agrawal-v100-1
#SBATCH --cpus-per-task=80
#SBATCH --mem=500G
#SBATCH --job-name=$job_name
#SBATCH --output=./slurm_logs/${job_name}/%j.out
#SBATCH --dependency=$2

$1
EOT