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
#SBATCH --partition=csail-shared
#SBATCH --qos=lab-free
#SBATCH --nodes=1
#SBATCH --time=1-00:00:00
#SBATCH --gpus-per-node=l40s:8
#SBATCH --exclude=agrawal-v100-1
#SBATCH --cpus-per-task=80
#SBATCH --mem=500G
#SBATCH --job-name=$job_name
#SBATCH --output=./slurm_logs/${job_name}/%j.out

$1
EOT