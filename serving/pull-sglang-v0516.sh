#!/bin/bash
#SBATCH --account=<account>
#SBATCH --job-name=pull-sglang-v0516
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --qos=1d
#SBATCH --time=1-00:00:00
#SBATCH --output=/shared/project/<account>/llm/logs/pull-sglang-v0516_%j.out
#SBATCH --error=/shared/project/<account>/llm/logs/pull-sglang-v0516_%j.err

module load apptainer/1.4.1
export APPTAINER_TMPDIR=/shared/project/<account>/llm/.cache/apptainer-tmp
mkdir -p $APPTAINER_TMPDIR
cd /shared/project/<account>/llm/containers
rm -f sglang-v0516.sif
echo "=== pulling lmsysorg/sglang:v0.5.16-cu130 (64GB RAM, 8 CPUs) ==="
apptainer pull --force sglang-v0516.sif docker://lmsysorg/sglang:v0.5.16-cu130
echo "=== done ==="
ls -la sglang-v0516.sif
