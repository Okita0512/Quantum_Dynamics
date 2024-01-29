#!/bin/bash
#SBATCH -p preempt
#SBATCH -o output.log
#SBATCH --mem=16GB
#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-task=1
#SBATCH --job-name=HEOM
#SBATCH --open-mode=append

time ../bin/1d-resp ./input.json

