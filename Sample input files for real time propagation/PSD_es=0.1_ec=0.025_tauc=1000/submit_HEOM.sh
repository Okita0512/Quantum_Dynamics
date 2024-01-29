# for i in 600 800 900 1000 1050 1100 1120 1140 1150 1160 1170 1180 1200 1250 1300 1400 1500 1600 1700 1800
for i in 1170
do

cd $i

cat << Eof > sbatch_HEOM.sh
#!/bin/bash
#SBATCH -p preempt
#SBATCH -o output.log
#SBATCH --mem=8GB
#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-task=1
#SBATCH --job-name=HEOM
#SBATCH --open-mode=append

time ../../../bin/rhot ./input.json

Eof

chmod +x sbatch_HEOM.sh
sbatch sbatch_HEOM.sh
cd ../

done
