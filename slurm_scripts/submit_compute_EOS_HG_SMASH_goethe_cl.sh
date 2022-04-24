#!/usr/bin/bash
#SBATCH --output=log_%x_%j
##SBATCH --partition=general1
##SBATCH --account=hyihp
#SBATCH --partition=fuchs
##SBATCH --partition=test
#SBATCH --account=fias
#SBATCH --nodes=1
#SBATCH --ntasks=20
##SBATCH --ntasks=6
#SBATCH --cpus-per-task=1
#SBATCH --time=2-23:0:0
##SBATCH --time=0-3:0:0
#SBATCH --mail-type=ALL
##SBATCH --array=0-99:20
#SBATCH --array=0-59:20

echo "*************************************"
echo SLURM_ARRAY_JOB_ID $SLURM_ARRAY_JOB_ID 
echo SLURM_ARRAY_TASK_COUNT $SLURM_ARRAY_TASK_COUNT 
echo SLURM_ARRAY_TASK_STEP $SLURM_ARRAY_TASK_STEP
echo SLURM_ARRAY_TASK_ID $SLURM_ARRAY_TASK_ID 
echo SLURM_JOB_ID $SLURM_JOB_ID 
echo SLURM_JOB_NAME $SLURM_JOB_NAME 
echo SLURM_LOCALID $SLURM_LOCALID 
echo SLURM_TASK_PID $SLURM_TASK_PID 
echo SLURM_ARRAY_TASK_MAX $SLURM_ARRAY_TASK_MAX 
echo SLURM_ARRAY_TASK_MIN $SLURM_ARRAY_TASK_MIN
echo SLURM_JOB_PARTITION $SLURM_JOB_PARTITION
echo SLURM_JOB_NODELIST $SLURM_JOB_NODELIST 
echo SLURMD_NODENAME $SLURMD_NODENAME
echo "*************************************"
echo "   "

source /home/hireaction/inghirami/enable_conda

export LC_NUMERIC="en_US.UTF-8"

qmin=-0.1
qmax=0.5

points=$(($SLURM_ARRAY_TASK_COUNT*$SLURM_ARRAY_TASK_STEP))

dq=$(echo "($qmax - $qmin)/($points - 1)" | bc -l)

endq=$(($SLURM_ARRAY_TASK_STEP-1))
for i in $(seq 0 $endq)
do
    qval=$(echo "$qmin + ($SLURM_ARRAY_TASK_ID + $i)*$dq" | bc -l)
    qvalS=$(printf "%04.3f" $qval)
    time python3 compute_EOS_HG_SMASH_v0.3.py out 0 $qval > log$qvalS&
done
wait
sleep 15
