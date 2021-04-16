B1;95;0c#### submit_job.sh START ####
#!/bin/bash
#$ -cwd
# error = Merged with joblog
#$ -o joblog.$JOB_ID
#$ -j y
## Edit the line below as needed:
#$ -l gpu,V100
#$ -l h_rt=24:00:00
#$ -l h_vmem=32G
# Email address to notify
#$ -M $USER@mail
# Notify when
#$ -m beas

# echo job info on joblog:
echo "Job $JOB_ID started on:   " `hostname -s`
echo "Job $JOB_ID started on:   " `date `
echo " "

# load the job environment:
. /u/local/Modules/default/init/modules.sh

## Edit the line below as needed:
module load python/anaconda3

## substitute the command to run your code
## in the two lines below:
echo "-------------------------------------------------"
echo "                     Verify Setup                "
echo "-------------------------------------------------"
git branch

echo "-------------------------------------------------"
echo "                     Environment Setup           "
echo "-------------------------------------------------"
echo "source activate pytorch-1.3.1-gpu"
source activate pytorch-1.3.1-gpu

echo "-------------------------------------------------"
echo "                     Running Search              "
echo "-------------------------------------------------"
EXPERIMENT="python3 search.py --epochs=10 --channels_start=8 --dataset=cifar10 --num_nodes_at_level='{0:3, 1:5}' --num_download_workers=8 --batch_size=32"

echo $EXPERIMENT
eval $EXPERIMENT

# echo job info on joblog:
echo "Job $JOB_ID ended on:   " `hostname -s`
echo "Job $JOB_ID ended on:   " `date `
echo " "
#### submit_job.sh STOP ####
