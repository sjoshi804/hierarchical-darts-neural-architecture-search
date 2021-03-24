#### submit_job.sh START ####
#!/bin/bash
#$ -cwd
# error = Merged with joblog
#$ -o joblog.$JOB_ID
#$ -j y
## Edit the line below as needed:
#$ -l gpu
# Email address to notify
#$ -M $USER@mail
# Notify when
#$ -m bea

# echo job info on joblog:
echo "Job $JOB_ID started on:   " `hostname -s`
echo "Job $JOB_ID started on:   " `date `
echo " "

# load the job environment:
. /u/local/Modules/default/init/modules.sh
## Edit the line below as needed:
module load python/3.7.2

## substitute the command to run your code
## in the two lines below:
echo "-------------------------------------------------"
echo "\t\tVerify Setup"
echo "-------------------------------------------------"
git branch
git log --name-status HEAD^..HEAD

echo "Running Search"
python3 search.py --dataset=cifar10 --epochs=1 --num_nodes_at_level='{0: 2, 1:2}' --channels_start=1 --weights_gradient_clip=5.0 --batch_size=64 --alpha_lr=0.01 --logdir=logs --learnt_model_path=mnist_learnt_models

# echo job info on joblog:
echo "Job $JOB_ID ended on:   " `hostname -s`
echo "Job $JOB_ID ended on:   " `date `
echo " "
#### submit_job.sh STOP ####
