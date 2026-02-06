# Remote script - run on JASMIN.
# Setup conda. Taken from .bashrc.jasmin.sh
__conda_setup="$('/home/users/mmuetz/miniforge3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)" 
eval "$__conda_setup"
unset __conda_setup

# Activate eng.
conda activate upflo_env

# Ensure correct plotting (forces non-interactive I think)
export MPLBACKEND=agg

cd /home/users/mmuetz/deploy/wescon-tools/ctrl/remakefiles
remake run -ESlurm wescon_radar_dev.py

# Poll until squeue is empty.
while [ -n "$(squeue --me -h)" ]; do echo "Polling..."; sleep 10; done
echo "All jobs finished"
