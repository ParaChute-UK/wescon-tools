# Remote script - run on JASMIN.
# Activate the shared uv env (built by ~/projects/local/radar_env/sync.sh).
source /home/users/mmuetz/projects/local/radar_env/.venv/bin/activate

# Ensure correct plotting (forces non-interactive I think)
export MPLBACKEND=agg

cd /home/users/mmuetz/deploy/wescon-tools/ctrl/remakefiles
remake info wescon_radar_dev.py

# Poll until squeue is empty.
# while [ -n "$(squeue --me -h)" ]; do sleep 10; done
