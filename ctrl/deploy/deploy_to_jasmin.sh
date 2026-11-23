# E.g. bash deploy_to_jasmin.sh sci-ph-02.jasmin.ac.uk jasmin_script.sh
# This will be run locally, and uses ssh to execute $SCRIPT on remote $SERVER.
SERVER=$1
SCRIPT=$2

cat $2 | ssh $1

# rsync figs back.
rsync -Rav --progress --include='*.png' --include='*.hdf' --include='*/' --exclude='*' mmuetz@xfer-vm-01.jasmin.ac.uk:/./gws/nopw/j04/mcs_prime/mmuetz/upflo/data/upflo_wp1_figs/wescon_radar_dev/v7/202308?? /home/markmuetz/mirrors/jasmin/
