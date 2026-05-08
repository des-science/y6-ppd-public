# y6-ppd-public
Public facing repository containing tools the Y6 Gaussian Mixture PPD methods.


# Working Readme for y6-3x2pt
* this contains references to DES-specific workspaces, needs editing! *

This directory will contail tools for running PPD calculations.
These calculations will take three steps
1. Subsample the original nautilus chains to create an equal weight
    subsample (not needed for polychord); script for this will be here.
*Instructions* At perlmutter, get on an interactive node, module load python and run python make_eqw_subsamples.py.
Comment out the chain name you just subsampled in infiles list in the python script. 
2. Run importance sampler using special PPP IS campaign run to save
   theory DV for each sample in the eqw subsample (will be in same dir as
   original campaing)
*Instructions* Copy, paste the subsample file to your machine. Setup ORIG_CHAIN_DIR environment variable to the location of that subsampled chain.
Run corresponding run in campaign_kp_is.yaml (same setting as running chain but can reduce walltime because just a few hours, 128/256 cpus might be enough) and once done, copy it back to NERSC to /global/cfs/cdirs/des/y6-modeling/chains/data/linear/is_chains. 
3. Run PPD calcs on the IS ouptut: use script run_ppd_calcs_y6unblnding.
    This uses funtions from ppd_calc_utils.py. An example of how to set up a
    job to run this on perlmutter is in job_example_run_ppd_calcs.sub
    (you could also run this interactively)
    > script is set up to skip calculations where chains can't be found
      and individual calculation are in if statement blocks
      that can be togggled on or off (using "if 1:" or "if 0")
      so we can run calculations for a subset of chains as needed
*Instructions* Set if to 1 in run_ppd_calcs_y6unblinding.py for the PPD you want to compute, either enter GPU interactive node (salloc --nodes 1 --qos interactive --time 01:00:00 --constraint gpu --gpus 4 --account des) or submit a GPU job at perlmutter, setup Y6WORKSPACE to point to the inference dir in y6-3x2pt and source setup_ppd_env_nersc.sh. Then run python run_ppd_calcs_y6unblinding.py. 
