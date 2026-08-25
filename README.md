# y6-ppd-public
Public facing repository containing tools the Y6 Gaussian Mixture PPD methods.
Contact Jessie Muir (muirjc@ucmail.uc.edu) with any questions!

## Overview

This directory will contain tools for running PPD calculations. To run PPD calculation, you first need to obtain a chain file that has theory predictions for your observables saved as derived parameters. Then you can read those in to estimate the PPD and relevant summary statistics from it.

## Saving theoretical predictions for data vector elements using cosmosis

For DES Y6, our pipeline is set up in Cosmosis, so we do the initial theory-prediction saving step as follows: 
1. Subsample the original nautilus chains to create an equal weight
    subsample (not needed for polychord); script for this is `make_eqw_subsamples.py`. (This is done to make the next step more efficient because nautilus tends to save millions of samples for a O(10K) effective sample size.)
2. Run cosmosis' importance sampler on that original chain, using the same model and theory prediction choices as the original chain, but adding 2pt theory elements as `EXTRA_OUTPUT`.
* For an example of how to set this up, see files in the subdirectory `examples_for_IS-save-DV_runs` for examples of how to do this either with cosmosis ini files or campaign runs.
* Note that you can probably run the importance sampling job with the same cluster settings as youused to run the chain but 128/256 cpus might be enough, and walltime will likely only be a few hours.

## Running PPD calculations

Once you have a chain file with your theory predictions saved for each sample (aka, you have a posterior estimate for your data vector elements), you're ready to do the actual PPD calculations. 
* The script `run_ppd_calcs_example.py` provides an example of how to use functions in the `ppd_calc_utils.py` library to run PPD tests.   This uses funtions from ppd_calc_utils.py. Note that this example does not necessarily use all the options that are adjustable for these calculations. To see more on this, look at the docstring for the function `do_ppd` in `ppd_calc_utils.py`. 
* An example of how to set up a a GPU job to run this on perlmutter can be found in  job_example_run_ppd_calcs.sub.
* You could also run this via an interactive job set up with something like this: `salloc --nodes 1 --qos interactive --time 01:00:00 --constraint gpu --gpus 1 --account mynerscallocation`
* If you're using GPUs, the calculation for each chain should only take a couple minutes. The calculations will also work on CPUS; but are likely to talk hours per chain rather than minutes.

## Output files
Running PPD calculations can produce the following output files for each PPD test:
* `outfile.ppdband-dat.txt`: A  file containing the data used to make the PPD "band plots" for visual inspection. Columns correspond to (0) data vector index, (1) the test data measurements, (2) the PPD mean, (3) the PPD standard deviation
* `outfile.ppdband.png` or `outfile.ppdband-norm.png`: A plot showing the comparison between test data and the PPD mean and standard deviation. For 3x2pt this will include some shading and labels to help idenfity which parts of the 3x2pt data vector are being plotted. If you set this to be normalized, the plot will subtract the PPD mean from all datapoints, and normalize them relative to the PPD standard deviation. There is an option to just remake the plot based on a saved `ppdband-dat.txt` file, if you want to adjust formatting without rerunning the whole calculation. Note that  file type can be changed to things other than `png`.
* `outfile.logp-dat.txt`: File containing data needed to compute the Delta_PPD metric and to make the associated histogram plot. This will be a single column file with a lot of rows. The first row (0) is the logp_dat value of the test data, (1) is Delta_PPD, the fraction of the posterior volume where logP<logp_dat, (2) is an estimate of the sampling error on Delta_PPD, and (3) if not None is a delta_PPD estimate using KDE smoothing of our histogram (as opposed to just counting fractions of samples). There will then be an empty line, and lines [5:] are all the logP samples drawn from the PPD in order to estimate the numbers above.
* `outfile.logp-hist.png`: plot of the logP histogram showing how the probability density of the test data compares to an ensemble of draws from the PPD. 

## Adapting to work with non-cosmosis pipeliens or chains in different formats

These tools have been set up to work in the DES Y6 ecosystem of cosmosis pipelines, and assuming the associated file format for chains, data files, and scale cut definition. However, the Gaussian mixture model, and the calculations set up in `ppd_calc_utils.py` can in principle be set up for other formats. You'd just need to add or adapt some of the helper functions to work with your pipeline. 
* `load_ppd_run` - This is the most importance piece for calculations that has pipeline-specific assumptions. It reads in DV theory predictions from a chain. In addition to reading in the chain samples and evaluating weights, it also does some evaluation to get the correct mapping between the indices of the data vector model elements saved in the input chain, and the data we're testing compared to the PPD. For DES chains, this uses the `.fits` file containing the 2pt measurements and covariance to learn about the ordering of data vector elements, and uses `.ini` files containing 2pt scale cuts to figure out what subset of those elements are in the chain file and in the test data. For a different pipeline format, you'd have to set up your own version of this operation.

If this is something you're interested in adapting to a different chain or pipeline set-up,  I (Jessie) am happy to chat with you to provide guidance as needed. Please feel free to get in touch!