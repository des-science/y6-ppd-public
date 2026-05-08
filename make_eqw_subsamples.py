"""
Given list of nautilus chains, if file exists, make a subsampled
version with equal weights. This isn't strictly necessary for PPD calculations,
but if you're using importance sampling to save model predictions for each
sample in a posterior, subsampling the chain can make this much more
efficient. (Note that this is feasible because  Nautilus sampler chains
typically save millions of samples, but have an effective sample size of order 10^4.)

Script by Jessie Muir, adapting from example script from Joe Zuntz, as
discussed in
https://github.com/cosmosis-developers/cosmosis/issues/145#issuecomment-2658977590

"""

import numpy as np
import os
import sys
import pathlib

def make_equal_weight_chain(fname,fdir='',save_N_times_neff=4,outtag='eqw',outdir=None):
    """
    fname - name of input chain, no directory path included
    fdir - path to where the input chains are stored
    save_N_times_neff - output chain will have N*neff samples, where neff
          is the effective number of samples in the origianl chains
    outtag - output chain name will be outtag_fname
    outdir - if specified, will be directory where output file goes
             if not specified, will put chains in same directory as original chains
    """

    infile = os.path.join(fdir,fname)

    if not outtag:
        raise ValueError("Need to put something for outtag to avoid overwriting files")

    print("  Looking at file:",fname)
    if not os.path.isfile(infile):
        print("    file not found")
        return
    # check if file ends with complete line:
    headerlines =[]
    footerlines = []
    f = open(infile)
    complete = False
    inheader=True
    for line in f:
        if line.startswith("#"):
            if inheader:
                headerlines.append(line.strip())
            else:
                footerlines.append(line.strip())
                if "complete=1" in line:
                    complete=True
                    break
        else:
            inheader=False
    f.close()
    if not complete:
        print("    not complete")
        return

    print('    getting subsample')
    dat = np.loadtxt(infile)
    logwind = -3
    logw = dat[:,logwind]
    w = np.exp(logw)
    w = w/np.sum(w)
    neff = int(1.0 / np.sum(w**2))

    # Select an equally-weighted subsample of the weights.
    index = np.arange(w.size)
    index = np.random.choice(index, size=save_N_times_neff*neff, p=w, replace=True)

    newdat = dat[index,:]
    newdat[:,logwind] = np.zeros(index.size)

    if outdir is not None: #we've specified the output directory
        outfile = os.path.join(outdir,outtag+'_'+fname)
    else: # put it in the same directory as the input chains
        outfile = os.path.join(fdir,outtag+'_'+fname)
    print('    Saving',outfile)
    np.savetxt(outfile,newdat,header='\n'.join(headerlines),footer='\n'.join(footerlines),comments='')


#==================================================================
def main():
    args = sys.argv[1:]
    if args is None:
        raise Exception("Please provide the path of the file to subsample.")
    elif len(args) == 1:
        raise Exception("It is missing the path of the file or the run name.")

    chaindir = args[0] #directory where to find the original chain
    fname = args[1] # filename of the original chain


    outdir = os.path.join(chaindir,'eqw_subsample/')
    path = pathlib.Path(outdir)
    path.mkdir(exist_ok=True)


    print("\nSUBSAMPLING",fname)
    make_equal_weight_chain(fname,chaindir,outdir=outdir)



#==================================================================
if __name__=="__main__":
    main()
