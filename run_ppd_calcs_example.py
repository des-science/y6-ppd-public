"""
This script provides an exmaple of how to use the PPD utilties
in ppd_calc_utils.py to conduct internal consistency tests
for a 3x2pt analysis. 

The script assumes
- Chains have been been uploaded to a project storage space
   at /project/path/forchains/
- Original chains (mostly run with nautilus) are in that original directory
- eqw_subsample subdirectory has subsampled versions of those chains
- is_chains subdirectory is where we have put outputs from cosmosis
   importance sampling runs saving theory predictions for the data vector
   at each sample (using the subsampled chains as input for practical purposes)
- ppd_output is where the output from this script will go

As input it will also need
- path to where to find the .fits file containing the data measurement
   (uses data there to map between data vector element index and
    labels for 2pt function, redshift bin, and angular bin)
- path to cosmosis ini file contining scale cuts being used in the analysis

"""


import os, sys
import ppd_calc_utils as ppd

# these couple lines should let you import ppd_calc_utils even if you're running from a different location
#ppdcalcdir = "/path/to/where/to/find/scripts/y6-ppd-public/"
#sys.path.insert(0,ppdcalcdir)
#ppd = __import__('ppd_calc_utils')

#============================================================
def main():

    # info about where to find scale cuts
    scutdir =  '/myworkspace/scale_cuts' # update to match your path
    scutini_name = 'y6-dv6-scales-ml6-3x2pt_9_6_TATT_chi2_5.00.ini'
    scutini = os.path.join(scutdir,scutini_name)

    
    #path to the data file containing covairance and 3x2pt measurements
    fitsdir =  '/myworkspace/data_vectors/')
    fitsfile = os.path.join(fitsdir, 'data_3x2pt.fits')

    # where to find chains
    chaindir = '/project/path/forchains'
    ischaindir = os.path.join(chaindir,'is_chains')
    
    # where to put calculation outputs and plots
    outdir= os.path.join(chaindir,'ppd_output')

    # chain base should be the is chain filenames without is_ and without .txt
    chainbase_32pt = '32pt_lin_lcdm' 
    chainbase_32pt_wcdm = '32pt_lin_wcdm' 
    chainbase_22pt = '22pt_lin_lcdm' 
    chainbase_12pt = '12pt_lin_lcdm' 
    chainbase_12pt_nla = '12pt_nla_lin_lcdm'
    chainbase_shear_ggl = '12pt_ggl_lin_lcdm'
    chainbase_shear_wtheta = '12pt_w_lin_lcdm'
    
    #############################
    # Note that calculations are set up in "if" statement blocks
    # so that they can be easily toggled on or off as needed
    
    #=========================================
    # Goodness of fit
    #=========================================
    # using if statement blocks to make it easier if we want to rerun
    # a subset of these
    #--------------
    # 1x2pt LCDM GOF
    #--------------
    if 10:
        labelstr = "PPD goodness-of-fit: 1x2pt LCDM"
        chainbase = chainbase_12pt
        chainf = 'is_{}.txt'.format(chainbase)
        chainfile = os.path.join(ischaindir,chainf)

        #labels for output files
        outf = 'ppd-gof_{}'.format(chainbase)
        outbase = os.path.join(outdir,outf)

        usecorrs = ['xip','xim']

        if not os.path.isfile(chainfile):
            print("No file found for",chainfile)
        else:
            ppd.do_ppd(chainfile,fitsfile,usecorrs,scutini,outbase,labelstr)


    #--------------
    # 2x2pt LCDM GOF
    #--------------
    if 10:
        labelstr = "PPD goodness-of-fit: 2x2pt LCDM"
        chainbase = chainbase_22pt
        chainf = 'is_{}.txt'.format(chainbase)
        chainfile = os.path.join(ischaindir,chainf)

        #labels for output files
        outf = 'ppd-gof_{}'.format(chainbase)
        outbase = os.path.join(outdir,outf)

        usecorrs = ['gammat','wtheta']
        if not os.path.isfile(chainfile):
            print("No file found for",chainfile)
        else:
            ppd.do_ppd(chainfile,fitsfile,usecorrs,scutini,outbase,labelstr)

    #--------------
    # 3x2pt LCDM GOF
    #--------------
    if 10:
        labelstr = "PPD goodness-of-fit: 3x2pt LCDM"
        chainbase = chainbase_32pt
        chainf = 'is_{}.txt'.format(chainbase)
        chainfile = os.path.join(ischaindir,chainf)

        #labels for output files
        outf = 'ppd-gof_{}'.format(chainbase)
        outbase = os.path.join(outdir,outf)

        usecorrs = ['xip','xim','gammat','wtheta']
        if not os.path.isfile(chainfile):
            print("No file found for",chainfile)
        else:
            ppd.do_ppd(chainfile,fitsfile,usecorrs,scutini,outbase,labelstr)

    #--------------
    # 3x2pt wCDM GOF
    #--------------
    if 10:
        labelstr = "PPD goodness-of-fit: 3x2pt wCDM"
        chainbase = chainbase_32pt_wcdm 
        chainf = 'is_{}.txt'.format(chainbase)
        chainfile = os.path.join(ischaindir,chainf)

        #labels for output files
        outf = 'ppd-gof_{}'.format(chainbase)
        outbase = os.path.join(outdir,outf)

        usecorrs = ['xip','xim','gammat','wtheta']
        if not os.path.isfile(chainfile):
            print("No file found for",chainfile)
        else:
            ppd.do_ppd(chainfile,fitsfile,usecorrs,scutini,outbase,labelstr)


    #=========================================
    # Combining for 3x2pt?
    #=========================================
    #--------------
    # 2x2pt | 3x2pt LCDM
    #--------------
    if 10:
        labelstr = "PPD for (2x2pt$|$3x2pt) LCDM"
        chainbase = chainbase_32pt
        chainf = 'is_{}.txt'.format(chainbase)
        chainfile = os.path.join(ischaindir,chainf)

        # what 2pt functions are involved in DV piece being tested?
        testcorrs =  ['gammat','wtheta']
        # what 2pt functs were used for the posterior?
        postcorrs = ['xip','xim','gammat','wtheta']

        outf = 'ppd-subset_22pt_given_{}'.format(chainbase)
        outbase = os.path.join(outdir,outf)

        if not os.path.isfile(chainfile):
            print("No file found for",chainfile)
        else:
            ppd.do_ppd(chainfile, fitsfile, postcorrs, scutini, outbase, labelstr,testcorrs)

    #--------------
    # 1x2pt | 3x2pt LCDM
    #--------------
    if 10:
        labelstr = "PPD for (1x2pt$|$3x2pt) LCDM"
        chainbase = chainbase_32pt
        chainf = 'is_{}.txt'.format(chainbase)
        chainfile = os.path.join(ischaindir,chainf)

        testcorrs =  ['xip','xim']
        postcorrs = ['xip','xim','gammat','wtheta']

        outf = 'ppd-subset_12pt_given_{}'.format(chainbase)
        outbase = os.path.join(outdir,outf)

        if not os.path.isfile(chainfile):
            print("No file found for",chainfile)
        else:
            ppd.do_ppd(chainfile, fitsfile, postcorrs, scutini, outbase, labelstr,testcorrs)

    #--------------
    # 1x2pt | 2x2pt LCDM
    #--------------
    if 10:
        labelstr = "PPD for (1x2pt$|$2x2pt) LCDM"
        chainbase = chainbase_22pt
        chainf = 'is_{}.txt'.format(chainbase)
        chainfile = os.path.join(ischaindir,chainf)

        testcorrs =  ['xip','xim']
        postcorrs = ['gammat','wtheta']

        outf = 'ppd-disjoint_12pt_given_{}'.format(chainbase)
        outbase = os.path.join(outdir,outf)

        if not os.path.isfile(chainfile):
            print("No file found for",chainfile)
        else:
            ppd.do_ppd(chainfile, fitsfile, postcorrs, scutini, outbase, labelstr,testcorrs)

    #--------------
    # w | shear + gammat LCDM
    #--------------
    if 10:
        labelstr = "PPD for (w$|$shear+gt) LCDM"
        chainbase = chainbase_shear_ggl
        chainf = 'is_{}.txt'.format(chainbase)
        chainfile = os.path.join(ischaindir,chainf)

        testcorrs =  ['wtheta']
        postcorrs = ['xip','xim','gammat']

        outf = 'ppd-disjoint_w_given_{}'.format(chainbase)
        outbase = os.path.join(outdir,outf)

        if not os.path.isfile(chainfile):
            print("No file found for",chainfile)
        else:
            ppd.do_ppd(chainfile, fitsfile, postcorrs, scutini, outbase, labelstr,testcorrs)

    #--------------
    # gammat | shear + w LCDM
    #--------------
    if 10:
        labelstr = "PPD for (gt$|$shear+w) LCDM"
        chainbase = chainbase_shear_wtheta
        chainf = 'is_{}.txt'.format(chainbase)
        chainfile = os.path.join(ischaindir,chainf)

        testcorrs = ['gammat']
        postcorrs = ['xip','xim','wtheta']

        outf = 'ppd-disjoint_gt_given_{}'.format(chainbase)
        outbase = os.path.join(outdir,outf)

        if not os.path.isfile(chainfile):
            print("No file found for",chainfile)
        else:
            ppd.do_ppd(chainfile, fitsfile, postcorrs, scutini, outbase, labelstr,testcorrs)
 
#============================================================
if __name__=="__main__":
    main()
