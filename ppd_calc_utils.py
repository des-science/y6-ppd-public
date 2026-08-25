"""
Utilities for performing PPD calculations, assuming you *already* have a
 chain with datavector model predictions saved as extra_output using the 
cosmosis importance sampler

Note that you will generally not run this script directly, rather you
should import it into another script or notebook where you set up the specific
calculations you would like to perform. Generally the only function you will need
to interface with is do_ppd(). 

See README for notes on how to use this, as well as example code in
run_ppd_calcs_example.py

Contact Jessie Muir with questions.
"""

import os
import numpy as np
import scipy as sp
from astropy.io import fits

# these are the modules used in Cyrille's PPD calc script
HAVETF=False
try:
    import tensorflow as tf
    import tensorflow_probability as tfp
    tfd = tfp.distributions
    from tqdm.auto import tqdm, trange
    HAVETF=True
except:
    print("Warning, couldn't find tensorflow modules. Can't compute PPD but can still use plotting functions.")
    HAVETF=False
    
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['font.family'] = 'serif'
from matplotlib import rc
rc('text',usetex=True)



#============================================================
# PPD calculations adapted from Cyrille Doux's proof-of-concept notebook
#============================================================
def do_ppd(chainfile, fitsfile, posterior_usecorrs = None, posterior_scutini = None, outfbase = 'ppdtest', legtitle = 'PPD test', test_usecorrs = None, test_scutini = None,   twopt_section='2pt_like', do_bandplot=True, do_deltappd=True ,ppdlabel="PPD MixGauss", datalabel='Data',ftype='png', Nprobsamples = 15000,plotsonly=False):
    """
    For a desired PPD test, read in importance sampled cosmosis chain
    set up PPD gaussian mixture, compute PPD value and create plots 
    of data vs ppd band and histogram of p-value. 

    Assume we have a chain originally run on data d
    put through importance sampler (KS) to save theory DVs for each sample
    evaluate comparison between that theory ensemble and 
    observables d' (dp=d prime). Can have d=dp for goodness of fit
          
    ARGUMENTS

    chainfile - path to chain with saved DV theory predcitions in 
                cosmosis importance sampler format
         -> Note that saved theory DV in IS chain file may not match DV used for
        intial chain posterior (e.g. could save 3x2pt predictions for 2x2pt chain)
    fitsfile - path to fitsfile where data measurements are. 
               This is also used to map data vector indices
               onto redshift bins and 2pt types (i.e. fits function appropriate
               for DV being considered)

    posterior_usecorrs - list of 2pt functions like ['xip','xim','gammat',wtheta'])
                 used to produce posterior for  original chain. 
                  Default is None, indicating we should do goodness-of-fit with
                  DV specified by test_usecorrs and test_scutini
                  
    posterior_scutini - ini file containing scale cuts used to generate input posterior

    test_usecorrs - Optional list of 2pt corrs that we want to assess PPD for
                     that are part of dtest (data whose fit we want to test)
                   Default is None, indicating we should do goodness-of-fit with
                     DV specified by posterior_usecorrs and posteior_scutini
                    Note: if test_ data is a subset of posterior_data; 
                     will use "goodness of fit" style calculation,
                     if it's disjoint will use conditional formulation of PPD
    test_scutini - path to ini file containing scale cuts specifying dtest. 
                  Optional; if None (default), is set to match posterior_scutini

    outfbase - string specifying base for output files (including directory path)
              with no file suffix ('.txt' and '.png' will be appended as
              appropriate for different dataproducts we want to save.

    legtitle - string used to label ppd run setup. Will show up as legend
             title in plots, and as part of header in data files. 
             E.g. "Redmagic shear-only goodness of fit" or
                  "Maglim PPD for 2x2pt given 3x2pt"
    
    twopt_section - header label for 2pt likelihood section in cosmosis files
             and thus in the chain header (probably can leave at default 2pt_like)

    do_bandplot - if True, will create data vector plot showing how measurements
             compare to PPD confidence interval. Will generate a plot file
             and a data file containing equivalent information. 

    do_deltappd- compute Delta_PPD for data given PPD distribution. Will save
             info both in a data file and in a histogram.  

    
    ppdlabel - labels ppd distribution band in plot (optional, default is "PPD MixGauss (mean, std)")
    datalabel - labels datapoints in plot  (optional, default is "data")
    
    ftype - filetype for plots. Suggest png for testing, pdf for paper-ready

    Npropsamples - number of samples from PPD Gaussian Mixture model to use
            when evaluating p-value. Default 15000 matches Cyrille's example code.

    plotsonly - assumes data has already been saved for ppd stats, just want to remake plots
    """
    if (not plotsonly) and (not HAVETF):
        raise ValueError("Can't run ppd calcs without tensorflow modules")

    if plotsonly:
        print("Remaking PPD plots for ",chainfile)
    else:
        print("Reading in IS samples from ",chainfile)
    print("  Evaluating PPD for posterior generated with",posterior_usecorrs," with scale cuts from",posterior_scutini)
    if test_usecorrs is None:
        print("  ...assessing goodness of fit.")
        test_usecorrs = posterior_usecorrs
        test_scutini = posterior_scutini
    else:
        if test_scutini is None:
            test_scutini = posterior_scutini
        print("  ...testing data",test_usecorrs," with scale cuts from",test_scutini)

    if not plotsonly:
        weights, dt_dat, dt_th, dt_cov = load_ppd_run(chainfile, fitsfile,  posterior_usecorrs, posterior_scutini, test_usecorrs, test_scutini, twopt_section)

        print("Getting PPD Gaussian mixture model")
        ppd = get_PPD_MixtureGaussian_helper(dt_th, dt_cov, weights)

    shortfits = os.path.split(fitsfile)[-1]
    shortchain = os.path.split(chainfile)[-1]
    plotheader = legtitle+'\ntest DV from: '+shortfits+'\nchain file: '+shortchain
    plt.title(plotheader,size=10)

    if do_bandplot:
        print("Plotting ppd datavector band")
        if plotsonly:
            plot_data_on_ppdband(None, None, outfbase, legtitle, ppdlabel, datalabel, ftype, fitsfile = fitsfile, corrs = test_usecorrs, scutini = test_scutini, readfromfile = True, chainfile=chainfile)
        else:
            plot_data_on_ppdband(dt_dat, ppd, outfbase, legtitle, ppdlabel, datalabel, ftype, fitsfile = fitsfile, corrs = test_usecorrs, scutini = test_scutini, chainfile=chainfile)

        # using data file info, make companion non-normed version
        #plot_data_on_ppdband(None, None, outfbase, plotheader, ppdlabel, datalabel, ftype, fitsfile = fitsfile, corrs = test_usecorrs, scutini = test_scutini, readfromfile = True, norm = False, chainfile=chainfile)

        # also make a non-normed version that shows negative datapoints if there are any
        #plot_data_on_ppdband(None, None, outfbase, plotheader, ppdlabel, datalabel, ftype, fitsfile = fitsfile, corrs = test_usecorrs, scutini = test_scutini, readfromfile = True, norm = False, chainfile=chainfile,showneg=True)

    if do_deltappd:
        print("Computing Delta_PPD")
        if not plotsonly:
            calc_ppd_metric(dt_dat, ppd, outfbase, plotheader, Nprobsamples,ftype=ftype)
        else:
            histplotf= outfbase+'.logp-hist.'+ftype
            histdatf = outfbase+'.logp-dat.txt'
            histdat = np.loadtxt(histdatf)
            logp_data = histdat[0]
            vol = histdat[1]
            err = histdat[2]
            vol_kde = histdat[3]
            logp_samples = histdat[5:]
            outf= outfbase+'.logp-hist.'+ftype
            kde = gaussian_kde(logp_samples)

            print("  Saving logP histogram to",histplotf)
            plot_logP_hist(logp_samples,logp_data, vol, err, vol_kde, histplotf, kde, plotheader)

#-------------------------
def get_PPD_MixtureGaussian_helper(loc, cov, weights):
    p = weights/np.sum(weights)
    p = p.astype(np.float32)
    loc = loc.astype(np.float32)
    cov = cov.astype(np.float32)
    return tfd.MixtureSameFamily(
        mixture_distribution=tfd.Categorical(probs=p),
        components_distribution=tfd.MultivariateNormalTriL(loc=loc, scale_tril=tf.linalg.cholesky(cov)))

#-------------------------
def load_ppd_run(chainfile, fitsfile,  posterior_usecorrs = None, posterior_scutini = None, test_usecorrs=None, test_scutini=None, twopt_section='2pt_like'):
    """
    Read info needed for PPD calc from cosmosis importance sampling chain run
    to save DV theory calcuulations

    assume we have a chain originally run on data dc 
    ("dcondition", called just "d" in Cyrille's notes/code)
    put through importance sampler to save theory DVs for each sample
    evaluate comparison between that theory ensemble and 
    observables dtest (dt, dprime in Cyrille's notes/code). 
    If test_usecorrs==None, sets dt=dc for goodness of fit

    posterior_usecorrs - list of 2pt functions we're conditioning on 
                         (aka, which were used to produce the input posterior
                          and so are part of dc)
    posterior_scutini - ini file containing scale cuts specifying conditional DV 
                         (aka, which were used to produce the input posterior)

    test_usecorrs - list of 2pt corrs (like ['xip','xim','gammat','wtheta'])
                  that are part of dtest (data whose fit we want to test)
    test_scutini - ini file containing scale cuts specifying dtest
    """
    if not os.path.isfile(chainfile):
        raise Warning("Can't open chain, skipping ppd calc for",chainfile)
        return
    pdict, plist = get_paramdict(chainfile,True) # read column headers

    dat = np.loadtxt(chainfile).astype(np.float32)
    
    # posterior sample weights
    if 'old_weight' in pdict.keys():
        print("  found column old_weight")
        weightind = pdict['old_weight']
        # old_weight is relevant here, not the importance sampling weight
        # since we want to weight based on the posterior of the original chain
        # (may have saved additional DV entries when running IS)
        weights =dat[:,weightind]
    elif 'old_log_weight' in pdict.keys(): # initial chain run with nautilius,
        print("  found column old_log_weight")
        # newer cosmosis versions will make this column when running IS on nautilus chains
        weightind = pdict['old_log_weight']
        weights = np.exp(dat[:,weightind])
    elif 'log_weight' in pdict.keys(): # initial chain was run with nautilus
        print("  no old weight column, check if there are two log_weight columns")
        # older versions of cosmosis will have two log_weight columns when running IS
        #    on nautilus chains. we want to use the leftmost one, which is how pdict
        #    is already set up
        if not plist.count('log_weight')>1:
            # if log_weight only shows up once and we didn't have old_log_weight
             # something weird has happened
            raise ValueError("beware, didn't find 'weight', 'old_log_weight', and only found 'log_weight' once")
        weightind = pdict['log_weight']
        weights = np.exp(dat[:,weightind])

    # normalize the weights!
    weights = weights/np.sum(weights)
    weights = weights.astype(np.float32) #should already be this type, but just make sure
    #print("checking weights - sum {}, mean {}, min {}, max{}".format(np.sum(weights),np.mean(weights),np.min(weights),np.max(weights)))
    
    # data vectors (data, theory, realizations)
    dvinds = []
    for p in plist:
        #print(p,pdict[p])
        if p.startswith('data_vector--2pt_theory'):
            dvinds.append(int(pdict[p]))
    dvinds = np.array(dvinds)
    dv_th = dat[:,dvinds] # [sample,dv entry]

    # sometimes there are rows where all the saved dv elements are nan
    # and posteiriors are -inf. This will get rid of them
    keeprows = np.where(np.isfinite(dv_th[:,0]))[0]
    dv_th = dv_th[keeprows,:]
    weights = weights[keeprows]

    # read full DV from fits file
    twopts,masks,dv,ang,b1,b2,cov = read_fits_file(fitsfile)

    # figure out what parts of that full DV were saved in the IS chain output
    saved_dv_mask = get_dv_mask_fromchain(chainfile,fitsfile)

    # need to figure out which parts of full DV in fits file correspond to d (data used for posterior)
    if posterior_scutini is not None:
        dc_scutdict = scutdict_fromini(posterior_scutini,twopt_section)
    else:
        print("No scale cut ini passed for d_posterior, assuming we should use all angular scales in IS chain file")
        dc_scutdict = get_scalecuts_fromchain(chainfile,twopt_section)
    if posterior_usecorrs is None:
        print("No posterior_usecorrs passed, assuming we should use all 2pt data in IS chain file")
        posterior_usecorrs = get_2ptsets_fromchain(chainfile,twopt_section)

    dc_dv_mask = get_datamask(fitsfile, posterior_usecorrs, dc_scutdict) #True for entries in full DV that are part of dc

    matchdv = test_scutini is None # d=dtest (dc=dt)
    if matchdv:
        dt_dv_mask = dc_dv_mask
    else: # need to figure out which parts of full DV in fits file correspond to dtest
        dt_scutdict = scutdict_fromini(test_scutini,twopt_section)
        dt_dv_mask = get_datamask(fitsfile, test_usecorrs, dt_scutdict) #True for entries in full DV that are part of dtest

    cov_tt = cov[dt_dv_mask,:][:,dt_dv_mask].astype(np.float32)# Cov entries for dtest
    dt_dat = dv[dt_dv_mask].astype(np.float32)           # DV meas from fits file corresponding to dtest

    # throw an error if there's a 1 in dt_dv_mask at an entry where saved_dv_mask is 0
    # (we need a theory prediction we didn't save)
    if np.any(np.logical_not(saved_dv_mask)*dt_dv_mask):
        raise ValueError("To do this calculation you need a data vector element not saved in chainfile (missing a test datapoint)")
    # which entries of saved_dv correspond to the dtest elements we want?
    dt_th = dv_th[:,dt_dv_mask[saved_dv_mask]].astype(np.float32) # Nsamples X Ndt

    # Use conditional setup for disjoint dataset if dt isn't a subset of dc
    #  otherwise we'll use the goodness-of-fit setup. 
    conditional = not np.all(dc_dv_mask[dt_dv_mask])

    if not conditional:
        return weights.astype(np.float32), dt_dat.astype(np.float32), dt_th.astype(np.float32),  cov_tt.astype(np.float32)
    else:
        # for conditional tests, we want to assess probability of dt given dc
        #  and need to account for covariance between the two disjoint datasets

        # notes for reference (wikipedia):
        # want observable x1 given x2, indep means mu1, mu2
        # mean(x1|x2) = mu1 + cov[12]*invcov[22](x2 - mu2)
        # cov(x1|x2) = cov[11] - cov[12]*invcov[22]*cov[21]
        #
        # in this code's notation
        # x1=dt_dat
        # x2=dc_dat
        # mean(x1|x2) = dt_th_cond (will depend on realization)
        # cov[11] = cov_tt
        # cov[22] = cov_cc
        # cov[12] = cov_tc
        # cov[21] = cov_ct
        # cov(x1|x2) = dt_cov_cond

        if np.any(np.logical_not(saved_dv_mask)*dc_dv_mask):
            raise ValueError("To do this calculation you need a data vector element not saved in chainfile (missing a conditional datapoint)")

        cov_cc = (cov[dc_dv_mask,:])[:,dc_dv_mask].astype(np.float32) #Ndc x Ndc
        icov_cc = np.linalg.inv(cov_cc)
        #icov_cc = tf.linalg.inv(cov_cc)

        cov_ct = cov[dc_dv_mask,:][:,dt_dv_mask].astype(np.float32) #Ndc x Ndt
        cov_tc = cov_ct.T #cov[dt_dv_mask,:][:,dc_dv_mask].astype(np.float32) #Ndc x Ndt # equals  cov_ct.T
        # cyrill's notation: cov_ddprime = cov_tc, inv_cov_d = icov_cc
        
        dc_dat = dv[dc_dv_mask].astype(np.float32) 
        dc_th = dv_th[:,dc_dv_mask[saved_dv_mask]].astype(np.float32) # Nsamples X Ndc

        # conditional mean
        dt_th_cond = dt_th + np.matmul(cov_tc, np.matmul(icov_cc, (dc_dat - dc_th).T)).T

        # conditional cov
        dt_cov_cond = cov_tt - np.matmul(cov_tc, np.matmul(icov_cc, cov_tc.T)) # note the covariance is always fixed, even here !

        cov_chol = tf.linalg.cholesky(dt_cov_cond)

        return weights.astype(np.float32), dt_dat.astype(np.float32), dt_th_cond.astype(np.float32), dt_cov_cond.astype(np.float32)

#============================================================
# Helpers for parsing chains
#============================================================
#-------------------------
def get_nsample(filename):
    """
    For multinest or polychord files, the last line tells you how many lines
    to keep. This function pulls that info from the file. 

    This info WILL NOT be in the bottom of the IS file lines with DV info for PPD;
    you need to get it from the original chain file. 
    (n.b. may be unnecessary, as chain files generally don't have 
     extra lines that need to be removed)
    """
    nsamples=None
    fi =  open(filename,"r")
    for ln in fi:
        if (ln.startswith("nsample=")) or (ln.startswith('#nsample=')):
            nsamples = int(ln.replace('nsample=','').replace('#',''))
            break
    fi.close()
    return nsamples
#-------------------------
def get_paramdict(filename,getlist=False):
    """
    Reads in first line of chain file to make dictionary
    to translate parameter names to column indices. 
    """
    f = open(filename,'r')
    firstline = f.readline().split()

    f.close()
    pdict = {}
    plist = []

    for i in range(len(firstline)):
        s = firstline[i].replace('#','')
        plist.append(s.lower())
            
        if s.lower() not in pdict.keys():
            # if we ran with IS and a parameter is saved as extra output
            # for both original and IS run, it may appear twice (doesn't seem to happen with cosmosis v>=2)
            # we want to use the leftmost one (that's the one that uses the IS calcs)
            # the original extra output and sampler output goes to the right of that in cosmosis v1,
            # thanks to Noah Weaverdyck and Otavio Alves for catching this 
            pdict[s.lower()]=i
            
    if getlist:
        return pdict, plist
    else:
        return pdict

#-------------------------
def get_defaults_fromchain(chainfile):
    """
    From chain header, get default parameters and return as a dictionary
    
    Note that if you run this on importance sampler output, default params will reflect IS run,
    not cuts used in intial input chain. 
    """
    secstart = "## ["
    checkfor="## [DEFAULT]"
    insec=False

    outdict = {}
    
    fi =  open(chainfile,"r")
    for ln in fi:
        if ln.startswith(checkfor):
            insec=True
        elif ln.startswith(secstart):
            if insec: # was reading desired sec, now it's done
                break
        elif insec: # we're in the desired section
            ln = ln.replace('##','').strip()
            if '=' in ln:
                words = ln.split('=')
                outdict[words[0].strip()]=words[1].strip()
                outdict[words[0].strip().upper()]=words[1].strip()
                # ^ looks like from chain files all keys are lowercase
                #   but references e.g. in 2pt_like are sometimes
                #   in uppercase, so let's add both versions to dictionary
        else: # haven't found desired section yet
            continue

    fi.close()

    return outdict

#-------------------------
def get_scalecuts_fromchain(chainfile,twopt_section='2pt_like'):
    """
    From chain header, read in scale cuts

    format will match scutdict format of scutdict_fromini

    Note that if you run this on importance sampler output, scale cuts will reflect IS run,
    not cuts used in intial input chain. 
    """
    secstart = "## ["
    checkfor="## [{}]".format(twopt_section)
    insec=False

    scutdict = {}
    
    fi =  open(chainfile,"r")
    for ln in fi:
        if ln.startswith(checkfor):
            insec=True
        elif ln.startswith(secstart):
            if insec: # was reading desired sec, now it's done
                break
        elif insec: # we're in the desired section
            if 'angle_range' in ln:
                ln = ln.replace('##','').strip()
                entries = ln.split()
                corrtype = entries[0].split('_')[2] #2pt name
                if corrtype not in scutdict.keys():
                    scutdict[corrtype]={}
                    
                b1,b2 = entries[0].split('_')[-2:] # bin numbers
                b1=int(b1)
                b2=int(b2)
                angmin = float(entries[-2])
                angmax = float(entries[-1])
                scutdict[corrtype][(b1,b2)] = (angmin,angmax)

            elif 'cut_' in ln:
                ln = ln.replace('##','').strip()
                entries = ln.split('=')
                corrtype = entries[0].split('_')[1].strip() #2pt name
                if corrtype not in scutdict.keys():
                    scutdict[corrtype] = {}
                binpairs = entries[1].split()
                for bp in binpairs:
                    p = bp.split(',')
                    b1= int(p[0])
                    b2= int(p[1])
                    scutdict[corrtype][(b1,b2)] = (999.0,999.0) # cut all measurements!
        else: # haven't found desired section yet
            continue

    fi.close()

    return scutdict
#-------------------------
def scutdict_fromini(scutini, twopt_section='2pt_like'):
    """
    given ini file containing scale cuts, read them in and 
    store them in a dictionary

    """
    if scutini is None:
        return {}
    f = open(scutini)
    scutdict = {}
    section=None
    for line in f:
        line = line.strip()
        if not line: #empty line
            continue
        elif line.startswith(';') or line.startswith('#'): #comment
            continue
        elif line.startswith('['): #section header
            section = line.replace('[','').replace(']','').lower()
        elif section!=twopt_section: #we're just looking for a line in the pipeline section
            continue
        else: # non-comment, non-include line in 2pt_like section
            if line.startswith('angle_range_'): # scale cut line
                entries = line.split()
                corrtype = entries[0].split('_')[2] #2pt name
                if corrtype not in scutdict.keys():
                    scutdict[corrtype]={}
                    
                b1,b2 = entries[0].split('_')[-2:] # bin numbers
                b1=int(b1)
                b2=int(b2)
                angmin = float(entries[-2])
                angmax = float(entries[-1])
                scutdict[corrtype][(b1,b2)] = (angmin,angmax)
            elif item[0].startswith('cut_'): # full bin combos are cut
                nameparts = item[0].split('_')
                corrtype = nameparts[1].strip() #2pt name
                if corrtype not in scutdict.keys():
                    scutdict[corrtype] = {}
                binpairs = item[1].split()
                for bp in binpairs:
                    p = bp.split(',')
                    b1= int(p[0])
                    b2= int(p[1])
                    scutdict[corrtype][(b1,b2)] = (999.0,999.0) # cut all measurements!
    f.close()

    return scutdict
#-------------------------
def mask_from_scutdict(scutdict,fitsfile=None,typemask=None,angle=None,bin1=None,bin2=None,cut_wtheta_cross=True):
    """
    If scale cuts have been read in from ini file into dictionary with format
    scutdict[corrtype][(b1,b2)] = (angmin,angmax)
    where corrtype = xip, xim (however things are labeled in the ini liness), b1 and b2 are inds
    and angmin, angmax are floats

    and given fits file (or DV<->label arrays from previously reading fits file),
    generate a boolean array which is True for DV entries kept after cuts, False otherwise

    if scutdict is empty (happens if scutini=None in scutdict_fromini), will return all Trues

    if cut_wtheta_cross, assumes theory DV in pipeline has no crossbins for wtheta
    even if they are present in the fits file
    """
    
    if (fitsfile is None ) and (typemask is None):
        raise ValueError("Need either fits file or index labels to get scut mask from dictionary")
    elif fitsfile is not None:
        # if given the fits file, readi n stuff so we can figure out indexing
        twopts,masks,fulldv,angle,bin1,bin2,cov = read_fits_file(fitsfile,cut_wtheta_cross=cut_wtheta_cross)
        
    N = bin1.size
    output = np.ones(N,dtype=bool)
    xmid = angle[:,1] # midpoint of angular bins; this is what is
                      # checked for evaluating scale cuts in 2pt_like

    keep = np.ones(N,dtype=bool)# we'll use this if we want to remove wtheta cross bin elements
    
    for corrtype in scutdict.keys():
        incorrtype=typemask[corrtype]
        for binpair in scutdict[corrtype].keys():
            b1,b2=binpair
            inbinpair = incorrtype*(bin1==b1)*(bin2==b2)
            if not np.any(inbinpair):
                continue
            
            angmin = scutdict[corrtype][binpair][0]
            angmax = scutdict[corrtype][binpair][1]
            outsiderange = (xmid<angmin)|(xmid>angmax)# True if outside range specified in scale cut ini

            cut = outsiderange*inbinpair
            output[cut]=False

            if cut_wtheta_cross and corrtype=='wtheta':
                if b1!=b2:
                    keep[inbinpair]=False
        
    return output[keep]


#-------------------------
def get_2ptsets_fromchain(chainfile,twopt_section='2pt_like',defaults=None):
    """
    Figure out which 2pt functions were used in likelihood
    from the chain header

    Note that if you run this on importance sampler output,2pt list  will reflect IS run,
    not those used in intial input chain. 
    """
    # look for "## data_sets = %(2PT_DATA_SETS)s"

    secstart = "## ["
    checkfor="## [{}]".format(twopt_section)
    insec=False

    twoptdat = None

    if not os.path.isfile(chainfile):
        print("    Trying to find 2pt corrs from chain, but can't find file:",chainfile)
        return None
    
    fi =  open(chainfile,"r")
    for ln in fi:
        if ln.startswith(checkfor):
            insec=True
        elif ln.startswith(secstart):
            if insec: # was reading desired sec, now it's done
                break
        elif insec: # we're in the desired section
            ln = ln.replace('##','').strip()
            if ln.startswith('data_sets'):
                words = ln.split('=')
                twoptdat = words[1]
                break
        else: # haven't found desired section yet
            continue

    fi.close()

    # is there something from the default section here?
    if '%(' in twoptdat:
        if defaults is None:
            defaults = get_defaults_fromchain(chainfile)
        twoptdat = twoptdat%defaults

    if twoptdat is not None:
        twoptdat = twoptdat.split()

    return twoptdat

#-------------------------
def get_dv_mask_fromchain(chainfile,fitsfile,twopt_section='2pt_like'):
    """
    Read chain header to figure out 2pt data sets and scale cuts
    Use this to associate extra output data_vector entries 
    with 2pt types, redshift bins, angles, etc

    return array of bools that are true for which entries in the fits file DV array
    (in same order as cov entries) correspond to what's saved in the IS chain file
    """

    scutdict = get_scalecuts_fromchain(chainfile,twopt_section)
    usecorrs = get_2ptsets_fromchain(chainfile,twopt_section)

    totalmask = get_datamask(fitsfile,usecorrs,scutdict)
    return totalmask

#-------------------------
def get_datamask(fitsfile,usecorrs=['xip','xim','gammat','wtheta'],scutdict={},cut_wtheta_cross=True):
    #print(">> ",fitsfile,usecorrs,scutdict)
    twopts,tempmasks,dv,ang,b1,b2,cov = read_fits_file(fitsfile,cut_wtheta_cross=cut_wtheta_cross)

    if usecorrs is None: # will happen if plotting script is run locally but chain isn't here
        usecorrs =['xip','xim','gammat','wtheta']

    # pick out array entries used in likelihood
    corrmask = np.zeros(b2.size,dtype=bool) # 1's for 2pt functions we're using
    for corr in usecorrs:
        corrmask[tempmasks[corr]]=True
    scutmask = mask_from_scutdict(scutdict,typemask=tempmasks,angle=ang,bin1=b1,bin2=b2)
    totalmask = corrmask*scutmask # one for points to keep, false otherwise

    return totalmask

#---------------------------------------------------------
def read_fits_file(fname,covname='COVMAT',justdv=False,cut_wtheta_cross=True):
    """
    if justdv, only return fulldv. Otherwise, returns
    DV along with arrays labeling the bin number, two pt types
    """
    fname = os.path.expandvars(fname)
    if not os.path.isfile(fname):
        print("Can't find fits file at",fname)
    else:
        pass
        #print("Reading",fname)
    datdict = {}
    fitsdat = fits.open(fname)
    
    # list of 2pt functions we have in the file
    twopts =[]
    startind = {} # starting index for 2PCFS in concatenated data evector
    endind = {}   # ending index
    masks = {}
    
    # pull info about about ordering and labels from the cov matrix table
    tab = fitsdat[covname]
    N = tab.header['NAXIS1'] #number of datapoints to hang onto

    cov = tab.data
    fulldv = np.nan*np.ones(N) # will hold concatenated DV
    angle = np.nan*np.ones((N,4)) # columns are angle index, midpoint, min, max
    bin1 = np.nan*np.ones(N) # z bin index 1
    bin2 = np.nan*np.ones(N) # z bin index 2

    keep = np.ones(N,dtype=bool) # for handling cut_wtheta_cross

    i=0
    allinds = np.arange(N)
    while i<10: #if we have more than 10 kinds of 2pt funcs need to adjust this
        if 'STRT_{}'.format(i) in tab.header:
            ytype = tab.header['NAME_{}'.format(i)]
            twopts.append(ytype)
            startind[ytype]=tab.header['STRT_{}'.format(i)]
            if 'STRT_{}'.format(i+1) in tab.header:
                endind[ytype] = tab.header['STRT_{}'.format(i+1)]
                i+=1
            else:
                endind[ytype] = N
                break
        else:
            break
    
    # now collect DV info
    for ytype in twopts:
        i1 = startind[ytype]
        i2 = endind[ytype]
        mask = np.zeros(N,dtype=bool)
        mask[i1:i2] = 1
        masks[ytype]=mask
        
        # Pull out relevant info from table
        tab = fitsdat[ytype]
        tab_y = tab.data['VALUE']
        fulldv[mask]=tab_y
        if not justdv:
            tab_bin1 = tab.data['BIN1']
            tab_bin2 = tab.data['BIN2']
            tab_angind = tab.data['ANGBIN']
            tab_ang = tab.data['ANG'] #will be in arcmin
            tab_angmin = tab.data['ANGLEMIN'] #will be in arcmin
            tab_angmax = tab.data['ANGLEMAX'] #will be in arcmin

            bin1[mask] = tab_bin1.astype(int)
            bin2[mask] = tab_bin2.astype(int)
            angle[mask,0] = tab_angind.astype(int)
            angle[mask,1] = tab_ang
            angle[mask,2] = tab_angmin
            angle[mask,3] = tab_angmax

        if cut_wtheta_cross and ytype=='wtheta':
            # find indices for wtheta cross bins
            keep[(bin1!=bin2)*mask] = False

            

    
    fitsdat.close()

    if justdv:
        return fulldv[keep]
    else:
        bin1=bin1.astype(int)[keep]
        bin2=bin2.astype(int)[keep]
        for ytype in twopts:
            masks[ytype] = masks[ytype][keep]
        return twopts,masks,fulldv[keep].astype(np.float32),angle[keep].astype(np.float32),bin1,bin2,(cov[keep,:][:,keep]).astype(np.float32)

#============================================================
# metric calculation
#============================================================
def calc_ppd_metric(data, ppd, outbase='ppdtest', runlabel = 'PPD test', Nprobsamples = 15000, dokde=True, savedat=True, plotdat=True, ftype='png', doprogbar =True):

    # To compute the metric, we need to sample the PPD and estimate their density to compare it to that of the data

    # First, we define a tf.func
    @tf.function
    def sample_log_prob(n):
        return ppd.log_prob(ppd.sample(n))

    # Then run it!
    if doprogbar: # show progress bar as we generate samples
        logp_samples = []
        # trange is just a range iterator that makes a progress bar
        Nblocks = 1000
        setsize=int(Nprobsamples/Nblocks)
        for _ in trange(Nblocks):
            logp_samples.append(sample_log_prob(setsize).numpy())
            # 100 times sample 150 log probabilities from distribution
            
            # for shear-only, takes about 10s per iteration
        logp_samples = np.concatenate(logp_samples) # will ahve 15000 log probs in it by default
    else: #just sample everything at once
        # note that trying to run with this setting on perlmutter
        # caused some OOM errors 
        logp_samples = sample_log_prob(Nprobsamples).numpy()
        
    logp_data = ppd.log_prob(data).numpy()
    
    # We can now compare it to the density at the observed data
    k = np.sum(logp_samples < logp_data)
    n = len(logp_samples)
    vol = k / n
    err = 1.96*np.sqrt(vol*(1.-vol)/n)
    # ^ confidence interval for binomial disribution
    # for a binomial distribuition (estiamte of sample proprotion vol) has standard deviation np.sqrt(vol*(1.-vol)/n)
    # the 1.96 coefficient gives the 95% confidence interval

    if dokde:
        # slightly more accurate (but slower) volume estimate
        kde = gaussian_kde(logp_samples)
        vol_kde = kde.integrate_box_1d(-np.inf, logp_data)
    else:
        vol_kde = None

    if savedat:
                        
        outf= outbase+'.logp-dat.txt'
        outdat = np.nan*np.ones(logp_samples.size+5)
        header = runlabel+"\nRows are logp_data,  posterior volume Delta_PPD with logp<logp_data, 95% CI for sampling error on Delta_PPD,  KDE estimate of Delta_PPD, nan, then logp samples"
        outdat[0] = logp_data
        outdat[1] = vol
        outdat[2] = err
        if vol_kde is not None:
            outdat[3] = vol_kde
        
        outdat[5:] = logp_samples
        

        print("Saving logp sampling data to",outf)
        np.savetxt(outf,outdat,header=header)

    if plotdat:
        outf= outbase+'.logp-hist.'+ftype
        print("  Saving logP histogram to",outf)
        plot_logP_hist(logp_samples,logp_data, vol, err, vol_kde, outf, kde, runlabel)

    return  vol, err, vol_kde
#============================================================
# Plotting 
#============================================================
def plot_data_on_ppdband(data=None,ppd=None,outbase=None,legtitle=None,ppdlabel=None,datalabel=None,ftype='png',savedat=True,readfromfile=None,norm=True,fitsfile=None,scutini=None,corrs = ['xip','xim','gammat','wtheta'],chainfile='',scalebytheta=True,showneg=False):
    """
    Given array of data measurements and ppd object that was produced by get_PPD_MixtureGaussian_helper
    Plot ppd info as band showing mean, stdev from ppd mixture
    and plot datpoints as red points
    with data vector entry index on x axis

    If readfromfile is a string filename, assume that it is an output
    from a previous run of plot_data_on_ppdband with savedat=True, ignore all the other 
    arguments, read in the data and remake the plot

    If norm==True, plot (data-ppdmean)/ppdstdev

    If fitsfile, scutini, corrs are passed, uses that to label data vector sections
    
    scalebytheta - if True and fitsfile, scutini, and corrs are passed
                   will scale unnormed DV plot entries by theta
    showneg- for non-normed dv plot, use symlog to show negative points (otherwise 
                  will only show positives)
    """
    if readfromfile is not None:
        if type(readfromfile)==bool:
            if readfromfile: # is true, set to filename
                readfromfile = outbase+'.ppdband-dat.txt'

        i, data, m, s, plotheader, datalabel, ppdlabel = read_ppdband_data(readfromfile)
        outbase = readfromfile.split('.')[0]
        savedat=False
    else:
        shortfits = os.path.split(fitsfile)[-1]
        shortchain = os.path.split(chainfile)[-1]
        plotheader = legtitle+'\ntest DV from: '+shortfits+'\nchain file: '+shortchain
        
        m = ppd.mean().numpy()
        s = ppd.stddev().numpy()
        i = np.arange(m.size)
        
    ndata = data.size
    width = max(ndata/50,6)
    plt.figure(figsize=(width, 4))

    if ppdlabel is None:
        ppdlabel = "PPD MixGauss (mean, std)"
    if datalabel is None:
        datalabel = "Data"

    if norm:
        outname = outbase+'.ppdband-norm.'+ftype
        plt.plot(i,np.zeros(i.size), c='orange', label=ppdlabel)
        
        plt.fill_between(i, -1, 1, color='orange', alpha=0.4, zorder=-9)
        plt.axhline(-2,color='orange',alpha=0.4,ls='--',zorder=-9)
        plt.axhline(2,color='orange',alpha=0.4,ls='--',zorder=-9)
        plt.scatter(i, (data - m)/s, c='r', s=3,  label=datalabel, zorder=5)

        maxdaty=np.max((data-m)/s)
        mindaty=np.min((data-m)/s)
        plt.ylim((mindaty*1.2,maxdaty*1.3))
        plt.ylabel('(Data vector - PPD mean)/(PPD stdev)')
    else:
        scaleby=1
        ylabel = "Data vector"
        if scalebytheta:
            if fitsfile is None:
                print("Can't scale by theta in DV plot if we don't have the fits file! Plotting without rescaling.")
            else:
                ylabel = r"Data vector $\times\theta$"
                twopts,typemasks,dv,ang,b1,b2,cov = read_fits_file(fitsfile)
                scutdict = scutdict_fromini(scutini) # will be empty dict if scutini is None
                totalmask = get_datamask(fitsfile,corrs,scutdict)
                # ^mask applied to full DV in fits file to set up test
                # get rid of info for unplotted points
                ang = ang[totalmask] # columns are angle index, midpoint, min, max
                scaleby = ang[:,1]

                # we'll also scale by some additional factors to get the
                # 2pt functions to be  more comparable on a plot
                for c in corrs:
                    typemasks[c] = typemasks[c][totalmask] # get points that are plotted
                    if c=='wtheta':
                        continue
                    elif c=='gammat':
                        scaleby[typemasks[c]]*=1.e2
                    else:
                        scaleby[typemasks[c]]*=1.e4
                
        if showneg:
            outname = outbase+'.ppdband-showneg.'+ftype
        else:
            outname = outbase+'.ppdband.'+ftype
        #plt.semilogy(m*scaleby, c='orange', label=ppdlabel)
        plt.plot(m*scaleby, c='orange', label=ppdlabel)

        
        plt.fill_between(i, (m - s)*scaleby, (m + s)*scaleby, color='orange', alpha=0.4, zorder=-9)
        plt.scatter(i, data*scaleby, c='r', s=3, label=datalabel, zorder=5)
        plt.ylabel(ylabel)

        maxdaty=np.max(data*scaleby)

        if showneg:
            mindaty=np.min(data*scaleby )
        else:
            mindaty=np.min(data[data>0]*scaleby[data>0])

        if mindaty>0:
            plt.ylim((mindaty/3.,maxdaty*5))
        else:
            plt.ylim((mindaty*3.,maxdaty*5))

        if showneg:
            if scalebytheta:
                minabs=1.e-2
            else:
                minabs=1.e-5
            #minabs = 1.e-5#np.min(np.fabs(data*scaleby)) # switch to linear close to zero
            plt.yscale('symlog',linscale=0.1,linthresh=10**(int(np.log10(minabs))))
            plt.axhline(0,ls='-',c='gray',lw=0.5)
        else:
            plt.yscale('log')
    
    plt.legend(frameon=True,fontsize=8,loc='lower right')
    
    plt.title(plotheader,size=10)
    plt.xlabel('Data vector index')
    plt.xlim(-2,ndata+2)

    if (fitsfile is not None):#  and (scutini is not None):
        # shade different redshift bins for readability
        g1 = '#f8f8f8'
        g2 = '#e8e8e8'
        g3 =  '#e0e0e0'#'#d9d9d9'#'#bdbdbd'
        
        if scalebytheta and (not norm): # add scaleby to the labels for clarity
            corrdict = {'xip':r'$10^4\theta\xi_+(\theta)$','xim':r'$10^4\theta\xi_-(\theta)$','gammat':r'$10^2\theta\gamma_T(\theta)$','wtheta':r'$\theta w(\theta)$'}
        else: #just label the different 2pt functions
            corrdict = {'xip':r'$\xi_+(\theta)$','xim':r'$\xi_-(\theta)$','gammat':r'$\gamma_T(\theta)$','wtheta':r'$w(\theta)$'}

        # let's add some labels to DV indices!
        twopts,typemasks,dv,ang,b1,b2,cov = read_fits_file(fitsfile)
        scutdict = scutdict_fromini(scutini) # will be empty dict if scutini is None
        totalmask = get_datamask(fitsfile,corrs,scutdict)
        # ^mask applied to full DV in fits file to set up test

        # get rid of info for unplotted points
        for c in corrs:
            typemasks[c] = typemasks[c][totalmask] # get points that are plotted
            #plt.axvline(firstind - 0.5, lw=2, color=gdark, zorder=-10)
        ang = ang[totalmask]
        b1 = b1[totalmask]
        b2 = b2[totalmask]

        # now shade regions and add labels
        for j,c in enumerate(corrs):
            #print(" >> on ",j,c)
            cinds = np.where(typemasks[c])[0]
            ymin,ymax=plt.ylim()
            plt.annotate(corrdict[c],xy=(cinds[0],ymax),xytext=(2,-2),fontsize=12,color='gray',ha='left',va='top',textcoords='offset points',zorder=-8)
            if j%2:
                #fill in background color for every other corr
                plt.axvspan(cinds[0]-0.5,cinds[-1]+0.5,color=g2,zorder=-11)
                bincol = g3
            else:
                bincol = g1
            # fill in bands for every bin combo
            binpair=0
            newpair=True
            for k in cinds:
                if newpair:
                    bstart = k
                    b1v = b1[k]
                    b2v = b2[k]
                    plt.annotate('{}\n{}'.format(b1v,b2v),xy=(bstart,ymax),xytext=(1,-18),fontsize=6,color='gray',ha='left',va='top',textcoords='offset points',zorder=-8)
                    newpair = False
                if (k==cinds[-1]) or (b1[k]!=b1[k+1]) or (b2[k]!=b2[k+1]): #last
                    bend=k
                    if binpair%2: #fill in every other bin pair 
                        plt.axvspan(bstart-0.5,bend+0.5,color=bincol,zorder=-10)
                    newpair = True
                    binpair+=1
                
                


    print("  Saving PPD band figure to",outname)
    plt.tight_layout()
    plt.savefig(outname,dpi=300)

    if savedat:
        outname = outname.replace('.'+ftype,'-dat.txt').replace('-norm','')
        shortfits = os.path.split(fitsfile)[-1]
        shortchain = os.path.split(chainfile)[-1]
        #header = legtitle+'\ntest DV from: '+shortfits+'\nchain file: '+shortchain+'\ndatalabel: '+datalabel+'\n ppdlabel: '+ppdlabel+"\nDVindex, data, ppd_mean, ppd_stdev"
        header = plotheader+'\ndatalabel: '+datalabel+'\n ppdlabel: '+ppdlabel+"\nDVindex, data, ppd_mean, ppd_stdev"
        outdat = np.nan*np.ones((ndata,4))
        outdat[:,0] = np.arange(ndata)
        outdat[:,1] = data
        outdat[:,2] = m
        outdat[:,3] = s
        print("  Saving PPD band data to",outname)
        np.savetxt(outname,outdat,header=header)


def read_ppdband_data(fname):
    """
    Given filename where data from plot_data_on_ppdband was saved
    """
    indat = np.loadtxt(fname)
    i = indat[:,0]
    data = indat[:,1]
    m = indat[:,2]
    s = indat[:,3]

    print("Reading in PPD DV bands from file:",fname)
    # read the header line

    f = open(fname)
    legtitlelines = []
    for testline in f:
        line = testline.replace('#','').strip()
        if 'datalabel:' in line:
            datalabel = line.split(':')[1].strip()
        elif 'ppdlabel:' in line:
            ppdlabel = line.split(':')[1].strip()
            # this is the last header line we want to read
            break
        else:
            legtitlelines.append(line)

    legtitle = '\n'.join(legtitlelines) #can handle multi-line labels
    
    f.close()

    print('    legtitle',legtitle)
    print('    datalabel',datalabel)
    print('    ppdlabel',ppdlabel)
    return i, data, m, s, legtitle, datalabel, ppdlabel

#-------------------------
def read_logP_hist_dat(fname):
    """
    Given filename where data from used for plot_logP_hist was saved
    read in that data
    """
    f = open(fname)
    runlabel = f.readline().replace("#",'').strip()
    f.close()
    
    dat = np.loadtxt(fname)
    logp_data = dat[0]
    vol = dat[1]
    err = dat[2]
    vol_kde = dat[3]
    logp_samples = dat[5:]
    kde = gaussian_kde(logp_samples)

    return logp_samples,logp_data, vol, err, vol_kde, kde, runlabel
    
#-------------------------
def plot_logP_hist(logp_samples,logp_data, vol, err, vol_kde, outf, kde=None, legtitle=''):
    
    plt.figure(figsize=(3.5,3))
    xx = np.linspace(logp_samples.min(), logp_samples.max(), 100)
    if kde is not None:
        plt.plot(xx, kde(xx), c='orange')
        
    plt.hist(logp_samples, bins='auto', color='orange', histtype='step', density=True)

    plt.axvline(logp_data, label=f'$\Delta_{{PPD}}={vol:.3f} \pm {err:.3f}\ ({vol_kde:.3f})$')
    plt.xlabel('$\log P$')
    plt.yticks([])
    plt.legend(frameon=False,loc='upper right',fontsize=10)#title=legtitle)
    ymin,ymax = plt.ylim()
    plt.ylim((ymin,ymax*1.1))
    plt.title(legtitle,size=8)
    print("Saving logp hist to plot",outf)
    plt.tight_layout()
    plt.savefig(outf,dpi=300)
    plt.clf()
    plt.close()
    
#============================================================
def main():

    """
    Calls for testing performance of functions above
    """
    print("See run_ppd_calcs_example.py for examples of how to use the functions in this file.")
    
    # Original script had Jessie's initial code testing functions,
    # but they're all specific to her NERSC workspace
    # and too messy to be helpful, so they've been removed. 


#==================================================================
if __name__=="__main__":
    main()

    
