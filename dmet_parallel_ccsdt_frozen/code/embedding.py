#!/usr/bin/python

import numpy as np
import scipy.linalg as sla


def construct_bath (dm, impurity_idx, nBath, threshold=None):
    if threshold is None:
        threshold = 1.0e-12

    non_imp  = ~(impurity_idx)
    print("non_imp", non_imp)
    embed_dm = dm[np.ix_(non_imp,non_imp)]
    print("dm", dm)
    print("np.ix(nonimp,nonimp)", np.ix_(non_imp,non_imp))
    print("embed_dm", embed_dm)

    nImp   = np.count_nonzero(impurity_idx)
    nTotal = len(impurity_idx)
    print("nImp, nTotal", nImp, nTotal)

    evals, evecs = sla.eigh(-embed_dm)
    idx = evals.argsort()
    evals = -evals[idx]
    evecs = evecs[:,idx]
    print("idx, evals, evecs", idx, evals, evecs)


    core_  = evals > 1.0-threshold
    virt_  = evals < threshold
    tokeep = np.logical_and(~(core_), \
                            ~(virt_))
    nvirt  = np.count_nonzero(virt_)

    ncore  = np.count_nonzero(core_)
    nkeep  = np.count_nonzero(tokeep)
    print("core, virt, tokeep, ncore, nkeep", core_, virt_, tokeep, ncore, nkeep)

    print ( "construct_bath :: original bath eigenvals: " )
    print ( evals[tokeep] )

    if nkeep < nBath:
        print ( "constructbath :: throwing out", \
            nBath-nkeep, "orbitals" )

    if nBath < nkeep:
        ic = 0
        iv = 1
        ncore_ = ncore
        nvirt_ = nvirt
        evalst = evals[tokeep]
        for k in range(nkeep-nBath):
            if ( 1.0-evalst[ic] < evalst[-iv] ):
                tokeep[ncore+ic] = False
                core_[ncore+ic]  = True
                ic += 1
                ncore_ += 1
            else:
                tokeep[ncore+nkeep-iv] = False
                virt_[ncore+nkeep-iv]  = True
                iv += 1
                nvirt_ += 1
        ncore = ncore_
        nvirt = nvirt_
        del k, iv, ic, evalst
        del ncore_, nvirt_
        print ( "construct_bath :: trimmed bath eigenvals: " )
        print ( evals[tokeep] )

    nBath = min(nkeep, nBath)
    cf = np.zeros((nTotal,nTotal,), dtype=float, order='F')

    # impurity orbitals first
    # then active bath
    # then core
    # then virtual
    cf[impurity_idx,:nImp]        = np.eye(nImp)
    cf[non_imp,nImp:nImp+nBath]   = evecs[:,tokeep]

    if ncore > 0:
        cf[non_imp,nImp+nBath:-nvirt] = evecs[:,core_]
    if nvirt > 0:
        cf[non_imp,-nvirt:]           = evecs[:,virt_]

    core_label = np.zeros((nTotal,), dtype=bool)
    core_label[nImp+nBath:-nvirt] = True
    print ("core lable", core_label)
    print("cf", cf)


    assert (np.allclose(np.eye(nTotal), np.dot(cf.T, cf)))
    return cf, nBath, core_label


def embedding (dm, impurity_idx, threshold=None, \
               transform_imp='hf'):

    if transform_imp is not None:
        assert (transform_imp in ['hf'])

    nTotal = len(impurity_idx)
    nImp   = np.count_nonzero(impurity_idx)
    nBath  = nImp
    print("imp index", impurity_idx)
    print("ntotal, nimp, nbath", nTotal, nImp, nBath)

    if nImp == nTotal:
        nBath = 0
        core_lab = np.zeros_like(impurity_idx)
        loc2dmet = np.eye(nImp)
    else:
        loc2dmet, nBath, core_lab = \
            construct_bath (dm, impurity_idx, nBath, threshold)
    ncore = np.count_nonzero(core_lab)
    nact  = nImp + nBath
    print("ncore, nact", ncore, nact)

    # organize orbitals
    cf = np.empty_like(loc2dmet)

    # .. core ..
    cf[:,:ncore] = loc2dmet[:,core_lab]
    print("cf[:,:ncore]",cf[:,:ncore], cf)

    # .. active (impurity+bath) ..
    # == define which active orbitals are impurity ==
    if transform_imp == None:
        cf[:,ncore:ncore+nImp] = loc2dmet[:,:nImp]
        cf[:,ncore+nImp:ncore+nact] = loc2dmet[:,nImp:nact]
        ImpOrbs = np.arange(nImp, dtype=int)
    elif transform_imp == 'hf':
        cf[:,ncore:ncore+nImp] = loc2dmet[:,:nImp]
        cf[:,ncore+nImp:ncore+nact] = loc2dmet[:,nImp:nact]
        cf_ = cf[:,ncore:ncore+nact]
        T   = np.dot(cf_.T, np.dot(dm, cf_))

        evals, evecs = sla.eigh(-T)
        idx = evals.argsort()
        evals = -evals[idx]
        evecs = evecs[:,idx]
        del T, evals, idx

        cf_[:,:] = np.dot(cf_, evecs)
        ImpOrbs  = evecs[:nImp,:].T
        del evecs, cf_

    core_lab[:nact] = True
    core_neg = ~(core_lab)

    # .. virtual ..
    cf[:,ncore+nact:] = loc2dmet[:,core_neg]

    del core_lab, core_neg
    assert (np.allclose(np.eye(nTotal), np.dot(cf.T, cf)))
    return cf, ncore, nact, ImpOrbs
