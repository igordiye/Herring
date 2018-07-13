#!/usr/bin/python

import numpy as np
import scipy.linalg as sla

from pyscf import gto, scf, fci, ao2mo


def solve (mol, nel, cf_core, cf_gs, ImpOrbs, chempot=0., n_orth=0):
    # cf_core : core orbitals (in AO basis, assumed orthonormal)
    # cf_gs   : guess orbitals (in AO basis)
    # ImpOrbs : cf_gs -> impurity orbitals transformation
    # n_orth  : number of orthonormal orbitals in cf_gs [1..n_orth]

    mol_ = gto.Mole()
    mol_.build (verbose=0)
    mol_.nelectron = nel
    mol_.incore_anyway = True

    cfx = cf_gs
    Sf  = mol.intor_symmetric('cint1e_ovlp_sph')
    Hc  = mol.intor_symmetric('cint1e_kin_sph') \
        + mol.intor_symmetric('cint1e_nuc_sph')

    occ = np.zeros((cfx.shape[1],))
    occ[:nel//2] = 2.

    # core contributions
    dm_core = np.dot(cf_core, cf_core.T)*2
    jk_core = scf.hf.get_veff (mol, dm_core)
    e_core  =     np.trace(np.dot(Hc, dm_core)) \
            + 0.5*np.trace(np.dot(jk_core, dm_core))

    # transform integrals
    Sp = np.dot(cfx.T, np.dot(Sf, cfx))
    Hp = np.dot(cfx.T, np.dot(Hc, cfx))
    jkp = np.dot(cfx.T, np.dot(jk_core, cfx))
    intsp = ao2mo.outcore.full_iofree (mol, cfx)

    # orthogonalize cf [virtuals]
    cf  = np.zeros((cfx.shape[1],)*2,)
    if n_orth > 0:
        assert (n_orth <= cfx.shape[1])
        assert (np.allclose(np.eye(n_orth), Sp[:n_orth,:n_orth]))
    else:
        n_orth = 0

    cf[:n_orth,:n_orth] = np.eye(n_orth)
    if n_orth < cfx.shape[1]:
        val, vec = sla.eigh(-Sp[n_orth:,n_orth:])
        idx = -val > 1.e-12
        U = np.dot(vec[:,idx]*1./(np.sqrt(-val[idx])), \
                   vec[:,idx].T)
        cf[n_orth:,n_orth:] = U

    # define ImpOrbs projection
    Xp = np.dot(ImpOrbs, ImpOrbs.T)

    # Si = np.dot(ImpOrbs.T, np.dot(Sp, ImpOrbs))
    # Mp = np.dot(ImpOrbs, np.dot(sla.inv(Si), ImpOrbs.T))
    Np = np.dot(Sp, Xp)
    # print np.allclose(Np, np.dot(Np, np.dot(Mp, Np)))

    # HF calculation
    mol_.energy_nuc = lambda *args: mol.energy_nuc() + e_core
    mf = scf.RHF(mol_)
    #mf.verbose = 4
    mf.mo_coeff  = cf
    mf.mo_occ    = occ
    mf.get_ovlp  = lambda *args: Sp
    mf.get_hcore = lambda *args: Hp + jkp - 0.5*chempot*(Np + Np.T)
    mf._eri = ao2mo.restore (8, intsp, cfx.shape[1])

    nt = scf.newton(mf)
    #nt.verbose = 4
    nt.max_cycle_inner = 1
    nt.max_stepsize = 0.25
    nt.ah_max_cycle = 32
    nt.ah_start_tol = 1.0e-12
    nt.ah_grad_trust_region = 1.0e8
    nt.conv_tol_grad = 1.0e-6

    nt.kernel()
    cf = nt.mo_coeff
    if not nt.converged:
        raise RuntimeError ('hf failed to converge')
    mf.mo_coeff  = nt.mo_coeff
    mf.mo_energy = nt.mo_energy
    mf.mo_occ    = nt.mo_occ

    # FCI solution
    fcisolver = fci.direct_spin0.FCISolver()
    _h1 = np.dot(cf.T, np.dot(mf.get_hcore(), cf))
    _h2 = ao2mo.incore.full (intsp, cf)
    fcisolver.verbose = 2
    Energy, fcivec = \
            fcisolver.kernel (_h1, _h2, cf.shape[1], (nel/2, nel/2))
    Energy += mol.energy_nuc() + e_core
    rdm1, rdm2 = \
            fcisolver.make_rdm12 (fcivec, cf.shape[1], (nel/2, nel/2))

    # transform rdm's to original basis
    tei  = ao2mo.restore (1, intsp, cfx.shape[1])
    rdm1 = np.dot(cf, np.dot(rdm1, cf.T))
    rdm2 = np.einsum('ai,ijkl->ajkl', cf, rdm2)
    rdm2 = np.einsum('bj,ajkl->abkl', cf, rdm2)
    rdm2 = np.einsum('ck,abkl->abcl', cf, rdm2)
    rdm2 = np.einsum('dl,abcl->abcd', cf, rdm2)

    ImpEnergy = +0.25 *np.einsum('ij,jk,ki->', 2*Hp+jkp, rdm1, Xp) \
                +0.25 *np.einsum('ij,jk,ki->', 2*Hp+jkp, Xp, rdm1) \
                +0.125*np.einsum('ijkl,ijkm,ml->', tei, rdm2, Xp) \
                +0.125*np.einsum('ijkl,ijml,mk->', tei, rdm2, Xp) \
                +0.125*np.einsum('ijkl,imkl,mj->', tei, rdm2, Xp) \
                +0.125*np.einsum('ijkl,mjkl,mi->', tei, rdm2, Xp)

    Nel = np.trace(np.dot(np.dot(rdm1, Sp), Xp))

    return Nel, ImpEnergy
