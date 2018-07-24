#!/usr/bin/python

import numpy as np
import scipy.linalg as sla
import pyscf
from pyscf import gto, scf, mp, ao2mo, df
#from mp2 import dfmp2
from pyscf.mp import dfmp2

''' This is a working version of DF-MP2 modification to the MP2 code
    The integrals need to calculated on the fly, without storing them
'''


def solve (mol, nel, cf_core, cf_gs, ImpOrbs, chempot=0., n_orth=0, FrozenPot=None):
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
        + mol.intor_symmetric('cint1e_nuc_sph') \
        + FrozenPot

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

    # density fitting ============================================================
    mf = scf.RHF(mol).density_fit()
    mf.with_df._cderi_to_save = 'saved_cderi.h5' # rank-3 decomposition
    mf.kernel()

    auxmol = df.incore.format_aux_basis(mol, auxbasis='weigend')
    j3c    = df.incore.aux_e2(mol, auxmol, intor='cint3c2e_sph', aosym='s1')
    nao    = mol.nao_nr()
    naoaux = auxmol.nao_nr()
    j3c    = j3c.reshape(nao,nao,naoaux) # (ij|L)
    j2c    = df.incore.fill_2c2e(mol, auxmol) #(L|M) overlap matrix between auxiliary basis functions

    #the eri is (ij|kl) = \sum_LM (ij|L) (L|M) (M|kl)
    omega = sla.inv(j2c)
    eps,U = sla.eigh(omega)
    #after this transformation the eri is (ij|kl) = \sum_L (ij|L) (L|kl)
    j3c   = np.dot(np.dot(j3c,U),np.diag(np.sqrt(eps)))


    conv = np.einsum('prl,pi,rj->ijl', j3c, cfx, cfx)
    df_eri = np.einsum('ijm,klm->ijkl',conv,conv)

    intsp_df = ao2mo.restore(4, df_eri, cfx.shape[1])
    print("shape of DF instp", intsp_df.shape)
    # =============================================================================

    intsp = ao2mo.outcore.full_iofree (mol, cfx)    # TODO: this we need to calculate on the fly using generator f'n
    print(intsp.shape)

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

    mf = scf.RHF(mol_).density_fit()
    print("all is well")
    #mf.verbose = 4
    # mf.mo_coeff  = cf
    # mf.mo_occ    = occ
    mf.get_ovlp  = lambda *args: Sp
    mf.get_hcore = lambda *args: Hp + jkp - 0.5*chempot*(Np + Np.T)
    mf._eri = ao2mo.restore (8, intsp, cfx.shape[1])         # ?why do we need to have it?

#    nt = scf.newton(mf)            # ?do we need this paragraph bit and the one above other than the scf calc?#
#    #nt.verbose = 4                 # ? why do we need the newton solver?
#    nt.max_cycle_inner = 1
#    nt.max_stepsize = 0.25
#    nt.ah_max_cycle = 32
#    nt.ah_start_tol = 1.0e-12
#    nt.ah_grad_trust_region = 1.0e8
#    nt.conv_tol_grad = 1.0e-6

#    nt.kernel()
#    cf = nt.mo_coeff
#    if not nt.converged:
#        raise RuntimeError ('hf failed to converge')
#    mf.mo_coeff  = nt.mo_coeff
#    mf.mo_energy = nt.mo_energy
#    mf.mo_occ    = nt.mo_occ
#    mf = nt
#    mo_coeff  = nt.mo_coeff
#    mo_energy = nt.mo_energy#
#    mo_occ    = nt.mo_occ
#    print("mo_energy", mo_energy)

    # mf           = scf.RHF(mol_) #.density_fit()
    # mf.verbose   = 4
    # # mf           = scf.newton(mf)
    # if(verbose): print ( 'Total SCF energy',mf.energy_tot() )
    # mo_coeff  = mf.mo_coeff
    # mo_energy = mf.mo_energy
    # mo_occ    = mf.mo_occ
    # mf.kernel()

    # print("mo_energy", mo_energy)
    #
    # cf = mf.mo_coeff
    # print("cf", cf)
    #
    # '''
    # mo_coeff  = mf.mo_coeff
    # mo_energy = mf.mo_energy
    # mo_occ    = mf.mo_occ
    #
    # print("mo_energy", mf.mo_energy)
    # '''


    # MP2 solution
    #mp2solver = dfmp2.MP2(mf)
    mf.with_df.fill_2c2e = lambda *args: Sp
    # mf.with_df.aux_e2    = lambda *args: np.zeros((3,3,5))
    mf.kernel()

    mp2solver = dfmp2.MP2(mf)
    mp2solver.verbose = 5
    mp2solver.kernel()

    nbas = Sp.shape[0]
    rdm1 = mp2solver.make_rdm1()
    rdm2 = mp2solver.make_rdm2()

    # transform rdm's to original basis
    tei  = ao2mo.restore(1, intsp, cfx.shape[1])
    rdm1 = np.dot(cf, np.dot(rdm1, cf.T))
    rdm2 = np.einsum('ai,ijkl->ajkl', cf, rdm2)
    rdm2 = np.einsum('bj,ajkl->abkl', cf, rdm2)
    rdm2 = np.einsum('ck,abkl->abcl', cf, rdm2)
    rdm2 = np.einsum('dl,abcl->abcd', cf, rdm2)

    ''' for istep, qov in enumerate(mp.loop_ao2mo(mo_coeff, nocc)):
        logger.debug(mp, 'Load cderi step %d', istep)
        for i in range(nocc):
            calculate impurity energy
    '''

    ImpEnergy = +0.25 *np.einsum('ij,jk,ki->', 2*Hp+jkp, rdm1, Xp) \
                +0.25 *np.einsum('ij,jk,ki->', 2*Hp+jkp, Xp, rdm1) \
                +0.125*np.einsum('ijkl,ijkm,ml->', tei, rdm2, Xp) \
                +0.125*np.einsum('ijkl,ijml,mk->', tei, rdm2, Xp) \
                +0.125*np.einsum('ijkl,imkl,mj->', tei, rdm2, Xp) \
                +0.125*np.einsum('ijkl,mjkl,mi->', tei, rdm2, Xp)

    Nel = np.trace(np.dot(np.dot(rdm1, Sp), Xp))

    return Nel, ImpEnergy
