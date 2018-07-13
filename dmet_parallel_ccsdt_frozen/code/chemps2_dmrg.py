#!/usr/bin/python

import numpy as np
import scipy.linalg as sla

from pyscf import scf, ao2mo

import PyCheMPS2
import ctypes


def solve (mol, nel, cf_core, cf_gs, ImpOrbs, chempot=0., n_orth=0):
    # cf_core : core orbitals (in AO basis, assumed orthonormal)
    # cf_gs   : guess orbitals (in AO basis)
    # ImpOrbs : cf_gs -> impurity orbitals transformation 
    # n_orth  : number of orthonormal orbitals in cf_gs [1..n_orth]

    cfx = cf_gs
    Sf  = mol.intor_symmetric('cint1e_ovlp_sph')
    Hc  = mol.intor_symmetric('cint1e_kin_sph') \
        + mol.intor_symmetric('cint1e_nuc_sph')

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

    _h1  = np.dot(cf.T, np.dot(Hp+jkp-0.5*chempot*(Np + Np.T), cf))
    _h2t = ao2mo.incore.full (intsp, cf)
    _h2  = ao2mo.restore (1, _h2t, cfx.shape[1])

    # Set the seed of the random number generator and cout.precision
    Initializer = PyCheMPS2.PyInitialize()
    Initializer.Init()

    # Setting up the Hamiltonian
    Group = 0
    Norb  = cfx.shape[1]
    orbirreps = np.zeros((Norb,), dtype=ctypes.c_int)
    HamCheMPS2 = PyCheMPS2.PyHamiltonian(Norb, Group, orbirreps)
    HamCheMPS2.setEconst(mol.energy_nuc()+e_core)
    for i1 in range(Norb):
        for i2 in range(Norb):
            HamCheMPS2.setTmat (i1, i2, _h1[i1,i2])

    for i1 in range (Norb):
        for i2 in range(Norb):
            for i3 in range(Norb):
                for i4 in range(Norb):
                    HamCheMPS2.setVmat (i1, i2, i3, i4, \
                        _h2[i1,i3,i2,i4]) # physics notation

    TwoS  = 0
    Irrep = 0
    Prob  = PyCheMPS2.PyProblem (HamCheMPS2, TwoS, nel, Irrep)

    OptScheme = PyCheMPS2.PyConvergenceScheme(3) # 3 instructions
    #OptScheme.setInstruction(instruction, D, Econst, maxSweeps, noisePrefactor)
    OptScheme.setInstruction(0,  512, 1e-10,  3, 0.05)
    OptScheme.setInstruction(1, 1024, 1e-10,  3, 0.05)
    OptScheme.setInstruction(2, 1024, 1e-10, 10, 0.00)
    # Last instruction a few iterations without noise

    theDMRG = PyCheMPS2.PyDMRG (Prob, OptScheme)
    EnergyCheMPS2 = theDMRG.Solve()
    theDMRG.calc2DMandCorrelations()
    rdm2 = np.zeros ((Norb,)*4, dtype=ctypes.c_double)
    for i1 in range(Norb):
        for i2 in range(Norb):
            for i3 in range(Norb):
                for i4 in range(Norb):
                    rdm2[i1,i3,i2,i4] = theDMRG.get2DMA(i1,i2,i3,i4)
                    #From physics to chemistry notation
    rdm1 = np.einsum('ijkk->ij', rdm2)/(nel-1)

    # theDMRG.deleteStoredMPS()
    theDMRG.deleteStoredOperators()
    del theDMRG
    del OptScheme
    del Prob

    del HamCheMPS2

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

