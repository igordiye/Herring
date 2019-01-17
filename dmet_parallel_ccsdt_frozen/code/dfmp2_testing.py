#!/usr/bin/python

import numpy as np
import scipy.linalg as sla
import pyscf
from pyscf import gto, scf, mp, ao2mo, df, lib
# from pyscf.mp import dfmp2_testing #(work)
from pyscf.mp import dfmp2 #(work) testing
# from pyscf.mp import dfmp2       #(home)

#from pyscf.mp.mp2 import make_rdm1, make_rdm2

''' This is a working version of DF-MP2 modification to the MP2 code
    The integrals need to calculated on the fly, without storing them
    POSTMAT: this version has problematic rdms, making a new working file with new test rdms
'''
    #TODO check the eris

def solve (mol, nel, cf_core, cf_gs, ImpOrbs, chempot=0., n_orth=0, FrozenPot=None, mf_tot=None):
    # cf_core : core orbitals (in AO basis, assumed orthonormal)
    # cf_gs   : guess orbitals (in AO basis)
    # ImpOrbs : cf_gs -> impurity orbitals transformation
    # n_orth  : number of orthonormal orbitals in cf_gs [1..n_orth]

    mol_ = gto.Mole()
    mol_.build (verbose=0)
    mol_.nelectron = nel
    mol_.incore_anyway = True

    cfx = cf_gs
    print("cfx shape", cfx.shape)
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
    # mf = scf.RHF(mol).density_fit()     # this should be moved out of to the parent directory, to avoid repetition
    # mf.with_df._cderi_to_save = 'saved_cderi.h5' # rank-3 decomposition
    # mf.kernel()                    ### moved these three lines to orbital_selection_fc

    auxmol = df.incore.format_aux_basis(mol, auxbasis='weigend')
    j3c    = df.incore.aux_e2(mol, auxmol, intor='cint3c2e_sph', aosym='s1')
    nao    = mol.nao_nr()
    naoaux = auxmol.nao_nr()
    j3c    = j3c.reshape(nao,nao,naoaux) # (ij|L)
    print("j3c shape", j3c.shape)
    j2c    = df.incore.fill_2c2e(mol, auxmol) #(L|M) overlap matrix between auxiliary basis functions

    #the eri is (ij|kl) = \sum_LM (ij|L) (L|M) (M|kl)
    omega = sla.inv(j2c)
    eps,U = sla.eigh(omega)
    #after this transformation the eri is (ij|kl) = \sum_L (ij|L) (L|kl)
    j3c   = np.dot(np.dot(j3c,U),np.diag(np.sqrt(eps)))

    #this part is slow, as we again store the whole eri_df
    # conv = np.einsum('prl,pi,rj->ijl', j3c, cfx, cfx)
    conv = np.einsum('prl,pi->irl',j3c,cfx)
    conv = np.einsum('irl,rj->ijl',conv,cfx)
    df_eri = np.einsum('ijm,klm->ijkl',conv,conv)

    intsp_df = ao2mo.restore(4, df_eri, cfx.shape[1])
    print("DF instp", intsp_df.shape)
    # =============================================================================

    # intsp = ao2mo.outcore.full_iofree (mol, cfx)    #TODO: this we need to calculate on the fly using generator f'n
    # print("intsp shape", intsp.shape)

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
    mf1 = scf.RHF(mol_) #.density_fit()
    #mf.verbose = 4
    mf1.mo_coeff  = cf
    mf1.mo_occ    = occ
    mf1.get_ovlp  = lambda *args: Sp
    mf1.get_hcore = lambda *args: Hp + jkp - 0.5*chempot*(Np + Np.T)
    mf1._eri = ao2mo.restore (8, intsp_df, cfx.shape[1]) #trying something

    nt = scf.newton(mf1)
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
    mo_coeff  = nt.mo_coeff
    mo_energy = nt.mo_energy
    mo_occ    = nt.mo_occ

    # dfMP2 solution
    nocc = nel//2
    # mp2solver = dfmp2_testing.MP2(mf_tot) #(work)  #we just pass the mf for the full molecule to dfmp2
    mp2solver = dfmp2.DFMP2(mf_tot) #(work)  #we just pass the mf for the full molecule to dfmp2
    # mp2solver = dfmp2.MP2(mf)  #(home)
    mp2solver.verbose = 5
    mp2solver.kernel(mo_energy=mo_energy, mo_coeff=mo_coeff, nocc=nocc)
    # exit()

    ''' Try generating j3c for the whole molecule, on the fly, then use that for rdms
    '''

    def make_rdm1(mp2solver, t2, mo_coeff, mo_energy, nocc):
        '''1-particle density matrix in MO basis.  The off-diagonal blocks due to
        the orbital response contribution are not included.
        '''
        mo = np.asarray(mo_coeff, order='F')
        nmo = mo.shape[1]
        nvir = nmo - nocc
        dm1occ = np.zeros((nocc,nocc))
        dm1vir = np.zeros((nvir,nvir))
        for i in range(nocc):
            dm1vir += np.einsum('jca,jcb->ab', t2[i], t2[i]) * 2 \
                    - np.einsum('jca,jbc->ab', t2[i], t2[i])
            dm1occ += np.einsum('iab,jab->ij', t2[i], t2[i]) * 2 \
                    - np.einsum('iab,jba->ij', t2[i], t2[i])
        rdm1 = np.zeros((nmo,nmo))
    # *2 for beta electron
        rdm1[:nocc,:nocc] =-dm1occ * 2
        rdm1[nocc:,nocc:] = dm1vir * 2
        for i in range(nocc):
            rdm1[i,i] += 2
        return rdm1

    def make_rdm2(mp2solver, t2, mo_coeff, mo_energy, nocc):
        '''2-RDM in MO basis'''
        mo = np.asarray(mo_coeff, order='F')
        nmo = mo.shape[1]
        nvir = nmo - nocc
        dm2 = np.zeros((nmo,nmo,nmo,nmo)) # Chemist notation
        #dm2[:nocc,nocc:,:nocc,nocc:] = t2.transpose(0,3,1,2)*2 - t2.transpose(0,2,1,3)
        #dm2[nocc:,:nocc,nocc:,:nocc] = t2.transpose(3,0,2,1)*2 - t2.transpose(2,0,3,1)
        for i in range(nocc):
            t2i = t2[i]
            dm2[i,nocc:,:nocc,nocc:] = t2i.transpose(1,0,2)*2 - t2i.transpose(2,0,1)
            dm2[nocc:,i,nocc:,:nocc] = dm2[i,nocc:,:nocc,nocc:].transpose(0,2,1)

        for i in range(nocc):
            for j in range(nocc):
                dm2[i,i,j,j] += 4
                dm2[i,j,j,i] -= 2
        return dm2

    mo = np.asarray(mo_coeff, order='F')
    nmo = mo.shape[1]
    nvir = nmo - nocc
    co = mo_coeff[:,:nocc]
    cv = mo_coeff[:,nocc:]
    # eri = mol_.intor('cint2e_sph', aosym='s8')
    _scf = mf1
    eri = _scf._eri
    eri = ao2mo.incore.general(eri, (co,cv,co,cv))
    print("eri shape", eri.shape)
    eri = ao2mo.load(eri)

    t2 = np.empty((nocc,nocc,nvir,nvir))
    eia = mo_energy[:nocc,None] - mo_energy[None,nocc:]
    with eri as ovov:
        for i in range(nocc):
            gi = np.asarray(ovov[i*nvir:(i+1)*nvir])
            gi = gi.reshape(nvir, nocc, nvir).transpose(1,0,2)
            t2[i] = gi/lib.direct_sum('jb+a->jba', eia, eia[i])

    rdm1 = make_rdm1(mp2solver, t2, mo_coeff, mo_energy, nocc)

    from scipy.linalg import eigh
    print("hermitian?", np.allclose(rdm1,rdm1.T))
    w,v = eigh(rdm1)
    print(w)
    print(np.trace(rdm1))
    print("rm1 shape", rdm1.shape)
    rdm2 = make_rdm2(mp2solver, t2, mo_coeff, mo_energy, nocc)
    print("rmd2 shape", rdm2.shape)
    print("cfx shape", cfx.shape)

# #  # -------------------------------------
#  # TEST: compare rdm dmet with dfmp2 rdms
#     # atoms_test=\
#     # [['C',( 0.0000, 0.0000, 0.7680)],\
#     #  ['C',( 0.0000, 0.0000,-0.7680)],\
#     #  ['H',(-1.0192, 0.0000, 1.1573)],\
#     #  ['H',( 0.5096, 0.8826, 1.1573)],\
#     #  ['H',( 0.5096,-0.8826, 1.1573)],\
#     #  ['H',( 1.0192, 0.0000,-1.1573)],\
#     #  ['H',(-0.5096,-0.8826,-1.1573)],\
#     #  ['H',(-0.5096, 0.8826,-1.1573)]]
#
#     atoms_test = [
#     ['O' , (0. , 0. , 0.)],\
#     ['H' , (0. , -0.757 , 0.587)],\
#     ['H' , (0. , 0.757  , 0.587)]]
#
#     mol_test = gto.M(atom=atoms_test,basis='cc-pvdz')
#     m_test = scf.RHF(mol_test).density_fit()
#     m_test.kernel()
#
#     mm_test = dfmp2.DFMP2(m_test)
#     mm_test.kernel()
#     from pyscf.mp import mp2
#     rdm1_test = mp2.make_rdm1(mm_test)
#     rdm2_test = mp2.make_rdm2(mm_test)
#
# # --------------plots ------------------------
#     # Plot sorted rdm1 values
#     x1 = rdm1
#     y1 = x1.flatten()
#     y1 = np.sort(y1)
#     import matplotlib.pyplot as plt
#     plt.plot(y1, 'r', label='rdm1 from dmet')
#     plt.ylabel('rdm1')
#     x2 = rdm1_test
#     y2 = x2.flatten()
#     y2 = np.sort(y2)
#     plt.plot(y2, 'b', label='rdm1 for dfmp2')
#     plt.ylabel('rdm1 sorted values')
#     # Place a legend above this subplot, expanding itself to
#     # fully use the given bounding box.
#     plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
#                ncol=2, mode="expand", borderaxespad=0.)
#     # plt.show()
#
#     print("deviations between sorted 1rdm in MO basis ")
#     print(np.abs(y1-y2).max())
#
#     # Plot sorted rdm2 values
#     x1 = rdm2
#     y1 = x1.flatten()
#     y1 = np.sort(y1)
#     import matplotlib.pyplot as plt
#     plt.plot(y1, 'r', label='rdm2 from dmet')
#     plt.ylabel('rdm2')
#     x2 = rdm2_test
#     y2 = x2.flatten()
#     y2 = np.sort(y2)
#     plt.plot(y2, 'b', label='rdm2 for dfmp2')
#     plt.ylabel('rdm1 sorted values')
#     # Place a legend above this subplot, expanding itself to
#     # fully use the given bounding box.
#     plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
#                ncol=2, mode="expand", borderaxespad=0.)
#     # plt.show()
#     print("deviations between sorted 1rdm in MO basis ")
#     print(np.abs(y1-y2).max())
#
#
#     print("deviations between 1rdm,2rdm in MO basis ")
#     print(np.abs(rdm1-rdm1_test).max())
#     print(np.abs(rdm2-rdm2_test).max())
# # ------------------------------------------------------------


    # transform rdm's to original basis
    tei  = ao2mo.restore(1, intsp_df, cfx.shape[1])
    print("tei shape", tei.shape)
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
