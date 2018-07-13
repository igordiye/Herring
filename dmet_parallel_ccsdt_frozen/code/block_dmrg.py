#!/usr/bin/python

import os
import subprocess
import shutil

import numpy as np
import scipy.linalg as sla

from pyscf import scf, ao2mo
from pyscf.tools import fcidump

block_exe = '/home/carlosjh/software/Block/block.spin_adapted'


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
    _h2  = ao2mo.incore.full (intsp, cf)

    # prepare BLOCK file, FCIDUMP
    fcidump.from_integrals ('FCIDUMP', _h1, _h2, cfx.shape[1], nel)

    target = open('dmrg.inp', 'w')
    output = open('dmrg.out', 'w')
    target.write('orbitals FCIDUMP\n')
    target.write('nelec %2d\n' % nel)
    target.write('spin  %2d\n' % 0)
    target.write('irrep %2d\n' % 1)

    target.write('\nhf_occ integral\n')
    target.write('schedule\n')
    target.write(' %2d %5d %9.1e %9.1e\n' % ( 0,  512, 1e-10, 1e-6,))
    target.write(' %2d %5d %9.1e %9.1e\n' % ( 3, 1024, 1e-10, 1e-6,))
    target.write(' %2d %5d %9.1e %9.1e\n' % ( 6, 1024, 1e-10,  0.0,))
    target.write('end\n')
    target.write('maxiter %3d\n' % 16)
    target.write('sweep_tol %9.1e\n' % 1e-8)
    target.write('onedot\n')

    target.write('\ntwopdm\n')
    target.write('outputlevel %2d\n' % 0)
    target.close()

    subprocess.call (['mpirun','-np','1',block_exe,'dmrg.inp'], stdout=output)
    output.close()

    os.remove('FCIDUMP')
    os.remove('dmrg.inp')
    os.remove('dmrg.out')

    rdm2i = np.loadtxt('node0/spatial_twopdm.0.0.txt', dtype=int, skiprows=1, usecols=(0,1,2,3,))
    rdm2t = np.loadtxt('node0/spatial_twopdm.0.0.txt', dtype=float, skiprows=1, usecols=(4,))

    shutil.rmtree('node0')

    rdm2 = np.zeros((cfx.shape[1],)*4)
    for i in range(rdm2t.shape[0]):
      rdm2[rdm2i[i,0],rdm2i[i,3],rdm2i[i,1],rdm2i[i,2]] = 2*rdm2t[i]

    rdm1  = np.einsum('ijkk->ij', rdm2)
    rdm1 /= (nel-1)
    del rdm2i, rdm2t

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

