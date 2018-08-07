#!/usr/bin/env python
# $Id$
# -*- coding: utf-8

'''
Make rmd1 and rdm2 for fragment.
'''

import time
import numpy
from sys import path
path.append('/home/yuliya/pyscf')
import pyscf
from pyscf import lib, scf
from pyscf.lib import logger
from pyscf.ao2mo import _ao2mo
from pyscf.mp.mp2 import make_rdm1, make_rdm2, make_rdm1_ao


def loop_ao2mo(self, mo_coeff, nocc):
    mo = numpy.asarray(mo_coeff, order='F')
    nmo = mo.shape[1]
    ijslice = (0, nocc, nocc, nmo)
    Lov = None

    for eri1 in self._scf.with_df.loop(): # this is the issue for the rdms.
        Lov = _ao2mo.nr_e2(eri1, mo, ijslice, aosym='s2', out=Lov)
        yield Lov

def make_rdm1(self, t2=None):
    nmo = self.nmo
    # nmo = len(self._scf.mo_energy)
    nocc = self.nocc
    nvir = nmo - nocc
    dm1occ = numpy.zeros((nocc,nocc))
    dm1vir = numpy.zeros((nvir,nvir))

    eia = lib.direct_sum('i-a->ia',self.mo_energy[:nocc],self.mo_energy[nocc:])
    for istep, qov in enumerate(self.loop_ao2mo(self.mo_coeff, nocc)):
        for i in range(nocc):
            buf = numpy.dot(qov[:,i*nvir:(i+1)*nvir].T,
                            qov).reshape(nvir,nocc,nvir)
            gi = numpy.array(buf, copy=False)
            gi = gi.reshape(nvir,nocc,nvir).transpose(1,2,0)
            t2i = gi/lib.direct_sum('jb+a->jba', eia, eia[i])
            # 2*ijab-ijba
#            theta = gi*2 - gi.transpose(0,2,1)
#            emp2 += numpy.einsum('jab,jab', t2i, theta)
            dm1vir += numpy.einsum('jca,jcb->ab', t2i, t2i) * 2 \
                    - numpy.einsum('jca,jbc->ab', t2i, t2i)
            dm1occ += numpy.einsum('iab,jab->ij', t2i, t2i) * 2 \
                    - numpy.einsum('iab,jba->ij', t2i, t2i)
    rdm1 = numpy.zeros((nmo,nmo))
# *2 for beta electron
    rdm1[:nocc,:nocc] =-dm1occ * 2
    rdm1[nocc:,nocc:] = dm1vir * 2
    for i in range(nocc):
        rdm1[i,i] += 2
    return rdm1


def make_rdm2(self, t2=None):
    '''2-RDM in MO basis'''
    nmo = self.nmo
    nocc = self.nocc
    nvir = nmo - nocc
    dm2 = numpy.zeros((nmo,nmo,nmo,nmo)) # Chemist notation
    #dm2[:nocc,nocc:,:nocc,nocc:] = t2.transpose(0,3,1,2)*2 - t2.transpose(0,2,1,3)
    #dm2[nocc:,:nocc,nocc:,:nocc] = t2.transpose(3,0,2,1)*2 - t2.transpose(2,0,3,1)
    eia = lib.direct_sum('i-a->ia',self.mo_energy[:nocc],self.mo_energy[nocc:])
    for istep, qov in enumerate(self.loop_ao2mo(self.mo_coeff, nocc)):
#        logger.debug(mp, 'Load cderi step %d', istep)
        for i in range(nocc):
            buf = numpy.dot(qov[:,i*nvir:(i+1)*nvir].T,
                            qov).reshape(nvir,nocc,nvir)
            gi = numpy.array(buf, copy=False)
            gi = gi.reshape(nvir,nocc,nvir).transpose(1,2,0)
            t2i = gi/lib.direct_sum('jb+a->jba', eia, eia[i])
            dm2[i,nocc:,:nocc,nocc:] = t2i.transpose(1,0,2)*2 - t2i.transpose(2,0,1)
            dm2[nocc:,i,nocc:,:nocc] = dm2[i,nocc:,:nocc,nocc:].transpose(0,2,1)

    for i in range(nocc):
        for j in range(nocc):
            dm2[i,i,j,j] += 4
            dm2[i,j,j,i] -= 2
    return dm2


if __name__ == '__main__':
    from pyscf import scf
    from pyscf import gto
    mol = gto.Mole()
    mol.verbose = 0
    mol.atom = [
        [8 , (0. , 0.     , 0.)],
        [1 , (0. , -0.757 , 0.587)],
        [1 , (0. , 0.757  , 0.587)]]

    mol.basis = 'cc-pvdz'
    mol.build()
    mf = scf.RHF(mol)
    mf.scf()
    pt = MP2(mf)
    pt.max_memory = .05
    emp2, t2 = pt.kernel()
    print(emp2 - -0.204254491987)

    mf = scf.density_fit(scf.RHF(mol))
    mf.scf()
    pt = MP2(mf)
    pt.max_memory = .05
    pt.ioblk = .05
    pt.verbose = 5
    emp2, t2 = pt.kernel()
    print(emp2 - -0.203986171133)
