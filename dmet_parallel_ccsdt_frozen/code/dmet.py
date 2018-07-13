#!/usr/bin/python

import numpy as np
import scipy.linalg as sla
import scipy.optimize as opt

from pyscf.tools import localizer

import embedding
import pyscf_hf
import pyscf_cc
import pyscf_mp2
import pyscf_dfmp2
import pyscf_ccsdt
import pyscf_fci

class dmet:

    def __init__ (self, mol, cf, imp_at, \
                  A_val, at_val, method='hf', thresh=1.0e-2, \
                  A_core=None, at_core=None, \
                  A_virt=None, at_virt=None, \
                  imp_atx=None, parallel=False,e_core=None, FrozenPot=None):
        # mol     - mol object
        # cf      - occupied (valence) orbitals
        # imp_at  - impurity <-> atom mapping [active atom]
        # imp_atx - impurity <-> atom mapping {overlapping}
        # method  - high level impurity solver
        # thresh  - bath threshold
        # A_??    - orbitals **
        # at_??   - atom <-> orbital mapping

        # ** core orbitals must be orthogonal to cf
        #    cf   must PARTIALLY span val orbitals
        #    virt orbitals must be orthogonal to cf
        # core and valence orbitals are assumed orthonormal

        self.mol = mol
        self.nb  = mol.nao_nr()
        self.nup = (mol.nelectron + mol.spin)//2
        self.ndn = mol.nelectron - self.nup

        if e_core is not None:
           self.e_core=e_core
        if FrozenPot is not None:
           self.FrozenPot=FrozenPot

        assert (method in ['hf','cc','ccsd(t)','mp2', 'dfmp2', 'fci', 'dmrg'])
        self.method = method
        assert (thresh > 0. and thresh < 1e-1)
        self.thresh = thresh
        self.parallel = parallel

        # orbital, impurity definition

        assert (self.nup == self.ndn)
        self.cf = cf

        self.nc = self.nvl = self.nvt = 0

        assert (A_val.shape[0] == self.nb)
        self.nvl   = A_val.shape[1]
        self.A_val = A_val

        assert (self.nvl > 0)
        assert (at_val.shape[0] == self.nvl)
        assert (np.all(at_val < mol.natm))
        self.at_val = at_val

        if A_core is not None:
            assert (A_core.shape[0] == self.nb)
            self.nc     = A_core.shape[1]
            self.A_core = A_core

            assert (at_core is not None)
            assert (at_core.shape[0] == self.nc)
            assert (np.all(at_core < mol.natm))
            self.at_core = at_core
        assert (cf.shape == (self.nb, self.nup-self.nc))

        if A_virt is not None:
            assert (A_virt.shape[0] == self.nb)
            self.nvt    = A_virt.shape[1]
            self.A_virt = A_virt

            assert (at_virt is not None)
            assert (at_virt.shape[0] == self.nvt)
            assert (np.all(at_virt < mol.natm))
            self.at_virt = at_virt

        self.nimp = len(imp_at)
        for i in range(self.nimp):
            assert (imp_at[i].shape[0] == mol.natm)
        self.imp_at = imp_at

        if imp_atx is not None:
            assert (len(imp_atx) == self.nimp)
            for i in range(self.nimp):
                assert (imp_atx[i].shape[0] == mol.natm)
                assert (np.array_equiv(imp_at[i], \
                                np.logical_and(imp_atx[i], imp_at[i])))
            self.imp_atx = imp_atx
        else:
            self.imp_atx = imp_at

        self.test_cluster()

        # get overlap integrals
        self.Sf = self.mol.intor_symmetric('cint1e_ovlp_sph')

    def test_cluster (self):

        quicktest = np.empty((self.mol.natm,), dtype=bool)

        for i in range(self.nimp):
            if i == 0:
                quicktest = self.imp_at[i].copy()
            else:
                assert (np.all(np.logical_not(np.logical_and\
                                        (quicktest, self.imp_at[i]))))
                quicktest = np.logical_or (quicktest, self.imp_at[i])

    def hl_solver (self, chempot=0., threshold=1.0e-12):
#        energy = self.mol.energy_nuc()
        energy = 0.
        nelec  = 0.

        rdm_ao  = np.dot(self.cf, self.cf.T)
        AX_val  = np.dot(self.Sf, self.A_val)
        rdm_val = np.dot(AX_val.T, np.dot(rdm_ao, AX_val))

        print ( "shapes" )
        print ( "cf",self.cf.shape )
        print ( "rdm_ao",rdm_ao.shape )
        print ( "AX_val",AX_val.shape )
        print ( "rdm_val",rdm_val.shape )

        if(not self.parallel):
           myrange = range(self.nimp)
        else:
           from mpi4py import MPI
           comm = MPI.COMM_WORLD
           rank = MPI.COMM_WORLD.Get_rank()
           size = MPI.COMM_WORLD.Get_size()
           myrange = range(rank,rank+1)

        for i in myrange:
            # prepare orbital indexing
            imp_val   = np.zeros((self.nvl,), dtype=bool)
            imp_val_  = np.zeros((self.nvl,), dtype=bool)
            if self.nc > 0:
                imp_core  = np.zeros((self.nc,),  dtype=bool)
                imp_core_ = np.zeros((self.nc,),  dtype=bool)
            if self.nvt > 0:
                imp_virt  = np.zeros((self.nvt,), dtype=bool)
                imp_virt_ = np.zeros((self.nvt,), dtype=bool)
            for k in range(self.mol.natm):
                if self.imp_atx[i][k]:
                    imp_val[self.at_val == k]   = True
                    if self.nc > 0:
                        imp_core[self.at_core == k] = True
                    if self.nvt > 0:
                        imp_virt[self.at_virt == k] = True
                if self.imp_at[i][k]:
                    imp_val_[self.at_val == k]   = True
                    if self.nc > 0:
                        imp_core_[self.at_core == k] = True
                    if self.nvt > 0:
                        imp_virt_[self.at_virt == k] = True

            # embedding
            cf_tmp, ncore, nact, ImpOrbs_x = \
                embedding.embedding (rdm_val, imp_val, \
                                     threshold=self.thresh, \
                                     transform_imp='hf')
            cf_tmp = np.dot(self.A_val, cf_tmp)

            # localize imp+bath orbitals
            if self.method == 'dmrg':
                XR = np.random.rand(nact,nact)
                XR -= XR.T
                XS = sla.expm(0.01*XR)
                cf_ib = np.dot(cf_tmp[:,ncore:ncore+nact], XS)
                loc = localizer.localizer (self.mol, cf_ib, 'boys')
                loc.verbose = 5
                cf_ib = loc.optimize(threshold=1.0e-5)
                del loc

                R = np.dot(cf_ib.T, \
                           np.dot(self.Sf, cf_tmp[:,ncore:ncore+nact]))
                print ( np.allclose(np.dot(cf_tmp[:,ncore:ncore+nact], \
                                         ImpOrbs_x), \
                                  np.dot(cf_ib, np.dot(R, ImpOrbs_x))) )
                ImpOrbs_x = np.dot(R, ImpOrbs_x)
                cf_tmp[:,ncore:ncore+nact] = cf_ib
                print ( cf_ib )

            # prepare ImpOrbs
            ni_val = nact
            nj_val = np.count_nonzero(imp_val_)
            if self.nc > 0:
                ni_core = np.count_nonzero(imp_core)
                nj_core = np.count_nonzero(imp_core_)
            else:
                ni_core = nj_core = 0
            if self.nvt > 0:
                ni_virt = np.count_nonzero(imp_virt)
                nj_virt = np.count_nonzero(imp_virt_)
            else:
                ni_virt = nj_virt = 0

            ii = 0
            ImpOrbs = np.zeros((ni_val+ni_core+ni_virt,\
                                nj_val+nj_core+nj_virt,))
            if self.nc > 0:
                j = 0
                for i in range(self.nc):
                    if imp_core[i] and imp_core_[i]:
                        ImpOrbs[j,ii] = 1.
                        ii += 1
                    if imp_core[i]:
                        j += 1
            j = 0
            for i in range(self.nvl):
                if imp_val[i] and imp_val_[i]:
                    ImpOrbs[ni_core:ni_core+ni_val,ii] = ImpOrbs_x[:,j]
                    ii += 1
                if imp_val[i]:
                    j += 1
            if self.nvt > 0:
                j = 0
                for i in range(self.nvt):
                    if imp_virt[i] and imp_virt_[i]:
                        ImpOrbs[ni_core+ni_val+j,ii] = 1.
                        ii += 1
                    if imp_virt[i]:
                        j += 1

            # prepare orbitals
            cf_core = cf_virt = None
            if self.nc > 0:
                cf_core = self.A_core[:,imp_core]
            if self.nvt > 0:
                cf_virt = self.A_virt[:,imp_virt]
            cf_val = cf_tmp[:,ncore:ncore+nact]

            if cf_core is not None and cf_virt is not None:
                cf = np.hstack ((cf_core, cf_val, cf_virt,))
            elif cf_core is not None:
                cf = np.hstack ((cf_core, cf_val,))
            elif cf_virt is not None:
                cf = np.hstack ((cf_val, cf_virt,))
            else:
                cf = cf_val

            # prepare core
            if self.nc > 0:
                Ac_ = self.A_core[:,~(imp_core)]
                X_core = np.hstack((Ac_, cf_tmp[:,:ncore],))
            else:
                X_core = cf_tmp[:,:ncore]

            n_orth = cf.shape[1]
            if cf_virt is not None:
                n_orth -= cf_virt.shape[1]

            if self.method == 'hf':
                nel_, en_ = \
                    pyscf_hf.solve (self.mol, \
                                2*(self.nup-X_core.shape[1]), \
                                X_core, cf, ImpOrbs, chempot=chempot, \
                                n_orth=n_orth)

            elif self.method == 'cc':
                nel_, en_ = \
                    pyscf_cc.solve (self.mol, \
                                2*(self.nup-X_core.shape[1]), \
                                X_core, cf, ImpOrbs, chempot=chempot, \
                                n_orth=n_orth,FrozenPot=self.FrozenPot)

            elif self.method == 'ccsd(t)':
                nel_, en_ = \
                    pyscf_ccsdt.solve (self.mol, \
                                2*(self.nup-X_core.shape[1]), \
                                X_core, cf, ImpOrbs, chempot=chempot, \
                                n_orth=n_orth)

            elif self.method == 'mp2':
                nel_, en_ = \
                    pyscf_mp2.solve (self.mol, \
                                2*(self.nup-X_core.shape[1]), \
                                X_core, cf, ImpOrbs, chempot=chempot, \
                                n_orth=n_orth,FrozenPot=self.FrozenPot)

            elif self.method == 'dfmp2':
                nel_, en_ = \
                    pyscf_dfmp2.solve (self.mol, \
                                2*(self.nup-X_core.shape[1]), \
                                X_core, cf, ImpOrbs, chempot=chempot, \
                                n_orth=n_orth,FrozenPot=self.FrozenPot)

            elif self.method == 'fci':
                nel_, en_ = \
                    pyscf_fci.solve (self.mol, \
                                2*(self.nup-X_core.shape[1]), \
                                X_core, cf, ImpOrbs, chempot=chempot, \
                                n_orth=n_orth)

            elif self.method == 'dmrg':
                nel_, en_ = \
                    dmrg.solve (self.mol, \
                                2*(self.nup-X_core.shape[1]), \
                                X_core, cf, ImpOrbs, chempot=chempot, \
                                n_orth=n_orth)

            nelec  += nel_
            energy += en_

        if(self.parallel):
           nelec_tot  = comm.reduce(nelec, op=MPI.SUM,root=0)
           energy_tot = comm.reduce(energy,op=MPI.SUM,root=0)
           if(rank==0):
              energy_tot    += self.mol.energy_nuc()+self.e_core
           nelec  = comm.bcast(nelec_tot, root=0)
           energy = comm.bcast(energy_tot,root=0)
           comm.barrier()
           if(rank==0): print ( 'DMET energy = ', energy )
        else:
           energy+=self.mol.energy_nuc()+self.e_core
           print ( 'DMET energy = ', energy )

        return nelec

    def eval (self):
        if self.method != 'dmrg':
            mu = opt.newton (self.nelec_diff, 0.0)
        else:
            mu = opt.newton (self.nelec_diff, 0.0, tol=1.0e-5)
        if(not self.parallel):
           print ( "converged chemical potential =", mu )
        else:
           from mpi4py import MPI
           comm = MPI.COMM_WORLD
           rank = MPI.COMM_WORLD.Get_rank()
           size = MPI.COMM_WORLD.Get_size()
           if(rank==0):
              print ( "converged chemical potential =", mu )

    def nelec_diff (self, chempot):
        Nelec_dmet   = self.hl_solver (chempot)
        Nelec_target = self.mol.nelectron
        if(not self.parallel):
           print ( "(chem pot , # electrons) = (", \
                   chempot, "," , Nelec_dmet ,")" )
        else:
           from mpi4py import MPI
           comm = MPI.COMM_WORLD
           rank = MPI.COMM_WORLD.Get_rank()
           size = MPI.COMM_WORLD.Get_size()
           if(rank==0):
              print ( "(chem pot , # electrons) = (", \
                   chempot, "," , Nelec_dmet ,")" )
        return Nelec_dmet - Nelec_target
