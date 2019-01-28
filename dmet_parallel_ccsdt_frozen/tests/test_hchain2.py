from sys import path
path.append('/home/yuliya/git/Herring/dmet_parallel_ccsdt_frozen')
import orbital_selection_fc as orb
import numpy,os
from pyscf import gto,scf,cc,mp
from pyscf.mp import dfmp2

R = 1.8 # Bonr units
N = 2
atoms = []
for i in range(N):
    atoms.append(['H', (i*R,0,0)])

mol = gto.M(atom=atoms, basis='sto-6g')
m   = scf.RHF(mol)
m.kernel()
mm  = cc.CCSD(m)
mm = mp.MP2(m)
mm.kernel()

# mdf = scf.RHF(mol).density_fit()
# mdf.kernel()
# print("E_dfscf")
# print(mdf.kernel())
#
# mp2_df = dfmp2.DFMP2(mdf)
# mp2_df.kernel()
# print("Edfmp2, t2")
# print(mp2_df.kernel())

del mol, m, mm #, mp2_df #,mm

# bs     = 'dz'
# basis  = {'H': 'cc-pv'+bs}
# shells = {'H': ['sto-6g','cc-pv'+bs]}
basis  = {'H': 'sto-6g'}
shells = {'H': ['sto-6g','sto-6g']}
charge = 0
spin   = 0
fragments = [[0,1]]
fragment_spins = [0]
thresh   = 1.0e-8
#method = 'cc'
# method   = 'dfmp2_testing4'
# method = 'mp2'
nfreeze  = 0
parallel = False


# orb.DMET_wrap(atoms,basis,charge,spin,fragments,fragment_spins,shells,nfreeze,method,thresh,parallel)
# print("|||||||||||| dfmp2 solver compeleted |||||||||||||||")

method2 = 'mp2'
orb.DMET_wrap(atoms,basis,charge,spin,fragments,fragment_spins,shells,nfreeze,method2,thresh,parallel)
print("||||||||||||||||||||| mp2 solver completed ||||||||||||||||")
