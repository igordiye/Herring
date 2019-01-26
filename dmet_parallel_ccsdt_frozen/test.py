from sys import path
path.append('/Users/yuliya/git/DMET/dmet_parallel_ccsdt_frozen')
import orbital_selection_fc as orb
import numpy,os
from pyscf import gto,scf,cc,mp
from pyscf.mp import dfmp2

#geometry in angstrom from cccbdb
# atoms=\
# [['C',( 0.0000, 0.0000, 0.7680)],\
#  ['C',( 0.0000, 0.0000,-0.7680)],\
#  ['H',(-1.0192, 0.0000, 1.1573)],\
#  ['H',( 0.5096, 0.8826, 1.1573)],\
#  ['H',( 0.5096,-0.8826, 1.1573)],\
#  ['H',( 1.0192, 0.0000,-1.1573)],\
#  ['H',(-0.5096,-0.8826,-1.1573)],\
#  ['H',(-0.5096, 0.8826,-1.1573)]]

atoms = [
['O' , (0. , 0. , 0.)],\
['H' , (0. , -0.757 , 0.587)],\
['H' , (0. , 0.757  , 0.587)]]

mol = gto.M(atom=atoms, basis='cc-pvdz')
m   = scf.RHF(mol)
m.kernel()
# mm  = cc.CCSD(m)
# mm = mp.MP2(m)
# mm.kernel()

mdf = scf.RHF(mol).density_fit()
mdf.kernel()

mp2_df = dfmp2.DFMP2(mdf)
mp2_df.kernel()


del mol, m,  mp2_df #,mm

bs     = 'dz'
basis  = {'O': 'cc-pv'+bs, 'H': 'cc-pv'+bs}
shells = {'O': ['sto-6g','cc-pv'+bs], 'H': ['sto-6g','cc-pv'+bs]}
charge = 0
spin   = 0

# fragments = [[0,2,3,4],[1,5,6,7]]
# fragment_spins = [1,-1]
# fragments = [[0,2,3,4,1,5,6,7]]
fragments = [[0,1,2]]
fragment_spins = [0]
thresh   = 1.0e-8
#method = 'cc'
method   = 'dfmp2_testing4'
# method = 'mp2'
nfreeze  = 0
parallel = False

orb.DMET_wrap(atoms,basis,charge,spin,fragments,fragment_spins,shells,nfreeze,method,thresh,parallel)
print("dfmp2 solver compeleted \
        ------")

# method2 = 'mp2'
# orb.DMET_wrap(atoms,basis,charge,spin,fragments,fragment_spins,shells,nfreeze,method2,thresh,parallel)
# print("mp2 solver completed")
