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

mm = mp.MP2(m)
mm.kernel()
del mol, m, mm

print()
print("Starting DMET")
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
# method   = 'dfmp2'
nfreeze  = 0
parallel = False


# orb.DMET_wrap(atoms,basis,charge,spin,fragments,fragment_spins,shells,nfreeze,method,thresh,parallel)
# print("|||||||||||| dfmp2 solver compeleted |||||||||||||||")

method2 = 'mp2'
orb.DMET_wrap(atoms,basis,charge,spin,fragments,fragment_spins,shells,nfreeze,method2,thresh,parallel)
print("||||||||||||||||||||| mp2 solver completed ||||||||||||||||")
