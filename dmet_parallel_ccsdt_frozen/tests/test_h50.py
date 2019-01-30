from sys import path
path.append('/home/yuliya/git/Herring/dmet_parallel_ccsdt_frozen')
import orbital_selection_fc as orb
import numpy,os
from pyscf import gto,scf,cc,mp
from pyscf.mp import dfmp2

R = 1 # Bohr units
N = 50
atoms = []
for i in range(N):
    atoms.append(['H', (i*R,0,0)])

mol = gto.M(atom=atoms,basis='sto-6g')
m   = scf.RHF(mol).density_fit()
m.kernel()
# mm  = cc.CCSD(m)
# mm = mp.MP2(m)

mm =  dfmp2.DFMP2(m)
mm.kernel()
del mol,m #,mm


print( )
print("Starting DMET")
basis  = {'H': 'sto-6g'}
shells = {'H': ['sto-6g','sto-6g']}
charge = 0
spin   = 0

fragments = []
for i in range (0,N,50):
    fragments.append(range(i,i+50)) #[i, i+1, i+2, i+3, i+4])

fragment_spins = [0]
thresh   = 1.0e-8
#method = 'cc'
method   = 'dfmp2'
# method = 'mp2'
nfreeze  = 0
parallel = False


orb.DMET_wrap(atoms,basis,charge,spin,fragments,fragment_spins,shells,nfreeze,method,thresh,parallel)
