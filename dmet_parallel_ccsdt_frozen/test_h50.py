from sys import path
path.append('/Users/yuliya/git/DMET/dmet_parallel_ccsdt_frozen')
import orbital_selection_fc as orb
import numpy,os
from pyscf import gto,scf,cc,mp
from pyscf.mp import dfmp2

#geometry in angstrom from cccbdb
R = 1.8 # Bonr units
N = 30
atoms = []
for i in range(N):
    atoms.append(['H', (i*R,0,0)])

mol = gto.M(atom=atoms,basis='sto-6g')
m   = scf.RHF(mol)
m.kernel()
# mm  = cc.CCSD(m)
#mm = mp.MP2(m)
# mm.kernel()
mm =  dfmp2.MP2(m)
mm.kernel()
exit()


del mol,m ,mm

bs     = 'dz'
basis  = {'C': 'cc-pv'+bs, 'H': 'cc-pv'+bs}
shells = {'C': ['sto-6g','cc-pv'+bs], 'H': ['sto-6g','cc-pv'+bs]}
charge = 0
spin   = 0

fragments = []
for i in range (0,N,2):
    fragments.append([i, i+1])

fragment_spins = [0 for x in range(0, N, 2)]
thresh   = 1.0e-8
#method = 'cc'
method   = 'dfmp2_testing'
# method = 'mp2'
nfreeze  = 0
parallel = False

orb.DMET_wrap(atoms,basis,charge,spin,fragments,fragment_spins,shells,nfreeze,method,thresh,parallel)
