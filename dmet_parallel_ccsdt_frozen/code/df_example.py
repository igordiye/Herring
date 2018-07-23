import numpy
from pyscf import gto, scf, df, ao2mo
from scipy import linalg as LA
from   pyscf.scf    import _vhf

#molecule
mol = gto.M(atom='H 0 0 0; H 0 0 1.2', basis='ccpvdz')
#now we do the calculations without the exact ERI but with the DF approximation
mf = scf.RHF(mol).density_fit()
#binary file where to save the rank-3 decomposition
mf.with_df._cderi_to_save = 'saved_cderi.h5'
mf.kernel()

#auxiliary molecule
auxmol = df.incore.format_aux_basis(mol,auxbasis='weigend')
# three-center integrals
j3c    = df.incore.aux_e2(mol, auxmol, intor='cint3c2e_sph', aosym='s1')
#number of basis functions
nao    = mol.nao_nr()
#number of auxiliary basis functions
naoaux = auxmol.nao_nr()
j3c    = j3c.reshape(nao,nao,naoaux) # (ij|L)
#overlap matrix between auxiliary basis functions
j2c    = df.incore.fill_2c2e(mol, auxmol) #(L|M)
#the eri is (ij|kl) = \sum_LM (ij|L) (L|M) (M|kl)

omega = LA.inv(j2c)
eps,U = LA.eigh(omega)
#after this transformation the eri is (ij|kl) = \sum_L (ij|L) (L|kl)
j3c   = numpy.dot(numpy.dot(j3c,U),numpy.diag(numpy.sqrt(eps)))
eriDF = numpy.einsum('ijm,klm->ijkl',j3c,j3c)

rho    = mf.make_rdm1()/2.0
gamma  = 4.0*numpy.einsum('pr,qs->prqs',rho,rho)-2.0*numpy.einsum('ps,qr->prqs',rho,rho)
E2     = 0.5*numpy.einsum('prqs,rpsq',eriDF,gamma)
print ("HF 2-body electronic energy (pyscf)",mf.energy_elec()[1])
print ("HF 2-body electronic energy (recomputed)",E2)

# with these lines of code we change basis from AOs to RHF orbitals
C = mf.mo_coeff
j3c  = numpy.einsum('abm,pa->pbm',j3c,C.T)
j3c  = numpy.einsum('abm,br->arm',j3c,C  )
eriDF = numpy.einsum('ijm,klm->ijkl',j3c,j3c)

rho    = mf.make_rdm1()/2.0
rho    = numpy.dot(LA.inv(C),numpy.dot(rho,LA.inv(C).T))
gamma  = 4.0*numpy.einsum('pr,qs->prqs',rho,rho)-2.0*numpy.einsum('ps,qr->prqs',rho,rho)
E2     = 0.5*numpy.einsum('prqs,rpsq',eriDF,gamma)
print ("HF 2-body electronic energy (pyscf)",mf.energy_elec()[1])
print ("HF 2-body electronic energy (recomputed)",E2)
