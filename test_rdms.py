from sys import path
# path.append('/Users/yuliya/pyscf_v5/pyscf')
path.append('/home/yuliya/pyscf_v5/pyscf')
from pyscf    import gto,scf,cc,mp,ao2mo,df, lib
from pyscf.mp import dfmp2
import numpy as np
from numpy    import sqrt,einsum
from scipy    import linalg as LA
from dmet_parallel_ccsdt_frozen.code import dfmp2_testing

'''Comparison between a molecular object treated at DFMP2 level
and a fake molecular object with the same Hamiltonian, but
treated with standard MP2.
'''


# define a molecule, and treat it with DFMP2

R=1.5
atoms = [['O',(0,0,0)],['H',(R,0,0)],['H',(-R*sqrt(3)/2,R/2,0)]]
mol   = gto.M(atom=atoms,basis='cc-pvdz',verbose=2)
m     = scf.RHF(mol).density_fit().run()
# m.kernel()
mo_coeff = m.mo_coeff
mo_energy = m.mo_energy
nocc = mol.nelectron//2

mm    =  dfmp2.DFMP2(m).run()
mp2solver = mm

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
eri = mol.intor('cint2e_sph', aosym='s8')
eri = ao2mo.incore.general(eri, (co,cv,co,cv))
eri = ao2mo.load(eri)

t2 = np.empty((nocc,nocc,nvir,nvir))
eia = mo_energy[:nocc,None] - mo_energy[None,nocc:]
with eri as ovov:
    for i in range(nocc):
        gi = np.asarray(ovov[i*nvir:(i+1)*nvir])
        gi = gi.reshape(nvir, nocc, nvir).transpose(1,0,2)
        t2[i] = gi/lib.direct_sum('jb+a->jba', eia, eia[i])

g1 = make_rdm1(mp2solver, t2, mo_coeff, mo_energy, nocc)
g2 = make_rdm2(mp2solver, t2, mo_coeff, mo_energy, nocc)


# now build a fake molecule where the rank-3 eri is
# folded back into a rank-4 object, and treat it with MP2

nbasis             = mol.nao_nr()
mol_               = gto.Mole()
mol_.build(verbose=2)
mol_.nelectron     = mol.nelectron
mol_.incore_anyway = True

# get the Hamiltonian
H0 = mol.energy_nuc()
S1 = mol.intor_symmetric('cint1e_ovlp_sph')
H1 = mol.intor_symmetric('cint1e_kin_sph')+mol.intor_symmetric('cint1e_nuc_sph')

m     = scf.RHF(mol).density_fit().run()
m.kernel()

auxmol = df.incore.format_aux_basis(mol,auxbasis=m.with_df.auxbasis)
j3c    = df.incore.aux_e2(mol,auxmol,intor='cint3c2e_sph',aosym='s1')
nao    = mol.nao_nr()
naoaux = auxmol.nao_nr()
j3c    = j3c.reshape(nao,nao,naoaux) # (ij|L)
j2c    = df.incore.fill_2c2e(mol,auxmol)
H2     = einsum('prL,LM,qsM->prqs',j3c,LA.inv(j2c),j3c)

mol_.energy_nuc = lambda *args: H0
m_              = scf.RHF(mol_)
m_.mo_coeff     = m.mo_coeff
m_.mo_occ       = m.mo_occ
m_.get_ovlp     = lambda *args: S1
m_.get_hcore    = lambda *args: H1
m_._eri         = ao2mo.restore(8,H2,nbasis)

E_hf = m_.kernel()
mm_  = mp.MP2(m_)
mm_.kernel()

print(np.linalg.det(np.einsum('ab,bc,cd->ad',m_.mo_coeff[:,m_.mo_occ>0].T,S1,m.mo_coeff[:,m.mo_occ>0])))
print(np.abs(m_.mo_coeff[:,m_.mo_occ>0]-m.mo_coeff[:,m.mo_occ>0]).max())

g1_ = mm_.make_rdm1()
g2_ = mm_.make_rdm2()

print(np.abs(g1-g1_).max())
print(np.abs(g2-g2_).max())
