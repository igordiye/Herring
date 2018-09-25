from sys import path
# path.append('/Users/yuliya/pyscf_v5/pyscf')
path.append('/home/yuliya/pyscf_v5/pyscf')
from pyscf    import gto,scf,cc,mp,ao2mo,df, lib
from pyscf.mp import dfmp2
from numpy    import sqrt,einsum
from scipy    import linalg as LA
import numpy as np

# ----- local functions ----- #

def get_t2(mp):
    '''basically identical to the DFMP2 kernel, returns t2'''
    from pyscf.mp import mp2
    mo_coeff  = mp2._mo_without_core(mp,mp.mo_coeff)
    mo_energy = mp2._mo_energy_without_core(mp, mp.mo_energy)
    nocc = mp.nocc
    nvir = mp.nmo - nocc
    eia  = mo_energy[:nocc,None] - mo_energy[None,nocc:]
    emp2 = 0
    t2   = []
    for istep, qov in enumerate(mp.loop_ao2mo(mo_coeff, nocc)):
        for i in range(nocc):
            buf = np.dot(qov[:,i*nvir:(i+1)*nvir].T,qov).reshape(nvir,nocc,nvir)
            gi  = np.array(buf,copy=False)
            gi  = gi.reshape(nvir,nocc,nvir).transpose(1,0,2)
            t2i = gi.conj()/lib.direct_sum('jb+a->jba',eia,eia[i])
            t2.append(t2i)
            emp2 += np.einsum('jab,jab',t2i,gi) * 2
            emp2 -= np.einsum('jab,jba',t2i,gi)
    return emp2,t2

def make_rdm1(mp2solver):
    '''rdm1 in the MO basis'''
    from pyscf.cc import ccsd_rdm
    doo, dvv = _gamma1_intermediates(mp2solver)
    nocc = doo.shape[0]
    nvir = dvv.shape[0]
    dov  = np.zeros((nocc,nvir), dtype=doo.dtype)
    dvo  = dov.T
    return ccsd_rdm._make_rdm1(mp,(doo,dov,dvo,dvv),with_frozen=False)

def _gamma1_intermediates(mp,t2=None):
    nmo  = mp.nmo
    nocc = mp.nocc
    nvir = nmo - nocc
    from pyscf.mp import mp2
    mo_coeff  = mp2._mo_without_core(mp, mp.mo_coeff)
    mo_energy = mp2._mo_energy_without_core(mp, mp.mo_energy)
    eia = mo_energy[:nocc,None] - mo_energy[None,nocc:]
    if(t2 is None):
        for istep, qov in enumerate(mp.loop_ao2mo(mo_coeff, nocc)):
            if(istep==0):
                dtype = qov.dtype
                dm1occ = np.zeros((nocc,nocc), dtype=dtype)
                dm1vir = np.zeros((nvir,nvir), dtype=dtype)
            for i in range(nocc):
                buf = np.dot(qov[:,i*nvir:(i+1)*nvir].T,
                               qov).reshape(nvir,nocc,nvir)
                gi = np.array(buf, copy=False)
                gi = gi.reshape(nvir,nocc,nvir).transpose(1,0,2)
                t2i = gi/lib.direct_sum('jb+a->jba', eia, eia[i])
                l2i = t2i.conj()
                dm1vir += np.einsum('jca,jcb->ba', l2i, t2i) * 2 \
                       - np.einsum('jca,jbc->ba', l2i, t2i)
                dm1occ += np.einsum('iab,jab->ij', l2i, t2i) * 2 \
                       - np.einsum('iab,jba->ij', l2i, t2i)
    else:
        dtype = t2[0].dtype
        dm1occ = np.zeros((nocc,nocc), dtype=dtype)
        dm1vir = np.zeros((nvir,nvir), dtype=dtype)
        for i in range(nocc):
            t2i = t2[i]
            l2i = t2i.conj()
            dm1vir += np.einsum('jca,jcb->ba', l2i, t2i) * 2 \
                  - np.einsum('jca,jbc->ba', l2i, t2i)
            dm1occ += np.einsum('iab,jab->ij', l2i, t2i) * 2 \
                  - np.einsum('iab,jba->ij', l2i, t2i)
    return -dm1occ, dm1vir

def make_rdm2(mp2solver):
    nmo  = nmo0  = mp2solver.nmo
    nocc = nocc0 = mp2solver.nocc
    nvir = nmo - nocc
    from pyscf.mp import mp2
    mo_coeff  = mp2._mo_without_core(mp2solver, mp2solver.mo_coeff)
    mo_energy = mp2._mo_energy_without_core(mp2solver, mp2solver.mo_energy)
    eia       = mo_energy[:nocc,None] - mo_energy[None,nocc:]

    moidx = oidx = vidx = None
    dm1   = make_rdm1(mp2solver)
    dm1[np.diag_indices(nocc0)] -= 2
    dm2   = np.zeros((nmo0,nmo0,nmo0,nmo0), dtype=dm1.dtype)

    if(t2 is None):
        for istep, qov in enumerate(mp2solver.loop_ao2mo(mo_coeff, nocc)):
            for i in range(nocc):
                buf = np.dot(qov[:,i*nvir:(i+1)*nvir].T,qov).reshape(nvir,nocc,nvir)
                gi  = np.array(buf,copy=False)
                gi  = gi.reshape(nvir,nocc,nvir).transpose(1,0,2)
                t2i = gi.conj()/lib.direct_sum('jb+a->jba',eia,eia[i])
                dovov = t2i.transpose(1,0,2)*2 - t2i.transpose(2,0,1)
                dovov *= 2
                if moidx is None:
                    dm2[i,nocc:,:nocc,nocc:] = dovov
                    dm2[nocc:,i,nocc:,:nocc] = dovov.conj().transpose(0,2,1)
                else:
                    dm2[oidx[i],vidx[:,None,None],oidx[:,None],vidx] = dovov
                    dm2[vidx[:,None,None],oidx[i],vidx[:,None],oidx] = dovov.conj().transpose(0,2,1)

    else:
        for i in range(nocc):
            t2i = t2[i]
            dovov = t2i.transpose(1,0,2)*2 - t2i.transpose(2,0,1)
            dovov *= 2
            if moidx is None:
                dm2[i,nocc:,:nocc,nocc:] = dovov
                dm2[nocc:,i,nocc:,:nocc] = dovov.conj().transpose(0,2,1)
            else:
                dm2[oidx[i],vidx[:,None,None],oidx[:,None],vidx] = dovov
                dm2[vidx[:,None,None],oidx[i],vidx[:,None],oidx] = dovov.conj().transpose(0,2,1)

    for i in range(nocc0):
        dm2[i,i,:,:] += dm1.T * 2
        dm2[:,:,i,i] += dm1.T * 2
        dm2[:,i,i,:] -= dm1.T
        dm2[i,:,:,i] -= dm1

    for i in range(nocc0):
        for j in range(nocc0):
            dm2[i,i,j,j] += 4
            dm2[i,j,j,i] -= 2

    return dm2

# ----- the main program ----- #

# define a molecule, and treat it with DFMP2

R     = 1.5
atoms = [['O',(0,0,0)],['H',(R,0,0)],['H',(-R*sqrt(3)/2,R/2,0)]]
mol   = gto.M(atom=atoms,basis='cc-pvqz',verbose=2)
m     = scf.RHF(mol).density_fit()
EhfDF = m.kernel()
print("DFHF energy ")
print(EhfDF)

mo_coeff   = m.mo_coeff
mo_energy  = m.mo_energy
nocc       = mol.nelectron//2
mm         = dfmp2.DFMP2(m)
EmmDF,t2DF = mm.kernel()
print("DFMP2 energy ")
print(EmmDF)

EmmDFb,t2 = get_t2(mm)
print("DFMP2 energy (from get t2)")
print(EmmDFb)

mp2solver = mm

t2 = None
g1 = make_rdm1(mp2solver)
g2 = make_rdm2(mp2solver)

# then treat it with MP2, but passing the same ERI

nbasis             = mol.nao_nr()
mol_               = gto.Mole()
mol_.build(verbose=2)
mol_.nelectron     = mol.nelectron
mol_.incore_anyway = True

H0 = mol.energy_nuc()
S1 = mol.intor_symmetric('cint1e_ovlp_sph')
H1 = mol.intor_symmetric('cint1e_kin_sph')+mol.intor_symmetric('cint1e_nuc_sph')

auxmol = df.incore.format_aux_basis(mol,auxbasis=m.with_df.auxbasis)
j3c    = df.incore.aux_e2(mol,auxmol,intor='cint3c2e_sph',aosym='s1')
nao    = mol.nao_nr()
naoaux = auxmol.nao_nr()
j3c    = j3c.reshape(nao,nao,naoaux)
j2c    = df.incore.fill_2c2e(mol,auxmol)
H2     = einsum('prL,LM,qsM->prqs',j3c,LA.inv(j2c),j3c)

mol_.energy_nuc = lambda *args: H0
m_              = scf.RHF(mol_)
m_.mo_coeff     = m.mo_coeff
m_.mo_occ       = m.mo_occ
m_.get_ovlp     = lambda *args: S1
m_.get_hcore    = lambda *args: H1
m_._eri         = ao2mo.restore(8,H2,nbasis)

print("HF energy (from fake molecule) ")
print(m_.kernel())
if(np.abs(m.mo_coeff-m_.mo_coeff).max()>1e-6):
 print("attention: HF orbitals from this calculation differ from the old ones")
 m_.mo_coeff     = m.mo_coeff
 m_.mo_occ       = m.mo_occ

mm_ = mp.MP2(m_)
print("DFMP2 energy (from fake molecule) ")
print(mm_.kernel()[0])

g1_ = mm_.make_rdm1()
g2_ = mm_.make_rdm2()

print("deviations between 1rdm,2rdm in MO basis ")
print(np.abs(g1-g1_).max())
print(np.abs(g2-g2_).max())
