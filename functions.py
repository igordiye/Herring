'''
These functions can be used to get (pr|qs) for only 4 selected orbitals,
which is useful if there is no memory for the whole eri tensor.
'''
import numpy,os
from pyscf import gto


#partition orbitals in shells
def aotosh(mol):

    nat=mol.natm
    nsh=len(range(mol.nbas))
    nao=mol.nao_nr()
    sis=numpy.zeros((nao,3),dtype=int)

    mu=0
    for sh_id in range(nsh):
        sh_size=gto.ao_loc_nr(mol)[sh_id+1]-gto.ao_loc_nr(mol)[sh_id]
        for ao_id in range(sh_size):
            sis[mu,:]=[sh_id,ao_id,sh_size]
            mu=mu+1
    return sis


#get the eri for 4 orbitals prqs
def eri_prqs(mol,sis,p,r,q,s):

    shls     = (sis[p,0],sis[r,0],sis[q,0],sis[s,0])
    eri_prqs = mol.intor_by_shell('cint2e_sph',shls)
    return eri_prqs[sis[p,1],sis[r,1],sis[q,1],sis[s,1]]



if __name__ == '__main__':
    from pyscf import gto

    atoms=\
    [['C',( 0.0000, 0.0000, 0.7680)],\
     ['C',( 0.0000, 0.0000,-0.7680)],\
     ['H',(-1.0192, 0.0000, 1.1573)],\
     ['H',( 0.5096, 0.8826, 1.1573)],\
     ['H',( 0.5096,-0.8826, 1.1573)],\
     ['H',( 1.0192, 0.0000,-1.1573)],\
     ['H',(-0.5096,-0.8826,-1.1573)],\
     ['H',(-0.5096, 0.8826,-1.1573)]]

    mol = gto.M(atom=atoms,basis='cc-pvdz')

    #usage
    sis = aotosh(mol)
    x     =  eri_prqs(mol,sis,1,1,1,1)
    print(x)
