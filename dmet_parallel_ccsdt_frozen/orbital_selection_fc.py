import numpy
import pyscf
from   pyscf       import gto,scf
from   pyscf.tools import localizer

from sys import path
path.append('/home/yuliya/git/DMET/dmet_parallel_ccsdt_frozen/code')
import dmet

#====================================================

def atomic_spins(s):
    if(s=='H'): return 1
    if(s=='C'): return 2
    if(s=='O'): return 2
    if(s=='N'): return 3

def core_orbitals(s):
    if(s=='H'): return 0
    if(s=='C'): return 1
    if(s=='O'): return 1
    if(s=='N'): return 1

def projector(C,S):
    return numpy.dot(C,numpy.dot(C.T,S))

def orthonormalize(C,S,task):
    from scipy import linalg as LA
    if(task=='orthonormalize'):
       M       = numpy.dot(C.T,numpy.dot(S,C))
       val,vec = LA.eigh(-M)
       idx     = -val > 1.e-12
       U       = numpy.dot(vec[:,idx]*1.0/numpy.sqrt(-val[idx]),vec[:,idx].T)
       C       = numpy.dot(C,U)
    if(task=='normalize'):
       val = numpy.diag(numpy.dot(C.T,numpy.dot(S,C)))
       C  /= numpy.sqrt(val)
    return C

def project(C,S,Cprime,task):
    import numpy as np
    from   scipy import linalg as sla
    if(task=='along'):
       M   = np.dot(np.dot(C.T,S),Cprime)
       Q,R = sla.qr(M,mode='economic')
       return np.dot(C,Q)
    else:
       M     = np.dot(np.dot(C.T,S),Cprime)
       U,s,V = sla.svd(M, full_matrices=True)
       return np.dot(C,U[:,Cprime.shape[1]:])

def matrixprint(M):
    for i in range(M.shape[0]):
        print ( M[i,:] )

#====================================================

def build_iAO_basis(mol,Cf,Cf_core,Cf_vale,nfreeze):
    import numpy as np
    import scipy.linalg as sla
    S_f  = mol.intor_symmetric('cint1e_ovlp_sph')
    nup  = mol.nelectron//2
    iao, Cf_core = build_iao(S_f,Cf[:,:nup],Cf_vale,P_core=Cf_core,nfreeze=nfreeze)
    loc          = localizer.localizer(mol,iao,'boys')
    loc.verbose  = 5
    iao_loc      = loc.optimize (threshold=1.0e-5)
    del loc
    # need to find occupied orbitals orthogonal to core
    if(Cf_core is not None):
       Cf_x = project(Cf[:,:nup],S_f,Cf_core,'out')
       return iao_loc,Cf_core,Cf_x
    else:
       Cf_x = Cf[:,:nup]
       return iao_loc,None,Cf_x

def build_iao(S, C_oc, P_valence, P_core=None, P_virt=None, nfreeze=None):
    # C_oc      :: occupied orbitals
    # P_valence :: valence basis functions
    # S         :: overlap matrix
    import numpy as np
    import scipy.linalg as sla
    assert (P_virt is None)

    nb = S.shape[0]

    if(nfreeze>0):
     A_core = C_oc[:,:nfreeze]
     XIX    = np.eye(nb)-projector(A_core,S)
     C_oc_  = C_oc[:,nfreeze:]
    else:
     if P_core is not None:
         Px = orthonormalize(P_core,S,'orthonormalize')
         # project putative core into occupied orbitals, orthonormalize
         A_core = project(C_oc,S,Px,'along')
         XIX    = np.eye(nb)-projector(A_core,S)
         C_oc_  = project(C_oc,S,A_core,'out')
     else:
         XIX    = np.eye(nb)
         C_oc_  = C_oc

    # find IAO's
    # [ B2 is a subset of B1 ]
    # symmetric orthonormalization of B2 subset
    Px = orthonormalize(np.dot(XIX,P_valence),S,'orthonormalize')

    # project occupied orbitals into B2 subset, defining orthonormal set C_oc_p
    C_oc_p = project(Px,S,C_oc_,'along')

    # apply iao construction
    M1 = projector(C_oc_, S)
    M2 = projector(C_oc_p,S)
    At = np.dot(np.dot(    M1,     M2), Px) \
       + np.dot(np.dot(XIX-M1, XIX-M2), Px)
    A  = orthonormalize(At,S,'orthonormalize')
    A_valence = A

    if P_core is None and P_virt is None:
        return A_valence, None
    elif P_virt is None:
        return A_valence, A_core



def orbital_partitioning(mol,fragments,shells,verbose):

    at_species,at_orbitals,species=[],[],[]
    natom=0

    for i,(sh0,sh1,ao0,ao1) in enumerate(mol.offset_nr_by_atom()):
        name = mol.atom[i][0]
        if(name not in species): species.append(name)

        at_orbitals.append((ao0,ao1))
        at_species.append(species.index(name))
        natom+=1

    if(verbose): print ( "number of atoms ",natom )
    if(verbose): print ( "atomic species  ",species )
    T_atom,n_core,n_vale,n_virt=[],[],[],[]

    for s in species:
        orbitals = []
        nc       = core_orbitals(s)
        nsh = len(shells[s])
        for ib in range(nsh):
            pmol          = gto.Mole()
            pmol.atom     = [[s,(0.0,0.0,0.0)]]
            pmol.basis    = {s: shells[s][ib]}
            pmol.charge   = 0
            pmol.spin     = atomic_spins(s)
            pmol.symmetry = False
            pmol.build()
            nbasis        = pmol.nao_nr()
            orbitals.append(nbasis-nc)

        pmf = scf.ROHF(pmol)
        pmf.max_cycle  = 5000
        pmf.conv_tol   = 1e-6
        pmf = scf.newton(pmf)
        pmf.scf()
        T_atom.append(pmf.mo_coeff)

        n_core.append(nc)
        n_vale.append(orbitals[ 0])
        n_virt.append(orbitals[0:])

    for i in range(natom):
       j = at_species[i]
       if(verbose): print ( "atom ",i," species ",species[j], \
                          " AOs ",at_orbitals[i], \
                          " core,vale,virt ",n_core[j],n_vale[j],n_virt[j] )

    #===============================================================

    nbasis=mol.nao_nr()
    n_core_tot,n_vale_tot,n_virt_tot=0,0,0
    for i in range(natom):
        j = at_species[i]
        n_core_tot += n_core[j]
        n_vale_tot += n_vale[j]
        n_virt_tot += n_virt[j][nsh-1]-n_vale[j]

    if(verbose): print ( "number of basis          orbitals: ",nbasis )
    if(verbose): print ( "number of core,vale,virt orbitals: ",n_core_tot,n_vale_tot,n_virt_tot )

    if(n_core_tot==0): Cf_core = None
    else:              Cf_core = numpy.zeros((nbasis,n_core_tot),dtype=float)

    imin,imax = 0,0
    for i in range(natom):
        j         = at_species[i]
        (ao0,ao1) = at_orbitals[i]
        nc = n_core[j]
        if(nc>0):
           if(verbose): print ( "atom ",i," AOs ",ao0," to ",ao1, \
                              "CORE block (",ao0,",",ao1-1,") x (",imin,",",imin+nc,")" )
           imax = imin+nc
           Cf_core[ao0:ao1,imin:imax]=T_atom[j][:,:nc]
           imin = imax

    Cf_vale = numpy.zeros((nbasis,n_vale_tot),dtype=float)

    imin,imax = 0,0
    for i in range(natom):
        j         = at_species[i]
        (ao0,ao1) = at_orbitals[i]
        nc = n_core[j]
        nv = n_vale[j]
        imax = imin+nv
        if(verbose): print ( "atom ",i," AOs ",ao0," to ",ao1, \
                           "VALE block (",ao0,",",ao1-1,") x (",imin,",",imax-1,")" )
        Cf_vale[ao0:ao1,imin:imax] = T_atom[j][:,nc:nv+nc]
        imin = imax

    Cf_virt = numpy.zeros((nbasis,n_virt_tot),dtype=float)

    imin,imax = 0,0
    for i in range(natom):
        j         = at_species[i]
        (ao0,ao1) = at_orbitals[i]
        ncr = n_core[j]
        nva = n_vale[j]
        nvt = n_virt[j][nsh-1]
        imax = imin+nvt-nva
        if(verbose): print ( "atom ",i," AOs ",ao0," to ",ao1, \
                           "VIRT block (",ao0,",",ao1-1,") x (",imin,",",imax-1,")" )
        Cf_virt[ao0:ao1,imin:imax] = T_atom[j][:,nva+ncr:]
        imin = imax

    return Cf_core,Cf_vale,Cf_virt



def RHF_calculation(mol,verbose):
    mf_mol           = scf.RHF(mol)
    mf_mol.verbose   = 4
    mf_mol.max_cycle = 5000
    mf_mol.conv_tol  = 1e-6
    mf_mol           = scf.newton(mf_mol)
    mf_mol.kernel()
    if(verbose): print ( 'Total SCF energy',mf_mol.energy_tot() )

#    from pyscf import cc
#    ccsolver = cc.CCSD(mf_mol)
#    ccsolver.verbose = 5
#    ccsolver.ccsd()
#    exit()

    return mf_mol.mo_coeff



def virtual_orbitals(mol,Cf_core,Cf_vale,Cf_virt,iAO_loc):
    import numpy as np
    nbasis  = mol.nao_nr()
    S_f     = mol.intor_symmetric('cint1e_ovlp_sph')
    if(Cf_core is not None): P_core = projector(Cf_core,S_f)
    else:                    P_core = np.zeros((nbasis,nbasis))
    P_iAO   = projector(iAO_loc,S_f)
    Cf_virt = np.dot(np.eye(nbasis)-P_iAO-P_core,Cf_virt)
    Cf_virt = orthonormalize(Cf_virt,S_f,'normalize')
    return Cf_virt



def atom_to_orb_mapping(mol,C):
    x_oper = mol.intor('cint1e_r_sph', comp=3)
    norb = C.shape[1]

    idx  = numpy.zeros(norb,dtype=int)
    for i in range(norb):
        x_aver=numpy.einsum('a,mab,b->m',C[:,i],x_oper,C[:,i])
        dmin,idmin=100000.0,-1
        for j in range(mol.natm):
            dx = x_aver - mol.atom_coord(j)
            dj = numpy.sqrt(numpy.dot(dx,dx))
            if(dj<dmin): dmin=dj; idmin=j
        idx[i]=idmin

    return idx



def atom_to_frg_mapping(fragments):
    ximp_at=[]
    natom=0
    for f in fragments: natom+=len(f)
    for f in fragments: v = [ (x in f) for x in range(natom) ]; ximp_at.append(numpy.array(v))

    return ximp_at



#----------------------------------------------------------------------------------------#

def DMET_wrap(atoms,basis,charge,spin,fragments,fragment_spins,shells,nfreeze,method,thresh,parallel):
    mol          = pyscf.gto.Mole()
    mol.verbose  = 4
    mol.output   = 'Mole'
    mol.atom     = atoms
    mol.charge   = charge
    mol.spin     = spin
    mol.basis    = basis
    mol.symmetry = False
    mol.ecp      = None
    mol.build()

    verbose=False
    if(parallel):
       from mpi4py import MPI
       comm = MPI.COMM_WORLD
       rank = MPI.COMM_WORLD.Get_rank()
       size = MPI.COMM_WORLD.Get_size()
       if(rank==0): verbose=True
    else:
       verbose=True

    Cf_core,Cf_vale,Cf_virt = orbital_partitioning(mol,fragments,shells,verbose)
    Cf                      = RHF_calculation(mol,verbose)

    iAO_loc,Cf_core,Cf_x    = build_iAO_basis(mol,Cf,Cf_core,Cf_vale,nfreeze)
    Cf_virt                 = virtual_orbitals(mol,Cf_core,Cf_vale,Cf_virt,iAO_loc)

    nb=Cf_core.shape[0]
    FrozenPot = numpy.zeros((nb,nb))
    e_core    = 0.0
    if(nfreeze>0):

       import numpy as np
       from pyscf import scf
       Hc  = mol.intor_symmetric('cint1e_kin_sph') \
           + mol.intor_symmetric('cint1e_nuc_sph')
       dm_core = np.dot(Cf_core,Cf_core.T)*2
       jk_core = scf.hf.get_veff(mol, dm_core)
       e_core  =     np.trace(np.dot(Hc, dm_core)) \
               + 0.5*np.trace(np.dot(jk_core, dm_core))
       FrozenPot = jk_core
       Cf_core = None
       mol.nelectron -= 2*nfreeze

    print ( nfreeze )

    idx_core = None
    if(Cf_core is not None): idx_core = atom_to_orb_mapping(mol,Cf_core)
    idx_vale = atom_to_orb_mapping(mol,Cf_vale)
    idx_virt = atom_to_orb_mapping(mol,Cf_virt)
    ximp_at  = atom_to_frg_mapping(fragments)

    if(parallel):

       from mpi4py import MPI
       comm = MPI.COMM_WORLD
       rank = MPI.COMM_WORLD.Get_rank()
       size = MPI.COMM_WORLD.Get_size()

       assert(len(fragments)==size)

       Cf_x     = comm.bcast(Cf_x,     root=0)
       iAO_loc  = comm.bcast(iAO_loc,  root=0)
       idx_vale = comm.bcast(idx_vale, root=0)
       Cf_core  = comm.bcast(Cf_core,  root=0)
       idx_core = comm.bcast(idx_core, root=0)
       Cf_virt  = comm.bcast(Cf_virt,  root=0)
       idx_virt = comm.bcast(idx_virt, root=0)
       ximp_at  = comm.bcast(ximp_at,  root=0)

    dmet_ = dmet.dmet(mol, Cf_x, ximp_at, \
                      iAO_loc, idx_vale, method=method, thresh=thresh, \
                      A_core  = Cf_core, at_core = idx_core, \
                      A_virt  = Cf_virt, at_virt = idx_virt, \
                      imp_atx = ximp_at, parallel = parallel, e_core=e_core, FrozenPot=FrozenPot)
    dmet_.eval()
