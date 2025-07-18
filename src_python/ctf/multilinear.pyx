import numpy as np
from ctf.tensor cimport Tensor, tensor, ctensor
from libcpp cimport bool
from libc.stdlib cimport malloc, free
from ctf.profile import timer

cdef extern from "ctf.hpp" namespace "CTF":
    cdef void TTTP_ "CTF::TTTP"[dtype](Tensor[dtype] * T, int num_ops, int * modes, Tensor[dtype] ** mat_list, bool aux_mode_first)
    cdef void MTTKRP_ "CTF::MTTKRP"[dtype](Tensor[dtype] * T, Tensor[dtype] ** mat_list, int mode, bool aux_mode_first)
    cdef void Solve_Factor_ "CTF::Solve_Factor"[dtype](Tensor[dtype] * T, Tensor[dtype] ** mat_list, Tensor[dtype] * RHS, int mode, Tensor[dtype] *regu, double epsilon, double barrier, bool proj,bool add_ones, bool aux_mode_first)
    cdef void Solve_Factor_with_RHS_ "CTF::Solve_Factor_with_RHS"[dtype](Tensor[dtype] * T, Tensor[dtype] ** mat_list, Tensor[dtype] * RHS, int mode, double regu, double barrier, bool aux_mode_first)
    cdef void Solve_Factor_Tucker_ "CTF::Solve_Factor_Tucker"[dtype](Tensor[dtype] * T, Tensor[dtype] ** mat_list, Tensor[dtype] * core, Tensor[dtype] * RHS, int mode, Tensor[dtype] *regu, double epsilon, double barrier, bool proj,bool add_ones, bool aux_mode_first)
    cdef void Sparse_add_ "CTF::Sparse_add"[dtype](Tensor[dtype] * T, Tensor[dtype] * M, double alpha, double beta)
    cdef void Sparse_mul_ "CTF::Sparse_mul"[dtype](Tensor[dtype] * T, Tensor[dtype] * M )
    cdef void Sparse_div_ "CTF::Sparse_div"[dtype](Tensor[dtype] * T, Tensor[dtype] * M )
    cdef double Sparse_inner_prod_ "CTF::Sparse_inner_prod"[dtype](Tensor[dtype] * T, Tensor[dtype] * M)
    cdef void Sparse_exp_ "CTF::Sparse_exp"[dtype](Tensor[dtype] * T, double alpha)
    cdef void Sparse_log_ "CTF::Sparse_log"[dtype](Tensor[dtype] * T)
    cdef void Sparse_sigmoid_ "CTF::Sparse_sigmoid"[dtype](Tensor[dtype] * T, double alpha, double eps)
    cdef void get_index_tensor_ "CTF::get_index_tensor"[dtype](Tensor[dtype] * T)
def TTTP(tensor A, mat_list):
    """
    TTTP(A, mat_list)
    Compute updates to entries in tensor A based on matrices in mat_list (tensor times tensor products)
    This routine is generally much faster then einsum when A is sparse.

    Parameters
    ----------
    A: tensor_like
       Input tensor of arbitrary ndim

    mat_list: list of size A.ndim
              Contains either None or matrix of dimensions m-by-k or vector,
              where m matches the corresponding mode length of A and k is the same for all 
              given matrices (or all are vectors)

    Returns
    -------
    B: tensor
        A tensor of the same ndim as A, updating by taking products of entries of A with multilinear dot products of columns of given matrices.
        For ndim=3 and mat_list=[X,Y,Z], this operation is equivalent to einsum("ijk,ia,ja,ka->ijk",A,X,Y,Z)
    """
    #B = tensor(A.shape, A.sp, A.sym, A.dtype, A.order)
    #s = _get_num_str(B.ndim+1)
    #exp = A.i(s[:-1])
    t_tttp = timer("pyTTTP")
    t_tttp.start()
    if len(mat_list) != A.ndim:
        raise ValueError('CTF PYTHON ERROR: mat_list argument to TTTP must be of same length as ndim')
    
    k = -1
    cdef int * modes
    modes = <int*>malloc(len(mat_list)*sizeof(int))
    tsrs = <Tensor[double]**>malloc(len(mat_list)*sizeof(ctensor*))
    imode = 0
    cdef tensor t
    ntsrs = 0
    for i in range(len(mat_list))[::-1]:
        if mat_list[i] is not None:
            ntsrs += 1
            modes[imode] = len(mat_list)-i-1
            t = mat_list[i]
            tsrs[imode] = <Tensor[double]*>t.dt
            imode += 1
            if mat_list[i].ndim == 1:
                if k != -1:
                    raise ValueError('CTF PYTHON ERROR: mat_list must contain only vectors or only matrices')
                if mat_list[i].shape[0] != A.shape[i]:
                    raise ValueError('CTF PYTHON ERROR: input vector to TTTP does not match the corresponding tensor dimension')
                #exp = exp*mat_list[i].i(s[i])
            else:
                if mat_list[i].ndim != 2:
                    raise ValueError('CTF PYTHON ERROR: mat_list operands has invalid dimension')
                if k == -1:
                    k = mat_list[i].shape[1]
                else:
                    if k != mat_list[i].shape[1]:
                        raise ValueError('CTF PYTHON ERROR: mat_list second mode lengths of tensor must match')
                #exp = exp*mat_list[i].i(s[i]+s[-1])
    B = tensor(copy=A)
    if A.dtype == np.float64:
        TTTP_[double](<Tensor[double]*>B.dt,ntsrs,modes,tsrs,1)
    else:
        raise ValueError('CTF PYTHON ERROR: TTTP does not support this dtype')
    free(modes)
    free(tsrs)
    t_tttp.stop()
    return B


def MTTKRP(tensor A, mat_list, mode):
    """
    MTTKRP(A, mat_list, mode)
    Compute Matricized Tensor Times Khatri Rao Product with output mode given as mode, e.g.
    MTTKRP(A, [U,V,W,Z], 2) gives W = einsum("ijkl,ir,jr,lr->kr",A,U,V,Z).
    This routine is generally much faster then einsum when A is sparse.

    Parameters
    ----------
    A: tensor_like
       Input tensor of arbitrary ndim

    mat_list: list of size A.ndim containing matrices that are n_i-by-R where n_i is dimension of ith mode of A,
              on output mat_list[mode] will contain the output of the MTTKRP 
    """
    t_mttkrp = timer("pyMTTKRP")
    t_mttkrp.start()
    if len(mat_list) != A.ndim:
        raise ValueError('CTF PYTHON ERROR: mat_list argument to MTTKRP must be of same length as ndim')
    k = -1
    tsrs = <Tensor[double]**>malloc(len(mat_list)*sizeof(ctensor*))
    #tsr_list = []
    imode = 0
    cdef tensor t
    for i in range(len(mat_list))[::-1]:
        t = mat_list[i]
        tsrs[imode] = <Tensor[double]*>t.dt
        imode += 1
        if mat_list[i].ndim == 1:
            if k != -1:
                raise ValueError('CTF PYTHON ERROR: mat_list must contain only vectors or only matrices')
            if mat_list[i].shape[0] != A.shape[i]:
                raise ValueError('CTF PYTHON ERROR: input vector to MTTKRP does not match the corresponding tensor dimension')
            #exp = exp*mat_list[i].i(s[i])
        else:
            if mat_list[i].ndim != 2:
                raise ValueError('CTF PYTHON ERROR: mat_list operands has invalid dimension')
            if k == -1:
                k = mat_list[i].shape[1]
            else:
                if k != mat_list[i].shape[1]:
                    raise ValueError('CTF PYTHON ERROR: mat_list second mode lengths of tensor must match')
    B = tensor(copy=A)
    if A.dtype == np.float64:
        MTTKRP_[double](<Tensor[double]*>B.dt,tsrs,A.ndim-mode-1,1)
    else:
        raise ValueError('CTF PYTHON ERROR: MTTKRP does not support this dtype')
    free(tsrs)
    t_mttkrp.stop()

def Solve_Factor(tensor A, mat_list, tensor R, mode, tensor regu, epsilon, barrier= None, proj = False, add_ones = False):
    """
    Solve_Factor(A, mat_list,R, mode,regu)
    solves for a factor matrix parallelizing over rows constructing rhs, sparse tensor and list of factor matrices
    eg. for mode=0 order 3 tensor Computes LHS = einsum("ijk,jr,jz,kr,kz->irz",T,B,B,C,C) and 
    RHS = einsum("ijk,jr,kr->ir",T,B,C) then solves the rows on the fly.
    Also adds barrier to the lhs and rhs for nonnegativity
    Parameters
    ----------
    A: tensor_like
       Input tensor of arbitrary ndim

    mat_list: list of size A.ndim containing matrices that are n_i-by-R where n_i is dimension of ith mode of A
    and mat_list[mode] will contain the output
    
    R: Like input tensor A, needed for RHS construction

    mode: integer for mode with 0 indexing

    """
    t_solve_factor = timer("pySolve_factor")
    t_solve_factor.start()
    if R.ndim != A.ndim:
        raise ValueError('CTF PYTHON ERROR: Both tensors should of same dimensions')
    if len(mat_list) != A.ndim:
        raise ValueError('CTF PYTHON ERROR: mat_list argument to Solve_Factor must be of same length as ndim')
    k = -1
    tsrs = <Tensor[double]**>malloc(len(mat_list)*sizeof(ctensor*))
    #tsr_list = []
    imode = 0
    cdef tensor t
    if barrier is None:
        barrier = -1.0

    
    for i in range(len(mat_list))[::-1]:
        t = mat_list[i]
        tsrs[imode] = <Tensor[double]*>t.dt
        imode += 1
        if mat_list[i].ndim == 1:
            if k != -1:
                raise ValueError('CTF PYTHON ERROR: mat_list must contain only vectors or only matrices')
            if mat_list[i].shape[0] != A.shape[i]:
                raise ValueError('CTF PYTHON ERROR: input vector to SOLVE_FACTOR does not match the corresponding tensor dimension')
            # if A.shape[i] != R.shape[i]:
            #     raise ValueError('CTF PYTHON ERROR: Tensor dimensions for mode',i,'do not match')
            #exp = exp*mat_list[i].i(s[i])
        else:
            if mat_list[i].ndim != 2:
                raise ValueError('CTF PYTHON ERROR: mat_list operands has invalid dimension')
            if k == -1:
                k = mat_list[i].shape[1]
            else:
                if k != mat_list[i].shape[1]:
                    raise ValueError('CTF PYTHON ERROR: mat_list second mode lengths of tensor must match')
    B = tensor(copy=A)
    RHS = tensor(copy=R)
    reg = tensor(copy=regu)
    if A.dtype == np.float64:
        Solve_Factor_[double](<Tensor[double]*>B.dt,tsrs,<Tensor[double]*>RHS.dt,A.ndim-mode-1,<Tensor[double]*>reg.dt, epsilon, barrier, proj, add_ones, 1)
    else:
        raise ValueError('CTF PYTHON ERROR: Solve_Factor does not support this dtype')
    free(tsrs)
    t_solve_factor.stop()

def Solve_Factor_with_RHS(tensor A, mat_list, tensor R, mode,regu, barrier= 0.0):
    """
    Solve_Factor(A, mat_list,R, mode,regu)
    when RHS is given to us
    
    Parameters
    ----------
    A: tensor_like
       Input tensor of arbitrary ndim

    mat_list: list of size A.ndim containing matrices that are n_i-by-R where n_i is dimension of ith mode of A
    and mat_list[mode] will contain the output
    
    R: Like input tensor A, needed for RHS construction

    mode: integer for mode with 0 indexing

    """
    t_solve_factor = timer("pySolve_factor")
    t_solve_factor.start()
    if len(mat_list) != A.ndim:
        raise ValueError('CTF PYTHON ERROR: mat_list argument to Solve_Factor must be of same length as ndim')
    k = -1
    tsrs = <Tensor[double]**>malloc(len(mat_list)*sizeof(ctensor*))
    #tsr_list = []
    imode = 0
    cdef tensor t
        
    for i in range(len(mat_list))[::-1]:
        t = mat_list[i]
        tsrs[imode] = <Tensor[double]*>t.dt
        imode += 1
        if mat_list[i].ndim == 1:
            if k != -1:
                raise ValueError('CTF PYTHON ERROR: mat_list must contain only vectors or only matrices')
            if mat_list[i].shape[0] != A.shape[i]:
                raise ValueError('CTF PYTHON ERROR: input vector to SOLVE_FACTOR does not match the corresponding tensor dimension')
            if A.shape[i] != R.shape[i]:
                raise ValueError('CTF PYTHON ERROR: Tensor dimensions for mode',i,'do not match')
            #exp = exp*mat_list[i].i(s[i])
        else:
            if mat_list[i].ndim != 2:
                raise ValueError('CTF PYTHON ERROR: mat_list operands has invalid dimension')
            if k == -1:
                k = mat_list[i].shape[1]
            else:
                if k != mat_list[i].shape[1]:
                    raise ValueError('CTF PYTHON ERROR: mat_list second mode lengths of tensor must match')
    B = tensor(copy=A)
    RHS = tensor(copy=R)
    if A.dtype == np.float64:
        Solve_Factor_with_RHS_[double](<Tensor[double]*>B.dt,tsrs,<Tensor[double]*>RHS.dt,A.ndim-mode-1,regu, barrier ,1)
    else:
        raise ValueError('CTF PYTHON ERROR: Solve_Factor does not support this dtype')
    free(tsrs)
    t_solve_factor.stop()

def Solve_Factor_Tucker(tensor A, mat_list, tensor C, tensor R, mode, tensor regu, epsilon, barrier= None, proj = False, add_ones = False):
    """
    solves for a factor matrix parallelizing over rows constructing rhs, sparse tensor and list of factor matrices
    eg. for mode=0 order 3 tensor Computes LHS = einsum("ijk,jr,jz,kr,kz->irz",T,B,B,C,C) and 
    RHS = einsum("ijk,jr,kr->ir",T,B,C) then solves the rows on the fly.
    Also adds barrier to the lhs and rhs for nonnegativity
    Parameters
    ----------
    A: tensor_like
       Input tensor of arbitrary ndim

    mat_list: list of size A.ndim containing matrices that are n_i-by-R where n_i is dimension of ith mode of A
    and mat_list[mode] will contain the output
    
    R: Like input tensor A, needed for RHS construction

    mode: integer for mode with 0 indexing

    """
    t_solve_factor_t = timer("pySolve_factor_Tucker")
    t_solve_factor_t.start()
    if R.ndim != A.ndim:
        raise ValueError('CTF PYTHON ERROR: Both tensors should of same dimensions')
    if len(mat_list) != A.ndim:
        raise ValueError('CTF PYTHON ERROR: mat_list argument to Solve_Factor must be of same length as ndim')
    k = -1
    tsrs = <Tensor[double]**>malloc(len(mat_list)*sizeof(ctensor*))
    #tsr_list = []
    imode = 0
    cdef tensor t
    if barrier is None:
        barrier = -1.0

    
    for i in range(len(mat_list))[::-1]:
        t = mat_list[i]
        tsrs[imode] = <Tensor[double]*>t.dt
        imode += 1
        if mat_list[i].ndim == 1:
            if k != -1:
                raise ValueError('CTF PYTHON ERROR: mat_list must contain only vectors or only matrices')
            if mat_list[i].shape[0] != A.shape[i]:
                raise ValueError('CTF PYTHON ERROR: input vector to SOLVE_FACTOR does not match the corresponding tensor dimension')
            # if A.shape[i] != R.shape[i]:
            #     raise ValueError('CTF PYTHON ERROR: Tensor dimensions for mode',i,'do not match')
            #exp = exp*mat_list[i].i(s[i])
        else:
            if mat_list[i].ndim != 2:
                raise ValueError('CTF PYTHON ERROR: mat_list operands has invalid dimension')
    B = tensor(copy=A)
    RHS = tensor(copy=R)
    reg = tensor(copy=regu)
    if A.dtype == np.float64:
        Solve_Factor_Tucker_[double](<Tensor[double]*>B.dt,tsrs,<Tensor[double]*>C.dt,<Tensor[double]*>RHS.dt,A.ndim-mode-1,<Tensor[double]*>reg.dt, epsilon, barrier, proj, add_ones, 1)
    else:
        raise ValueError('CTF PYTHON ERROR: Solve_Factor does not support this dtype')
    free(tsrs)
    t_solve_factor_t.stop()

def Sparse_add(tensor A, tensor B,alpha=1.0,beta=1.0):
    """
    Add two sparse tensors A and B with identical distributions

    """
    t_sp_add = timer("pySparse_add")
    t_sp_add.start()
    
    if A.dtype == np.float64:
        Sparse_add_[double](<Tensor[double]*>A.dt,<Tensor[double]*>B.dt,alpha,beta)
    else:
        raise ValueError('CTF PYTHON ERROR: Sparse_add does not support this dtype')
    t_sp_add.stop()

def Sparse_mul(tensor A, tensor B):
    """
    Multiply two sparse tensors A and B with identical distributions

    """
    t_sp_add = timer("pySparse_mul")
    t_sp_add.start()
    
    if A.dtype == np.float64:
        Sparse_mul_[double](<Tensor[double]*>A.dt,<Tensor[double]*>B.dt)
    else:
        raise ValueError('CTF PYTHON ERROR: Sparse_mul does not support this dtype')
    t_sp_add.stop()
    
def Sparse_div(tensor A, tensor B):
    """
    Multiply two sparse tensors A and B with identical distributions

    """
    t_sp_add = timer("pySparse_div")
    t_sp_add.start()
    
    if A.dtype == np.float64:
        Sparse_div_[double](<Tensor[double]*>A.dt,<Tensor[double]*>B.dt)
    else:
        raise ValueError('CTF PYTHON ERROR: Sparse_div does not support this dtype')
    t_sp_add.stop()

def Sparse_inner_prod(tensor A, tensor B):
    """
    Multiply two sparse tensors A and B with identical distributions

    """
    t_sp_add = timer("pySparse_mul")
    t_sp_add.start()
    
    if A.dtype == np.float64:
        val = Sparse_inner_prod_[double](<Tensor[double]*>A.dt,<Tensor[double]*>B.dt)
    else:
        raise ValueError('CTF PYTHON ERROR: Sparse_mul does not support this dtype')

    t_sp_add.stop()

    return val



def Sparse_exp(tensor A, alpha=1.0):
    """
    Multiply two sparse tensors A and B with identical distributions

    """
    t_sp_exp = timer("pySparse_exp")
    t_sp_exp.start()
    
    if A.dtype == np.float64:
        Sparse_exp_[double](<Tensor[double]*>A.dt, alpha)
    else:
        raise ValueError('CTF PYTHON ERROR: Sparse_exp does not support this dtype')
    t_sp_exp.stop()

def Sparse_log(tensor A):
    """
    Multiply two sparse tensors A and B with identical distributions

    """
    t_sp_log = timer("pySparse_log")
    t_sp_log.start()
    
    if A.dtype == np.float64:
        Sparse_log_[double](<Tensor[double]*>A.dt)
    else:
        raise ValueError('CTF PYTHON ERROR: Sparse_log does not support this dtype')
    t_sp_log.stop()

def Sparse_sigmoid(tensor A, alpha = 1.0, eps = 0.0):
    """
    Take reciprocal

    """
    t_sp_log = timer("pySparse_sigmoid")
    t_sp_log.start()
    
    if A.dtype == np.float64:
        Sparse_sigmoid_[double](<Tensor[double]*>A.dt, alpha, eps)
    else:
        raise ValueError('CTF PYTHON ERROR: Sparse_log does not support this dtype')
    t_sp_log.stop()

def get_index_tensor(tensor A):
    """
    Get index tensor 

    """
    t_ind_tnsr = timer("pyget_index_tensor")
    t_ind_tnsr.start()
    
    if A.dtype == np.float64:
        get_index_tensor_[double](<Tensor[double]*>A.dt)
    else:
        raise ValueError('CTF PYTHON ERROR: get_index_tensor does not support this dtype')
    t_ind_tnsr.stop()
