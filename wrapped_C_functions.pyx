cimport cython
import numpy as np
cimport numpy as np

cdef extern from 'mkl.h' nogil:
    int LAPACK_COL_MAJOR
    int LAPACK_ROW_MAJOR
    
    int LAPACKE_dstemr( int matrix_layout, #Specifies whether matrix storage layout is row major (LAPACK_ROW_MAJOR) or column major (LAPACK_COL_MAJOR).
                              char jobz, # 'N' for eigenvalues, 'V' for vects too
                              char range, #'A' all eigenvalues.
                              int n, #size of the matrices
                              double * d, #Contains diagonal elements of tridiagonal matrix T.
                              double* e, #contains off diagonals, length n, the last element used internally as scratch space
                              double vl, #If range = 'V', interval to be searched for eigenvalues
                              double vu,
                              int il, # If range = 'I', the indices in ascending order of the smallest and largest eigenvalues.
                              int iu,
                              int* m, #OUTPUT: The total number of eigenvalues found
                              double* w, #OUTPUT: eigenvalues in ascending order
                              double* z, #If jobz = 'V', and info = 0, then the first m columns of z contain the orthonormal eigenvectors of the matrix T corresponding to the selected eigenvalues, with the i-th column of z holding the eigenvector associated with w(i)
                              int ldz, #The leading dimension of the output array z
                              int nzc, #The number of eigenvectors to be held in the array z
                              int* isuppz, #Array, size (2*max(1, m)). The support of the eigenvectors in z,
                              int* tryrac, #If tryrac is true, it indicates that the code should check whether the tridiagonal matrix defines its eigenvalues to high relative accuracy
                      )
#This function returns a value info.
#If info = 0, the execution is successful.
#If info = -i, the i-th parameter had an illegal value.
#If info > 0, an internal error occurred.

    int LAPACKE_dstedc( int matrix_layout, #
                       char compz, # 'I' for both vals and vecs
                       int n, # shape of matrix
                       double* d, # diagaonal 
                       double* e, # offdiagonal
                       double* z, #output vecs
                       int ldz, #size of z
                      )
    

def diagonalise_mkl_dstedc(double [::1] diagonals, double [::1] offdiagonals,
                   double [::1] eigenvalues, double [:, ::1] eigenvects):
    
    eigenvalues[...] = diagonals
    info = LAPACKE_dstedc(LAPACK_ROW_MAJOR, b'I', eigenvalues.shape[0],
                       &eigenvalues[0], # diagaonal 
                       &offdiagonals[0], # offdiagonal
                       &eigenvects[0,0], #output vecs
                       eigenvalues.shape[0], #size of z
                      )
    

def diagonalise_mkl(double [::1] diagonals, double [::1] offdiagonals,
                   double [::1] eigenvalues, double [:, ::1] eigenvects):
    N = diagonals.shape[0]
    cdef int eigenvalues_found = 0
    cdef int[::1] isuppz = np.zeros(2*N, dtype = np.intc)
    cdef int tryrac = 0
    
    info = LAPACKE_dstemr(LAPACK_ROW_MAJOR, b'N', b'A', N, &diagonals[0], &offdiagonals[0], 0, 0, 0, 0,
                                  &eigenvalues_found, &eigenvalues[0], &eigenvects[0,0],
                                  N, N, &isuppz[0], &tryrac)
    print(f"""
    info: {info}
    eigenvalues_found: {eigenvalues_found}
    eigenvalues: {list(eigenvalues)}
    isuppz: {list(isuppz)}
    tryrac: {tryrac}
    """)
    
from scipy.linalg import eigh_tridiagonal
cpdef diagonalise_scipy(double [::1] diagonals, double [::1] offdiagonals,
                   double [::1] eigenvalues, double [:, ::1] eigenvects):
    
    cdef int N = diagonals.shape[0]
    numpy_diag = np.asarray(<np.double_t[:N]> &diagonals[0])
    numpy_offdiag = np.asarray(<np.double_t[:N-1]> &offdiagonals[0])
    
    cdef double[:] vals #so that cython knows what's being returned from eigh_tridiagonal
    cdef double[:, :] vecs
    
    vals, vecs = eigh_tridiagonal(d=numpy_diag, e=numpy_offdiag)
    
    eigenvalues[:] = vals
    eigenvects[:, :] = vecs