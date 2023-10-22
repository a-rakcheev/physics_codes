# coding: utf-8
import numpy as np
def comm(A, B):
    return A @ B - B @ A
    
def diag(size):
    return np.diag(np.random.rand(size))
    
print(diag(4))
def herm(size):
    mat = np.random.rand(size, size) + 1.j * np.random.rand(size, size)
    mat = mat + mat.T.conj()
    mat = mat - np.diag(np.diag(mat))
    return mat
    
print(herm(4))
np.set_printoptions(3, linewidth+200)
np.set_printoptions(3, linewidth=200)
print(herm(4))
print(comm(diag(4), herm(4)))
print(comm(diag(4), herm(4)))
print(comm(diag(4), herm(4)))
get_ipython().run_line_magic('save', 'commutator_diag_offdiag 1-13')
L=5
print(comm(diag(L), herm(L)))
L=10
L=10
print(comm(diag(L), herm(L)))
print(np.diag(comm(diag(L), herm(L))))
get_ipython().run_line_magic('save', 'commutator_diag_offdiag 1-20')
