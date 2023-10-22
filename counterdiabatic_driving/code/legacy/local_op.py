from quspin.operators import hamiltonian
import scipy.sparse
from numpy import kron
def locop(sp,i,N):
  '''
  Local operator
  sp - operator type: {x,y,z,i,+,-}
  i  - site index
  N  - Total number of sites
  '''
  if sp=='1':
      sp = 'i'
      
  if sp=='i':
      return scipy.sparse.identity(2**N,format='csc')
  elif sp=='+':
      return 0.5*(locop('x',i,N) + 1j*locop('y',i,N))
  elif sp=='-':
      return 0.5*(locop('x',i,N) - 1j*locop('y',i,N))
  else:
      return hamiltonian([[sp,[[1,i]]]],[],N = int(N),check_symm=False,check_herm=False).tocsc()

def mat_product(mat):
  '''
  Tensor product of two matrices mat[1] \otimes mat[0]
  '''
  if scipy.sparse.issparse(mat[0]):
    return scipy.sparse.kron(mat[1],mat[0])
  else:
    return kron(mat[1],mat[0])
