# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 13:46:10 2019

Copied and modified from
"""

def krilov_exponentiate(HH,psi_iter,tt,kk=20):
    '''
    HH       - Linear Operator with function .dot(X) which returns the dot product
    psi_iter - wave-function to exponentiate
    tt       - time to evolve to
    kk       - Number of Krilov basis vectors to use
    
    RETURNS
    
    '''
    krilov_space = 1j*zeros([kk]+list(psi_iter.shape))
    krilov_space[0] = psi_iter
                
    krilov_span0 = 1j*zeros([kk]+list(psi_iter.shape))
    krilov_span0[0] = psi_iter
    for i in range(1,kk):
        vecin = HH.dot(krilov_space[i-1])
        krilov_space[i] = vecin/sqrt(sum(vecin*conj(vecin),0))
    if len(psi_iter.shape)==1:
        krilov_span = linalg.qr(krilov_space.T)[0].T
        A = dot(conj(krilov_span),HH.dot(krilov_span.T))
        Q = scipy.linalg.expm(tt*A)[:,0]
        
        recommended_iters = array([dot(abs(Q)**2,range(kk))/sum(abs(Q)**2)])
        psiout = dot(krilov_span.T,Q)
    elif len(psi_iter.shape)==2:
        psiout = 1j*zeros(psi_iter.shape)
        recommended_iters = zeros(psi_iter.shape[1])
        for ii in range(psi_iter.shape[1]):
            krilov_span = linalg.qr(krilov_space[:,:,ii].T)[0].T
            A = dot(conj(krilov_span),HH.dot(krilov_span.T))
            Q = scipy.linalg.expm(tt*A)[:,0]
            
            psiout[:,ii] = dot(krilov_span.T,Q)
            try:
                recommended_iters[ii] = dot(abs(Q)**2,range(kk))/sum(abs(Q)**2)
            except:
                print('Recommendations failed.')
                recommended_iters = array([-1])
        #return psiout,max(recommended_iters)
    else:
        raise NotImplemented('I have not done wavefunctions with dimension greater then 2. Please flatten _ \ | / _')
    
    #print(max(recommended_iters),kk)
    if max(recommended_iters)>0.3*kk:
        print('NOT ENOUGH PRECISION! Recursing with {:0.0f} elements'.format(2*kk))
        psiout,recommended_iters = krilov_exponentiate(HH,psi_iter,tt,2*kk)
    
    return psiout,recommended_iters.max()



# TEST IT
if __name__=='__main__':
    print('TESTING on a small system!')
    N = 8
    D = 2**N
    k = 10 # Krilov subspace size
    
    #psi = random.normal(size=D) + 1j*random.normal(size=D)
    #psi /= sqrt(dot(psi,conj(psi)))
    
    psi = random.normal(size=[D,10]) + 1j*random.normal(size=[D,10])
    psi = sum(psi*conj(psi),0)**-0.5*psi
    
    # Generate Hamiltonian
    Nels = 10
    #H = scipy.sparse.random(D,D,density=0.5*Nels/D) + 1j*scipy.sparse.random(D,D,density=0.5*Nels/D)
    #H += conj(H.T)
    H = scipy.sparse.coo_matrix((random.normal(size=Nels*D),(random.choice(D,Nels*D),random.choice(D,Nels*D))),shape=[D,D])
    H = 1j*scipy.sparse.coo_matrix((random.normal(size=Nels*D),(random.choice(D,Nels*D),random.choice(D,Nels*D))),shape=[D,D])
    
    H += conj(H.T)
    spectral_width = scipy.sparse.linalg.eigsh(H,which='LA',k=1)[0][0] - scipy.sparse.linalg.eigsh(H,which='SA',k=1)[0][0]
    # Normalize spectral width to 1... (maximum eigenvalue is 0.5)
    H /= 0.5*spectral_width
    
    H = H.tocoo()
    t = 1j*N*2
    
    # Builtin version
    psi_iter_builtin = scipy.sparse.linalg.expm_multiply(t*H,psi.copy())
    psi_iter,AA = krilov_exponentiate(H,psi.copy(),t,k)
    
    # ED version
    psi2 = psi_iter
    psi3 = psi_iter_builtin
    if False:#D<1000:
        # Compare to ED...
        eigees = linalg.eigh(H.toarray())
        psi1 = einsum('ax,x,x...->a...',eigees[1],exp(t*(jj+1)*eigees[0]),dot(conj(eigees[1].T),psi))
        #psi22 = scipy.linalg.expm(H.toarray()).dot(psi) # The above is equivalent...
        
        print(average(abs(1-sum(psi2*conj(psi1),0)**2)),'ED with Krilov')
        print(average(abs(1-sum(psi3*conj(psi1),0)**2)),'ED with builtin')
    print(average(abs(1-sum(psi2*conj(psi3),0)**2)),'Krilov with builtin')      