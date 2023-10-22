# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 15:27:46 2019

Yet Another iteration to compute the variational gauge potential

@author: jwurtz
"""

def find_Alambda5(H,Akeys,energyvariance=False):
    '''
    H - Time-dependent object
    Akeys - List of equations to guess by
    Energyvariance - True:  MIN ||[H,X]||
                   - False: MIN ||X||
    '''
    tsteps = len([*H.values()][0])
    print('Computing Commutators...')
    comm = []
    for i in tqdm.tqdm(range(len(Akeys))):
        if energyvariance==False:
            comm.append(H*Akeys[i] - Akeys[i]*H)
        else:
            tmp = H*Akeys[i] - Akeys[i]*H
            comm.append(H*tmp - tmp*H)
    
    
    M = 1j*zeros([tsteps,len(Akeys),len(Akeys)])
    
    B = 1j*zeros([tsteps-1,len(Akeys)])
    
    print("computing M...")
    for i in tqdm.tqdm(range(len(Akeys))):
        for j in range(len(Akeys)):
            M[:,i,j] = m.tracedot(comm[i],comm[j])
    print("Computing B...")
    dH = H.astype(diff).tol(1e-10)*(tsteps)
    
    if energyvariance==True:
        dH = H.astype(lambda x:x[1::])*dH - dH*H.astype(lambda x:x[1::])

    for i in tqdm.tqdm(range(len(Akeys))):
        B[:,i] = m.tracedot(comm[i].astype(lambda x:x[1::]),dH)
        
    # Now we must compute the pseudo-inverse by ignoring the zeros
    print('Computing Eigen-system...')
    eigees = linalg.eigh(M) # $$$ This is the expensive bit... $$$
    
    print('Computing Pseudo-inverse...')
    dout = []
    dout_chi = []
    for i in tqdm.tqdm(range(eigees[0].shape[0]-1)):
        
        nz = nonzero(abs(eigees[0][i])>1e-12*abs(eigees[0]).max())[0]
        #print(nz.shape,eigees[1][i,:,nz].shape,eigees[0][i,nz].shape)
        pseudoinverse = einsum('au,u,bu->ab',conj(eigees[1][i][:,nz]),1/eigees[0][i,nz],eigees[1][i][:,nz])
        #pseudoinverse = einsum('au,u,bu->ab',conj(eigees[1][i]),1/(eigees[0][i]-cutoff),eigees[1][i])
        
        chi = dot(pseudoinverse,B[i])
        dout.append(dot(Akeys,chi).tol(1e-8))
        
        #print(chi.shape,M.shape,B.shape,m.tracedot(dH,dH).shape)
        try:
            chinorm = m.tracedot(dH,dH)[i] - dot(dot(chi,M[i]),chi) + dot(B[i],chi)
        except:
            #print('WARNING: Chinorm Borked. I guess its zero')
            chinorm = 0
        
        dout_chi.append(chinorm)
    
    print('Reducing A...')
    keys = set.union(*[set(q.keys()) for q in dout])
    Aout = equation({})
    for key in keys:
        timetrace = []
        for i in range(tsteps-1):
            timetrace.append(dout[i].get(key,0))
        Aout[key] = array(timetrace)
        
    return (M,B),dout_chi,dout,Aout

class dooer():
    def __init__(self,Hamiltonian,Ansatz):
        '''
        Hamiltonian - Parameterized Hamiltonian along the path (dict equation)
        Ansatz   - List of dict equations as the ansatz for the GP
        '''
        self.ham = Hamiltonian
        self.ansatz = Ansatz
        
        self.state = {}
        self.state['GP_computed'] = False
        self.state['magnus_computed'] = False
        self.state['Htilde_computed'] = False
        self.state['ptilde_computed'] = False
        self.state['AAop_computed'] = False
        
    def compute_GP(self,energyvariance=True):
        '''
        Computes the Gauge Potential via variational minimization
        gaugepotential - the Gauge Potential
        '''
        MM,BB,dout,AA = find_Alambda5(self.ham,self.ansatz,energyvariance)
        self.gaugepotential = AA
        self.chinorm = array(BB)
        self.state['GP_computed'] = True
        return True
    
    def compute_magnus_expansion(self,order=2):
        '''
        Computes the Magnus Expansion of the gauge potential
        omega - Magnus expanded GP
        '''
        if self.state['GP_computed']==False:
            print("The GP has yet to be computed. I'm doing it now before I forget")
            self.compute_GP()
        
        self.omega = m.magnus(1j*self.gaugepotential,1,order=order)
        self.state['magnus_computed'] = True
    
    def compute_AAop(self,basis=None,tolarance=1e-5):
        '''
        Computes the GP operator as a sparse matrix
        '''
        if self.state['AAop_computed']==True:
            return self.AAop
        else:
            print('Computing the Matrix form of the GP!')
            
            if self.state['GP_computed']==False:
                print("The GP has yet to be computed. I'm doing it now before I forget")
                self.compute_GP()
            nsteps = len([*self.ham.values()][0])
        
            # Multiprocessing gets annoyed and cannot pickle scoped functions
            #def tmp(ii):
            #    return self.gaugepotential((i,)).tol(1e-5).make_operator()
            #p = multiprocessing.Pool(processes=11)
            #self.AAop = p.map(tmp,range(nsteps-1))
            #p.close()
            self.AAop = [self.gaugepotential((i,)).tol(tolarance).make_operator(basis=basis) for i in tqdm.tqdm(range(nsteps-1))]
            
            self.state['AAop_computed'] = True
            return self.AAop
    
    def compute_Htilde(self):
        '''
        Computes the rotated Hamiltonian via Heisenberg evolution
        Htilde - H rotated by the GP
        '''
        
        if self.state['GP_computed']==False:
            print("The GP has yet to be computed. I'm doing it now before I forget")
            self.compute_GP()
        
        if len([*self.ham.keys()][0])>11:
            raise(BaseException('Too Big, I quit'))
        print('Computing Aop')
        nsteps = len([*self.ham.values()][0])
        #AAop = self.gaugepotential.tol(1e-5).make_operator(sparse=False)
        #AAop = [self.gaugepotential((i,)).tol(1e-5).make_operator() for i in tqdm.tqdm(range(nsteps-1))]
        AAop = self.compute_AAop()
        def differ(t,X):
            tind = argmin(abs(linspace(0,1,nsteps-1)-t))
            shapes = int(sqrt(len(X)))
            #return -(dot(AAop[:,:,tind],X.reshape(shapes,shapes)) - dot(X.reshape(shapes,shapes),AAop[:,:,tind])).flatten()
            
            return -(AAop[tind].dot(X.reshape(shapes,shapes)) - (AAop[tind].T.dot(X.reshape(shapes,shapes).T)).T).flatten()
        
        print('Integrating...')
        inter = scipy.integrate.complex_ode(differ)
        inter.set_initial_value((self.ham((0,))).make_operator().toarray().flatten())
        success = []
        for tt in tqdm.tqdm(linspace(0,1,101)[1::]):
            Hr_tot = inter.integrate(tt).reshape(2**N_small,2**N_small)
            success.append(inter.successful())
        print('Was Successful:',inter.successful())
        
        self.Htilde = Hr_tot
        self.state['Htilde_computed'] = True
    
    def compute_eigenbasis(self):
        '''
        Computes Htilde in the eigenbasis of H0
        Htilde_basis - Htilde in basis of H0
        H_basis      - H in basis of H0
        '''
        
        if self.state['Htilde_computed'] == False:
            print("Htilde has yet to be computed. I'm doing it now before I forget")
            self.compute_Htilde()
        
        eigeesS = linalg.eigh(self.ham((-1,)).make_operator().toarray())
        self.Htilde_basis = dot(conj(eigeesS[1].T),dot(self.Htilde,eigeesS[1]))
        self.H_basis = dot(conj(eigeesS[1].T),dot((self.ham((0,))).make_operator().toarray(),eigeesS[1]))

    
    def compute_projector_OLD(self,P,direction='b',basis=None):
        '''
        Computes the rated projective subspace. P is the original subspace.
        direction - 'f': goes from 0 to 1
                  - 'b': goes from 1 to 0
        '''
        
            
        if len([*self.ham.keys()][0])>18:
            raise(BaseException('Too Big, I quit'))
        pshape = P.shape
        #AAop = self.gaugepotential.tol(1e-5).make_operator(sparse=False)
        #AAop = [self.gaugepotential((i,)).tol(1e-5).make_operator() for i in tqdm.tqdm(range(nsteps-1))]
        AAop = self.compute_AAop()
        nsteps = len(AAop)
        def differ(t,X):
            #tind = argmin(abs(linspace(0,1,nsteps)-t))
            tind = max([min([int(floor(t*nsteps)),nsteps-1]),0]) # AGPs map BETWEEN indices
            # The bounding is necessary as sometimes the integrator
            # queries outside of the range for some reason
            #print(t,tind,nsteps)
            return -AAop[tind].dot(X.reshape(pshape)).flatten()
        
        print('Integrating...')
        inter = scipy.integrate.complex_ode(differ)

        inter.set_initial_value(P.flatten(),t={'b':1,'f':0}[direction])

            
        success = []
        Prot_agg = []
        self.P_errors = []
        
        times = linspace({'b':1,'f':0}[direction],{'b':0,'f':1}[direction],nsteps+1)[1::]
        for ii in tqdm.tqdm(range(nsteps)):
            Prot = inter.integrate(times[ii]).reshape(pshape)
            #Prot_agg.append(Prot) # These are 100 evenly spaced intervals excluding 0
            success.append(inter.successful())
            
            # Compute the commutator of Ptilde with H(lamb)
            if direction=='b':
                tind = -ii-2#argmin(abs(linspace(0,1,nsteps)-tt))+1
            elif direction=='f':
                tind = ii+1
            
            H_start_op = self.ham((tind,)).make_operator(basis=basis)
              
            HdotP = H_start_op.dot(P)
            HdotPtilde = H_start_op.dot(Prot)
            
            Hproj = dot(conj(P.T),HdotP)
            H2proj = dot(conj(HdotP.T),HdotP)
            
            Hproj_tilde = dot(conj(Prot.T),HdotPtilde)
            H2proj_tilde = dot(conj(HdotPtilde.T),HdotPtilde)
            
            naive_error = -2*(sum(abs(Hproj)**2) - trace(H2proj))/P.shape[1]
            fixed_error = -2*(sum(abs(Hproj_tilde)**2) - trace(H2proj_tilde))/P.shape[1]
            self.P_errors.append([naive_error,fixed_error])
            
        print('Was Successful:',inter.successful())
        self.Prot = Prot
        #self.Prot_agg = array(Prot_agg)
        self.P_errors = array(self.P_errors)
        self.state['ptilde_computed'] = True
    
    def compute_projector(self,P,direction='b',basis=None):
        '''
        Indistinguishable from compute_projector_old
        except that it uses a built-in exponentiator instead of a general-purpose integrator
        
        Computes the rated projective subspace. P is the original subspace.
        direction - 'f': goes from 0 to 1
                  - 'b': goes from 1 to 0
        '''
        
        if len([*self.ham.keys()][0])>18:
            raise(BaseException('Too Big, I quit'))
        if direction!='b':
            raise NotImplementedError('The Forward condition has not been implemented. See the old version')

        print('Integrating...')
        #AAop = self.compute_AAop()
        
        P0 = P.copy()
        Pi = P.copy()
        
        def compute_Perrors(PP,ham,basis):
            H_start_op = ham.make_operator(basis=basis)
              
            HdotP = H_start_op.dot(PP)
            
            Hproj = dot(conj(PP.T),HdotP)
            H2proj = dot(conj(HdotP.T),HdotP)
            
            return -2*(sum(abs(Hproj)**2) - trace(H2proj))/PP.shape[1]
            
            
        nsteps = array([*self.gaugepotential.values()]).shape[1]#len(AAop)
        
        self.P_errors = [2*[compute_Perrors(Pi,self.ham((-1,)),basis)]]
        
        krilov_steps = 10
        self.krilov_step_history = zeros(nsteps)
        self.ent_ent = []
        for ii in tqdm.tqdm(range(nsteps)):
            # Old way: use builtin. New way: use homebuilt Krilov method
            #Pi = scipy.sparse.linalg.expm_multiply(self.AAop[-ii-1]/(nsteps+1),Pi)
            
            
            # Error accumulates as the stepsize^2 for time-independent gauge potentials.
            # Error accumulates as stepsize for time-dependent potentials, due to
            #  Magnus-expansion type effects
            
            # C:\Users\jwurtz\Documents\python3\KRILOV.py
            Pi,errs = krilov_exponentiate(self.gaugepotential.make_LinearOperator(basis,index=-ii-1),Pi,1./(nsteps+1),krilov_steps)
            krilov_steps = max([3,4*errs])
            self.krilov_step_history[ii] = errs
            
            naive_error = compute_Perrors(P0,self.ham((-ii-2,)),basis)
            fixed_error = compute_Perrors(Pi,self.ham((-ii-2,)),basis)
            
            self.P_errors.append([naive_error,fixed_error])
            
            # Total Entanglement Entropy between halves
            #  in units of bits
            self.ent_ent.append(basis.ent_entropy(Pi,arange(int(basis.N/2)))['Sent_A']*int(basis.N/2)/log(2))
        
        self.Prot = Pi
        #self.Prot_agg = array(Prot_agg)
        self.P_errors = array(self.P_errors)
        self.ent_ent = array(self.ent_ent)
        self.state['ptilde_computed'] = True
if True:
    '''
    The following is a check of computing the exact variational gauge potential
    along an arbitrary path to show that it exactly diagonalizes the rotated
    Hamiltonian Htilde in the basis of the original Hamiltonian.
    '''
    close('all')
    figure(figsize=(20,12))
    
    H_full = equation({})
    nsteps = 1001
    N_small = 3
    nn = N_small
    
    nparams = 5
    t = linspace(0,2,nsteps)
    freqs = linspace(0,10,1000)
    BB = []
    
    random.seed(8)
    for i in range(nparams):
        Bi = dot(random.normal(size=1000)*exp(-freqs),cos(outer(t,freqs)).T)
        Bi += dot(random.normal(size=1000)*exp(-freqs),sin(outer(t,freqs)).T)
        Bi /= std(Bi)
        BB.append(Bi)
    BB = array(BB)
    BB /= sqrt(sum(BB*BB,0))
    
    for i in range(N_small):
        op1 = ''.join(roll(list('xx'+'1'*(N_small-2)),i))
        op1b = ''.join(roll(list('yy'+'1'*(N_small-2)),i))
        opZ = ''.join(roll(list('z'+'1'*(N_small-1)),i))
        opX = ''.join(roll(list('x'+'1'*(N_small-1)),i))
        opY = ''.join(roll(list('y'+'1'*(N_small-1)),i))
    
        H_full[op1] = BB[0]
        H_full[op1b] =BB[1]
        H_full[opZ] = BB[2]
        H_full[opX] = BB[3]
        H_full[opY] = BB[4]
    
    
    subplot(1,4,3)
    plot(t,array([*H_full.values()]).T)
    title('Hamiltonian Parameters')
    axis([0,max(t),axis()[2],axis()[3]])
    
    Akeys = []
    
    for q in itertools.product(*(nn*[['x','y','z','1']])):
        Akeys.append(equation({''.join(roll(list(q) + ['1']*(N_small-nn),i)):1 for i in range(N_small)}))
    Akeys = list(set(Akeys))
    
    
    
    doo = dooer(H_full,Akeys)
    doo.compute_GP(True)
    subplot(1,4,4)
    plot(t[1::],array([*doo.gaugepotential.values()]).T*1j)
    title('Gauge Potential Parameters')
    axis([0,max(t),axis()[2],axis()[3]])
    subplots_adjust(left=0.05,right=0.99)
    draw()
    show()
    pause(0.1)
    time.sleep(0.1)
    
    doo.compute_eigenbasis()
    
    eigees = linalg.eigh(H_full((-1,)).make_operator().toarray())
    doo.compute_projector_OLD(eigees[1])
    Htilde_basis2 = dot(conj(doo.Prot.T),dot((doo.ham((0,))).make_operator().toarray(),doo.Prot))
    
    vmax = .1
    subplot(2,4,1)
    ims(abs(doo.Htilde_basis),vmin=0,vmax=vmax,aspect='equal')
    title('Heisenberg Picture Rotated')
    
    subplot(2,4,5)
    ims(abs(Htilde_basis2),vmin=0,vmax=vmax,aspect='equal')
    xlabel('Schrodinger Picture Rotated')
    
    subplot(2,4,2)
    ims(abs(doo.H_basis),vmin=0,vmax=vmax,aspect='equal')
    title('Unrotated')
    
    


'''
MM,BB,dout,AA = find_Alambda5(H_full,Akeys)





#%%
#omega = m.magnus(1j*AA,1,order=1)
#eigees = linalg.eigh(omega.tol(1e-5).make_operator().toarray()) # Calculate Unitary via eigen-decomposition (faster...?)
#Ub = dot(conj(eigees[1]),(exp(-1j*eigees[0])*eigees[1]).T).T
print('Computing Aop')
AAop = AA.tol(1e-5).make_operator(sparse=False)
def differ(t,X):
    tind = argmin(abs(linspace(0,1,AAop.shape[2])-t))
    shapes = int(sqrt(len(X)))
    return -(dot(AAop[:,:,tind],X.reshape(shapes,shapes)) - dot(X.reshape(shapes,shapes),AAop[:,:,tind])).flatten()

print('Integrating')
inter = scipy.integrate.complex_ode(differ)
inter.set_initial_value((H_full((0,))).make_operator().toarray().flatten())
success = []
for tt in linspace(0,1,101)[1::]:
    Hr_tot = inter.integrate(tt).reshape(2**N_small,2**N_small)
    success.append(inter.successful())
print('Was Successful:',inter.successful())

print('Rotating to ~ basis...')
#Hr_center = dot(conj(Ub.T),dot((H_center).make_operator().toarray(),Ub))
#Hr_tot2 = dot(conj(Ub.T),dot((H_full((0,))).make_operator().toarray(),Ub))

#%% Project to dict-equations
# As it turns out, this step is the slowest one. It is easily paralizable
#  but it is not done to be so.
extents = 4
print('Projecting to dicts...')

print('Hr_center...')
#Hr_center_dict = m.from_matrix(Hr_center,max_extent=extents,precision=1e-10)
print("Hr_total...")
#Hr_tot_dict = m.from_matrix(Hr_tot,max_extent=extents,precision=1e-10)

#raise(BaseException('Stop Here!'))
#%%
print('Computing EigenSystem')
eigeesS = linalg.eigh(H_full((-1,)).make_operator().toarray())
jumps = nonzero(diff(eigeesS[0])>1e-5)[0]+0.5

print('Plotting...')
vmax = .1
subplot(1,2,1)
#Hr_tot_rot = einsum('au,ab,bv',conj(eigeesS[1]),Hr_tot,eigeesS[1])
Hr_tot_rot = dot(conj(eigeesS[1].T),dot(Hr_tot,eigeesS[1]))
ims(abs(Hr_tot_rot),vmin=0,vmax=vmax,aspect='equal')
ylabel('From Commutator dynamics')
title('Rotated')

for jump in jumps:
    plot([0,2**N_small-1],[jump,jump],'r-',linewidth=0.2)
    plot([jump,jump],[0,2**N_small-1],'r-',linewidth=0.2)

subplot(2,4,5)
#Hr_tot_rot2 = einsum('au,ab,bv',conj(eigeesS[1]),Hr_tot2,eigeesS[1])#dot(conj(eigeesS[1].T),dot(Hr_tot,eigeesS[1]))
ims(abs(Hr_tot_rot2),vmin=0,vmax=vmax,aspect='equal')
ylabel('From Magnus Expansion')


subplot(1,2,2)
#Hr_tot_unrot = einsum('au,ab,bv',conj(eigeesS[1]),(H_full((0,))).make_operator().toarray(),eigeesS[1])#dot(conj(eigeesS[1].T),dot((H_full((0,))).make_operator().toarray(),eigeesS[1]))
Hr_tot_unrot = dot(conj(eigeesS[1].T),dot((H_full((0,))).make_operator().toarray(),eigeesS[1]))
ims(abs(Hr_tot_unrot),vmin=0,vmax=vmax,aspect='equal')
title('Unrotated')
for jump in jumps:
    plot([0,2**N_small-1],[jump,jump],'r-',linewidth=0.2)
    plot([jump,jump],[0,2**N_small-1],'r-',linewidth=0.2)



subplot(1,4,4)
plot(t[1::],array([*AA.values()]).T*1j)
title('Gauge Potential Parameters')
axis([0,max(t),axis()[2],axis()[3]])
subplots_adjust(left=0.05,right=0.99)
'''