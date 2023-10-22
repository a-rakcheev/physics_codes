# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 16:18:49 2018

@author: Jonathan Wurtz
"""
import itertools
from numpy import *
import collections
import tqdm

class core:
    def __init__(self):
        self.lookup = {}
        self.lookup['xx'] = {'1':1}   ##
        self.lookup['xy'] = {'z':1j}  #
        self.lookup['xz'] = {'y':-1j} #
        
        self.lookup['yx'] = {'z':-1j} ##
        self.lookup['yy'] = {'1':1}   ##
        self.lookup['yz'] = {'x':1j}  ##
        
        self.lookup['zx'] = {'y':1j}  ##
        self.lookup['zy'] = {'x':-1j} #
        self.lookup['zz'] = {'1':1}   ##
        
        self.lookup['x1'] = {'x':1}
        self.lookup['y1'] = {'y':1}
        self.lookup['z1'] = {'z':1}
        
        self.lookup['1x'] = {'x':1}
        self.lookup['1y'] = {'y':1}
        self.lookup['1z'] = {'z':1}
        
        self.lookup['11'] = {'1':1}
                   
        self.biglookup = {}
        
        self.intlookup = {'x':0,'y':1,'z':2,'1':3}
        self.revlookup = array(['x','y','z','1'])
        self.keylookup = array([[3,2,1,0],[2,3,0,1],[1,0,3,2],[0,1,2,3]],dtype=uint8)
        self.numlookup = array([[0,-1,1,0],[1,0,-1,0],[-1,1,0,0],[0,0,0,0]],dtype=int8)
    
    def primative_product(self,a,b):
        '''
        Product of two strings a and b
        '''
        if len(a)!=len(b):
            raise BaseException("Ill-Shaped Strings!")
            
        if a+';'+b in self.biglookup:
            return self.biglookup[a+';'+b]
        
        
        dat = [self.lookup[a[i]+b[i]] for i in range(len(a))]
        payload = ''.join([q.keys()[0] for q in dat]),prod([q.values() for q in dat])
        if len(self.biglookup)<1000000:
            self.biglookup[a+';'+b] = payload
        return payload
    
    def product(self,A,B):
        '''
        Product of two equations A and B
        '''
        dout = {}
        for a in A.keys():
            for b in B.keys():
                pp,val = self.primative_product(a,b)
                dout[pp] = dout.get(pp,0) + A[a]*B[b]*val
        
        return {a:b for a,b in dout.iteritems() if average(abs(array(b)))>1e-16}
    
    def fastcommute(self,AA,BB,verbose=True):
        '''
        Fast commutator of two equations  [AA,BB]
        '''
        N = len(AA.keys()[0])
        
        A = array([[self.intlookup[q] for q in r] for r in AA.keys()],dtype=uint8)
        B = array([[self.intlookup[q] for q in r] for r in BB.keys()],dtype=uint8)
        Aval = array(AA.values())
        Bval = array(BB.values())
        if len(A)>len(B):
          A,B = B,A
          Aval,Bval = Bval,Aval
        
        if sqrt(len(A)*len(B))>100 and verbose==True:
            iterprint = tqdm.tqdm
        else:
            iterprint = lambda x:x
        ctr = 0
        
        dout = collections.defaultdict(lambda: 0)
        #print 'SHAPES:',A.shape,B.shape
        for aa in iterprint(A): # Parallelizable?

            tmp1 = ((1j)**self.numlookup.T[aa][range(N),B].sum(1) - (1j)**self.numlookup[aa][range(N),B].sum(1))
            #val = einsum('a,a...->a...',tmp1,Bval*Aval[ctr])
            val = ((Bval*Aval[ctr]).T*tmp1).T
            
            ctr += 1
            nz = nonzero(abs(val)>1e-15)[0] # Only lookup relevent values...
            key = [''.join(self.revlookup[r]) for r in self.keylookup[aa][range(N),B[nz]]]
            for a,b in zip(key,val[nz]): # This is the expensive thing =/
                dout[a] += b
            
        return dout#{a:b for a,b in dout.iteritems() if average(abs(array(b)))>1e-16}


    
    
    def add(self,A,B,a=1,b=1):
        '''
        Addition of two equations a*A + b*B
        '''
        keysout = set(A.keys()).union(set(B.keys()))
        dout = {}
        for key in keysout:
            #print 'Adding...'
            #print type(a),type(b)
            #print type(A.get(key,0)),type(B.get(key,0))
            dd = a*A.get(key,0) + b*B.get(key,0)
            if type(dd)!=ndarray:
                if abs(dd)>1e-15:
                    dout[key] = dd
            else:
                if average(abs(dd))>1e-15:#array(abs(dd)>1e-15).prod():#
                    dout[key] = dd
        
        # Purge all zeros...
        return dout

    
    