#coding=utf-8

from numpy import *

class RBM():
    def __init__(self, nv, nh):
        self.nv = nv
        self.b = zeros(nv)
        self.nh = nh
        self.c = zeros(nh)
        self.w = ones( (nv,nh) )

    def validate(self, name, value):
        if name == 'b' or name == 'v':
            return True if len(value) == self.nv else False
        if name == 'c' or name == 'h':
            return True if len(value) == self.nh else False
        if name == 'w':
            return True if shape(value) == (self.nv, self.nh) else False
    
    def setParam(self, name, value):
        if self.validate(name, value):
            if name == 'b':
                self.b = value
            if name == 'c':
                self.c = value
            if name == 'w':
                self.w = value
            return True
        else:
            print 'invalid argument'
            return False
            
    def confVisible(self, h):
        return 1 / (1 + exp( self.b + dot(self.w, h) ))
    
    def confHidden(self, v):
        return 1 / (1 + exp( self.c + dot(v, self.w) ))

    def energy(self, v, h):
        visible = dot( v, self.b )
        hidden = dot( h, self.c )
        whole = dot( v, dot(self.w, h) )
        return -visible -hidden -whole 

    def gibbs_sampling(self, v, A=1):
        vt = v
        for t in range(A):
            ht = self.confHidden(vt)
            vt = self.confVisible(ht)
        return vt

    def train(self, data, eta, bound=10000): # w,b,cの学習率をetaで統一
        N = len(data)
        count, del_w, del_b, del_c = 0,0,0,0
        sig = sigmoid(1.0)
        while(1):
            print '\n'
            for v in data:
                gv = self.gibbs_sampling(v,A=1)
                sig_c_vw = sig(self.c + dot(v,self.w))
                sig_c_gvw = sig(self.c + dot(gv,self.w))
                print 'gv',gv
                print 'sig_c_vw',sig_c_vw
                print 'sig_c_gvw',sig_c_gvw
                
                del_w += ( dot(matrix(v).T, matrix(sig_c_vw)) - dot(matrix(gv).T, matrix(sig_c_gvw)) ).A
                del_b += v - gv
                del_c += sig_c_vw - sig_c_gvw

            # divide N, num of data
            del_w /= N
            del_b /= N
            del_c /= N

            # debug
            print '\n'
            print 'eta * del_w',eta * del_w
            print 'eta * del_b',eta * del_b
            print 'eta * del_c',eta * del_c

            # update
            self.w -= eta * del_w
            print 'w',self.w
            self.b -= eta * del_b
            print 'b',self.b
            self.c -= eta * del_c
            print 'c',self.c

            if count > bound:
                print 'loop couner is over upper bound(%d)' % bound
                return True
            else:
                count += 1

def sigmoid(beta):
    def f(x):
        return 1 / (1 + exp(-beta*x))
    return f

if __name__=='__main__':
    rbm = RBM(3,2)
    data = array([[0,0,1],
                  [0,0,1],
                  [0,0,1],
                  [0,0,1],
                  [0,0,1],
                  [0,0,1],
                  [0,0,1],
                  [0,0,1],
                  [0,1,1],
                  [1,1,1]])
    rbm.train(data, 0.1, bound=1000)

    print '\n'
    v = array([0,0,1])
    h = rbm.confHidden(v)
    v = rbm.confVisible(h)
    print 'confHidden:', h
    print 'confVisible:', v
    v = array([1,1,1])
    h = rbm.confHidden(v)
    v = rbm.confVisible(h)
    print 'confHidden:', h
    print 'confVisible:', v
