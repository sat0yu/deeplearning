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

    def train(self, data, eta, mu, bound=10000):
        count = 0
        while(1):

            # 更新式

            if count > bound:
                print 'loop couner is over upper bound(%d)' % bound
                return True

def sigmoid(beta):
    def f(x):
        return 1 / (1 + exp(-beta*x))
    return f

if __name__=='__main__':
    v = array([1,2,3])
    h = array([1,1,0,1])

    rbm = RBM(3,4)
    print 'confVisible:',rbm.confVisible(h)
    print 'confHidden:',rbm.confHidden(v)
    print 'energy:',rbm.energy(v,h)

    b = array([-10, 1, 3])
    rbm.setParam('b', b)
    c = array([0, 1, 1, 0.5])
    rbm.setParam('c', c)
    print 'confVisible:',rbm.confVisible(h)
    print 'confHidden:',rbm.confHidden(v)
    print 'energy:',rbm.energy(v,h)
    
    data = array([[1,2,3],
                  [2,3,4],
                  [3,2,1],
                  [0,0,0],
                  [1,1,1],
                  [2,2,2]])
    rbm.train(data)
    v = arran([1,2,3])
    print 'confHidden:',rbm.confHidden(v)
    v = arran([2,3,4])
    print 'confHidden:',rbm.confHidden(v)
    v = arran([0,0,0])
    print 'confHidden:',rbm.confHidden(v)
