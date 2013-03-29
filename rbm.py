#coding=utf-8

from numpy import *
import math

class RBM():
    def __init__(self, num_v, num_h, vbias=None, hbias=None, W=None):
        if vbias is None:
            vbias = zeros(num_v)

        if hbias is None:
            hbias = zeros(num_h)
        
        if W is None:
            width = 1. / num_v
            W = random.uniform(
                low = -width,
                high = width,
                size = (num_v, num_h) )

        # initialize
        self.num_v = num_v
        self.num_h = num_h
        self.vbias = vbias
        self.hbias = hbias
        self.W = W
            
    def visible_given_hidden(self, h, datatype="binary"):
        if datatype is "binary":
            v_means = sigmoid( self.vbias + dot(h, self.W.T) )
            v_instance = random.binomial(n=1, p=v_means)
            return (v_means, v_instance)
        
        if datatype is "continuous":
            # supposed, variance is 1.0
            v_instance = v_means = random.normal(self.vbias + dot(h, self.W.T), 1.)
            return (v_means, v_instance)

        else:
            print "given an invalid argument: datatype=%s" % datatype
            return False
    
    def hidden_given_visible(self, v):
        h_means = sigmoid( self.hbias + dot(v, self.W) )
        h_instance = random.binomial(n=1, p=h_means)
        return (h_means, h_instance)

    def reconstruct(self, v):
        return self.visible_given_hidden( self.hidden_given_visible(v)[1] )

    def gibbs_sampling(self, data, datatype="binary"):
        h0_means, h0_sample = self.hidden_given_visible(data)
        # print 'h0_means',h0_means
        # print 'h0_sample',h0_sample

        v1_means, v1_sample = self.visible_given_hidden(h0_sample, datatype=datatype)
        h1_means, h1_sample = self.hidden_given_visible(v1_sample)

        return (v1_means, v1_sample, h1_means, h1_sample)

    def training(self, data, datatype="binary", learning_rate=0.1, bound=1000, CD_A=1):
        for step in range(bound):
            h_means, h_sample = self.hidden_given_visible(data)
        
            for A in range(CD_A):
                if A == 0:
                    vn_means, vn_samples, hn_means, hn_samples = self.gibbs_sampling(data, datatype)
                else:
                    vn_means, vn_samples, hn_means, hn_samples = self.gibbs_sampling(vn_samples, datatype)

            self.W += learning_rate * ( dot(data.T, h_means) - dot(vn_samples.T, hn_means) )
            self.vbias += learning_rate * mean(data - vn_samples, axis=0)
            self.hbias += learning_rate * mean(h_means - hn_means, axis=0)

            print self.log_likelihood(data)

    def log_likelihood(self, data):
        v1_means = self.reconstruct(data)[0]
        # print data
        # print v1_means
        # print data * v1_means + (1 - data) * (1-v1_means)
        likelihood = data * v1_means + (1 - data) * (1-v1_means)
        return  sum( log( mean(likelihood , axis=1) ) )

def sigmoid(x):
    return 1. / (1 + exp(-x))

def gauss(x, mean, variance=1.):
    # need to import "erf" function
    # that included python v2.7 or later,
    # and also included scipy.special
    return ( 1. + math.erf( x - mean / (2*variance)**0.5 ) ) / 2.

if __name__=='__main__':
    rbm = RBM(2,6)

    # train
    traindata = array([[12.0, -10.1],
                       [11.0, -9.6],
                       [11.5, -10.2],

                       [10.5, 10.0],
                       [10.1, 8.9],
                       [13.0, 9.9],
                       ])
    rbm.training(traindata, datatype="continuous", learning_rate=0.05, bound=10000, CD_A=1)

    # test
    testdata = array([[10.0, -10.0],
                      [10.0, 10.0],
                      
                      [0.0, -10.0],
                      [10.0, 0.0],

                      [0.0, 0.0],
                      ])
    for td in testdata:
        print ''
        print 'td',td
        h = rbm.hidden_given_visible(td)[1]
        v = rbm.visible_given_hidden(h, datatype="continuous")[1]
        print 'h',h
        print 'v',v
