# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 13:51:59 2015

@author: itanveer
"""
from collections import defaultdict as ddict
import numpy as np
#import matplotlib as mpl
#mpl.use('Agg')
#import matplotlib.pyplot as plt

# Train IBM model-1 using EM algorithm
def train(eng,fr,prtable,debug=False):
    ec_num = ddict(lambda:1e-6)
    ec_den = ddict(lambda:1e-6)
    # E step
    if debug:
        print 'E Step'
    for eSent,fSent in zip(eng,fr):
        # split the words and insert NULL word
        e = eSent.strip().split(' ')
        e.insert(0,'NULL')
        f = fSent.strip().split(' ')
        l = len(e)
        m = len(f)
        # Calculate the expected counts for E step
        for j in xrange(m):
            a = np.zeros(l)
            for i in xrange(l):
                a[i]=prtable[f[j],e[i]]
                if debug:
                    print 'Pr. Lookup:',e[i],f[j],prtable[f[j],e[i]],np.sum(a)
            a = a/np.sum(a)
            for i in xrange(l):
                ec_num[f[j],e[i]] += a[i]
                ec_den[e[i]]+=a[i]
    # M Step
    if debug:
        print 'M Step'
    for fj,ei in prtable.keys():
        # calculate p(fj|ei) table from expected counts
        prtable[fj,ei] = ec_num[fj,ei]/ec_den[ei]
        if debug:
            print ei,fj,ec_num[fj,ei],prtable[fj,ei]
    return prtable

def calclikelihood(prtable,eng,fr):
    L = 0.0
    for eSent,fSent in zip(eng,fr):
        # split the words and insert NULL word
        e = eSent.strip().split(' ')
        e.insert(0,'NULL')
        f = fSent.strip().split(' ')
        l = len(e)
        m = len(f)
        for j in xrange(m):
            k = 0.0
            for i in xrange(l):
                k+=prtable[f[j],e[i]]/l
            L+=np.log(k)
    return np.float(L)

def readfile(filename):
    sentList=[]
    with open(filename,'r') as f:
        for aline in f:
            sentList.append(aline.strip())
    return sentList

def savettable(filename,ttable):
    with open(filename,'w') as f:
        for item in ttable.items():
            print>>f,item[0][0],',',item[0][1],',',item[1]
        
#def decode_alignment(ttable,)


# Run this module to check the sample results against the given test case
def sampleresults():
    prtable = ddict(lambda: 1e-16)
    for iter in xrange(10):
        prtable = train(['a b','a c'],['A B','A C'],prtable,True)
#        xx=0
#        for akey in prtable.keys():
#            if akey
#            prtable[akey]
        print
        print 'Iteration #',iter
        print '================'
        print 'train loglikelihood',calclikelihood(prtable,['a b','a c'],['A B','A C'])
        print 'test loglikelihood',calclikelihood(prtable,['b c'],['B C'])
        print

# Run this module to apply the algorithm on a small training and test data
def runondata(train_eng,train_fra,test_eng,test_fra,output,iterNum):
    prtable = ddict(lambda: 1e-6)
    # read training data
    trainlist_eng = readfile(train_eng)
    trainlist_fra = readfile(train_fra)
    # read test data
    testlist_eng = readfile(test_eng)
    testlist_fra = readfile(test_fra)    

#    plt.figure(1)    
    # iter
    testll_=-1*np.inf
    for iter in xrange(iterNum):
        print 'Iteration #',iter
        prtable1 = train(trainlist_eng,trainlist_fra,prtable)
        trainll = calclikelihood(prtable1,trainlist_eng,trainlist_fra)
        testll = calclikelihood(prtable1,testlist_eng,testlist_fra)
        print 'train_loglikelihood',trainll,'test_loglikelihood',testll
        if testll_<testll:
            testll_ = testll
            prtable = prtable1.copy()
        else:
            break
    savettable('ttable',prtable1)
    return prtable1
#        # visualization
#        if iter>0:
#            plt.subplot(211)
#            plt.scatter(iter,trainll,c='r')
#            plt.hold(True)
#            plt.xlabel('Iteration')
#            plt.ylabel('Log Likelihood (Train)')
#            
#            plt.subplot(212)
#            plt.scatter(iter,testll,c='b')
#            plt.hold(True)
#            plt.xlabel('Iteration')
#            plt.ylabel('Log Likelihood (Test)')
#            
#            if train_eng=='training_short.eng':
#                plt.suptitle('IBM Model 1 Log Likelihood for a small dataset (~100 sentences)')
#            else:
#                plt.suptitle('IBM Model 1 Log Likelihood for whole dataset')
#            plt.draw()
#            plt.pause(0.01)
#            plt.savefig(output,dpi=300)
#            plt.pause(0.01)


if __name__=='__main__':
    # Check the results against given test case
    #sampleresults()
    # Run the IBM model on a small data
    #besttable = runondata('training_short.eng','training_short.fra','test_short.eng','test_short.fra',\
    #    'result_small.png',10)
    runondata('training.eng','training.fra','test.eng','test.fra','result_Full.png',20)
