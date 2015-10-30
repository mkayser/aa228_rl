import numpy as np
from sklearn import gaussian_process 
import argparse
import os
import sys
import itertools
import json
import matplotlib.pyplot as plt

# Transition mean
# Independent transition variance
# Impute transition means
# Value iteration
#def value_iteration(T, R_sa, 

def save_array(A,fn):
    with open(fn, 'w') as f:
        json.dump(A.tolist(), f, allow_nan = True, indent=4)
        #for i in range(A.shape[0]):
        #    for j in range(A.shape[1]):
        #        f.write("{:4d} {:4d} [{:4d} {:4d}] [{:4d} {:4d}]\n".format(i,j,A[i,j,0,0],A[i,j,0,1],A[i,j,1,0],A[i,j,1,1]))

def load_array(fn):
    with open(fn,"r") as f:
        A = np.array(json.load(f))
    

def sorted_uniq_vals(c):
    return sorted(set(c))

def make_vocab(t):
    uniq_by_col = [sorted_uniq_vals(t[:,i]) for i in range(t.shape[1])]
    all_combinations = itertools.product(*uniq_by_col)
    vocab = {t : i for i,t in enumerate(all_combinations)}
    return vocab

def scored_neighbor_indices(X,xv,k,kernel):
    # Find k nearest neighbors
    ninds = np.argpartition(np.abs((X-xv).sum(axis=1)),k)[:k]
    scores = kernel(X[ninds],xv)
    return (ninds,scores)

def fix_circular_discontinuity(vals,maxval):
    while(max(vals) - min(vals) > (min(vals)+maxval - max(vals))):
        source = min(vals)
        target = min(vals) + maxval
        vals = [k if k!=source else target for k in vals]
    return vals

def locally_linear_regression(X,y,xeval,k,kernel,iscirc,maxval):
    assert len(y.shape)==1
    print("Xeval shape={}".format(xeval.shape))
    yeval = np.zeros((xeval.shape[0]))
    for i,xv in enumerate(xeval):
        print "\r   {}".format(i),
        ninds,ksub = scored_neighbor_indices(X,xv,k,kernel)
        Xsub,ysub = (X[ninds],y[ninds])
        Xsub = np.hstack((Xsub.copy(),np.ones((Xsub.shape[0],1))))
        if iscirc:
            ysub = np.array(fix_circular_discontinuity(list(ysub)))
        # Weighted linear regression is equivalent to reweighting the inputs/outputs
        Xsubw = Xsub * np.sqrt(ksub)[:,None]
        ysubw = ysub * np.sqrt(ksub)
        c = np.linalg.lstsq(Xsubw,ysubw)[0]
        xv = np.concatenate((xv,[1.0]))
        yeval[i] = xv.dot(c)
    return yeval

# iscirc denotes whether the first and last state values are identified
# These cases are handled specially
def compute_mean_int_tuple(vals, iscirc, maxvals):
    means = []
    for i in range(len(vals[0])):
        ivals = [v[i] for v in vals]
        if iscirc[i]:
            while(max(ivals) - min(ivals) > (min(ivals)+maxvals[i] - max(ivals))):
                source = min(ivals)
                target = min(ivals) + maxvals[i]
                ivals = [k if k!=source else target for k in ivals]
        means.append(int(round(float(sum(ivals))/float(len(ivals)))) % (maxvals[i]+1))
    return tuple(means)
        

def make_incomplete_transition_and_reward_tables(instates, outstates, actions, rewards, statecirc, statemaxvals):
    isv = make_vocab(instates)
    osv = make_vocab(instates)
    av = make_vocab(actions)
    assert(isv == osv)

    lens = tuple(len(set(instates[:,i])) for i in range(instates.shape[1]))
    assert(len(statemaxvals) == len(statecirc))

    print("Statemaxvals={}   Statecirc={}".format(statemaxvals,statecirc))

    # Calculate mean state values
    H = {}
    for i in range(instates.shape[0]):
        key = tuple(list(instates[i]) + list(actions[i]))
        val = tuple(outstates[i])
        H.setdefault(key,[]).append(val)
    
    T = np.empty(lens + (len(av),instates.shape[1]), dtype=np.int32)
    T[:] = -1
    for key,vals in H.iteritems():
        meanval = compute_mean_int_tuple(vals,statecirc,statemaxvals)
        skey,akey = tuple(key[:instates.shape[1]]),tuple(key[instates.shape[1]:])
        #sind = isv[skey]
        aind = av[akey]
        #meansind = isv[meanval]
        #print("Key={}  KeyS={}  KeyA={}  Meanval={}  MeanS={}".format(key,sind,aind,meanval,meansind))
        T[skey + (aind,)] = meanval

    return T



def make_bool(s):
    return (s.lower() in ["true","yes","1","t","y"])

def learn_policy():
    parser = argparse.ArgumentParser()
    parser.add_argument("-csv", type=str, required=True, help="Input CSV file")
    parser.add_argument("-instatecols", type=int, nargs="+", required=True)
    parser.add_argument("-outstatecols", type=int, nargs="+", required=True)
    parser.add_argument("-actioncol", type=int, required=True)
    parser.add_argument("-rewardcol", type=int, required=True)
    parser.add_argument("-statecirc", type=make_bool, nargs="+", required=True)
    parser.add_argument("-outprefix", type=str, required=True)
    
    args = parser.parse_args()

    print(args.statecirc)
    assert(len(args.instatecols) == len(args.outstatecols))
    assert(len(args.instatecols) == len(args.statecirc))

    table = np.loadtxt(args.csv, dtype=np.int32)

    instates = table[:,args.instatecols]
    actions = table[:,args.actioncol][:,None]
    outstates = table[:,args.outstatecols]
    rewards = table[:,args.rewardcol][:,None]

    statemaxvals = list(instates.max(axis=0))

    T = make_incomplete_transition_and_reward_tables(instates, outstates, actions, rewards, args.statecirc)

    #np.savetxt("{}.t_partial".format(args.outprefix), T, fmt="%d")
    save_array(T, "{}.t_partial".format(args.outprefix))

if __name__ == "__main__": 
    learn_policy()

