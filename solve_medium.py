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

def sorted_uniq_vals(c):
    return sorted(set(c))

class CsvTable(object):
    def __init__(self, file_name, instatecols, actioncol, outstatecols, rewardcol, statecirc):
        print(statecirc)
        assert(len(instatecols) == len(outstatecols))
        assert(len(instatecols) == len(statecirc))
        
        self.table = np.loadtxt(file_name, dtype=np.int32)
        
        self.instates = table[:,instatecols]
        self.actions = table[:,actioncol][:,None]
        self.outstates = table[:,outstatecols]
        self.rewards = table[:,rewardcol][:,None]
    
        self.statemaxvals = list(self.instates.max(axis=0))

        self.statevoc,self.i_to_state = self.make_row_voc(self.instates)
        outstatevoc,i_to_ostate = self.make_row_voc(self.outstates)
        assert(self.statevoc == outstatevoc)
        self.actionvoc,self.i_to_action = self.make_row_voc(self.actions)
        
        self.all_states = np.array(self.i_to_state)

        self.NA = len(instatecols)
        self.NS = self.all_states.shape[0]
        self.SC = len(instatecols)

    def make_row_voc(self,t):
        uniq_by_col = [sorted_uniq_vals(t[:,i]) for i in range(t.shape[1])]
        all_combinations = list(itertools.product(*uniq_by_col))
        vocab = {t : i for i,t in enumerate(all_combinations)}
        return vocab,all_combinations

    def row_index_of_state_action(self, s_ind, a_ind):
        if not self.is_full:
            raise Exception("Table has not imputed missing values")
        index = (a_ind * self.NS) + s_ind
        assert(self.statevoc[tuple(self.instates[index])] == s_ind)
        assert(self.actionvoc[tuple(self.actions[index])] == a_ind)

        return index

    def impute_missing(self, k, kernel):
        outstates_new = np.empty((NS*NA,S), dtype=np.int32)
        outstates_new[...] = -1

        Xeval = self.all_states

        instates_new, actions_new, rewards_new = ([],[])
    
        # for each action, for each outstate component, regress
        for a in range(self.NA):
            mask = (self.actions == self.i_to_action[a]).flatten()
            X = instates[mask]
            ys = outstates[mask]
        
            full_instates.append(all_states)
            for j in range(ys.shape[1]):
                yeval = outstates_new[a*NS:(a+1)*NS,j]
                y = ys[:,j]
                yeval[:] = locally_linear_regression(X, y, Xeval, k, kernel, self.statecirc[j], self.statemaxvals[j])
            actions_new.append(np.ones((NS,1)) * a)

        self.instates = np.vstack(instates_new)
        self.actions = np.vstack(actions_new)
        self.outstates = outstates_new
        self.rewards = TODO
    
        assert(self.instates.shape[0] == self.actions.shape[0])
        assert(self.instates.shape[0] == self.outstates.shape[0])

class ImplicitTransitionMatrix(object):
    def __init__(self, table, d, f):
        self.table = table
        self.d = d
        self.f = f
        
    def product(self, p, v):
        # find outstate mean for each instate according to the policy
        # v is a numpy array (i.e. vector) of dimension (NS,)
        # p is a list of action values of length NS
        for s_ind in range(self.table.NS):
            a_val = p[s_ind]
            a_ind = self.table.actionvoc[(a_val,)]
            sa_ind = self.table.row_index_of_state_action(s_ind,a_ind)
            mean_out_state = self.outstates[sa_ind]
            r = self.rewards[sa_ind][0]

            out_states_and_probs = self.compute_out_states_and_probs(mean_out_state)
            total = 0
            for osval, prob in out_states_and_probs:
                
            

# returns a matrix of neighbor vals and a vector of scores using kernel
def compute_scored_neighbors(xv, maxvals, is_circ, k, kernel, normalize_scores=True):
    offsets = np.array(list(itertools.product(range(-k,k+1),range(-k,k+1))))
    neighbors = xv + offsets
    assert(len(maxvals)==len(is_circ))
    assert(len(maxvals)==xv.size)
    assert(len(maxvals)==neighbors.shape[1])

    for j in range(neighbors.shape[1]):
        _max = maxvals[j]
        _iscirc = is_circ[j]
        _min = 0

        # Fix less than _min
        while True:
            col = neighbors[:,j]
            I = col < _min
            if not any(list(I)):
                break
            else:
                if _iscirc:
                    col[I] += _max
                else:
                    notI = col >= _min
                    neighbors = neighbors[notI]

        # Fix less than _min
        while True:
            col = neighbors[:,j]
            I = col > _max
            if not any(list(I)):
                break
            else:
                if _iscirc:
                    col[I] -= _max
                else:
                    notI = col <= _max
                    neighbors = neighbors[notI]

    scores = kernel(neighbors, xv)
    assert(scores.shape == (neighbors.shape[0],))
    assert(all(list(scores>=0)))

    if normalize_scores:
        scores = scores / np.sum(scores)
    
    return (neighbors, scores)

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

def random_initial_policy(NS,NA):
    return np.random.randint(NA,size=NS)

def one_step_policy_iteration(p, T, R, gamma):
    NS = len(p)
    NA = R.shape[1]

    p_tuples = list(zip(list(np.arange(p.size)), list(p)))

    Tp = np.vstack([T[i] for i in p_tuples])
    Rp = np.array([R[i] for i in p_tuples])
    assert Tp.shape == (NS,NS), "Tp.shape={}, p_tuples[0]={}, T[p_tuples[0]]={}".format(Tp.shape,p_tuples[0], T[p_tuples[0]])
    assert Rp.shape == (NS,), "Rp.shape={}".format(Rp.shape)
    
    Up = np.linalg.solve(np.eye(NS)-gamma*Tp, Rp)
    
    Usa = R + gamma * np.dot(T,Rp)
    assert(Usa.shape == (NS,NA))
    
    p_new = np.argmax(Usa,axis=1)
    assert(p_new.shape == (NS,))

    return p_new

def policy_iteration(p0, T, R, gamma):
    p = p0
    while True:
        pPrev = p
        p = one_step_policy_iteration(pPrev, T, R, gamma)
        if np.array_equal(pPrev,p):
            break
    return p

def locally_linear_regression(X,y,xeval,k,kernel,iscirc,maxval):
    assert len(y.shape)==1
    print("Xeval shape={}".format(xeval.shape))
    yeval = np.zeros((xeval.shape[0]))
    for i,xv in enumerate(xeval):
        print "\r   {}".format(i),
        ninds,ksub = scored_neighbor_indices(X,xv,k,kernel)
        Xsub,ysub = (X[ninds],y[ninds])
        Xsub = np.hstack((Xsub.copy(),np.ones((Xsub.shape[0],1))))

        assert(len(sorted_uniq_vals(X_sub)) != 1)
        if iscirc:
            ysub = np.array(fix_circular_discontinuity(list(ysub),maxval))
        # Weighted linear regression is equivalent to reweighting the inputs/outputs
        Xsubw = Xsub * np.sqrt(ksub)[:,None]
        ysubw = ysub * np.sqrt(ksub)
        c = np.linalg.lstsq(Xsubw,ysubw)[0]
        xv = np.concatenate((xv,[1.0]))
        yv = int(xv.dot(c))
        if iscirc:
            yv = yv % (maxval+1)
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
    parser.add_argument("-k", type=int, required=True)
    parser.add_argument("-tau", type=float, required=True)
    
    parser.add_argument("-outprefix", type=str, required=True)
    
    args = parser.parse_args()

    csv = CsvTable(args.csv, args.instatecols, args.actioncol, args.outstatecols, args.rewardcol, args.statecirc):
    expanded = ExpandedTable(csv, 
        #TODO

    #Original: NS*NA x SC+AC+SC
    state_indexer = lambda row: statevoc[tuple(row)]
    neighbor_lister = lambda 

    scored_neighbor_ids = compute_all_scored_neighbor_ids(all_states, statemaxvals, args.statecirc, args.knn, lambda X,xv: args.F ** np.abs(X-xv).sum(axis=1))

        

    T = convert_table_to_sparse_transition_matrix(all_states, mean_outstate, state_indexer, 


    #T,R = max_likelihood_T_and_R_tables(instates_I, actions_I, outstates_I, rewards)
    
    p0 = random_initial_policy(len(statevoc),len(actionvoc))
    p = policy_iteration(p0, T, R, args.gamma)

    

    #np.savetxt("{}.t_partial".format(args.outprefix), T, fmt="%d")

if __name__ == "__main__": 
    learn_policy()

