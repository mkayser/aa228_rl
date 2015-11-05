import numpy as np
from sklearn import gaussian_process 
import argparse
import os
import sys
import itertools
import json
import pickle
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
        
        self.instates = self.table[:,instatecols]
        self.actions = self.table[:,actioncol][:,None]
        self.outstates = self.table[:,outstatecols]
        self.rewards = self.table[:,rewardcol][:,None]
    
        self.statemaxvals = list(self.instates.max(axis=0))
        self.statecirc = statecirc

        self.statevoc,self.i_to_state = self.make_row_voc(self.instates)
        outstatevoc,i_to_ostate = self.make_row_voc(self.outstates)
        assert(self.statevoc == outstatevoc)
        self.actionvoc,self.i_to_action = self.make_row_voc(self.actions)
        
        self.all_states = np.array(self.i_to_state)

        self.NA = len(instatecols)
        self.NS = self.all_states.shape[0]
        self.SC = len(instatecols)
        self.is_full = False

    @classmethod
    def load_full_from_file(cls, file_name):
        with open(file_name, "r") as fin:
            return pickle.load(fin)

    def save_full_to_file(self, file_name):
        assert(self.is_full)
        with open(file_name, "w") as fout:
            pickle.dump(self, fout)

    def make_row_voc(self,t):
        uniq_by_col = [sorted_uniq_vals(t[:,i]) for i in range(t.shape[1])]
        all_combinations = list(itertools.product(*uniq_by_col))
        vocab = {t : i for i,t in enumerate(all_combinations)}
        return vocab,all_combinations

    def state_row_to_ind(self, row):
        return self.statevoc[tuple(row)]

    def action_row_to_ind(self, row):
        return self.actionvoc[tuple(row)]

    def row_index_of_state_action(self, s_ind, a_ind):
        if not self.is_full:
            raise Exception("Table has not imputed missing values")
        index = (a_ind * self.NS) + s_ind
        #assert(self.statevoc[tuple(self.instates[index])] == s_ind), "Bad sind/aind pair: {}, {}".format(s_ind,a_ind)
        #assert(self.actionvoc[tuple(self.actions[index])] == a_ind), "Bad sind/aind pair: {}, {}".format(s_ind,a_ind)

        return index
    
    def convert_states_to_canonical(self, states):
        states = states.copy()
        assert(states.shape[1] == self.SC)
        for i in range(self.SC):
            max_i = self.statemaxvals[i]
            if self.statecirc[i]:
                states[:,i] = states[:,i] % max_i
            else:
                states[:,i] = np.minimum(np.maximum(states[:,i],0),max_i)
        return states

    def impute_missing(self, k, kernel, debug_mode=False):
        outstates_new = np.empty((self.NS*self.NA,self.SC), dtype=np.int32)
        outstates_new[...] = -1

        Xeval = self.all_states

        instates_new, actions_new = ([],[])
    
        # for each action, for each outstate component, regress
        for a in range(self.NA):
            mask = (self.actions == self.i_to_action[a]).flatten()
            X = self.instates[mask]
            ys = self.outstates[mask]
        
            instates_new.append(self.all_states)
            for j in range(ys.shape[1]):
                yeval = outstates_new[a*self.NS:(a+1)*self.NS,j]
                y = ys[:,j]
                yeval[:],err = locally_linear_regression(X, y, Xeval, k, kernel, self.statecirc[j], self.statemaxvals[j], debug_mode)
                if debug_mode:
                    acol = np.ones((Xeval.shape[0],1))*self.i_to_action[a]
                    outindexcol = np.ones((Xeval.shape[0],1))*j
                    err = np.hstack((Xeval,acol,outindexcol,err))
                    np.savetxt("ERR.a{}.sc{}".format(a,j), err, fmt="%.2f")
            actions_new.append(np.ones((self.NS,1),dtype=np.int32) * self.i_to_action[a])

        instates_new = np.vstack(instates_new)
        actions_new = np.vstack(actions_new)
        
        # Handle rewards conservatively: do not perform regression but instead preserve existing reward values
        rewards_new = np.empty((instates_new.shape[0],1))
        rewards_new[...] = np.NAN  # Sentinel value for undefined
        nonzero_reward_row_indices = list(np.nonzero(self.rewards.flatten())[0])

        self.is_full = True
    
        for i in nonzero_reward_row_indices:
            s_ind = self.state_row_to_ind(self.instates[i])
            a_ind = self.action_row_to_ind(self.actions[i])
            sa_ind = self.row_index_of_state_action(s_ind,a_ind)
            assert(np.isnan(rewards_new[sa_ind,0]) or rewards_new[sa_ind,0] == self.rewards[i,0])
            rewards_new[sa_ind,0] = self.rewards[i,0]
            assert(np.array_equal(instates_new[sa_ind], self.instates[i]))
            assert(np.array_equal(actions_new[sa_ind], self.actions[i])), "New action {} not equal to old action {}".format(str(actions_new[sa_ind]),str(self.actions[i]))

        rewards_new[np.isnan(rewards_new)] = 0

        self.instates = instates_new
        self.actions = actions_new
        self.outstates = outstates_new
        self.rewards = rewards_new

        assert(self.instates.shape[0] == self.actions.shape[0])
        assert(self.instates.shape[0] == self.outstates.shape[0])

class ImplicitTransitionMatrix(object):
    def __init__(self, table, d, f):
        self.table = table
        d = abs(d)
        components = [list(range(-d,d+1)) for i in range(self.table.SC)]
        self.neighbors = np.array(list(itertools.product(*components)))
        exps = self.neighbors.sum(axis=1)
        self.probs = f ** exps
        self.probs = self.probs / self.probs.sum()
        assert self.probs.shape == ((2*d+1) ** self.table.SC,)
        
    def product(self, p, v):
        # find outstate mean for each instate according to the policy
        # v is a numpy array (i.e. vector) of dimension (NS,)
        # p is a list of action values of length NS
        result = np.zeros(self.table.NS)

        for s_ind in range(self.table.NS):
            if s_ind % 5000 == 0:
                print("    {}   ".format(s_ind)),
            a_ind = p[s_ind]
            sa_ind = self.table.row_index_of_state_action(s_ind,a_ind)
            mean_out_state = self.table.outstates[sa_ind]

            out_states, out_probs = self.compute_out_states_and_probs(mean_out_state)
            total = 0.0
            for i in range(out_states.shape[0]):
                osval = out_states[i]
                prob = out_probs[i]
                os_ind = self.table.statevoc[tuple(osval)]
                total += v[os_ind] * prob

            result[s_ind] = total
        return result

    def product_as_matrix(self, v):
        # find outstate mean for each instate according to the policy
        # v is a numpy array (i.e. vector) of dimension (NS,)
        result = np.zeros((self.table.NS,self.table.NA))

        for s_ind in range(self.table.NS):
            for a_ind in range(self.table.NA):
                sa_ind = self.table.row_index_of_state_action(s_ind,a_ind)
                mean_out_state = self.table.outstates[sa_ind]

                out_states, out_probs = self.compute_out_states_and_probs(mean_out_state)
                total = 0.0
                for i in range(out_states.shape[0]):
                    osval = out_states[i]
                    prob = out_probs[i]
                    os_ind = self.table.statevoc[tuple(osval)]
                    total += v[os_ind] * prob

                result[s_ind,a_ind] = total
        return result

    def compute_out_states_and_probs(self, mean_out_state):
        out_states = mean_out_state + self.neighbors
        out_probs = self.probs
        out_states = self.table.convert_states_to_canonical(out_states)
        return out_states,out_probs

class ImplicitRewardMatrix(object):
    def __init__(self, table):
        self.table = table
        
    def rewards_given_policy(self, p):
        # find reward for each state according to the policy
        # p is a list of action values of length NS
        result = np.zeros(self.table.NS)

        for s_ind in range(self.table.NS):
            a_ind = p[s_ind]
            sa_ind = self.table.row_index_of_state_action(s_ind,a_ind)
            result[s_ind] = self.table.rewards[sa_ind]
        return result

    def explicit(self):
        result = np.zeros((self.table.NS,self.table.NA))

        for s_ind in range(self.table.NS):
            for a_ind in range(self.table.NA):
                sa_ind = self.table.row_index_of_state_action(s_ind,a_ind)
                result[s_ind,a_ind] = self.table.rewards[sa_ind]
        return result
        

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
    ninds = np.argpartition(np.abs((X-xv)).sum(axis=1),k)[:k]
    scores = kernel(X[ninds],xv)
    return (ninds,scores)

def fix_circular_discontinuity(vals,maxval):
    while(max(vals) - min(vals) > (min(vals)+maxval - min(i for i in vals if i != min(vals)))):
        source = min(vals)
        target = min(vals) + maxval
        vals = [k if k!=source else target for k in vals]
        assert np.amax(vals)<1000
    return vals

def random_initial_policy(NS,NA):
    return np.random.randint(NA,size=NS)

def one_step_policy_iteration(p, T, R, gamma, kinner):
    print("Policy iteration step")
    NS = len(p)
    NA = R.table.NA

    p_tuples = list(zip(list(np.arange(p.size)), list(p)))

    #Tp = np.vstack([T[i] for i in p_tuples])
    Rp = R.rewards_given_policy(p)
    #assert Tp.shape == (NS,NS), "Tp.shape={}, p_tuples[0]={}, T[p_tuples[0]]={}".format(Tp.shape,p_tuples[0], T[p_tuples[0]])
    assert Rp.shape == (NS,), "Rp.shape={}".format(Rp.shape)
    
    Up = np.zeros((NS,))
    for i in range(kinner):
        print("   Inner step")
        Up = Rp + gamma * T.product(p,Up)

    Usa = R.explicit() + gamma * T.product_as_matrix(Up)
    assert(Usa.shape == (NS,NA))
    
    p_new = np.argmax(Usa,axis=1)
    assert(p_new.shape == (NS,))

    return p_new

def policy_iteration(p0, T, R, gamma, kinner, kouter):
    p = p0
    ps = [p]
    for i in range(kouter):
        pPrev = p
        p = one_step_policy_iteration(pPrev, T, R, gamma, kinner)
        ps.append(p)
        if np.array_equal(pPrev,p):
            break
    return (p,np.vstack(ps).T)

def locally_linear_regression(X,y,xeval,k,kernel,iscirc,maxval,debug_mode=False):
    assert len(y.shape)==1
    print("Xeval shape={}".format(xeval.shape))
    yeval = np.zeros((xeval.shape[0]))
    err = np.empty((xeval.shape[0],3))
    err[...] = np.nan
    
    for i,xv in enumerate(xeval):
        if i%10000 == 0:
            print "   {}".format(i)
        #if i%80!=1:
        #    continue
        ninds,ksub = scored_neighbor_indices(X,xv,k,kernel)
        Xsub,ysub = (X[ninds],y[ninds])
        Xsub = np.hstack((Xsub.copy(),np.ones((Xsub.shape[0],1))))

        assert(len(sorted_uniq_vals([tuple(row) for row in Xsub])) != 1)
        if iscirc:
            ysub = np.array(fix_circular_discontinuity(list(ysub),maxval))
        # Weighted linear regression is equivalent to reweighting the inputs/outputs
        Xsubw = Xsub * np.sqrt(ksub)[:,None]
        ysubw = ysub * np.sqrt(ksub)
        c = np.linalg.lstsq(Xsubw,ysubw)[0]
        xv_ = np.concatenate((xv,[1.0]))
        yv = int(xv_.dot(c))
        if iscirc:
            yv = yv % maxval
        yeval[i] = yv
        if debug_mode:
            mask = (X==xv).all(axis=1)
            if mask.any():
                if (xv==[33,108]).all():
                    print "Ymask=", y[mask]
                    print "Ypred=", yv
                    print "Xsub=",Xsub
                    print "ysub=",ysub
                avgy = np.mean(y[mask])
                err[i] = [yv, avgy, np.abs(yv - avgy)]

    return yeval,err

def gaussian_kernel(tau):
    return lambda X,v: np.exp(-1.0 * np.sum((X - v)**2,axis=1)/(2*tau**2))

def make_bool(s):
    return (s.lower() in ["true","yes","1","t","y"])

def learn_policy():
    parser = argparse.ArgumentParser()
    parser.add_argument("-csv", type=str, required=True, help="Input CSV file")
    parser.add_argument("-load", action="store_true")
    parser.add_argument("-debug", action="store_true")
    parser.add_argument("-instatecols", type=int, nargs="+", required=True)
    parser.add_argument("-outstatecols", type=int, nargs="+", required=True)
    parser.add_argument("-actioncol", type=int, required=True)
    parser.add_argument("-rewardcol", type=int, required=True)
    parser.add_argument("-statecirc", type=make_bool, nargs="+", required=True)
    parser.add_argument("-k", type=int, required=True, help="Number of neighbors for local linear approximation")
    parser.add_argument("-ninner", type=int, required=True, help="Number of iterations for inner loop of policy iteration")
    parser.add_argument("-nouter", type=int, required=True, help="Number of iterations for outer loop of policy iteration")
    parser.add_argument("-d", type=int, required=True, help="Max distance (in discretized units) for assigning probability to neighbor states")
    parser.add_argument("-f", type=float, required=True)
    parser.add_argument("-gamma", type=float, required=True)
    
    parser.add_argument("-outprefix", type=str, required=True)
    
    args = parser.parse_args()

    full_csv_fn = "{}.cached_full_table".format(args.outprefix)

    if args.load:
        csv = CsvTable.load_full_from_file(full_csv_fn)
    else:
        csv = CsvTable(args.csv, args.instatecols, args.actioncol, args.outstatecols, args.rewardcol, args.statecirc)
        csv.impute_missing(args.k, gaussian_kernel(0.1), args.debug)
        csv.save_full_to_file(full_csv_fn)

    T = ImplicitTransitionMatrix(csv, args.d, args.f)
    R = ImplicitRewardMatrix(csv)

    p0 = random_initial_policy(len(csv.statevoc),len(csv.actionvoc))
    p,phist = policy_iteration(p0, T, R, args.gamma, args.ninner, args.nouter)

    with open("{}.policy".format(args.outprefix),"w") as fout:
        for i in (p):
            action = csv.i_to_action[i][0]
            fout.write("{}\n".format(action))

    np.savetxt("{}.policy_history".format(args.outprefix), phist, fmt="%d")
    

    #np.savetxt("{}.t_partial".format(args.outprefix), T, fmt="%d")

if __name__ == "__main__": 
    learn_policy()

