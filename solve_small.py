import numpy as np
import argparse
import itertools

def sorted_uniq_vals(c):
    return sorted(set(c))

def reverse_each_tuple_in_list(l):
    return [tuple(reversed(i)) for i in l]

def make_row_voc(t,aa228_sorting=False):
    uniq_by_col = [sorted_uniq_vals(t[:,i]) for i in range(t.shape[1])]
    all_combinations = list(itertools.product(*uniq_by_col))
    if aa228_sorting:
        all_combinations = reverse_each_tuple_in_list(sorted(reverse_each_tuple_in_list(all_combinations)))
    vocab = {t : i for i,t in enumerate(all_combinations)}
    return vocab,all_combinations

def max_likelihood_T_and_R_tables(instates_I, actions_I, outstates_I, rewards):
    NS = len(set(instates_I))
    NA = len(set(actions_I))

    T = np.zeros((NS,NA,NS),dtype=np.float_)
    R = np.zeros((NS,NA))
    R[...] = np.NAN

    for s,a,r,sprime in zip(instates_I,actions_I,rewards,outstates_I):
        T[s,a,sprime] += 1
        assert(np.isnan(R[s,a]) or r == R[s,a])
        R[s,a] = r
    #Normalize
    T = T / T.sum(axis=2,keepdims=True)
    return (T,R)
       
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
    
    Usa = R + gamma * np.dot(T,Up)
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

def learn_policy():
    parser = argparse.ArgumentParser()
    parser.add_argument("-csv", type=str, required=True, help="Input CSV file")
    parser.add_argument("-instatecols", type=int, nargs="+", required=True)
    parser.add_argument("-outstatecols", type=int, nargs="+", required=True)
    parser.add_argument("-actioncol", type=int, required=True)
    parser.add_argument("-rewardcol", type=int, required=True)
    parser.add_argument("-gamma", type=float, required=True)
    parser.add_argument("-outprefix", type=str, required=True)
    
    args = parser.parse_args()

    assert(len(args.instatecols) == len(args.outstatecols))

    table = np.loadtxt(args.csv, delimiter=",",skiprows=1, dtype=np.int32)

    instates = table[:,args.instatecols]
    actions = table[:,args.actioncol][:,None]
    outstates = table[:,args.outstatecols]
    rewards = table[:,args.rewardcol][:,None]

    statevoc,i_to_state = make_row_voc(instates,aa228_sorting=True)
    outstatevoc,i_to_ostate = make_row_voc(outstates,aa228_sorting=True)
    assert(statevoc == outstatevoc)
    actionvoc,i_to_action = make_row_voc(actions)

    instates_I = [statevoc[tuple(r)] for r in instates]
    actions_I = [actionvoc[tuple(r)] for r in actions]
    outstates_I = [statevoc[tuple(r)] for r in outstates]

    T,R = max_likelihood_T_and_R_tables(instates_I, actions_I, outstates_I, rewards)
    p0 = random_initial_policy(len(statevoc),len(actionvoc))
    p = policy_iteration(p0, T, R, args.gamma)

    #ptuples = [sv+(pv,) for sv,pv in zip(i_to_state,list(p))]
    #print "\n".join([" ".join([str(j) for j in i]) for i in ptuples])
    with open("{}.policy".format(args.outprefix),"w") as fout:
        for i in (p):
            action = i_to_action[i][0]
            fout.write("{}\n".format(action))

    #np.savetxt("{}.t_partial".format(args.outprefix), T, fmt="%d")
    #np.savetxt("{}.policy".format(args.outprefix), p)

if __name__ == "__main__": 
    learn_policy()


