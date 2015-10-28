import numpy as np
from sklearn import gaussian_process 
import argparse
import os
import sys
import itertools
import matplotlib.pyplot as plt

def locally_linear_regression(X,y,xeval,K):
    print("Xeval shape={}".format(xeval.shape))
    yeval = np.zeros((xeval.shape[0],y.shape[1]))
    for i,xv in enumerate(xeval):
        print "\r   {}".format(i),
        kall = K(X,xv)
        I = np.argpartition(kall,-10)[-10:]
        ksub,Xsub,ysub = (kall[I],X[I],y[I])
        Xsub = np.hstack((Xsub.copy(),np.ones((Xsub.shape[0],1))))
        # Weighted linear regression is equivalent to reweighting the inputs/outputs
        Xsubw = Xsub * np.sqrt(ksub)[:,None]
        ysubw = ysub * np.sqrt(ksub)[:,None]
        c = np.linalg.lstsq(Xsubw,ysubw)[0]
        xv = np.concatenate((xv,[1.0]))
        yeval[i] = xv.dot(c)
    return yeval
        

def global_linear_regression(X,y,Xeval,deg):
    def expand_basis(Xorig):
        return np.hstack([Xorig**k for k in range(1,deg+1)] + [np.ones((Xorig.shape[0],1))])
    assert(deg <= 4)
    Xaug = expand_basis(X)
    Xevalaug = expand_basis(Xeval)

    coeff, resid, _, _ = np.linalg.lstsq(Xaug, y)

    ypred = Xaug.dot(coeff)
    resid_computed = np.mean((ypred-y)**2, axis=0)

    print("Coeff = {}   Resid = {}  Resid_Computed = {}".format(coeff, resid, resid_computed))
    yeval = Xevalaug.dot(coeff)
    return yeval


def plot(x1,x2,y):
    uniq_x2 = sorted(set(x2))
    colors = np.linspace(0,1,num=len(uniq_x2))

    for v,c in zip(uniq_x2, colors):
        cstr = str(c)
        mask = x2==v
        x1sub = x1[mask]
        ysub = y[mask]
        plt.plot(x1sub,ysub,color=cstr,marker=".",ls="None")
    plt.show()
        
def gaussian_kernel(tau):
    return lambda X,v: np.exp(-1.0 * np.sum((X - v)**2,axis=1)/(2*tau**2))


def integerize_columns(table, index_multiple_pairs):
    for index,mult in index_multiple_pairs:
        table[:,index] = np.vectorize(lambda x: int(round(x)))(table[:,index] / mult)
    table = table.astype(np.int32)
    table = table - table.min(axis=0)
    return table

def integerize_table():
    modes = ["local_linear","global_linear"]

    parser = argparse.ArgumentParser()
    parser.add_argument("-csv", type=str, required=True, help="Input CSV file")
    parser.add_argument("-instatecols", type=int, nargs="+", required=True)
    parser.add_argument("-outstatecols", type=int, nargs="+", required=True)
    parser.add_argument("-actioncol", type=int, required=True)
    parser.add_argument("-rewardcols", type=int, nargs="+", required=True)
    parser.add_argument("-output_csv", type=str, required=True, help="Input CSV file")
    
    args = parser.parse_args()

    table = np.loadtxt(args.csv, delimiter=",",skiprows=1)

    #.0125664 1 4 .04 2 5
    a=.0125664
    b=.04
    pairs = [(0,a),(1,b)]

    instates = table[:,args.instatecols]
    actions = table[:,args.actioncol][:,None]
    outstates = table[:,args.outstatecols]
    rewards = table[:,args.rewardcols]

    instates = integerize_columns(instates, pairs)
    outstates = integerize_columns(outstates, pairs)
    table = np.hstack([instates, actions.astype(np.int32), outstates, rewards.astype(np.int32)])

    np.savetxt(args.output_csv, table, fmt="%d")

def main():
    modes = ["local_linear","global_linear"]

    parser = argparse.ArgumentParser()
    parser.add_argument("-csv", type=str, required=True, help="Input CSV file")
    parser.add_argument("-output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("-output_prefix", type=str, required=True, help="Output file prefix")
    parser.add_argument("-instatecols", type=int, nargs="+", required=True)
    parser.add_argument("-outstatecols", type=int, nargs="+", required=True)
    parser.add_argument("-actioncol", type=int, required=True)
    parser.add_argument("-rewardcols", type=int, nargs="+", required=True)
    parser.add_argument("-mode", type=str, required=True, help="Mode. One of {}".format(modes))
    parser.add_argument("-tau", type=float, default=0.1)
    
    args = parser.parse_args()

    table = np.loadtxt(args.csv, delimiter=",",skiprows=1)

    #.0125664 1 4 .04 2 5
    a=.0125664
    b=.04
    pairs = [(0,a),(3,a),(1,b),(4,b)]

    

    instates = table[:,args.instatecols]
    outstates = table[:,args.outstatecols]
    actions = table[:,args.actioncol]
    rewards = table[:,args.rewardcols]


    uniq_state_lists = [list(enumerate(sorted(set(instates[:,i])))) for i in range(instates.shape[1])]
    uniq_actions = list(enumerate(sorted(set(actions))))
    all_states_list = list(itertools.product(*uniq_state_lists))
    S = len(all_states_list)

    all_states_as_ids = np.array([[pair[0] for pair in state] for state in all_states_list])
    all_states_as_vals = np.array([[pair[1] for pair in state] for state in all_states_list])

    all_conditions_ids = np.vstack([all_states_as_ids] * len(uniq_actions))
    all_conditions_vals = np.vstack([np.hstack([all_states_as_vals, np.ones((S,1)) * pair[1]]) for pair in uniq_actions])

    if args.mode == "local_linear_OLD":
        print("Lists: {}".format(uniq_state_lists))

        output_matrices = []
        for i,val in uniq_actions:
            mask = actions==val
            actioncol = val * np.ones((all_states_as_vals.shape[0],1))
            yeval = locally_linear_regression(instates[mask], outstates[mask,0], all_states_as_vals, gaussian_kernel(args.tau))
            output_matrix = np.hstack((all_states_as_ids,all_states_as_vals,action_col,yeval))
            output_matrices.append(output_matrix)
        output_matrix = np.vstack(output_matrices)
    elif args.mode == "local_linear":
        X = np.hstack([instates, actions[:,None]])
        y = outstates
        Xeval = all_conditions_vals
        ypred = locally_linear_regression(X, y, Xeval, gaussian_kernel(args.tau))
        output_matrix = np.hstack((Xeval,ypred))

    elif args.mode == "global_linear":
        X = np.hstack([instates, actions[:,None]])
        y = outstates
        Xeval = all_conditions_vals
        ypred = global_linear_regression(X, y, Xeval, 1)
        output_matrix = np.hstack((Xeval,ypred))
        
    else:
        raise Exception("Unknown mode: {}".format(args.mode))
        

    np.savetxt("{}/{}.predicted".format(args.output_dir, args.output_prefix), output_matrix, fmt="%.4f")


def test_regression():
    #def locally_linear_regression(X,y,xeval,K):
    X = np.linspace(0,100,num=50)[:,None]
    ytrue = (X**2).flatten()
    yerr = np.random.normal(0,400,ytrue.shape)
    yobs = ytrue + yerr
    plt.plot(X,yobs,color='b',marker="x", ls="None")
    xeval = np.linspace(0,100,num=300)[:,None]

    colors = np.linspace(0,.7,num=10)
    kernels = np.linspace(.1,10,num=10)

    for c,k in zip(colors,kernels):
        yeval = locally_linear_regression(X, yobs, xeval, gaussian_kernel(k))
        plt.plot(xeval,yeval,color=str(c),marker=".")
    plt.show()

if __name__ == "__main__": 
    #main()
    #test_regression()
    integerize_table()

