import numpy as np
import argparse
import os
import sys

def integerize_columns(table, index_multiple_pairs):
    for index,mult in index_multiple_pairs:
        table[:,index] = np.vectorize(lambda x: int(round(x)))(table[:,index] / mult)
    table = table.astype(np.int32)
    table = table - table.min(axis=0)
    return table

def integerize_table():
    parser = argparse.ArgumentParser()
    parser.add_argument("-csv", type=str, required=True, help="Input CSV file")
    parser.add_argument("-instatecols", type=int, nargs="+", required=True)
    parser.add_argument("-outstatecols", type=int, nargs="+", required=True)
    parser.add_argument("-actioncol", type=int, required=True)
    parser.add_argument("-rewardcol", type=int, required=True)
    parser.add_argument("-output_numpy", type=str, required=True, help="Output numpy txt file")
    
    args = parser.parse_args()

    table = np.loadtxt(args.csv, delimiter=",",skiprows=1)

    #.0125664 1 4 .04 2 5
    a=.0125664
    b=.04
    pairs = [(0,a),(1,b)]

    instates = table[:,args.instatecols]
    actions = table[:,args.actioncol][:,None]
    outstates = table[:,args.outstatecols]
    rewards = table[:,args.rewardcol][:,None]

    instates = integerize_columns(instates, pairs)
    outstates = integerize_columns(outstates, pairs)
    table = np.hstack([instates, actions.astype(np.int32), outstates, rewards.astype(np.int32)])

    np.savetxt(args.output_numpy, table, fmt="%d")


if __name__ == "__main__": 
    #main()
    #test_regression()
    integerize_table()

