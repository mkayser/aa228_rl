python solve_small.py -csv small.csv -instatecols 0 1 -actioncol 2 -outstatecols 3 4 -rewardcol 5 -outprefix small -gamma .9
python integerize_pendulum_table.py -instatecols 0 1 -actioncol 2 -outstatecols 3 4 -rewardcol 5 -csv medium.csv -output_numpy medium.ints.txt

python -u solve_medium.py -csv medium.ints.txt -instatecols 0 1 -actioncol 2 -outstatecols 3 4 -rewardcol 5 -statecirc True False -k 10 -ninner 3 -nouter 3 -d 2 -f .3 -gamma .9 -outprefix TEST_MEDIUM -debug
