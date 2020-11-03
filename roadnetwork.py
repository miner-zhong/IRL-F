import numpy as np
import csv

import argparse
from args import appropo_args
parser = argparse.ArgumentParser()
appropo_args(parser)
args = parser.parse_args()



class RoadNetwork(object):
    def __init__(self, args):
        self.n_states = args.nb_state + len(args.ls_destination)
        self.n_actions = args.nb_action + 1

        self.transition_probability = np.array([[[self._transition_probability(i, j, k, args)
                                                  for k in range(self.n_states)]
                                                 for j in range(self.n_actions)]
                                                for i in range(self.n_states)])


    def feature_matrix_all(self, args):
        with open(args.state_feature_file, 'r') as csvfile:
            reader = csv.reader(csvfile)
            rows = [row for row in reader]

        ls_info = []

        for a in range(args.nb_feature):
            feature = 'f' + str(a)
            idx = rows[0].index(feature)
            info = []
            for r in rows[1:]:
                info.append(float(r[idx]))
            ls_info.append(info)

        for j in range(args.nb_obs_state):
            idx = rows[0].index('L'+str(j))
            info = []
            for r in rows[1:]:
                info.append(float(r[idx]))
            ls_info.append(info)

        features = []
        for s in range(args.nb_state):
            f=[]
            for k in range(len(ls_info)):
                f.append(ls_info[k][s])
            features.append(f)

        for s in range(len(args.ls_destination)):
            f=[]
            for k in range(len(ls_info)):
                f.append(0)
            features.append(f)

        return (np.array(features))





    def feature_matrix_traj(self, args):
        with open(args.state_feature_file, 'r') as csvfile:
            reader = csv.reader(csvfile)
            rows = [row for row in reader]

        ls_info = []

        for a in range(args.nb_feature):
            feature = 'f' + str(a)
            idx = rows[0].index(feature)
            info = []
            for r in rows[1:]:
                info.append(float(r[idx]))
            ls_info.append(info)

        features = []
        for s in range(args.nb_state):
            f=[]
            for k in range(len(ls_info)):
                f.append(ls_info[k][s])
            features.append(f)

        for s in range(len(args.ls_destination)):
            f=[]
            for k in range(len(ls_info)):
                f.append(0)
            features.append(f)

        return (np.array(features))





    def feature_matrix_count(self, args):
        with open(args.state_feature_file, 'r') as csvfile:
            reader = csv.reader(csvfile)
            rows = [row for row in reader]

        ls_info = []

        for j in range(args.nb_obs_state):
            idx = rows[0].index('L'+str(j))
            info = []
            for r in rows[1:]:
                info.append(float(r[idx]))
            ls_info.append(info)

        features = []
        for s in range(args.nb_state):
            f=[]
            for k in range(len(ls_info)):
                f.append(ls_info[k][s])
            features.append(f)

        for s in range(len(args.ls_destination)):
            f=[]
            for k in range(len(ls_info)):
                f.append(0)
            features.append(f)

        return (np.array(features))




    def _transition_probability(self, i, j, k, args):
        tran_info = args.tran_info

        def find_next(is_dest):
            next = None
            FLAG = False
            if(is_dest):
                if(j == args.nb_action):
                    next = args.dest_absorb_info[str(i)]
                else:
                    info = tran_info[str(i)]
                    for pair in info['action_next_state_pairs']:
                        if(j == pair[0]):
                            next = pair[1]
                            FLAG = True
                    if not (FLAG):
                        next = i
            else:
                if(j == args.nb_action):
                    next = i
                else:
                    if(str(i) in list(tran_info.keys())):
                        info = tran_info[str(i)]
                        for pair in info['action_next_state_pairs']:
                            if(j == pair[0]):
                                next = pair[1]
                                FLAG = True
                        if not (FLAG):
                            next = i
                    else:
                        next = i
            return next

        ls_destinations = args.ls_destination
        ls_absorbing = []
        for key in args.dest_absorb_info.keys():
            ls_absorbing.append(args.dest_absorb_info[key])

        if(i in ls_absorbing):
            next_s = i
        else:
            if (i in ls_destinations):
                next_s = find_next(is_dest=True)
            else:
                next_s = find_next(is_dest=False)


        if not (next_s==k):
            return 0
        else:
            return 1





    def load_trajectories(self, args):
        with open(args.expert_trajectory_file, 'r') as csvfile:
            reader = csv.reader(csvfile)
            rows = [row for row in reader]
        ls_traj = []
        ls_nb = []
        idx_1 = rows[0].index('ls_states')
        idx_2 = rows[0].index('observed_nb')
        for r in rows[1:]:
            ls_traj.append(r[idx_1])
            ls_nb.append(r[idx_2])
        trajectories_by_s = []
        for i in range(len(ls_traj)):
            traj = ls_traj[i]
            for j in range(int(ls_nb[i])):
                t = []
                if(',' in traj):
                    info = traj.split(',')
                    t.append(int(info[0][1:]))
                    for k in range(1,len(info)-1):
                        t.append(int(info[k]))
                    t.append(int(info[-1][:-1]))
                else:
                    t.append(int(traj[1:-1]))
                trajectories_by_s.append(t)
        return (np.array(trajectories_by_s))





    def load_traffic_count(self, args):
        with open(args.traffic_count_file, 'r') as csvfile:
            reader = csv.reader(csvfile)
            rows = [row for row in reader]
        counts={}
        idx_1 = rows[0].index('obs_state')
        idx_2 = rows[0].index('observed_nb')
        for r in rows[1:]:
            counts[str(r[idx_1])] = int(r[idx_2])
        return counts
