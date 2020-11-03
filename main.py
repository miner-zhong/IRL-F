

'''

Implementation of IRL-F
Written by Miner Zhong.


first sepcify input data in args.py

    For Nguyen-Dupuis network:
    there are 4 observation scenarios for:
        all of these scenarios share the same set of observed link flow (30% of the entire link set)
        for observed trajectories:
        -- s1: All path sampling rate = 100%, and observed trajectories cover all possible transitions in the network
        -- s2: All path sampling rate = 100%
        -- s3: Among the entire path set, sampling rate range from 20%-40%
        -- s4: Among the entire path set, sampling rate range from 5%-20%

    For Berlin network:
    there are 4 observation scenarios for:
        all of these scenarios share the same set of observed link flow (30% of the entire link set)
        for observed trajectories:
        -- s1: Trajectory sampling rate =20%-40% + use real road features to define state features
        -- s2: Trajectory sampling rate =5%-20% + use real road features to define state features
        -- s3: Trajectory sampling rate =20%-40% + use one-hot vector of the state index to define state features)
        -- s4: Trajectory sampling rate =5%-20% + use one-hot vector of the state index to define state features)

then run this program

when finished, the output file contains state visit frequency & link flow estimation results

'''





import roadnetwork
import algo_irl as IRL
from util import *

import argparse
from args import appropo_args
parser = argparse.ArgumentParser()
appropo_args(parser)
args = parser.parse_args()




def main(epochs, learning_rate):

    # get road network
    rn = roadnetwork.RoadNetwork(args)

    # run IRL-F
    (reward, svf) = IRL.irl(rn, args, epochs, learning_rate)

    # get svf result
    result_svf = []
    for i in range(args.nb_state):
        result_svf.append(svf[i])

    result_svf_normalized = []
    for r in result_svf:
        result_svf_normalized.append(r / sum(result_svf))
    for j in range(len(result_svf_normalized)):
        if (result_svf_normalized[j] == 0):
            result_svf_normalized[j] = 1e-12

    truth_svf_normalized = args.svf_truth

    svf_result = {}
    for i in range(args.nb_state):
        svf_result['s' + str(i)] = {'svf_true':truth_svf_normalized[i], 'svf_estimated':result_svf_normalized[i]}


    # find scalar
    ratio = []
    for k in args.link_flow_obs.keys():
        state = 's' + k
        if (svf_result[state]['svf_estimated'] > args.scalar_tolerance):
            if (args.link_flow_obs[k] > 0):
                ratio.append(args.link_flow_obs[k] / svf_result[state]['svf_estimated'])
    scalar = sum(ratio) / len(ratio)

    # get estimation result
    estimation_result = {}
    for k in svf_result.keys():
        estimation_result[k] = svf_result[k]['svf_estimated'] * scalar


    # output
    output_svf(svf_result, result_svf_normalized, truth_svf_normalized, args)
    output_estimation(estimation_result, args)
    print('done')




if __name__ == '__main__':
    main(args.iteration, args.learning_rate)
