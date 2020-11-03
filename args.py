from util import *


path_1 = './data/nguyen-dupuis'
path_2 = './data/nguyen-dupuis/s3'

# network information & true flow
f_network_info = path_1 + '/network_info.csv'
f_transition_info = path_1 + '/transition_info.csv'
f_link_true = path_1 + '/link_flow_true.csv'

# scenario-specific data
f_trajectories = path_2 + '/INPUT/trajectory.csv'
f_link_obs = path_2 + '/INPUT/link_flow_obs.csv'
f_features = path_2 + '/INPUT/state_feature.csv'

# output files
f_output_1 = path_2 + '/RESULT/svf_result.csv'
f_output_2 = path_2 + '/RESULT/estimation_result.csv'



def appropo_args(parser):
    parser.add_argument('--iteration', type=int, default=3)                  # number of iterations
    parser.add_argument('--learning_rate', type=float, default=0.001)        # learning rate
    parser.add_argument('--discount', type=float, default=0.99)              # discount factor for MDP
    parser.add_argument('--trajectory_horizon', type=int, default=20)        # trajectory horizon for calculating expected svf
    parser.add_argument('--scalar_tolerance', type=float, default=1e-04)     # tolerance for calculating scaling factor

    # environment initialization
    nb_state, nb_action, nb_feature = load_network_properties(path=f_network_info)
    trajectories, trajs_number, trajs_number_truth = load_trajectories(path=f_trajectories)
    parser.add_argument('--nb_state', type=int, default=nb_state)
    parser.add_argument('--nb_action', type=int, default=nb_action)
    parser.add_argument('--nb_feature', type=int, default=nb_feature)

    counts_obs = load_traffic_count_1(path=f_link_obs)
    parser.add_argument('--link_flow_obs', type=dict, default=counts_obs)
    parser.add_argument('--nb_obs_state', type=int, default=len(counts_obs))

    counts_true = load_traffic_count_2(path=f_link_true)
    parser.add_argument('--link_flow_true', type=dict, default=counts_true)

    tran_info = get_transition_info(path=f_transition_info)
    parser.add_argument('--tran_info', type=dict, default=tran_info)

    ls_destination = get_ls_destination(trajectories)
    parser.add_argument("--ls_destination", type=list, default=ls_destination)

    dest_absorb_info = get_dest_absorb_info(ls_destination, nb_state)
    parser.add_argument("--dest_absorb_info", type=dict, default=dest_absorb_info)

    truth_svf_normalized = get_truth_svf_normalized(trajectories, trajs_number_truth, nb_state)
    parser.add_argument('--svf_truth', type=list, default=truth_svf_normalized)

    # input
    parser.add_argument('--state_feature_file', type=str, default= f_features)
    parser.add_argument('--expert_trajectory_file', type=str, default= f_trajectories)
    parser.add_argument('--traffic_count_file', type=str, default= f_link_obs)
    parser.add_argument('--transition_info_file', type=str, default= f_transition_info)

    # output
    parser.add_argument('--result_svf', type=str, default=f_output_1)
    parser.add_argument('--result_estimation', type=str, default=f_output_2)