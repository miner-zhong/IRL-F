import csv
import numpy as np
import pandas as pd
from scipy.stats import entropy




def load_network_properties(path):
    with open(path,'r') as csvfile:
        reader = csv.reader(csvfile)
        rows= [row for row in reader]
    idx_3 = rows[0].index('nb_state')
    nb_state = int(rows[1][idx_3])
    idx_4 = rows[0].index('nb_action')
    nb_action = int(rows[1][idx_4])
    idx_5 = rows[0].index('nb_feature')
    nb_feature = int(rows[1][idx_5])
    return (nb_state, nb_action, nb_feature)



def load_trajectories(path):
    with open(path,'r') as csvfile:
        reader = csv.reader(csvfile)
        rows= [row for row in reader]
    trajectories = {}
    trajs_number = {}
    trajs_number_truth = {}

    ls_traj = []
    ls_nb = []
    ls_nb_truth = []
    idx_1 = rows[0].index('ls_states')
    for i in range(1, len(rows)):
        ls_traj.append(rows[i][idx_1])
    idx_2 = rows[0].index('observed_nb')
    for i in range(1, len(rows)):
        ls_nb.append(int(rows[i][idx_2]))
    idx_3 = rows[0].index('ground_truth')
    for i in range(1, len(rows)):
        ls_nb_truth.append(int(rows[i][idx_3]))

    for i in range(len(ls_traj)):
        traj = ls_traj[i]
        t = []
        if(',' in traj):
            info = traj.split(',')
            t.append(int(info[0][1:]))
            for k in range(1, len(info) - 1):
                t.append(int(info[k]))
            t.append(int(info[-1][:-1]))
        else:
            t.append(int(traj[1:-1]))
        trajectories['t' + str(i)] = t
        trajs_number['t' + str(i)] = ls_nb[i]
        trajs_number_truth['t' + str(i)] = ls_nb_truth[i]
    return (trajectories, trajs_number, trajs_number_truth)




def load_traffic_count_1(path):
    with open(path,'r') as csvfile:
        reader = csv.reader(csvfile)
        rows= [row for row in reader]
    counts = {}
    ls_state = []
    ls_nb = []
    idx_1 = rows[0].index('obs_state')
    for i in range(1, len(rows)):
        ls_state.append(rows[i][idx_1])
    idx_2 = rows[0].index('observed_nb')
    for i in range(1, len(rows)):
        ls_nb.append(rows[i][idx_2])
    for i in range(len(ls_nb)):
        counts[str(ls_state[i])] = int(ls_nb[i])
    return counts



def load_traffic_count_2(path):
    data = pd.read_csv(path)
    counts_true = {}
    for i in range(len(data)):
        state = str(list(data['state'])[i])
        flow = int(list(data['flow'])[i])
        counts_true[state] = flow
    return counts_true




def get_ls_destination(trajectories):
    ls_destination = []
    for k in trajectories.keys():
        if not (trajectories[k][-1] in ls_destination):
            ls_destination.append(trajectories[k][-1])
    ls_destination.sort()
    return ls_destination




def get_dest_absorb_info(ls_destination, nb_state):
    dest_absorb_info = {}
    for i in range(len(ls_destination)):
        des = ls_destination[i]
        absorb = nb_state + i
        dest_absorb_info[str(des)] = absorb
    return dest_absorb_info



def get_transition_info(path):
    with open(path,'r') as csvfile:
        reader = csv.reader(csvfile)
        rows= [row for row in reader]
    tran_info = {}
    idx_1 = rows[0].index('state_from')
    idx_2 = rows[0].index('state_to')
    idx_3 = rows[0].index('action')
    for r in rows[1:]:
        if not (str(r[idx_1]) in tran_info.keys()):
            tran_info[str(r[idx_1])] = {'action_next_state_pairs': [[int(r[idx_3]), int(r[idx_2])]]}
        else:
            tran_info[str(r[idx_1])]['action_next_state_pairs'].append([int(r[idx_3]), int(r[idx_2])])
    return tran_info



def find_traj_number(all_trajs_number, trajs):
    path_info_truth = []
    ls_key = list(all_trajs_number.keys())
    for k in ls_key:
        temp = {}
        temp['idx'] = k
        temp['path'] = trajs[k]
        temp['num'] = all_trajs_number[k]
        path_info_truth.append(temp)
    return path_info_truth


def get_count_truth(all_trajs_number, trajs, nbstates):
    ls_states={}
    for i in range(nbstates):
        ls_states['s' + str(i)] = 0

    path_info = find_traj_number(all_trajs_number, trajs)

    ls_key = list(ls_states.keys())
    for k in ls_key:
        for info in path_info:
            for s in info['path']:
                if('s' + str(s) == k):
                    ls_states[k] += info['num']
    ls_name=[]
    ls_count=[]
    for i in range(nbstates):
        ls_name.append('s' + str(i))
        ls_count.append(ls_states['s' + str(i)])
    dict={'state':ls_name, 'counts_truth':ls_count}

    traffic_counts = {}
    for i in range(len(dict['state'])):
        key = dict['state'][i].strip('s')
        traffic_counts[str(key)] = int(dict['counts_truth'][i])

    return traffic_counts


def get_state_repr(sidx, nbstates):
    info = np.zeros(nbstates)
    info[sidx] = 1
    return info


def get_truth_svf_normalized(trajs, all_trajs_number, nbstates):
    traffic_counts = get_count_truth(all_trajs_number, trajs, nbstates)
    result = np.zeros(nbstates)
    sum = 0
    for k in traffic_counts.keys():
        result += (get_state_repr(int(k), nbstates) * traffic_counts[k])
        sum += traffic_counts[k]
    result /= sum

    for i in range(len(result)):
        if(result[i]==0):
            result[i] = 1e-12

    truth_svf = list(result)

    return truth_svf




def output_svf(svf_result, result_svf_normalized, truth_svf_normalized, args):
    result={'state':[], 'svf_true':[], 'svf_estimated':[], 'KL_divergence_1':[], 'KL_divergence_2':[]}
    for k in svf_result.keys():
        result['state'].append(k)
        result['svf_true'].append(svf_result[k]['svf_true'])
        result['svf_estimated'].append(svf_result[k]['svf_estimated'])
        result['KL_divergence_1'].append(entropy(result_svf_normalized, truth_svf_normalized))
        result['KL_divergence_2'].append(entropy(truth_svf_normalized, result_svf_normalized))
    df = pd.DataFrame(result)
    df.to_csv(args.result_svf, index=False,
              columns=['state', 'svf_true', 'svf_estimated', 'KL_divergence_1', 'KL_divergence_2'])




def output_estimation(estimation_result, args):
    output = {'unobs_state': [], 'flow_estimated': [], 'flow_true': []}
    for k in args.link_flow_true.keys():
        if not (k in list(args.link_flow_obs.keys())):
            output['unobs_state'].append(k)
            output['flow_estimated'].append(estimation_result['s'+k])
            output['flow_true'].append(args.link_flow_true[k])
    pd.DataFrame(output).to_csv(args.result_estimation, index=False,
                                columns=['unobs_state', 'flow_true', 'flow_estimated'])