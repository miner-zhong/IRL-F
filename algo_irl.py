from itertools import product
import numpy as np



def irl(rn, args, epochs, learning_rate):

    feature_matrix = rn.feature_matrix_all(args)

    feature_matrix_1 = rn.feature_matrix_traj(args)
    feature_matrix_2 = rn.feature_matrix_count(args)

    transition_probability = rn.transition_probability

    n_states = rn.n_states
    d_states = feature_matrix.shape[1]
    n_actions = rn.n_actions

    discount = args.discount
    trajectory_horizon = args.trajectory_horizon

    obs_trajectories = rn.load_trajectories(args)
    traffic_counts = rn.load_traffic_count(args)

    # Initialise weights.
    alpha = np.random.uniform(size=(d_states,))

    # Calculate the feature expectations \tilde{phi}.
    feature_expectations = find_feature_expectations(feature_matrix, obs_trajectories, traffic_counts, args)

    # Gradient descent on alpha.
    last_svf = []
    reward= None
    for i in range(epochs):
        r = feature_matrix.dot(alpha)
        expected_svf = find_expected_svf(n_states, r, n_actions, discount,
                                         transition_probability, obs_trajectories, trajectory_horizon, args)

        true_feature = feature_expectations

        expected_feature = calculate_expected_feature_expectation(feature_matrix, feature_matrix_1, feature_matrix_2,
                                                                  expected_svf, traffic_counts, args)

        grad = true_feature - expected_feature

        alpha += learning_rate * grad

        print("epoch: {}".format(i))

        last_svf=expected_svf

        reward = feature_matrix.dot(alpha).reshape((n_states,))

    return (reward, last_svf)




def calculate_expected_feature_expectation(feature_matrix, feature_matrix_1, feature_matrix_2, expected_svf, traffic_counts, args):
    expected_feature = np.zeros(feature_matrix.shape[1])

    expected_feature[:feature_matrix_1.shape[1]] = feature_matrix_1.T.dot(expected_svf)

    obs_states = []
    for k in traffic_counts.keys():
        obs_states.append(int(k))
    expected_svf_counts = []
    for j in range(len(expected_svf)):
        if (j in obs_states):
            expected_svf_counts.append(expected_svf[j])
        else:
            expected_svf_counts.append(0)
    expected_svf_counts_normalized = []
    for j in range(len(expected_svf)):
        expected_svf_counts_normalized.append(expected_svf_counts[j] / sum(expected_svf_counts))
    expected_svf_counts_normalized = np.array(expected_svf_counts_normalized)

    expected_feature[feature_matrix_1.shape[1]:] = feature_matrix_2.T.dot(expected_svf_counts_normalized)

    return expected_feature






def find_feature_expectations(feature_matrix, obs_trajectories, traffic_counts, args):
    feature_expectations = np.zeros(feature_matrix.shape[1])

    for trajectory in obs_trajectories:
        for state in trajectory:
            feature_expectations[:args.nb_state] += feature_matrix[state][:args.nb_state]
    feature_expectations[:args.nb_state] /= obs_trajectories.shape[0]

    sum = 0
    for k in traffic_counts.keys():
        feature_expectations[args.nb_state:] += (feature_matrix[int(k)][args.nb_state:] * traffic_counts[k])
        sum += traffic_counts[k]
    feature_expectations[args.nb_state:] /= sum

    return feature_expectations





def find_expected_svf(n_states, r, n_actions, discount,
                      transition_probability, trajectories, trajectory_horizon, args):

    n_trajectories = trajectories.shape[0]
    trajectory_length = trajectory_horizon

    policy = find_policy(n_states, n_actions,transition_probability, r, discount, stochastic=True)

    start_state_count = np.zeros(n_states)
    for trajectory in trajectories:
        start_state_count[trajectory[0]] += 1
    p_start_state = start_state_count/n_trajectories


    expected_svf = np.tile(p_start_state, (trajectory_length, 1)).T
    for t in range(1, trajectory_length):
        expected_svf[:, t] = 0
        for i, j, k in product(range(n_states), range(n_actions), range(n_states)):
            expected_svf[k, t] += (expected_svf[i, t-1] *
                                  policy[i, j] *
                                  transition_probability[i, j, k])



    LAST_STEP_SVF = {}
    count = 0
    for a in range(n_states):
        LAST_STEP_SVF[str(a)] = expected_svf[a, trajectory_length-1]
        count += expected_svf[a, trajectory_length-1]


    ABSORB_VF = {}
    count2 = 0
    for a in range(args.nb_state, n_states):
        ABSORB_VF[str(a)] = expected_svf[a, trajectory_length-1]
        count2 += expected_svf[a, trajectory_length - 1]


    return expected_svf.sum(axis=1)




def optimal_value(n_states, n_actions, transition_probabilities, reward,
                  discount, threshold=1e-2):

    v = np.zeros(n_states)

    diff = float("inf")
    while diff > threshold:
        diff = 0
        for s in range(n_states):
            max_v = float("-inf")
            for a in range(n_actions):
                tp = transition_probabilities[s, a, :]
                max_v = max(max_v, np.dot(tp, reward + discount*v))

            new_diff = abs(v[s] - max_v)
            if new_diff > diff:
                diff = new_diff
            v[s] = max_v

    return v


def find_policy(n_states, n_actions, transition_probabilities, reward, discount, threshold=1e-2, v=None, stochastic=True):
    if v is None:
        v = optimal_value(n_states, n_actions, transition_probabilities, reward,
                          discount, threshold)

    if stochastic:
        Q = np.zeros((n_states, n_actions))
        for i in range(n_states):
            for j in range(n_actions):
                p = transition_probabilities[i, j, :]
                Q[i, j] = p.dot(reward + discount*v)
        Q -= Q.max(axis=1).reshape((n_states, 1))
        Q = np.exp(Q)/np.exp(Q).sum(axis=1).reshape((n_states, 1))
        return Q

    def _policy(s):
        return max(range(n_actions),
                   key=lambda a: sum(transition_probabilities[s, a, k] * (reward[k] + discount * v[k]) for k in range(n_states)))
    policy = np.array([_policy(s) for s in range(n_states)])
    return policy



