import argparse
from configparser import ConfigParser
import os

import torch
from torch.utils.data import DataLoader
from dca import analysis, data_util

import numpy as np
from sklearn.linear_model import LinearRegression as LR
from data_util import CrossValidate
from utils.cov_util import calc_pi_from_cross_cov_mats, form_lag_matrix
import pickle


class myconf(ConfigParser):
    def __init__(self, defaults=None):
        ConfigParser.__init__(self, defaults=None)
    def optionxform(self, optionstr):
        return optionstr


def linear_decode_r2(X_train, Y_train, X_test, Y_test, decoding_window=1, offset=0):
    """Train a linear model on the training set and test on the test set.
    This will work with batched training data and/or batched test data.
    X_train : ndarray (time, channels) or (batches, time, channels)
        Feature training data for regression.
    Y_train : ndarray (time, channels) or (batches, time, channels)
        Target training data for regression.
    X_test : ndarray (time, channels) or (batches, time, channels)
        Feature test data for regression.
    Y_test : ndarray (time, channels) or (batches, time, channels)
        Target test data for regression.
    decoding_window : int
        Number of time samples of X to use for predicting Y (should be odd). Centered around
        offset value.
    offset : int
        Temporal offset for prediction (0 is same-time prediction).
    """

    if isinstance(X_train, np.ndarray) and X_train.ndim == 2:
        X_train = [X_train]
    if isinstance(Y_train, np.ndarray) and Y_train.ndim == 2:
        Y_train = [Y_train]

    if isinstance(X_test, np.ndarray) and X_test.ndim == 2:
        X_test = [X_test]
    if isinstance(Y_test, np.ndarray) and Y_test.ndim == 2:
        Y_test = [Y_test]

    X_train_lags = [form_lag_matrix(Xi, decoding_window) for Xi in X_train]
    X_test_lags = [form_lag_matrix(Xi, decoding_window) for Xi in X_test]

    Y_train = [Yi[decoding_window // 2:] for Yi in Y_train]
    Y_train = [Yi[:len(Xi)] for Yi, Xi in zip(Y_train, X_train_lags)]
    if offset >= 0:
        Y_train = [Yi[offset:] for Yi in Y_train]
    else:
        Y_train = [Yi[:Yi.shape[0] + offset] for Yi in Y_train]

    Y_test = [Yi[decoding_window // 2:] for Yi in Y_test]
    Y_test = [Yi[:len(Xi)] for Yi, Xi in zip(Y_test, X_test_lags)]
    if offset >= 0:
        Y_test = [Yi[offset:] for Yi in Y_test]
    else:
        Y_test = [Yi[:Yi.shape[0] + offset] for Yi in Y_test]

    if offset >= 0:
        X_train_lags = [Xi[:Xi.shape[0] - offset] for Xi in X_train_lags]
        X_test_lags = [Xi[:Xi.shape[0] - offset] for Xi in X_test_lags]
    else:
        X_train_lags = [Xi[-offset:] for Xi in X_train_lags]
        X_test_lags = [Xi[-offset:] for Xi in X_test_lags]

    if len(X_train_lags) == 1:
        X_train_lags = X_train_lags[0]
    else:
        X_train_lags = np.concatenate(X_train_lags)

    if len(Y_train) == 1:
        Y_train = Y_train[0]
    else:
        Y_train = np.concatenate(Y_train)

    if len(X_test_lags) == 1:
        X_test_lags = X_test_lags[0]
    else:
        X_test_lags = np.concatenate(X_test_lags)

    if len(Y_test) == 1:
        Y_test = Y_test[0]
    else:
        Y_test = np.concatenate(Y_test)

    model = LR().fit(X_train_lags, Y_train)
    r2 = model.score(X_test_lags, Y_test)
    return r2


def run_analysis_cpic(X, Y, T_pi_vals, dim_vals, offset_vals, decoding_window, X_train=None, X_test=None, Y_test=None, Y_train=None,
                      n_init=1, verbose=False, Kernel=None, xdim=None, beta=1e-3, beta1=1, beta2=0, good_ts=None,
                      standardize_Y=False, train_test_ratio=0.8, regularization_weight=0, linear_encoding=True):
    """
    :param X: N x XDim
    :param Y: N x YDim
    :param T_pi_vals: window size for DCA and CPIC
    :param dim_vals: compressed dimension
    :param offset_vals: Temporal offsets for prediction (0 is same-time prediction)
    :param decoding_window: Number of time samples of X to use for predicting Y (should be odd). Centered around
        offset value.
    :param n_init: the number of initialization for DCA
    :param verbose:
    :return:
    """
    results_r2_size = (len(dim_vals), len(offset_vals), len(T_pi_vals))
    results_MI_size = (len(dim_vals), len(T_pi_vals), 2)
    results_r2 = np.zeros(results_r2_size)
    results_MI = np.zeros(results_MI_size)
    min_std = 1e-6

    if X_train is not None and X_test is not None and Y_train is not None and Y_test is not None:
        pass
    elif X is not None and Y is not None:
        print("Generate X_train, X_test, Y_train, Y_test from X and Y")
        good_cols = (X.std(axis=0) > min_std)
        X = X[:, good_cols]
        if good_ts is not None:
            X = X[:good_ts]
            Y = Y[:good_ts]

        n = X.shape[0]
        n_train = int(n * train_test_ratio)
        X_train = X[:n_train]
        X_test = X[n_train:]
        Y_train = Y[:n_train]
        Y_test = Y[n_train:]
    else:
        raise ValueError("Must provide either (X_train, X_test, Y_train, Y_test) or (X, Y)")

    if Kernel is not None:
        xdim = int(xdim + xdim * (xdim + 1) / 2)  # polynomial

    if Kernel is not None:
        X_train = [Kernel(Xi) for Xi in X_train]
        X_test = Kernel(X_test)

    # mean-center X and Y
    X_mean = np.concatenate(X_train).mean(axis=0, keepdims=True)
    X_train_ctd = [Xi - X_mean for Xi in X_train]
    X_train_ctd = np.stack(X_train_ctd)
    if X_train_ctd.ndim == 2:
        X_train_ctd = [X_train_ctd]
    X_test_ctd = X_test - X_mean
    if standardize_Y:
        Y_mean = np.concatenate(Y_train).mean(axis=0, keepdims=True)
        Y_train_ctd = [Yi - Y_mean for Yi in Y_train]
        Y_test_ctd = Y_test - Y_mean
        Y_train = Y_train_ctd
        Y_test = Y_test_ctd
    Y_train = np.stack(Y_train)
    if Y_train.ndim == 2:
        Y_train = [Y_train]

    # loop over dimensionalities
    for dim_idx in range(len(dim_vals)):
        dim = dim_vals[dim_idx]
        if verbose:
            print("dim", dim_idx + 1, "of", len(dim_vals))

        # # No compression
        # results_original_r2 = np.zeros(len(offset_vals))
        # for offset_idx in range(len(offset_vals)):
        #     offset = offset_vals[offset_idx]
        #     r2 = linear_decode_r2(X_train, Y_train, X_test, Y_test, decoding_window=decoding_window, offset=offset)
        #     results_original_r2[offset_idx] = r2
        # print("Original data, R2: {}".format(results_original_r2))
        
        # loop over T_pi vals
        for T_pi_idx in range(len(T_pi_vals)):
            T_pi = T_pi_vals[T_pi_idx]
            critic_params = {"x_dim": T_pi * ydim, "y_dim": T_pi * ydim, "hidden_dim": hidden_dim}
            critic_params_YX = {"x_dim": T_pi * ydim, "y_dim": T_pi * xdim, "hidden_dim": hidden_dim}
            # train data
            if do_dca_init and linear_encoding:
                init_weights = DCA_init(np.concatenate(X_train_ctd, axis=0), T=T_pi, d=dim, n_init=n_init)
            else:
                init_weights = None

            train_data = PastFutureDataset(X_train_ctd, window_size=T_pi)
            train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

            CPIC, I_compress, I_predictive = train_CPIC(beta, xdim, dim, mi_params, critic_params, baseline_params, num_epochs,
                              train_dataloader, T=T_pi,
                              signiture=args.config, deterministic=deterministic, linear_encoding=linear_encoding, 
                              init_weights=init_weights, lr=lr,
                              num_early_stop=num_early_stop, device=device, beta1=beta1, beta2=beta2,
                              critic_params_YX=critic_params_YX, regularization_weight=regularization_weight,
                              return_mutual_information=True)
            CPIC = CPIC.to(device)
            results_MI[dim_idx, T_pi_idx, 0] = I_compress
            results_MI[dim_idx, T_pi_idx, 1] = I_predictive

            # encode train data and test data via CPIC
            X_train_cpic = CPIC.encode(torch.from_numpy(X_train_ctd).to(torch.float).to(device))
            X_train_cpic = X_train_cpic.cpu().detach().numpy() 
            # X_train_dca = [np.dot(Xi, init_weights) for Xi in X_train_ctd]
            X_test_cpic = CPIC.encode(torch.from_numpy(X_test_ctd).to(torch.float).to(device))
            X_test_cpic = X_test_cpic.cpu().detach().numpy()
            # X_test_dca = np.dot(X_test_ctd, init_weights)
            ### save encoded test data


            for offset_idx in range(len(offset_vals)):
                offset = offset_vals[offset_idx]
                r2_cpic = linear_decode_r2(X_train_cpic, Y_train, X_test_cpic, Y_test, decoding_window=decoding_window, offset=offset)
                # r2_dca = linear_decode_r2(X_train_dca, Y_train, X_test_dca, Y_test, decoding_window=decoding_window, offset=offset)
                results_r2[dim_idx, offset_idx, T_pi_idx] = r2_cpic
        print("dim_idx: {}, R2: {}".format(dim_vals[dim_idx], results_r2[dim_idx]))
    return results_r2, results_MI


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CPIC for real data.')
    parser.add_argument('--config', type=str, default="hc_stochastic_infonce_alt")
    parser.add_argument('--model', type=str, default="CPIC")
    parser.add_argument('--nonlinear_encoding', action='store_true')
    args = parser.parse_args()
    if args.model == "CPIC":
        from CPIC import PastFutureDataset, train_CPIC, DCA_init, Polynomial_expand
    if args.model == "PFPC_RC":
        from PFPC_RC import PastFutureDataset, train_CPIC, DCA_init, Polynomial_expand
    if args.config == 'm1_stochastic_infonce':
        config_file = 'config/config_m1_stochastic_infonce.ini'
        ydims = np.array([5]).astype(int)
    elif args.config == 'm1_stochastic_infonce_alt':
        config_file = 'config/config_m1_stochastic_infonce_alt.ini'
        ydims = np.array([5]).astype(int)
    elif args.config == 'm1_deterministic_infonce_alt':
        config_file = 'config/config_m1_deterministic_infonce_alt.ini'
        ydims = np.array([5]).astype(int)
    elif args.config == 'm1_stochastic_infonce_alt_v2':
        config_file = 'config/config_m1_stochastic_infonce_alt_v2.ini'
        ydims = np.array([5]).astype(int)
    elif args.config == 'hc_stochastic_infonce':
        config_file = 'config/config_hc_stochastic_infonce.ini'
        ydims = np.array([5]).astype(int)
    elif args.config == 'hc_stochastic_infonce_alt':
        config_file = 'config/config_hc_stochastic_infonce_alt.ini'
        ydims = np.array([5]).astype(int)
    elif args.config == 'hc_deterministic_infonce_alt':
        config_file = 'config/config_hc_deterministic_infonce_alt.ini'
        ydims = np.array([5]).astype(int)
    elif args.config == 'hc_stochastic_infonce_alt_v2':
        config_file = 'config/config_hc_stochastic_infonce_alt_v2.ini'
        ydims = np.array([5]).astype(int)
    elif args.config == 'temp_stochastic_infonce':
        config_file = 'config/config_temp_stochastic_infonce.ini'
        ydims = np.array([5]).astype(int)
    elif args.config == 'temp_stochastic_infonce_alt':
        config_file = 'config/config_temp_stochastic_infonce_alt.ini'
        ydims = np.array([5]).astype(int)
    elif args.config == 'temp_deterministic_infonce_alt':
        config_file = 'config/config_temp_deterministic_infonce_alt.ini'
        ydims = np.array([5]).astype(int)
    elif args.config == 'ms_stochastic_infonce':
        config_file = 'config/config_ms_stochastic_infonce.ini'
        ydims = np.array([5]).astype(int)
    elif args.config == 'ms_stochastic_infonce_alt':
        config_file = 'config/config_ms_stochastic_infonce_alt.ini'
        ydims = np.array([5]).astype(int)
    elif args.config == 'ms_deterministic_infonce_alt':
        config_file = 'config/config_ms_deterministic_infonce_alt.ini'
        ydims = np.array([5]).astype(int)
    elif args.config == 'mc_maze_stochastic_infonce_alt_v2':
        config_file = 'config/config_mc_maze_stochastic_infonce_alt_v2.ini'
        ydims = np.array([5]).astype(int)
    elif args.config == 'mc_maze_deterministic_infonce_alt':
        config_file = 'config/config_mc_maze_deterministic_infonce_alt.ini'
        ydims = np.array([5]).astype(int)
    elif args.config == 'mc_maze_stochastic_infonce_alt':
        config_file = 'config/config_mc_maze_stochastic_infonce_alt.ini'
        ydims = np.array([5]).astype(int)
    elif args.config == 'mc_maze_stochastic_infonce_alt_v2_cond0':
        config_file = 'config/config_mc_maze_stochastic_infonce_alt_v2_cond0.ini'
        ydims = np.array([5,10,20,40]).astype(int)
    else:
        raise ValueError("{} has not been implemented!".format(args.config))

    cfg = myconf()
    cfg.read(config_file)

    RESULTS_FILENAME = cfg.get('User', 'RESULTS_FILENAME')
    saved_root = cfg.get('User', 'saved_root')
    if not os.path.exists(saved_root):
        os.mkdir(saved_root)

    # set hyper-parameters
    beta = cfg.getfloat('Hyperparameters', 'beta')
    beta1 = cfg.getfloat('Hyperparameters', 'beta1')
    beta2 = cfg.getfloat('Hyperparameters', 'beta2')
    xdim = cfg.getint('Hyperparameters', 'xdim')
    Ts = cfg.get('Hyperparameters', 'T')
    Ts = Ts.split(' ')
    Ts = [int(T) for T in Ts]
    hidden_dim = cfg.getint('Hyperparameters', 'hidden_dim')

    estimator_compress = cfg.get('Hyperparameters', 'estimator_compress')
    estimator_predictive = cfg.get('Hyperparameters', 'estimator_predictive')
    critic = cfg.get('Hyperparameters', 'critic')
    baseline = cfg.get('Hyperparameters', 'baseline')
    kernel = cfg.get('Hyperparameters', 'kernel')
    mi_params = {'estimator_compress': estimator_compress, "estimator_predictive": estimator_predictive,
                 "critic": critic,
                 "baseline": baseline}
    # critic_params = {"x_dim": T * ydim, "y_dim": T * ydim, "hidden_dim": hidden_dim}
    baseline_params = {"hidden_dim": hidden_dim}
    deterministic = cfg.getboolean('Hyperparameters', 'deterministic')
    # set training parameters
    do_vis_latent_trials = cfg.getboolean('Training', 'do_vis_latent_trials')
    batch_size = cfg.getint('Training', 'batch_size')
    num_epochs = cfg.getint('Training', 'num_epochs')
    num_early_stop = cfg.getint('Training', 'num_early_stop')
    num_vis = cfg.getint('Training', 'num_vis')
    do_dca_init = cfg.getboolean('Training', 'do_dca_init')
    device = cfg.get('Training', 'device')
    lr = cfg.getfloat('Training', 'lr')

    X_train = None; X_test = None; Y_train = None; Y_test = None

    if args.config.startswith("m1"):
        # M1 = data_util.load_sabes_data('/home/rui/Data/M1/indy_20160627_01.mat')
        M1 = data_util.load_sabes_data('data/M1/indy_20160627_01.mat')
        X, Y = M1['M1'], M1['cursor']
        good_ts = None
        standardize_Y = False
    if args.config.startswith('hc'):
        # HC = data_util.load_kording_paper_data('/home/rui/Data/HC/example_data_hc.pickle')
        HC = data_util.load_kording_paper_data('data/HC/example_data_hc.pickle')
        X, Y = HC['neural'], HC['loc']
        good_ts = 22000
        # good_ts = None
        standardize_Y = False
    if args.config == "temp_stochastic_infonce" or args.config == "temp_stochastic_infonce_alt"\
            or args.config == "temp_deterministic_infonce_alt":
        # weather = data_util.load_weather_data('/home/fan/Data/TEMP/temperature.csv')
        weather = data_util.load_weather_data('/home/rui/Data/TEMP/temperature.csv')
        X, Y = weather, weather
        good_ts = None
        standardize_Y = True
    if args.config == "ms_stochatic_infonce" or args.config == "ms_stochastic_infonce_alt"\
            or args.config == "ms_deterministic_infonce_alt":
        ms = data_util.load_accel_data('/home/rui/Data/motion_sense/A_DeviceMotion_data/std_6/sub_19.csv')
        X, Y = ms, ms
        good_ts = None
        standardize_Y = True
    if args.config.startswith("mc_maze"):  # only consider single condition for mc_maze.
        ### INITIALIZE DATA FOR THE FIRST TIME
        # from nlb_tools.nwb_interface import NWBDataset
        ## Load dataset
        ## Initial mc_maze data
        # # mc_maze = NWBDataset("/mnt/d/Data/neural_latents/mc_maze/000128/sub-Jenkins/", "*train", split_heldout=False)
        # mc_maze = NWBDataset("/home/rmmeng/data/neural/000128/sub-Jenkins/", "*train", split_heldout=False)
        
        # mc_maze.smooth_spk(50, name='smth_50')

        ## generate data for Cond0
        # conds = mc_maze.trial_info.set_index(['trial_type', 'trial_version']).index.unique().tolist()
        # cond = conds[0]
        ###spikes_rate for cond0
        # mask = np.all(mc_maze.trial_info[['trial_type', 'trial_version']] == cond, axis=1)
        # trial_data = mc_maze.make_trial_data(align_field='move_onset_time', align_range=(-50, 450), ignored_trials=(~mask))
        # t = np.arange(-50, 450, mc_maze.bin_width)
        # rates = []
        # vels = []
        # for _, trial in trial_data.groupby('trial_id'):
        #     trial_spike = trial['spikes'].to_numpy()
        #     trial_vel = trial['hand_vel'].to_numpy()
        #     rates.append(trial_spike)
        #     vels.append(trial_vel)
        # rates = np.stack(rates) # batch_size, time_stamps, neurons
        # vels = np.stack(vels) # batch_size, time_stamps, hand_vels(x, y)
        # import pickle
        # with open("/mnt/d/Data/neural_latents/mc_maze/mc_maze_nerual_vs_handvel_cond0.pickle", "wb") as f:
        #     pickle.dump({"rates": rates, "vels": vels}, f)

        # with open("/mnt/d/Data/neural_latents/mc_maze/mc_maze_nerual_vs_handvel_cond0.pickle", "rb") as f:
        #     data = pickle.load(f)
        #     rates, vels = data["rates"], data["vels"]


        ###spikes_smth_50 for cond0
        # mask = np.all(mc_maze.trial_info[['trial_type', 'trial_version']] == cond, axis=1)
        # trial_data = mc_maze.make_trial_data(align_field='move_onset_time', align_range=(-50, 450), ignored_trials=(~mask))
        # t = np.arange(-50, 450, mc_maze.bin_width)
        # rates = []
        # vels = []
        # for _, trial in trial_data.groupby('trial_id'):
        #     trial_spike = trial['spikes_smth_50'].to_numpy()
        #     trial_vel = trial['hand_vel'].to_numpy()
        #     rates.append(trial_spike)
        #     vels.append(trial_vel)
        # rates = np.stack(rates) # batch_size, time_stamps, neurons
        # vels = np.stack(vels) # batch_size, time_stamps, hand_vels(x, y)
        # import pickle
        # with open("/home/rmmeng/data/neural/mc_maze_nerual_spikes_smth_50_vs_handvel_cond0.pickle", "wb") as f:
        #     pickle.dump({"rates": rates, "vels": vels}, f)

        with open("/home/rmmeng/data/neural/mc_maze_nerual_spikes_smth_50_vs_handvel_cond0.pickle", "rb") as f:
            data = pickle.load(f)
            rates, vels = data["rates"], data["vels"]
        X, Y = rates, vels
        train_test_ratio = 0.8
        n = X.shape[0]
        n_train = int(n * train_test_ratio)
        X_train = X[:n_train]
        X_test = X[n_train:]
        Y_train = Y[:n_train]
        Y_test = Y[n_train:]
        good_ts = None
        standardize_Y = False
    
    # if args.config == "mc_maze_stochastic_infonce_alt_v2":
    #     # !pip install git+https://github.com/neurallatents/nlb_tools.git
    #     from nlb_tools.nwb_interface import NWBDataset
    #     ## Load dataset
    #     ## Initial mc_maze data
    #     # mc_maze = NWBDataset("/mnt/d/Data/neural_latents/mc_maze/000128/sub-Jenkins/", "*train", split_heldout=False)
    #     mc_maze = NWBDataset("/home/rmmeng/data/neural/000128/sub-Jenkins/", "*train", split_heldout=False)
        
    #     mc_maze.smooth_spk(50, name='smth_50')

        
    #     ## generate data for all
    #     # trial_data = mc_maze.make_trial_data(align_field='move_onset_time', align_range=(-50, 450))
    #     # t = np.arange(-50, 450, mc_maze.bin_width)
    #     # rates = []
    #     # vels = []
    #     # for _, trial in trial_data.groupby('trial_id'):
    #     #     trial_spike = trial['spikes'].to_numpy()
    #     #     trial_vel = trial['hand_vel'].to_numpy()
    #     #     rates.append(trial_spike)
    #     #     vels.append(trial_vel)
    #     # rates = np.stack(rates) # batch_size, time_stamps, neurons
    #     # vels = np.stack(vels) # batch_size, time_stamps, hand_vels(x, y)
    #     # import pickle
    #     # with open("/mnt/d/Data/neural_latents/mc_maze/mc_maze_nerual_vs_handvel.pickle", "wb") as f:
    #     #     pickle.dump({"rates": rates, "vels": vels}, f)
        
    #     # with open("/mnt/d/Data/neural_latents/mc_maze/mc_maze_nerual_vs_handvel.pickle", "rb") as f:
    #     #     data = pickle.load(f)
    #     #     rates, vels = data["rates"], data["vels"]
        
    #     ###spikes_smth_50 for cond0
    #     trial_data = mc_maze.make_trial_data(align_field='move_onset_time', align_range=(-50, 450))
    #     t = np.arange(-50, 450, mc_maze.bin_width)
    #     rates = []
    #     vels = []
    #     for _, trial in trial_data.groupby('trial_id'):
    #         trial_spike = trial['spikes_smth_50'].to_numpy()
    #         trial_vel = trial['hand_vel'].to_numpy()
    #         rates.append(trial_spike)
    #         vels.append(trial_vel)
    #     rates = np.stack(rates) # batch_size, time_stamps, neurons
    #     vels = np.stack(vels) # batch_size, time_stamps, hand_vels(x, y)
    #     import pickle
    #     with open("/home/rmmeng/data/neural/mc_maze_nerual_spikes_smth_50_vs_handvel.pickle", "wb") as f:
    #         pickle.dump({"rates": rates, "vels": vels}, f)

    #     # with open("/home/rmmeng/data/neural/mc_maze_nerual_spikes_smth_50_vs_handvel.pickle", "rb") as f:
    #     #     data = pickle.load(f)
    #     #     rates, vels = data["rates"], data["vels"]

        X, Y = rates, vels
        train_test_ratio = 0.8
        n = X.shape[0]
        n_train = int(n * train_test_ratio)
        X_train = X[:n_train]
        X_test = X[n_train:]
        Y_train = Y[:n_train]
        Y_test = Y[n_train:]

        good_ts = None
        standardize_Y = True

    T_pi_vals = np.array(Ts)
    offsets = np.array([5, 10, 15])

    win = 3
    n_init = 5

    # rewrite ydim
    # m1_ydims = np.array([5,10,20,30])
    # hc_ydims = np.array([5,10,15,25])
    # temp_ydims = np.array([3, 4, 5, 6])
    # ydims = np.array([5])
    if kernel == "Linear":
        Kernel = None
    elif kernel == "Polynomial":
        Kernel = Polynomial_expand
    else:
        raise ValueError("This kernel is not available.")

    if args.nonlinear_encoding:
        linear_encoding=False
    else:
        linear_encoding=True

    for ydim in ydims:
        regularzation_weight = 0
        result_r2, result_MI = run_analysis_cpic(X, Y, T_pi_vals, dim_vals=[ydim], offset_vals=offsets, decoding_window=win,
                          X_train=X_train, X_test=X_test, Y_train=Y_train, Y_test=Y_test,
                          n_init=n_init, verbose=True, Kernel=Kernel, xdim=xdim, beta=beta, beta1=beta1, beta2=beta2,
                          good_ts=good_ts, standardize_Y=standardize_Y, regularization_weight=regularzation_weight, linear_encoding=linear_encoding)

        saved_file = "result_dim{}.pkl".format(ydim) if linear_encoding else "result_dim{}_nonlinear.pkl".format(ydim)
        with open(saved_root + "/" + saved_file, "wb") as f:
            pickle.dump([result_r2, result_MI], f)
        # import pdb; pdb.set_trace()
