import pickle
import numpy as np
import matplotlib.pyplot as plt
import argparse


def summary_statistics(x, axis=0):
    return np.mean(x, axis=axis), np.std(x, axis=axis)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="m1", help="dataset")
    args = parser.parse_args()
    dataset = args.dataset

    m1_dims = np.array([5])
    hc_dims = np.array([5])
    temp_dims = np.array([5])
    ms_dims = np.array([5])
    mc_maze_dims = np.array([5, 10, 20, 40])

    # for dim in m1_dims:
    #     with open("res/m1_stochastic_infonce_alt/result_dim{}_standard.pkl".format(dim), "rb") as f:
    #         m1_sto_infonce_R2s = pickle.load(f)
    #     m1_sto_infonce_R2s = np.squeeze(m1_sto_infonce_R2s)
    #     print("m1(stochastic), dim:{}, R2s:{}".format(dim, m1_sto_infonce_R2s))
    #     with open("res/m1_deterministic_infonce_alt/result_dim{}_standard.pkl".format(dim), "rb") as f:
    #         m1_det_infonce_R2s = pickle.load(f)
    #     m1_det_infonce_R2s = np.squeeze(m1_det_infonce_R2s)
    #     m1_det_infonce_R2_mean, m1_det_infonce_R2_std = summary_statistics(m1_det_infonce_R2s, axis=0)
    #     print("m1(deterministic), dim:{}, mean: {}, std: {}".format(dim, m1_det_infonce_R2_mean, m1_det_infonce_R2_std))
    # for dim in hc_dims:
    #     with open("res/hc_stochastic_infonce_alt/result_dim{}_standard.pkl".format(dim), "rb") as f:
    #         hc_sto_infonce_R2s = pickle.load(f)
    #     hc_sto_infonce_R2s = np.squeeze(hc_sto_infonce_R2s)
    #     print("hc(stochastic), dim:{}, R2s:{}".format(dim, hc_sto_infonce_R2s))
    #     with open("res/hc_deterministic_infonce_alt/result_dim{}_standard.pkl".format(dim), "rb") as f:
    #         hc_det_infonce_R2s = pickle.load(f)
    #     hc_det_infonce_R2s = np.squeeze(hc_det_infonce_R2s)
    #     hc_det_infonce_R2_mean, hc_det_infonce_R2_std = summary_statistics(hc_det_infonce_R2s, axis=0)
    #     print("hc(deterministic), dim:{}, mean: {}, std: {}".format(dim, hc_det_infonce_R2_mean, hc_det_infonce_R2_std))
    # for dim in temp_dims:
    #     with open("res/temp_stochastic_infonce/result_dim{}_standard.pkl".format(dim), "rb") as f:
    #         temp_sto_infonce_R2s = pickle.load(f)
    #     temp_sto_infonce_R2s = np.squeeze(temp_sto_infonce_R2s)
    #     temp_sto_infonce_R2_mean, temp_sto_infonce_R2_std = summary_statistics(temp_sto_infonce_R2s, axis=0)
    #     print("temp(stochastic), dim:{}, mean: {}, std: {}".format(dim, temp_sto_infonce_R2_mean, temp_sto_infonce_R2_std))
    # for dim in ms_dims:
    #     with open("res/ms_stochastic_infonce_alt/result_dim{}_standard.pkl".format(dim), "rb") as f:
    #         ms_sto_infonce_R2s = pickle.load(f)
    #     ms_sto_infonce_R2s = np.squeeze(ms_sto_infonce_R2s)
    #     print("ms(stochastic), dim:{}, R2s:{}".format(dim, ms_sto_infonce_R2s))

    # Experiments for varying window sizes:
    if dataset == "m1":
        for dim in m1_dims:
            with open("res/m1_stochastic_infonce_alt_v2/result_dim{}_standard.pkl".format(dim), "rb") as f:
                m1_sto_infonce_R2s, m1_sto_infonce_MI = pickle.load(f)
            m1_sto_infonce_R2s = np.squeeze(m1_sto_infonce_R2s)
            print("m1(stochastic), dim:{}, R2s:{}".format(dim, m1_sto_infonce_R2s))
        # plot trials 
        lags = [5, 10, 15]
        fig = plt.figure()
        # Plot each trial with a label
        for i in range(m1_sto_infonce_R2s.shape[0]):
            plt.plot(m1_sto_infonce_R2s[i], label=f'Lag {lags[i]}')
        # plt.xticks([0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 6])
        # plt.xlabel("Window size")
        plt.xticks([0, 1, 2, 3, 4, 5], [50, 100, 150, 200, 250, 300])
        plt.xlabel("Milliseconds")
        plt.ylabel("R2")
        plt.title("R2 scores for M1 datasets with varying window size")
        # Add legend to show trial labels
        plt.legend()
        plt.savefig("fig/varying_window_sizes/M1_R2_varying_WS.png")
    if dataset == "hc":
        for dim in hc_dims:
            with open("res/hc_stochastic_infonce_alt_v2/result_dim{}_standard.pkl".format(dim), "rb") as f:
                hc_sto_infonce_R2s, hc_sto_infonce_MI = pickle.load(f)
            hc_sto_infonce_R2s = np.squeeze(hc_sto_infonce_R2s)
            print("hc(stochastic), dim:{}, R2s:{}".format(dim, hc_sto_infonce_R2s))
        # plot trials 
        lags = [5, 10, 15]
        fig = plt.figure()
        # Plot each trial with a label
        for i in range(hc_sto_infonce_R2s.shape[0]):
            plt.plot(hc_sto_infonce_R2s[i], label=f'Lag {lags[i]}')
        # plt.xticks([0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 6])
        # plt.xlabel("Window size")
        plt.xticks([0, 1, 2, 3, 4, 5], [50, 100, 150, 200, 250, 300])
        plt.xlabel("Milliseconds")
        plt.ylabel("R2")
        plt.title("R2 scores for HC datasets with varying window size")
        # Add legend to show trial labels
        plt.legend()
        plt.savefig("fig/varying_window_sizes/HC_R2_varying_WS.png")
    if dataset == "mc_maze":
        mc_maze_sto_infonce_R2s_list = []
        for dim in mc_maze_dims:
            with open("res/mc_maze_stochastic_infonce_alt_v2_cond0/result_dim{}_standard.pkl".format(dim), "rb") as f:
                mc_maze_sto_infonce_R2s, mc_maze_sto_infonce_MI = pickle.load(f)
            mc_maze_sto_infonce_R2s_list.append(np.squeeze(mc_maze_sto_infonce_R2s))
            print("mc_maze(stochastic), dim:{}, R2s:{}".format(dim, mc_maze_sto_infonce_R2s_list[-1]))
        lags = [5, 10, 15]
        for i in range(len(mc_maze_sto_infonce_R2s_list)):
            mc_maze_sto_infonce_R2s = mc_maze_sto_infonce_R2s_list[i]    
            fig = plt.figure()
            # Plot each trial with a label
            for j in range(mc_maze_sto_infonce_R2s.shape[0]):
                plt.plot(mc_maze_sto_infonce_R2s[j], label=f'Lag {lags[j]}')
            # plt.xticks([0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 6])
            # plt.xlabel("Window size")
            plt.xticks([0, 1, 2, 3, 4, 5], [50, 100, 150, 200, 250, 300])
            plt.xlabel("Milliseconds")
            plt.ylabel("R2")
            plt.title("R2 scores for MC-MAZE datasets with varying window size")
            # Add legend to show trial labels
            plt.legend()
            plt.savefig("fig/varying_window_sizes/MC-MAZE_cond0_R2_varying_WS_dim{}.png".format(mc_maze_dims[i]))
