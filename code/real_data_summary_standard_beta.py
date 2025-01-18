import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib


def plot_CPIC(I_compress, I_predictive, save_file="fig/m1_CPIC_IC.png", xticks=None, yticks=None):
    fig = plt.figure()
    plt.plot(I_compress, I_predictive)
    plt.xlabel("Compression Complexity I(X(T),Y(T))")
    plt.ylabel("Predictive Information I(Y(-T),Y(T))")
    if xticks:
        plt.xticks(xticks)
    if yticks:
        plt.yticks(yticks)
    fig.savefig(save_file)
    plt.show()


if __name__ == "__main__":

    # m1_dims = np.array([5])
    # hc_dims = np.array([5])
    mc_dims = np.array([5])
    betas = [1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 1e-1, 2e-1, 5e-1, 1, 2]

    # for dim in m1_dims:
    #     I_compress = []
    #     I_predictive = []
    #     for beta in betas:
    #         with open("res/m1_stochastic_infonce_beta/result_dim{}_standard_beta_{}.pkl".format(dim, beta), "rb") as f:
    #             m1_sto_infonce_R2s, m1_sto_infonce_MIs = pickle.load(f)
    #         I_compress.append(m1_sto_infonce_MIs[0,0,0])
    #         I_predictive.append(m1_sto_infonce_MIs[0,0,1])
    #     plot_CPIC(I_compress, I_predictive, save_file="fig/m1_dim{}_CPIC_IC.png".format(dim), xticks=[0, 30, 60, 90], yticks=[0, 0.6, 1.2, 1.8])

    # for dim in hc_dims:
    #     I_compress = []
    #     I_predictive = []
    #     for beta in betas:
    #         with open("res/hc_stochastic_infonce_beta/result_dim{}_standard_beta_{}.pkl".format(dim, beta), "rb") as f:
    #             hc_sto_infonce_R2s, hc_sto_infonce_MIs = pickle.load(f)
    #         I_compress.append(hc_sto_infonce_MIs[0, 0, 0])
    #         I_predictive.append(hc_sto_infonce_MIs[0, 0, 1])
    #     plot_CPIC(I_compress, I_predictive, save_file="fig/hc_dim{}_CPIC_IC.png".format(dim), xticks=[0, 167, 334, 501], yticks=[0, 1, 2, 3])

    # for dim in mc_dims:
    #     I_compress = []
    #     I_predictive = []
    #     for beta in betas:
    #         with open("res/mc_maze_stochastic_infonce_alt_v2_cond0/result_dim{}_standard_beta_{}.pkl".format(dim, beta), "rb") as f:
    #             hc_sto_infonce_R2s, hc_sto_infonce_MIs = pickle.load(f)
    #         I_compress.append(hc_sto_infonce_MIs[0, 0, 0])
    #         I_predictive.append(hc_sto_infonce_MIs[0, 0, 1])
    #     plot_CPIC(I_compress, I_predictive, save_file="fig/mc_dim{}_CPIC_IC.png".format(dim), )

    import pdb; pdb.set_trace()

