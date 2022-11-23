import numpy as np
from natsort import natsorted
from glob import glob
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams.update({'font.size': 32})


GREY = "#d1cfcf"
SOFTRED = '#F47174'
DARKGREY = "#6d6d6d"
SOFTBLUE= "#93CAED"

alpha = 0.5

mpl.rcParams["axes.prop_cycle"] = mpl.cycler(
    color=["#4da9cf", "#153565", "#e9952b", "#ce1a0a", "#864a83", "#51968f"]
)


GREY = "#d1cfcf"
SOFTRED = '#F47174'
DARKGREY = "#6d6d6d"
SOFTBLUE= "#93CAED"


SOFTCOLORS = [SOFTBLUE,SOFTRED]

mrkcyc = ['o', "s", "^", '*', "P", 'X', '']
markersize = 3
ha2kcal = 627.5

#set global font size



def get_crypto(file, python_pred=False):
    if python_pred:

        curr_res = open(file, 'r')
        pytpred = float(curr_res.readlines()[-1].strip())
        curr_res.close()
        curr_res = open(file, 'r')
        for line in curr_res:
            line = line.split()
            if line[0] == "CRYPTO_PRED":
                crypto_pred = float(line[-1])
            if line[0] == "Time":
                time = float(line[2])
            if line[0] == "Global":
                traffic = float(line[4])


        
        return crypto_pred,pytpred, time, traffic
    

    else:
        curr_res = open(file, 'r')
        for line in curr_res:
            line = line.split()
            if line[0] == "CRYPTO_PRED":
                crypto_pred = float(line[-1])
            if line[0] == "Time":
                time = float(line[2])
            if line[0] == "Global":
                traffic = float(line[4])
        return crypto_pred, time, traffic


def get_crypto_lc(reptype, A, result_path):

    all_n_lc = np.array([32,    64,   128,   256,   512,
                    1024,  2048,  4096,  8192,  16384])
    n_lc = []

    all_results = []
    for n in all_n_lc:
        try:
            
            results = natsorted(
                glob("{}/{}/{}/out_{}_*".format(result_path, A, reptype, n)))
                
            crpto = []
            for res in results:
                crpto.append(get_crypto(res))

            crpto = np.array(crpto)
            crpto_prds, crpto_times, crpto_traffic = crpto[:, 0], crpto[:, 1], crpto[:, 2]
            all_results.append([crpto_prds, crpto_times, crpto_traffic])
            n_lc.append(n)

        except Exception as e:
            print(e)

    n_lc, all_results = np.array(n_lc), np.array(all_results)
    return n_lc, all_results


def get_crypto_single(result_path, python_pred=False):
    if python_pred:
        results = natsorted(glob("{}/out_*".format(result_path)))
        n_rep = np.array([  int(n.split("_")[-1]) for n in results ])
        crpto = []
        for res in results:
            crpto.append(get_crypto(res, python_pred=True))

        crpto = np.array(crpto)
        crpto_prds,pyt_pred, crpto_times, crpto_traffic = crpto[:, 0], crpto[:, 1], crpto[:, 2],crpto[:, 3]
        all_results = np.array([crpto_prds,pyt_pred, crpto_times, crpto_traffic])
        return all_results, n_rep

    else:
        results = natsorted(glob("{}/out_*".format(result_path)))
        n_rep = np.array([  int(n.split("_")[-1]) for n in results ])
        crpto = []
        for res in results:
            crpto.append(get_crypto(res))

        crpto = np.array(crpto)
        crpto_prds, crpto_times, crpto_traffic = crpto[:, 0], crpto[:, 1], crpto[:, 2]
        all_results = np.array([crpto_prds, crpto_times, crpto_traffic])
        return all_results, n_rep        



def extract_crpto(rep,A, result_path, mean=True):
    """crypto_pred
    extract all results
    """

    num_devs = []
    times = []
    traffics = []
    python_input1 = natsorted(
        glob("./gen_input/{}/python_input/REP.npz".format(rep)))

    y_test = np.load(python_input1[0], allow_pickle=True)["y_test"]


    python_input2 = natsorted(
        glob("./gen_input/{}/python_input/RESULTS.npz".format(rep)))

    n_lc, crypto_pred = get_crypto_lc(rep, A, result_path)
    crypto_pred_n_max = crypto_pred[:, 0].shape[0]
    python_pred = np.load(python_input2[0], allow_pickle=True)[
        "predictions"][:crypto_pred_n_max]

    cryto_lc = np.array([np.mean(np.abs(y_test - cp)) for cp in crypto_pred[:, 0]])
    python_lc = np.load(python_input2[0], allow_pickle=True)["lrncrv"]
    num_devs.append(python_pred - crypto_pred[:, 0])
    times.append(crypto_pred[:, 1])
    traffics.append(crypto_pred[:, 2])

    num_devs, times, traffics = np.array(
        num_devs), np.array(times), np.array(traffics)

    if mean:
        return n_lc, np.std(np.abs(num_devs), axis=2)[0], np.mean(np.abs(num_devs), axis=2)[0], np.mean(times, axis=2)[0], np.mean(traffics, axis=2)[0], crypto_pred[:, 0], cryto_lc, python_lc
    else:
        return n_lc, np.abs(num_devs)[0], times[0], traffics[0], crypto_pred[:, 0], cryto_lc, python_lc


def fix_axes(ax):
    plt.setp(ax.spines.values(), linewidth=1.5)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    # ax.spines["left"].set_edgecolor(GREY)
    # ax.spines["bottom"].set_edgecolor(GREY)
    ax.tick_params(length=5, color=DARKGREY, width=1.5, labelcolor="black")
    ax.tick_params(which="minor", length=5, color=GREY, width=1.5, labelcolor="white")
    for loc, spine in ax.spines.items():
        if loc in ["left", "bottom"]:
            spine.set_position(("outward", 0))
    ax.set_zorder(-100)
