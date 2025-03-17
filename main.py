import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import warnings

from NMF import nmf
import GNMFLIB
from clustering import k_means
import get_data
import meta_nmf as METANMF
import meta_gnmf as METAGNMF

warnings.filterwarnings("ignore")

def run_experiment(data_name, results_num, K, maxiter_outer, maxiter_inner, nb_tasks, task_size, params):
    """
    Runs NMF and GNMF experiments on a given dataset.

    Args:
        data_name (str): Name of the dataset.
        results_num (int): Number for results directory.
        K (list): List of rank values.
        maxiter_outer (int): Max iterations for outer loop.
        maxiter_inner (int): Max iterations for inner loop.
        nb_tasks (int): Number of tasks for meta methods.
        task_size (int): Size of each task for meta methods.
        params (dict): Dictionary of parameters for the dataset.
    """

    X, y, Xts, yts, c = get_data.load_data(data_name)
    print(f"Training data shape: {X.shape}, Test data shape: {Xts.shape}")
    print(f"NMI Score for K-means Original: {k_means(X, y, nb_clstr=c)}")

    m, n = Xts.shape[1], Xts.shape[0]

    results = {
        'NMF random': {'mean': [], 'median': [], 'std': []},
        'Meta-NMF random': {'mean': [], 'median': [], 'std': []},
        'GNMF': {'mean': [], 'median': [], 'std': []},
        'Meta-GNMF': {'mean': [], 'median': [], 'std': []},
        'GNMF NNDSVD': {'mean': [], 'median': [], 'std': []},
        'Meta-GNMF NNDSVD': {'mean': [], 'median': [], 'std': []}
    }

    for k in K:
        print(f"----> Rank: {k}")
        for algorithm in results:
            NMI = []
            for i in range(5):
                print(f"{i}-", end='')
                if algorithm == 'NMF random':
                    Wts = np.random.rand(m, k)
                    Hts = np.random.rand(k, n)
                    Hts, _, _ = nmf(Xts.T, k, thresh=0.001, L=maxiter_inner, W=None, H=None, init='random')
                elif algorithm == 'Meta-NMF random':
                    _, Wt = METANMF.fit(X, y, k=k, nb_tasks=nb_tasks, task_size=task_size,
                                         maxiter_outer=maxiter_outer, maxiter_inner=maxiter_inner)
                    Hts, _ = METANMF.predict(Xts, Wt, k=k, maxiter=maxiter_inner)
                elif algorithm == 'GNMF':
                    Wts = np.random.rand(m, k)
                    Hts = np.random.rand(k, n)
                    Wts, Hts, _ = GNMFLIB.gnmf(Xts.T, Wts, Hts, n_components=k, max_iter=maxiter_inner)
                elif algorithm == 'Meta-GNMF':
                    _, Wt = METAGNMF.fit(X, y, k=k, nb_tasks=nb_tasks, task_size=task_size,
                                          maxiter_outer=maxiter_outer, maxiter_inner=maxiter_inner)
                    Hts, _ = METAGNMF.predict(Xts, Wt, k=k, maxiter=maxiter_inner)
                    Hts = np.asarray(Hts)
                elif algorithm == 'GNMF NNDSVD':
                    Wts = np.random.rand(m, k)
                    Hts = np.random.rand(k, n)
                    Wts, Hts, _ = GNMFLIB.gnmf(Xts.T, Wts, Hts, nndsvd_init=True, n_components=k, max_iter=maxiter_inner)
                elif algorithm == 'Meta-GNMF NNDSVD':
                    _, Wt = METAGNMF.fit(X, y, nndsvd_init=True, k=k, nb_tasks=nb_tasks, task_size=task_size,
                                          maxiter_outer=maxiter_outer, maxiter_inner=maxiter_inner)
                    Hts, _ = METAGNMF.predict(Xts, Wt, k=k, maxiter=maxiter_inner)
                    Hts = np.asarray(Hts)

                nmi = k_means(Hts.T, yts, nb_clstr=c)
                NMI.append(nmi)

            results[algorithm]['mean'].append(np.mean(NMI))
            results[algorithm]['median'].append(np.median(NMI))
            results[algorithm]['std'].append(np.std(NMI))
            print(f"\n-> {algorithm}: {np.mean(NMI):.2f}, {np.median(NMI):.2f}, {np.std(NMI):.2f}")

    save_results(results, K, data_name, results_num, params)

def save_results(results, K, data_name, results_num, params):
    """Saves results to CSV and generates plots."""

    for stat in ['mean', 'median', 'std']:
        data = {alg: results[alg][stat] for alg in results}
        df = pd.DataFrame(data, index=K)
        print(df)
        df.plot(title=f"{stat.capitalize()} NMI")

        filename = f"Results_{results_num}/{data_name.capitalize()}/{data_name}_{'_'.join(f'{k}{v}' for k, v in params.items())}_iter{maxiter_outer}_{maxiter_inner}_{nb_tasks}_{task_size}_N00_{stat}.png"
        csv_filename = filename.replace(".png", ".csv")

        plt.savefig(filename)
        df.to_csv(csv_filename)
        plt.close()

# Main execution
data_name = 'blobs'
results_num = 3
K = [2, 3, 5, 10, 14, 18]
maxiter_outer = 50
maxiter_inner = 100
nb_tasks = 5
task_size = 50

params = {
    'samples': 1000,
    'dims': 50,
    'c': 5,
    'noise': 0.1
}

if data_name == 'faces':
    params = {'noise':0.6}
elif data_name == 'digits':
    params = {'noise':0.4}
elif data_name == 'fashion_mnist':
    params = {'size':0.10, 'noise':0.0}
elif data_name == 'coil20':
    params = {'noise':0.0}

run_experiment(data_name, results_num, K, maxiter_outer, maxiter_inner, nb_tasks, task_size, params)