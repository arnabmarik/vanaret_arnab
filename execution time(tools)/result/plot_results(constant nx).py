import numpy as np
import pandas as pd
import matplotlib.pylab as plt

counter0 = 1
for nx in [3, 4, 5, 6, 7]:
    fig2 = plt.figure(counter0)
    counter0 += 1
    fig2.suptitle('nx = ' + str(nx), fontsize=8)
    plt.tight_layout()
    result = pd.read_csv("results.csv")
    result = result[result["4.total iterations[MDF]"] < 20]
    print(result)
    counter1 = 0
    for d in [0.3, 0.4, 0.5, 0.6, 0.7]:
        counter1 += 1
        ax = fig2.add_subplot(3, 2, counter1)
        a_mdf = []
        a_idf = []
        for ny in [3, 4, 5, 6, 7]:
            d_idf_srt = result[(result["2.ny"] == ny) & (result["1.nx"] == nx) & (result["3.d"] == d)]["7.str_count[MDF]"].to_numpy()
            d_idf_aer = result[(result["2.ny"] == ny) & (result["1.nx"] == nx) & (result["3.d"] == d)]["8.aer_count[MDF]"].to_numpy()
            d_idf_pro = result[(result["2.ny"] == ny) & (result["1.nx"] == nx) & (result["3.d"] == d)]["9.pro_count[MDF]"].to_numpy()
            total_func_calls_idf = d_idf_srt + d_idf_aer + d_idf_pro
            # remove the outliers
            # d_idf = d_idf[d_idf != max(d_idf)]
            # d_idf = d_idf[d_idf != max(d_idf)]
            a_idf.append(np.median(total_func_calls_idf))
            ax.scatter((np.ones(np.size(total_func_calls_idf)) * ny).astype(int), total_func_calls_idf, marker='o', facecolor='blue',
                       edgecolor='black', linewidths=.2)

            d_mdf_str = result[(result["2.ny"] == ny) & (result["1.nx"] == nx) & (result["3.d"] == d)]["13.str_count[IDF_tol]"].to_numpy()
            d_mdf_aer = result[(result["2.ny"] == ny) & (result["1.nx"] == nx) & (result["3.d"] == d)]["14.aer_count[IDF_tol]"].to_numpy()
            d_mdf_pro = result[(result["2.ny"] == ny) & (result["1.nx"] == nx) & (result["3.d"] == d)]["15.pro_count[IDF_tol]"].to_numpy()
            total_func_calls_mdf = d_mdf_str + d_mdf_aer + d_mdf_pro
            # remove the outliers
            # d_mdf = d_mdf[d_mdf != max(d_mdf)]
            # d_mdf = d_mdf[d_mdf != max(d_mdf)]
            a_mdf.append(np.median(total_func_calls_mdf))
            ax.scatter((np.ones(np.size(total_func_calls_mdf)) * ny).astype(int), total_func_calls_mdf, marker='*', facecolor='red',
                       edgecolor='black', linewidths=.2)
        plt.xlabel('ny', fontsize=8)
        plt.ylabel('total_function_calls')
        ax.plot(range(3, 8), a_idf)
        ax.plot(range(3, 8), a_mdf)
        if d == 0.3:
            plt.gca().legend(('IDF', 'MDF'))
    plt.tight_layout()
plt.show()