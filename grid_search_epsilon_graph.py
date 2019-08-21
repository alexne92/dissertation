import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("epsilon_grid_search/psnr_ssim_results.csv")
x = list(reversed([str(i) for i in df[df.columns[0]]]))
title = "Effect of the parameter on epsilon"
title_train = " on the training set"
title_test = " on the test set"
names = ["psnr_train.png", "ssim_train.png", "psnr_test.png", "ssim_test.png"]
for i in range(1,df.shape[1]):
    y = list(reversed([j for j in df[df.columns[i]]]))
    plt.bar(x, y, width = 0.5)
    plt.xlabel("Parameters")
    if i < 3:
        plt.title(title + title_train)
    else:
        plt.title(title + title_test)
    if i%2 ==0:
        plt.ylabel("PSNR")
    else:
        plt.ylabel("SSIM")
    plt.savefig("epsilon_grid_search/" + names[i-1])
    plt.clf()
