import glob
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas
from scipy.optimize import curve_fit
from scipy.stats import sem
from sklearn.metrics import r2_score


def func(x, a):
    return a * x


lred = (0.85, 0.5, 0.2)
lblue = (0.3, 0.3, 0.55)
lred2 = (1.0, 0.75, 0.45)
lblue2 = (0.55, 0.65, 0.9)
dred = (0.65, 0.3, 0.0)
dblue = (0.0, 0.0, 0.5)

parent_model_path = '/nemo/stp/lm/working/barryd/hpc/python/zf_reg/outputs/'
model_list = glob.glob(parent_model_path + os.sep + "simple_regression_multi_gpu_custom_augmentation_2023-06*")
model_path = model_list[int(sys.argv[1])]

datasets = (('Zebrafish_Test_Regression', 'Zebrafish_25C', 'Crick'),
            ('20232103 ZF 15 mins 28.5 C', '20232803 ZF 15 mins 25', 'Princeton'))

model_name = os.path.basename(model_path)
print('Working on results for model ' + model_name)
plot_path = parent_model_path + os.sep + model_name + os.sep + 'plots'
os.makedirs(plot_path)
training_log = glob.glob(parent_model_path + os.sep + model_name + os.sep + '*training.log')
trainingProgressData = pandas.read_csv(training_log[0])

plt.figure(figsize=(9.0, 3.0), dpi=200)
plt.plot(trainingProgressData['epoch'], trainingProgressData['loss'], linewidth=1.0, color=dblue,
         label='Training Loss')
plt.plot(trainingProgressData['epoch'], trainingProgressData['val_loss'], linewidth=1.0, color=dred,
         label='Validation Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
# plt.xlim(left=0, right=55)
# plt.ylim(top=60, bottom=0)
plt.legend(fontsize=8, markerscale=1.5)
plt.savefig(plot_path + os.sep + 'Training_Progress.png')
plt.close()

for wt_folder, mut_folder, data_label in datasets:
    print('Producing plots for ' + wt_folder + ' and ' + mut_folder)
    wt_data_file = glob.glob(
        parent_model_path + os.sep + model_name + os.sep + 'test_outputs' + os.sep + wt_folder + os.sep + '*_predictions.csv')
    wtData = pandas.read_csv(wt_data_file[0])
    mut_data_file = glob.glob(
        parent_model_path + os.sep + model_name + os.sep + 'test_outputs' + os.sep + mut_folder + os.sep + '*_predictions.csv')
    mutData = pandas.read_csv(mut_data_file[0])

    wtData = wtData[wtData['Label'] >= 4.5]
    mutData = mutData[mutData['Label'] >= 4.5]

    wt_linear_model = np.polyfit(wtData['Label'], wtData['Prediction'], 1)
    wt_linear_model_fn = np.poly1d(wt_linear_model)
    kimmel_wt = np.polynomial.polynomial.Polynomial([0, 1])
    wtpopt1, wtpcov1 = curve_fit(func, wtData['Label'], wtData['Prediction'])

    r2 = r2_score(wtData['Prediction'], func(wtData['Label'], wtpopt1))

    # print(wtpopt1, np.sqrt(np.diag(wtpcov1)))

    # wty1 = (wtpopt1 - np.sqrt(np.diag(wtpcov1))) * x_s
    # wty2 = (wtpopt1 + np.sqrt(np.diag(wtpcov1))) * x_s
    mut_linear_model = np.polyfit(mutData['Label'], mutData['Prediction'], 1)
    mut_linear_model_fn = np.poly1d(mut_linear_model)
    kimmel_mut = np.polynomial.polynomial.Polynomial([0, 0.805])
    mutpopt, mutpcov = curve_fit(func, mutData['Label'], mutData['Prediction'])
    x_s = np.arange(0, 53)

    r2 = r2_score(mutData['Prediction'], func(mutData['Label'], mutpopt))

    plt.figure(figsize=(5.0, 5.0), dpi=200)
    plt.plot(wtData['Label'], wtData['Prediction'], 'o', markersize=1, alpha=0.5, mfc=lblue, mec=lblue,
             label='28.5C')
    plt.plot(mutData['Label'], mutData['Prediction'], 'o', markersize=1, alpha=0.5, mfc=lred, mec=lred,
             label='25.0C')
    plt.plot(x_s, func(x_s, wtpopt1), linewidth=1.5, color=dblue, label='28.5C fit')
    # plt.fill_between(x_s, y1, y2, color="blue", alpha = 0.3)
    plt.plot(x_s, kimmel_wt(x_s), linewidth=1.5, linestyle='--', color=dblue, label='28.5C Kimmel')
    plt.plot(x_s, func(x_s, mutpopt), linewidth=1.5, color=dred, label='25.0C fit')
    plt.plot(x_s, kimmel_mut(x_s), linewidth=1.5, linestyle='--', color=dred, label='25.0C Kimmel')
    plt.xlabel("Actual HPF")
    plt.ylabel("Predicted HPF")
    plt.xlim(left=0, right=55)
    plt.ylim(top=60, bottom=0)
    plt.legend(fontsize=8, markerscale=1.5)
    plt.savefig(plot_path + os.sep + data_label + '_Prediction_Accuracy.png')
    plt.close()

    wterrs = wtData['Prediction'] - wtData['Label'] * wtpopt1
    print(np.mean(wterrs))
    print(sem(wterrs))
    print(np.std(wterrs))
    muterrs = mutData['Prediction'] - mutData['Label'] * mutpopt
    print(np.mean(muterrs))
    print(sem(muterrs))
    print(np.std(muterrs))
    wt_kim_errs = wtData['Prediction'] - wtData['Label']
    mut_kim_errs = mutData['Prediction'] - mutData['Label'] * 0.805

    errs = [wterrs, wt_kim_errs]

    plt.figure(figsize=(5.0, 5.0), dpi=200)
    plt.hist(errs, bins=50, range=[-40, 40], color=[dblue, lblue2], label=['Best Fit', 'Kimmel'], density=True)
    plt.xlabel("Prediction Error")
    plt.ylabel("Relative Frequency")
    plt.xlim(left=-40, right=40)
    plt.ylim(bottom=0, top=0.1)
    plt.legend(fontsize=8, markerscale=1.5)
    plt.savefig(plot_path + os.sep + data_label + '_WT_Prediction_Errors.png')
    plt.close()

    errs = [muterrs, mut_kim_errs]

    plt.figure(figsize=(5.0, 5.0), dpi=200)
    plt.hist(errs, bins=50, range=[-40, 40], color=[dred, lred2], label=['Best Fit', 'Kimmel'], density=True)
    plt.xlabel("Prediction Error")
    plt.ylabel("Relative Frequency")
    plt.xlim(left=-40, right=40)
    plt.ylim(bottom=0, top=0.1)
    plt.legend(fontsize=8, markerscale=1.5)
    plt.savefig(plot_path + os.sep + data_label + '_MUT_Prediction_Errors.png')
    plt.close()

    mMin = 1
    mMax = -1
    for i in range(10000):
        wtDataSubset = wtData.sample(200)
        wtpopt2, wtpcov2 = curve_fit(func, wtDataSubset['Label'], wtDataSubset['Prediction'])
        if wtpopt2 < mMin:
            mMin = wtpopt2
        if wtpopt2 > mMax:
            mMax = wtpopt2

    wty1 = mMin * x_s
    wty2 = mMax * x_s

    mMin = 1
    mMax = -1
    for i in range(10000):
        wtDataSubset = wtData.sample(50)
        wtpopt3, wtpcov3 = curve_fit(func, wtDataSubset['Label'], wtDataSubset['Prediction'])
        if wtpopt3 < mMin:
            mMin = wtpopt3
        if wtpopt3 > mMax:
            mMax = wtpopt3

    wty3 = mMin * x_s
    wty4 = mMax * x_s

    mMin = 1
    mMax = -1
    for i in range(10000):
        mutDataSubset = mutData.sample(200)
        mutpopt2, mutpcov2 = curve_fit(func, mutDataSubset['Label'], mutDataSubset['Prediction'])
        if mutpopt2 < mMin:
            mMin = mutpopt2
        if mutpopt2 > mMax:
            mMax = mutpopt2

    muty1 = mMin * x_s
    muty2 = mMax * x_s

    mMin = 1
    mMax = -1
    for i in range(10000):
        mutDataSubset = mutData.sample(50)
        mutpopt3, mutpcov3 = curve_fit(func, mutDataSubset['Label'], mutDataSubset['Prediction'])
        if mutpopt3 < mMin:
            mMin = mutpopt3
        if mutpopt3 > mMax:
            mMax = mutpopt3

    muty3 = mMin * x_s
    muty4 = mMax * x_s

    plt.figure(figsize=(5.0, 5.0), dpi=200)
    plt.fill_between(x_s, wty3, wty4, color=lblue2, label='28.5C Outer Confidence Interval')
    plt.fill_between(x_s, wty1, wty2, color=lblue, label='28.5C Inner Confidence Interval')
    plt.plot(x_s, func(x_s, wtpopt1), linewidth=1.5, color=dblue, label='28.5C fit')
    plt.plot(x_s, kimmel_wt(x_s), linewidth=1.5, linestyle='--', color=dblue, label='28.5C Kimmel')
    plt.fill_between(x_s, muty3, muty4, color=lred2, label='25.0C Outer Confidence Interval')
    plt.fill_between(x_s, muty1, muty2, color=lred, label='25.0C Inner Confidence Interval')
    plt.plot(x_s, kimmel_mut(x_s), linewidth=1.5, linestyle='--', color=dred, label='25.0C Kimmel')
    plt.plot(x_s, func(x_s, mutpopt), linewidth=1.5, color=dred, label='25.0C fit')

    plt.xlabel("Actual HPF")
    plt.ylabel("Predicted HPF")
    plt.xlim(left=0, right=55)
    plt.ylim(top=60, bottom=0)
    plt.legend(fontsize=8, markerscale=1.5, loc='upper left')
    plt.savefig(plot_path + os.sep + data_label + '_Confidence_Intervals.png')
    plt.close()
