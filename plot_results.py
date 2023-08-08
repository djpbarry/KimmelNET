import glob
import os
import shutil
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

model_path = 'KimmelNet_Model/published_model_multi_gpu_custom_augmentation_trained_model'

suffix = sys.argv[3]

datasets = (('Zebrafish_Test_Regression', 'Zebrafish_25C', 'Crick'),
            ('Zebrafish_Test_Princeton_Regression', '20232803 ZF 15 mins 25', 'Princeton'))

model_name = os.path.basename(model_path)
print('Working on results for model ' + model_name)
plot_path = parent_model_path + os.sep + model_name + os.sep + 'plots'

if os.path.exists(plot_path):
    shutil.rmtree(plot_path)

os.makedirs(plot_path)
training_log = glob.glob(parent_model_path + os.sep + model_name + os.sep + '*training.log')
trainingProgressData = pandas.read_csv(training_log[0])

plt.figure(figsize=(5.0, 5.0), dpi=300)
plt.plot(trainingProgressData['epoch'], trainingProgressData['loss'], linewidth=1.0, color=dblue,
         label='Training Loss')
plt.plot(trainingProgressData['epoch'], trainingProgressData['val_loss'], linewidth=1.0, color=dred,
         label='Validation Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss (Mean Squared Error)")
# plt.xlim(left=0, right=55)
# plt.ylim(top=60, bottom=0)
plt.legend(fontsize=8, markerscale=1.5)
plt.savefig(plot_path + os.sep + 'Training_Progress_' + suffix + '.png')
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

    wt_r2 = r2_score(wtData['Prediction'], func(wtData['Label'], wtpopt1))

    # print(wtpopt1, np.sqrt(np.diag(wtpcov1)))

    # wty1 = (wtpopt1 - np.sqrt(np.diag(wtpcov1))) * x_s
    # wty2 = (wtpopt1 + np.sqrt(np.diag(wtpcov1))) * x_s
    mut_linear_model = np.polyfit(mutData['Label'], mutData['Prediction'], 1)
    mut_linear_model_fn = np.poly1d(mut_linear_model)
    kimmel_mut = np.polynomial.polynomial.Polynomial([0, 0.805])
    mutpopt1, mutpcov1 = curve_fit(func, mutData['Label'], mutData['Prediction'])
    x_s = np.arange(0, 53)

    mut_r2 = r2_score(mutData['Prediction'], func(mutData['Label'], mutpopt1))

    plt.figure(figsize=(6.0, 5.0), dpi=300)
    plt.plot(wtData['Label'], wtData['Prediction'], 'o', markersize=1, alpha=0.5, mfc=lblue, mec=lblue,
             label='28.5C')
    plt.plot(mutData['Label'], mutData['Prediction'], 'o', markersize=1, alpha=0.5, mfc=lred, mec=lred,
             label='25.0C')
    plt.plot(x_s, func(x_s, wtpopt1), linewidth=1.5, color=dblue, label='28.5C fit')
    # plt.fill_between(x_s, y1, y2, color="blue", alpha = 0.3)
    plt.plot(x_s, kimmel_wt(x_s), linewidth=1.5, linestyle='--', color=dblue, label='28.5C Kimmel')
    plt.plot(x_s, func(x_s, mutpopt1), linewidth=1.5, color=dred, label='25.0C fit')
    plt.plot(x_s, kimmel_mut(x_s), linewidth=1.5, linestyle='--', color=dred, label='25.0C Kimmel')
    plt.xlabel("Actual HPF")
    plt.ylabel("Predicted HPF")
    plt.xlim(left=0, right=55)
    plt.ylim(top=60, bottom=0)
    plt.legend(fontsize=8, markerscale=1.5)
    plt.savefig(plot_path + os.sep + data_label + '_Prediction_Accuracy_' + suffix + '.png')
    plt.close()

    wterrs = wtData['Prediction'] - wtData['Label'] * wtpopt1
    muterrs = mutData['Prediction'] - mutData['Label'] * mutpopt1
    wt_kim_errs = wtData['Prediction'] - wtData['Label']
    mut_kim_errs = mutData['Prediction'] - mutData['Label'] * 0.805

    errs = [wterrs, wt_kim_errs]

    plt.figure(figsize=(6.0, 5.0), dpi=300)
    plt.hist(errs, bins=50, range=[-40, 40], color=[dblue, lblue2], label=['Best Fit', 'Kimmel'], density=True)
    plt.xlabel("Prediction Error")
    plt.ylabel("Relative Frequency")
    plt.xlim(left=-30, right=30)
    plt.ylim(bottom=0, top=0.25)
    plt.legend(fontsize=8, markerscale=1.5)
    plt.savefig(plot_path + os.sep + data_label + '_WT_Prediction_Errors_' + suffix + '.png')
    plt.close()

    errs = [muterrs, mut_kim_errs]

    plt.figure(figsize=(6.0, 5.0), dpi=300)
    plt.hist(errs, bins=50, range=[-40, 40], color=[dred, lred2], label=['Best Fit', 'Kimmel'], density=True)
    plt.xlabel("Prediction Error")
    plt.ylabel("Relative Frequency")
    plt.xlim(left=-30, right=30)
    plt.ylim(bottom=0, top=0.25)
    plt.legend(fontsize=8, markerscale=1.5)
    plt.savefig(plot_path + os.sep + data_label + '_MUT_Prediction_Errors_' + suffix + '.png')
    plt.close()

    mMin = 1
    mMax = -1
    for i in range(10000):
        wtDataSubset = wtData.sample(100)
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
        mutDataSubset = mutData.sample(100)
        mutpopt3, mutpcov3 = curve_fit(func, mutDataSubset['Label'], mutDataSubset['Prediction'])
        if mutpopt3 < mMin:
            mMin = mutpopt3
        if mutpopt3 > mMax:
            mMax = mutpopt3

    muty3 = mMin * x_s
    muty4 = mMax * x_s

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
        mutDataSubset = mutData.sample(200)
        mutpopt2, mutpcov2 = curve_fit(func, mutDataSubset['Label'], mutDataSubset['Prediction'])
        if mutpopt2 < mMin:
            mMin = mutpopt2
        if mutpopt2 > mMax:
            mMax = mutpopt2

    muty1 = mMin * x_s
    muty2 = mMax * x_s

    plt.figure(figsize=(6.0, 5.0), dpi=300)
    plt.fill_between(x_s, muty3, muty4, color=lred2, label='25.0C Confidence Interval (n=100)')
    plt.fill_between(x_s, wty3, wty4, color=lblue2, label='28.5C Confidence Interval (n=100)')
    plt.fill_between(x_s, muty1, muty2, color=lred, label='25.0C Confidence Interval (n=200)')
    plt.fill_between(x_s, wty1, wty2, color=lblue, label='28.5C Confidence Interval (n=200)')
    plt.plot(x_s, func(x_s, wtpopt1), linewidth=1.5, color=dblue, label='28.5C fit')
    plt.plot(x_s, kimmel_wt(x_s), linewidth=1.5, linestyle='--', color=dblue, label='28.5C Kimmel')
    plt.plot(x_s, kimmel_mut(x_s), linewidth=1.5, linestyle='--', color=dred, label='25.0C Kimmel')
    plt.plot(x_s, func(x_s, mutpopt1), linewidth=1.5, color=dred, label='25.0C fit')

    plt.xlabel("Actual HPF")
    plt.ylabel("Predicted HPF")
    plt.xlim(left=0, right=55)
    plt.ylim(top=60, bottom=0)
    plt.legend(fontsize=8, markerscale=1.5, loc='upper left')
    plt.savefig(plot_path + os.sep + data_label + '_Confidence_Intervals_' + suffix + '.png')
    plt.close()

    with open(plot_path + os.sep + 'errors_stats.txt', 'a') as fh:
        fh.write('\n\nDataset: ' + wt_folder)
        fh.write('\nMean: ' + str(np.mean(wterrs)))
        fh.write('\nSEM: ' + str(sem(wterrs)))
        fh.write('\nSD: ' + str(np.std(wterrs)))
        fh.write('\nm: ' + str(wtpopt1))
        fh.write('\nR^2: ' + str(wt_r2))
        fh.write('\n\nDataset: ' + mut_folder)
        fh.write('\nMean: ' + str(np.mean(muterrs)))
        fh.write('\nSEM: ' + str(sem(muterrs)))
        fh.write('\nSD: ' + str(np.std(muterrs)))
        fh.write('\nm: ' + str(mutpopt1))
        fh.write('\nR^2: ' + str(mut_r2))
