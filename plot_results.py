import matplotlib.pyplot as plt
import numpy as np
import pandas
from scipy.optimize import curve_fit


def func(x, a):
    return a * x


lred = (0.85, 0.5, 0.2)
lblue = (0.3, 0.3, 0.55)
lred2 = (1.0, 0.65, 0.35)
lblue2 = (0.45, 0.45, 0.7)
dred = (0.65, 0.3, 0.0)
dblue = (0.0, 0.0, 0.5)

trainingProgressData = pandas.read_csv('Z:/working/barryd/hpc/python/zf_reg/outputs'
                                       '/simple_regression_multi_gpu_added_augmentation_2022-06-28-13-45-31'
                                       '/simple_regression_multi_gpu_added_augmentation_training.log')
wtData = pandas.read_csv('Z:/working/barryd/hpc/python/zf_reg/outputs'
                         '/zf_regression_test_multi_gpu_added_augmentation_2022-06-29-09-01-19'
                         '/zf_regression_test_multi_gpu_added_augmentation_predictions.csv')
mutData = pandas.read_csv('Z:/working/barryd/hpc/python/zf_reg/outputs'
                          '/zf_regression_test_25C_multi_gpu_added_augmentation_2022-06-29-10-11-43'
                          '/zf_regression_test_25C_multi_gpu_added_augmentation_predictions.csv')

plt.figure(figsize=(6.4, 4.8), dpi=200)
plt.plot(trainingProgressData['epoch'], trainingProgressData['loss'], linewidth=1.0, color=dblue, label='Training Loss')
plt.plot(trainingProgressData['epoch'], trainingProgressData['val_loss'], linewidth=1.0, color=dred, label='Validation Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
#plt.xlim(left=0, right=55)
#plt.ylim(top=60, bottom=0)
plt.legend(fontsize=8, markerscale=1.5)
plt.show()

wt_linear_model = np.polyfit(wtData['Label'], wtData['Prediction'], 1)
wt_linear_model_fn = np.poly1d(wt_linear_model)
kimmel_wt = np.polynomial.polynomial.Polynomial([0, 1])
wtpopt1, wtpcov1 = curve_fit(func, wtData['Label'], wtData['Prediction'])

# print(wtpopt1, np.sqrt(np.diag(wtpcov1)))

# wty1 = (wtpopt1 - np.sqrt(np.diag(wtpcov1))) * x_s
# wty2 = (wtpopt1 + np.sqrt(np.diag(wtpcov1))) * x_s
mut_linear_model = np.polyfit(mutData['Label'], mutData['Prediction'], 1)
mut_linear_model_fn = np.poly1d(mut_linear_model)
kimmel_mut = np.polynomial.polynomial.Polynomial([0, 0.805])
mutpopt, mutpcov = curve_fit(func, mutData['Label'], mutData['Prediction'])
x_s = np.arange(0, 53)

plt.figure(figsize=(4.5, 4.5), dpi=200)
plt.plot(wtData['Label'], wtData['Prediction'], 'o', markersize=1, alpha=0.5, mfc=lblue, mec=lblue, label='28.5C')
plt.plot(mutData['Label'], mutData['Prediction'], 'o', markersize=1, alpha=0.5, mfc=lred, mec=lred, label='25.0C')
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
plt.show()

wterrs = wtData['Prediction'] - wtData['Label']
muterrs = mutData['Prediction'] - mutData['Label'] * mutpopt

errs = [wterrs, muterrs]

plt.figure(figsize=(5.5, 4.5), dpi=200)
plt.hist(errs, bins=50, color=[dblue, dred], label=['28.5C', '25.0C'], density=True)
plt.xlabel("Prediction Error")
plt.ylabel("Relative Frequency")
plt.xlim(left=-20, right=20)
plt.ylim(bottom=0, top=0.25)
plt.legend(fontsize=8, markerscale=1.5)
plt.show()

plt.figure(figsize=(5.5, 4.5), dpi=200)
plt.hist(wterrs, bins=100, color=dblue, density=True, alpha=0.7)
plt.xlabel("Prediction Error")
plt.ylabel("Frequency")
plt.xlim(left=-20, right=20)
plt.ylim(bottom=0, top=0.25)
plt.show()

plt.figure(figsize=(5.5, 4.5), dpi=200)
plt.hist(muterrs, bins=100, color=dred, density=True, alpha=0.7)
plt.xlabel("Prediction Error")
plt.ylabel("Frequency")
plt.xlim(left=-20, right=20)
plt.ylim(bottom=0, top=0.25)
plt.show()

mMin = 1
mMax = -1
for i in range(10000):
    wtDataSubset = wtData.sample(100)
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
    wtDataSubset = wtData.sample(20)
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
    mutDataSubset = mutData.sample(10)
    mutpopt3, mutpcov3 = curve_fit(func, mutDataSubset['Label'], mutDataSubset['Prediction'])
    if mutpopt3 < mMin:
        mMin = mutpopt3
    if mutpopt3 > mMax:
        mMax = mutpopt3

muty3 = mMin * x_s
muty4 = mMax * x_s

plt.figure(figsize=(4.5, 4.5), dpi=200)
plt.fill_between(x_s, wty3, wty4, color=lblue2, label='28.5C Uncertainty Band 1')
plt.fill_between(x_s, wty1, wty2, color=lblue, label='28.5C Uncertainty Band 2')
plt.plot(x_s, func(x_s, wtpopt1), linewidth=1.5, color=dblue, label='28.5C fit')
plt.plot(x_s, kimmel_wt(x_s), linewidth=1.5, linestyle='--', color=dblue, label='28.5C Kimmel')
plt.fill_between(x_s, muty3, muty4, color=lred2, label='25.0C Uncertainty Band 1')
plt.fill_between(x_s, muty1, muty2, color=lred, label='25.0C Uncertainty Band 2')
plt.plot(x_s, kimmel_mut(x_s), linewidth=1.5, linestyle='--', color=dred, label='25.0C Kimmel')
plt.plot(x_s, func(x_s, mutpopt), linewidth=1.5, color=dred, label='25.0C fit')

plt.xlabel("Actual HPF")
plt.ylabel("Predicted HPF")
plt.xlim(left=0, right=55)
plt.ylim(top=60, bottom=0)
plt.legend(fontsize=8, markerscale=1.5)
plt.show()
