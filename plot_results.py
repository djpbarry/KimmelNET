import matplotlib.pyplot as plt
import numpy as np
import pandas
from scipy.optimize import curve_fit


def func(x, a):
    return a * x


wtData = pandas.read_csv('Z:/working/barryd/hpc/python/zf_reg/outputs/zf_regression_test_2022-06-27-08-57-27'
                         '/zf_regression_test_on_training_data_predictions.csv')
mutData = pandas.read_csv('Z:/working/barryd/hpc/python/zf_reg/outputs/zf_regression_test_25C_2022-06-27-16-17-36'
                          '/zf_regression_test_25C_predictions.csv')

wt_linear_model = np.polyfit(wtData['Label'], wtData['Prediction'], 1)
wt_linear_model_fn = np.poly1d(wt_linear_model)
kimmel_wt = np.polynomial.polynomial.Polynomial([0, 1])
wtpopt1, wtpcov1 = curve_fit(func, wtData['Label'], wtData['Prediction'])


#print(wtpopt1, np.sqrt(np.diag(wtpcov1)))

#wty1 = (wtpopt1 - np.sqrt(np.diag(wtpcov1))) * x_s
#wty2 = (wtpopt1 + np.sqrt(np.diag(wtpcov1))) * x_s
mut_linear_model = np.polyfit(mutData['Label'], mutData['Prediction'], 1)
mut_linear_model_fn = np.poly1d(mut_linear_model)
kimmel_mut = np.polynomial.polynomial.Polynomial([0, 0.805])
mutpopt, mutpcov = curve_fit(func, mutData['Label'], mutData['Prediction'])
x_s = np.arange(0, 53)

lred = (0.85, 0.5, 0.2)
lblue = (0.3, 0.3, 0.55)
dred = (0.65, 0.3, 0.0)
dblue = (0.0, 0.0, 0.5)

plt.figure(figsize=(4.5, 4.5), dpi=200)
plt.plot(wtData['Label'], wtData['Prediction'], 'o', markersize=3, alpha=0.1, mfc=lblue, mec=lblue)
plt.plot(mutData['Label'], mutData['Prediction'], 'o', markersize=3, alpha=0.1, mfc=lred, mec=lred)
plt.plot(x_s, func(x_s, wtpopt1), linewidth=2, color=dblue)
#plt.fill_between(x_s, y1, y2, color="blue", alpha = 0.3)
plt.plot(x_s, kimmel_wt(x_s), linewidth=2, linestyle='--', color=dblue)
plt.plot(x_s, func(x_s, mutpopt), linewidth=2, color=dred)
plt.plot(x_s, kimmel_mut(x_s), linewidth=2, linestyle='--', color=dred)
plt.xlabel("Actual HPF")
plt.ylabel("Predicted HPF")
plt.xlim(left=0, right=55)
plt.ylim(top=60, bottom=0)
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
    mutDataSubset = mutData.sample(20)
    mutpopt3, mutpcov3 = curve_fit(func, mutDataSubset['Label'], mutDataSubset['Prediction'])
    if mutpopt3 < mMin:
        mMin = mutpopt3
    if mutpopt3 > mMax:
        mMax = mutpopt3

muty3 = mMin * x_s
muty4 = mMax * x_s

plt.figure(figsize=(4.5, 4.5), dpi=200)
plt.plot(wtData['Label'], wtData['Prediction'], 'o', markersize=3, alpha=0.1, mfc=lblue, mec=lblue)
plt.plot(mutData['Label'], mutData['Prediction'], 'o', markersize=3, alpha=0.1, mfc=lred, mec=lred)
plt.plot(x_s, func(x_s, wtpopt1), linewidth=2, color=dblue)
plt.fill_between(x_s, wty3, wty4, color=dblue, alpha=0.4)
plt.fill_between(x_s, wty1, wty2, color=dblue, alpha=0.8)
plt.plot(x_s, kimmel_wt(x_s), linewidth=2, linestyle='--', color=dblue)
plt.plot(x_s, func(x_s, mutpopt), linewidth=2, color=dred)
plt.fill_between(x_s, muty3, muty4, color=dred, alpha=0.4)
plt.fill_between(x_s, muty1, muty2, color=dred, alpha=0.8)
plt.plot(x_s, kimmel_mut(x_s), linewidth=2, linestyle='--', color=dred)

plt.xlabel("Actual HPF")
plt.ylabel("Predicted HPF")
plt.xlim(left=0, right=55)
plt.ylim(top=60, bottom=0)
plt.show()