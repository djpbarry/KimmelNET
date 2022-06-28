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
wtpopt, wtpcov = curve_fit(func, wtData['Label'], wtData['Prediction'])
print(wtpopt, np.sqrt(np.diag(wtpcov)))
mut_linear_model = np.polyfit(mutData['Label'], mutData['Prediction'], 1)
mut_linear_model_fn = np.poly1d(mut_linear_model)
kimmel_mut = np.polynomial.polynomial.Polynomial([0, 0.805])
mutpopt, mutpcov = curve_fit(func, mutData['Label'], mutData['Prediction'])
x_s = np.arange(4, 53)

plt.figure()
plt.title("Prediction Accuracy")
plt.plot(wtData['Label'], wtData['Prediction'], 'o', markersize=2)
plt.plot(mutData['Label'], mutData['Prediction'], 'o', markersize=2)
plt.plot(x_s, func(x_s, wtpopt), color="blue")
y1 = (wtpopt - np.sqrt(np.diag(wtpcov))) * x_s
y2 = (wtpopt + np.sqrt(np.diag(wtpcov))) * x_s
plt.fill_between(x_s, y1, y2, color="blue", alpha = 0.3)
plt.plot(x_s, kimmel_wt(x_s), color="green")
plt.plot(x_s, func(x_s, mutpopt), color="red")
plt.plot(x_s, kimmel_mut(x_s), color="yellow")
plt.xlabel("True label")
plt.ylabel("Predicted label")
plt.show()
