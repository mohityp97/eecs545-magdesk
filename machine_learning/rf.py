import numpy as np
import matplotlib
import matplotlib.pyplot as plt
# from sklearnex import patch_sklearn
# patch_sklearn()
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
import pandas as pd
from copy import deepcopy
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.neural_network import MLPRegressor
from joblib import dump, load
from xgboost import XGBRegressor
from filterpy.kalman import FixedLagSmoother, KalmanFilter
from sklearn.model_selection import train_test_split

data = np.load('LSTM_data1.npz')
X = data['X']
y = data['y']

reg = ExtraTreesRegressor(n_jobs=-1,n_estimators=200, random_state=42)

sel_model = reg
# clf = SVC(kernel='linear', gamma='auto')
kf = KFold(n_splits=10, shuffle=False)
kf.get_n_splits(X)

error = np.array([])
total_y_test=np.array([])
total_y_pred=np.array([])

for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    sel_model.fit(X_train,y_train)
    y_pred = sel_model.predict(X_test)
    total_y_test = np.append(total_y_test, y_test)
    total_y_pred = np.append(total_y_pred, y_pred)
    error = np.append(error, np.abs(y_pred-y_test))

error = np.reshape(error, (-1, 3))
print("Mean: ",np.mean(error, axis = 0))
print("Median: ", np.median(error, axis = 0))
print("Max: ", np.max(error, axis = 0))
print("Min: ", np.min(error, axis = 0))
print("SD: ", np.std(error, axis = 0))

y_test = np.reshape(total_y_test, (-1,3))
y_pred = np.reshape(total_y_pred, (-1,3))
fig,ax = plt.subplots()
# ax.plot(y_test[13000:14330,0], label='Ground Truth')
# ax.plot(y_pred[13000:14330,0], label='Prediction')
ax.plot(y_test[13950:15530,1], label='Ground Truth')
ax.plot(y_pred[13950:15530,1], label='Prediction')
ax.set_ylabel("Z Axis Height")
ax.set_xlabel("samples")
# ax.set_ylim(-0.1, 2)
# ax.set_xlim(-1, 10)
ax.legend()
ax.set_title('Random Forest 1 cm Voxel 9 magnetometers 50 plus removed')
# format_axes(ax)
plt.tight_layout()
# plt.savefig('Magdesk_ml_rf_1cm_9_mag_50_plus_removed.pdf',bbox_inches='tight')
plt.show()