from statistics import median_grouped
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import cppsolver as cs
from config import pSensor_7_line_elevated_z_1cm
from sklearn.metrics import r2_score

timestamp_of_file = '2022_01_27-07_48_PM_1'

# params = np.array([40 / np.sqrt(2) * 1e-6, 40 / np.sqrt(2) * 1e-6,
#                    0, np.log(2.2), 1e-2 * 75, 1e-2 * (0), 1e-2 * (7), np.pi, 0])
params = np.array([40 / np.sqrt(2) * 1e-6, 40 / np.sqrt(2) * 1e-6, 0, 0.966, 1e-2 * 75, 1e-2 * (0), 1e-2 * (7), np.pi, 0])
pSensor = pSensor_7_line_elevated_z_1cm


SPINE_COLOR = 'gray'
def latexify(fig_width=36, fig_height=15, columns=1):
    assert(columns in [1,2])

    MAX_HEIGHT_INCHES = 8.0
    if fig_height > MAX_HEIGHT_INCHES:
        fig_height = MAX_HEIGHT_INCHES

    params = {'backend': 'ps',
              'axes.labelsize': 30, # fontsize for x and y labels (was 10)
              'axes.titlesize': 30,
              'figure.titlesize': 30, # was 10
              'legend.fontsize': 30, # was 10
              'xtick.labelsize': 30,
              'ytick.labelsize': 30,
              'axes.labelpad': 1,
              'figure.figsize': [fig_width,fig_height],
              'font.family': 'calibri',
              'savefig.pad_inches':0.1,
              'figure.constrained_layout.h_pad':0,
              'figure.constrained_layout.w_pad':0,
              'axes.xmargin': 0,
              'axes.ymargin': 0,
              'xtick.major.pad':0.2,
              'ytick.major.pad':0.2,
              'pdf.fonttype': 42
    }

    matplotlib.rcParams.update(params)


def format_axes(ax):

    for spine in ['top', 'right']:
        ax.spines[spine].set_color(SPINE_COLOR)
        ax.spines[spine].set_linewidth(0.5)

    for spine in ['left', 'bottom']:
        ax.spines[spine].set_color(SPINE_COLOR)
        ax.spines[spine].set_linewidth(0.5)

    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    for axis in [ax.xaxis, ax.yaxis]:
        axis.set_tick_params(direction='out', color=SPINE_COLOR)
    return ax


def analytical_model_location_predictor(mag_data, mag_to_use, seed_params):

    mag_data = mag_data.reshape(-1,3)
    mag_data[:,0] = -mag_data[:,0]
    mag_data[:,1] = -mag_data[:,1]
    mag_data_for_calc = (mag_data.reshape(-1))[mag_to_use]
    pSensor_for_calc = (pSensor.reshape(-1))[mag_to_use]
    new_params = cs.solve_1mag(mag_data_for_calc, pSensor_for_calc, seed_params)
    location_data = [new_params[4] * 1e2, new_params[5] * 1e2, new_params[6] * 1e2]
    return location_data, new_params


def main():
    filename = './magdesk_dataset_background_noise_SA_' + timestamp_of_file + '.csv'
    csv = pd.read_csv(filename, delimiter=',', header=1)
    background_data = csv.values[1:,:]
    # filename = './magdesk_dataset_' + timestamp_of_file + '.csv'
    filename = './magdesk_dataset_SA_' + timestamp_of_file + '.csv'
    csv = pd.read_csv(filename, delimiter=',', header=1)
    data = csv.values[1:,:]

    y = data[:,1:4]
    X = data[:, 7:]
    background_X = background_data[:,1:]
    mean_background = np.mean(background_X, axis=0)
    # std_background = np.std(background_X, axis=0)
    # print(mean_background)
    # print(std_background)
    X = X - mean_background
    # X = X / std_background
    # X = X[(y[:,2] > 5)]
    # y = y[(y[:,2] > 5)]
    X = X[(y[:,2] < 25)]
    y = y[(y[:,2] < 25)]
    # X = X[(y[:,2] < 50)]
    # y = y[(y[:,2] < 50)]
    seed_params = params

    y_pred = np.zeros(np.shape(y))

    error  = np.array([])
    # z_error_percentage  = np.array([])
    seed_params[4] = y[0, 0] * 1e-2
    seed_params[5] = y[0, 1] * 1e-2
    seed_params[6] = y[0, 2] * 1e-2
    for sample_iter in range(len(X)):
        # print(seed_params)
        seed_params[4] = y[sample_iter, 0] * 1e-2
        seed_params[5] = y[sample_iter, 1] * 1e-2
        seed_params[6] = y[sample_iter, 2] * 1e-2
        # print(seed_params)
        y_pred[sample_iter], seed_params = analytical_model_location_predictor(X[sample_iter], range(len(X[sample_iter])), seed_params)
        # print(f'Ground Truth Location: {y[sample_iter]}')
        # print(f'Predicted Location: {y_pred[sample_iter]}')
        error = np.append(error, np.abs(y_pred[sample_iter]-y[sample_iter]))
        # z_error_percentage = np.append(z_error_percentage, np.abs(((y_pred[sample_iter, 2]-y[sample_iter, 2]) * 100) / y[sample_iter, 2]))
        print(sample_iter, end = "\r")
    
    error = np.reshape(error, (-1, 3))

    median_axis_error = np.median(error, axis = 0)
    # error[:, 2] = error[:, 2] - median_axis_error[2]
    # error = error - median_axis_error
  
    z_error_percentage = (error[:, 2] / y[:, 2]) * 100

    max_r2_iter = 0
    delay = True
    max_r2_score = -1000
    r2_score_forward = np.array([])
    r2_score_reverse = np.array([])
    for iter in range(10000):
        r2_score_value = r2_score(y[(iter+1):,:], y_pred[0:-(iter+1),:])
        # r2_score_value = r2_score(y[(iter+1):,:], y[(iter+1):,:])
        if r2_score_value > max_r2_score:
            max_r2_score = r2_score_value
            max_r2_iter = iter
            delay = True
        r2_score_forward = np.append(r2_score_forward, r2_score_value)
    for iter in range(10000):
        r2_score_value = r2_score(y[0:-(iter+1),:], y_pred[(iter+1):,:])
        # r2_score_value = r2_score(y[0:-(iter+1),:], y[0:-(iter+1),:])
        if r2_score_value > max_r2_score:
            max_r2_score = r2_score_value
            max_r2_iter = iter
            delay = False
        r2_score_reverse = np.append(r2_score_reverse, r2_score_value)

    print(f"Max R2 Value: {max_r2_iter}, delay = {delay}, offset = {max_r2_iter}")

    fig2,ax2 = plt.subplots()
    ax2.plot(r2_score_forward)
    ax2.plot(r2_score_reverse)
    ax2.set_ylabel("correlation")
    ax2.set_xlabel("Offset")
    # ax2.set_ylim(-0.1, 70)
    # # ax.set_xlim(-1, 10)
    # ax1.legend()
    ax2.set_title('R2 with offset')
    format_axes(ax2)
    plt.tight_layout()
    # plt.savefig('magdesk_analytical_model_z_axis_error.pdf',bbox_inches='tight')
    plt.show()

    # # y_pred[:, 2] =  y_pred[:, 2] - median_axis_error[2]
    # r2_score_value = r2_score(y[-2500:-1500,:], y_pred[-2550:-1550,:])
    # print(f"R2 Value: {r2_score_value}")


    # # y_pred[:, 2] =  y_pred[:, 2] - median_axis_error[2]
    # r2_score_value = r2_score(y[-2500:-1500,:], y_pred[-2500:-1500,:])
    # print(f"R2 Value: {r2_score_value}")

    print("Mean: ",np.mean(error, axis = 0))
    print("Median: ", np.median(error, axis = 0))
    print("Max: ", np.max(error, axis = 0))
    print("Min: ", np.min(error, axis = 0))
    print("SD: ", np.std(error, axis = 0))

    # y_test = np.reshape(total_y_test, (-1,3))
    # y_pred = np.reshape(total_y_pred, (-1,3))
    latexify()
    fig,ax = plt.subplots()
    ax.plot(y[-2500:-1500,2], label='Ground Truth')
    ax.plot(y_pred[-2550:-1550,2], label='Prediction')
    ax.set_ylabel("Z Axis Height")
    ax.set_xlabel("samples")
    ax.set_ylim(-0.1, 70)
    # ax.set_xlim(-1, 10)
    ax.legend()
    ax.set_title('Analytical Model')
    format_axes(ax)
    plt.tight_layout()
    plt.savefig('magdesk_analytical_model.pdf',bbox_inches='tight')
    plt.show()
    fig1,ax1 = plt.subplots()
    ax1.scatter(y[:, 2] , z_error_percentage)
    ax1.set_ylabel("Z axis Percentage Error")
    ax1.set_xlabel("Z Axis Height")
    ax1.set_ylim(-0.1, 70)
    # # ax.set_xlim(-1, 10)
    # ax1.legend()
    ax1.set_title('Analytical Model Z axis error')
    format_axes(ax)
    plt.tight_layout()
    plt.savefig('magdesk_analytical_model_z_axis_error.pdf',bbox_inches='tight')
    plt.show()




if __name__ == '__main__':

    main()