from asyncio.events import get_event_loop
from os import read, ttyname, path
import queue
import re
# from magdesk_ml_tracker import NUM_BACKGROUND_DATAPOINTS
from src.preprocess.data_reader import read_calibrate_data
from codetiming import Timer
import asyncio
import matplotlib.pyplot as plt
import numpy as np
import sys
import random
from itertools import count
import time
from matplotlib.animation import FuncAnimation
from numpy.core.numeric import True_
import matplotlib
import queue
import asyncio
import struct
import sys
import time
import datetime
import atexit
import time
import numpy as np
from bleak import BleakClient
import matplotlib.pyplot as plt
from bleak import exc
import pandas as pd
from multiprocessing import Pool
import multiprocessing
import threading
from src.solver import Solver_jac, Solver
from src.filter import Magnet_KF, Magnet_UKF
from src.preprocess import Calibrate_Data
from config import pSensor_smt, pSensor_large_smt, pSensor_small_smt, pSensor_median_smt, pSensor_2_line, pSensor_7_line, pSensor_7_line_elevated_z_5cm, pSensor_7_line_elevated_z_1cm
import cppsolver as cs
import csv
import numpy.linalg
import serial
import serial.tools.list_ports
import binascii
import triad_openvr
import atexit
import math
from datetime import datetime

matplotlib.use('Qt5Agg')

BAUDRATE = 921600
filename_timestamp = datetime.now().strftime("%Y_%m_%d-%I:%M_%p")
magnet_name = 'SA'

dataset_filename = 'data/ml_dataset/magdesk_dataset_' + magnet_name + "_" + filename_timestamp + '.csv'
dataset_file = open(dataset_filename, mode='w')
dataset_writer = csv.writer(dataset_file, delimiter=',')

background_noise_dataset_filename = 'data/ml_dataset/magdesk_dataset_background_noise_' + magnet_name + "_" + filename_timestamp + '.csv'
background_noise_dataset_file = open(background_noise_dataset_filename, mode='w')
background_noise_dataset_writer = csv.writer(background_noise_dataset_file, delimiter=',')
NUM_BACKGROUND_DATAPOINTS = 5000

magnetometer_data_dict = multiprocessing.Manager().dict()
vive_data_dict = multiprocessing.Manager().dict()

pSensor = pSensor_7_line_elevated_z_1cm

params = np.array([40 / np.sqrt(2) * 1e-6, 40 / np.sqrt(2) * 1e-6,
                   0, np.log(2.2), 1e-2 * 75, 1e-2 * (0), 1e-2 * (7), np.pi, 0])

v = triad_openvr.triad_openvr()
VIVE_CALIB_DONE = False
VIVE_RECALIB = False
vive_magdesk_tracking_point_diff = 18 * 1e-2
rigid_transform_rotation = None
rigid_transform_translation = None
vive_init_angles = None
vive_calib_file_name = 'vive_calib.npz'
if (path.exists(vive_calib_file_name) and (not(VIVE_RECALIB))):
    VIVE_CALIB_DONE =  True

if(VIVE_CALIB_DONE):
    rigid_transform_rotation = np.load(vive_calib_file_name)['rotation']
    rigid_transform_translation = np.load(vive_calib_file_name)['translation']
    vive_init_angles = np.load(vive_calib_file_name)['init_angles']


all_lines_data_read = multiprocessing.Value("i", 0)

def dataWriter():
    global all_lines_data_read
    global magnetometer_data_dict
    global vive_data_dict
    # past_time  = time.time_ns()

    while len(magnetometer_data_dict) < 2 * 7:
        time.sleep(0.001)
    
    line_miss = np.zeros(7)

    print("Please remove all magnetic elements from vicinty of MagDesk")
    time.sleep(10)
    for background_noise_writer_iter in range(NUM_BACKGROUND_DATAPOINTS):
        while(not(magnetometer_data_dict['line1_data_read'] and magnetometer_data_dict['line2_data_read'] and magnetometer_data_dict['line3_data_read']
         and magnetometer_data_dict['line4_data_read'] and magnetometer_data_dict['line5_data_read']  and magnetometer_data_dict['line6_data_read']
          and magnetometer_data_dict['line7_data_read'])):
            time.sleep(0.001)
        
        magdesk_line_1 = magnetometer_data_dict['line 1']
        magdesk_line_2 = magnetometer_data_dict['line 2']
        magdesk_line_3 = magnetometer_data_dict['line 3']
        magdesk_line_4 = magnetometer_data_dict['line 4']
        magdesk_line_5 = magnetometer_data_dict['line 5']
        magdesk_line_6 = magnetometer_data_dict['line 6']
        magdesk_line_7 = magnetometer_data_dict['line 7']
        magdesk_data = np.concatenate(
            (magdesk_line_1, magdesk_line_2, magdesk_line_3, magdesk_line_4, magdesk_line_5, magdesk_line_6, magdesk_line_7), axis=0)
        magdesk_data = magdesk_data.reshape(-1) 
        magdesk_data_list = np.ndarray.tolist(magdesk_data)

        print(f"Number of background datapoints collected: {background_noise_writer_iter} / {NUM_BACKGROUND_DATAPOINTS}", end = "\r")

        csv_row = [time.time_ns()] + magdesk_data_list
        background_noise_dataset_writer.writerow(csv_row)
        
        magnetometer_data_dict['line1_data_read'] = False
        magnetometer_data_dict['line2_data_read'] = False
        magnetometer_data_dict['line3_data_read'] = False
        magnetometer_data_dict['line4_data_read'] = False
        magnetometer_data_dict['line5_data_read'] = False
        magnetometer_data_dict['line6_data_read'] = False
        magnetometer_data_dict['line7_data_read'] = False

    past_time_dict = {}
    frequency = {}

    print("Starting Data Collection")

    while True:
        while(not(magnetometer_data_dict['line1_data_read'] and magnetometer_data_dict['line2_data_read'] and magnetometer_data_dict['line3_data_read']
         and magnetometer_data_dict['line4_data_read'] and magnetometer_data_dict['line5_data_read']  and magnetometer_data_dict['line6_data_read']
          and magnetometer_data_dict['line7_data_read'])):
            if(not magnetometer_data_dict['line1_data_read']):
                line_miss[0] = line_miss[0] + 1
            if(not magnetometer_data_dict['line2_data_read']):
                line_miss[1] = line_miss[1] + 1
            if(not magnetometer_data_dict['line3_data_read']):
                line_miss[2] = line_miss[2] + 1
            if(not magnetometer_data_dict['line4_data_read']):
                line_miss[3] = line_miss[3] + 1
            if(not magnetometer_data_dict['line5_data_read']):
                line_miss[4] = line_miss[4] + 1
            if(not magnetometer_data_dict['line6_data_read']):
                line_miss[5] = line_miss[5] + 1
            if(not magnetometer_data_dict['line7_data_read']):
                line_miss[6] = line_miss[6] + 1

            time.sleep(0.001)
        
        magdesk_line_1 = magnetometer_data_dict['line 1']
        magdesk_line_2 = magnetometer_data_dict['line 2']
        magdesk_line_3 = magnetometer_data_dict['line 3']
        magdesk_line_4 = magnetometer_data_dict['line 4']
        magdesk_line_5 = magnetometer_data_dict['line 5']
        magdesk_line_6 = magnetometer_data_dict['line 6']
        magdesk_line_7 = magnetometer_data_dict['line 7']
        magdesk_data = np.concatenate(
            (magdesk_line_1, magdesk_line_2, magdesk_line_3, magdesk_line_4, magdesk_line_5, magdesk_line_6, magdesk_line_7), axis=0)
        magdesk_data = magdesk_data.reshape(-1) 
        magdesk_data_list = np.ndarray.tolist(magdesk_data)

        x = vive_data_dict['x']
        y = vive_data_dict['y']
        z = vive_data_dict['z']
        yaw = vive_data_dict['yaw']
        pitch = vive_data_dict['pitch']
        roll = vive_data_dict['roll']

        vive_data_list = [x* 1e2, y* 1e2, z* 1e2, yaw, pitch, roll]

        csv_row = [time.time_ns()] + vive_data_list + magdesk_data_list

        if(not (abs(vive_data_dict['roll']) > 65 and abs(vive_data_dict['roll']) < 115)):
            dataset_writer.writerow(csv_row)

        # with all_lines_data_read.get_lock():
        #     all_lines_data_read.value = 0
        if magnetometer_data_dict['line1_data_read'] == True:
            magnetometer_data_dict['line1_data_read'] = False
            if 'line1' in past_time_dict.keys():
                frequency['line1'] = 1e9 / (time.time_ns() - past_time_dict['line1'])
            past_time_dict['line1']  = time.time_ns()
        if magnetometer_data_dict['line2_data_read'] == True:
            magnetometer_data_dict['line2_data_read'] = False
            if 'line2' in past_time_dict.keys():
                frequency['line2'] = 1e9 / (time.time_ns() - past_time_dict['line2'])
            past_time_dict['line2']  = time.time_ns()
        if magnetometer_data_dict['line3_data_read'] == True:
            magnetometer_data_dict['line3_data_read'] = False
            if 'line3' in past_time_dict.keys():
                frequency['line3'] = 1e9 / (time.time_ns() - past_time_dict['line3'])
            past_time_dict['line3']  = time.time_ns()
        if magnetometer_data_dict['line4_data_read'] == True:
            magnetometer_data_dict['line4_data_read'] = False
            if 'line4' in past_time_dict.keys():
                frequency['line4'] = 1e9 / (time.time_ns() - past_time_dict['line4'])
            past_time_dict['line4']  = time.time_ns()
        if magnetometer_data_dict['line5_data_read'] == True:
            magnetometer_data_dict['line5_data_read'] = False
            if 'line5' in past_time_dict.keys():
                frequency['line5'] = 1e9 / (time.time_ns() - past_time_dict['line5'])
            past_time_dict['line5']  = time.time_ns()
        if magnetometer_data_dict['line6_data_read'] == True:
            magnetometer_data_dict['line6_data_read'] = False
            if 'line6' in past_time_dict.keys():
                frequency['line6'] = 1e9 / (time.time_ns() - past_time_dict['line6'])
            past_time_dict['line6']  = time.time_ns()
        if magnetometer_data_dict['line7_data_read'] == True:
            magnetometer_data_dict['line7_data_read'] = False
            if 'line7' in past_time_dict.keys():
                frequency['line7'] = 1e9 / (time.time_ns() - past_time_dict['line7'])
            past_time_dict['line7']  = time.time_ns()
        
        print(frequency, end = "\r")

    

class mangnetometerArray(object):

    def __init__(self, arrayPort=None, arrayBaudrate=None):
        self.ser = serial.Serial(port=arrayPort, baudrate=arrayBaudrate)
        self.startTime = time.time_ns()
        self.lastTime = time.time_ns()
        time.sleep(5)
        self.ser.flushInput()
        self.ser.flush()
        self.ser.write(binascii.unhexlify('FFFFFFFFFFFFFFFF'))
        self.ser.read_until(binascii.unhexlify('525D'))
        self.ser.write(binascii.unhexlify('FFFFFFFFFFFFFFFF'))
        print(f'COM Port: {arrayPort} initialized')
        while(self.ser.inWaiting()):
            garbage_data = self.ser.read(self.ser.inWaiting())

    def readWriteMeasurement(self):
        num = 16
        global magnetometer_data_dict
        # global all_lines_data_read
        sensors = np.zeros((num, 3))

        while(self.ser.inWaiting() < 136):
            time.sleep(0.001)
        self.ser.read_until(binascii.unhexlify('424D'))
        dataframe = self.ser.read(134)
        serData = dataframe[0:1]
        self.arrayID = int.from_bytes(serData, byteorder='big')
        line_index = 6 - self.arrayID
        serData = dataframe[1:2]
        self.num_sensors = int.from_bytes(serData, byteorder='big')
        serData = dataframe[2:6]
        self.sampleTimeOffsetUS = int.from_bytes(serData, byteorder='big')
        for sensorNumber in range(self.num_sensors):
            local_checksum = 0
            sensorXAxisData = int.from_bytes(
                dataframe[6+(sensorNumber*8): 8+(sensorNumber*8)], byteorder='big', signed=True)
            local_checksum += abs(sensorXAxisData)
            sensorYAxisData = int.from_bytes(
                dataframe[8+(sensorNumber*8): 10+(sensorNumber*8)], byteorder='big', signed=True)
            local_checksum += abs(sensorYAxisData)
            sensorZAxisData = int.from_bytes(
                dataframe[10+(sensorNumber*8): 12+(sensorNumber*8)], byteorder='big', signed=True)
            local_checksum += abs(sensorZAxisData)
            checksum = int.from_bytes(
                dataframe[12+(sensorNumber*8): 14+(sensorNumber*8)], byteorder='big', signed=True)
            if(checksum == (local_checksum & 0xFFFF)):
                sensors[sensorNumber, 0] = sensorXAxisData
                sensors[sensorNumber, 1] = sensorYAxisData
                sensors[sensorNumber, 2] = sensorZAxisData

        sensors = sensors.reshape(-1)
        if line_index == 0:
            magnetometer_data_dict['line 1'] = sensors
            magnetometer_data_dict['line1_data_read'] = True
        elif line_index == 1:
            magnetometer_data_dict['line 2'] = sensors
            magnetometer_data_dict['line2_data_read'] = True
        elif line_index == 2:
            magnetometer_data_dict['line 3'] = sensors
            magnetometer_data_dict['line3_data_read'] = True
        elif line_index == 3:
            magnetometer_data_dict['line 4'] = sensors
            magnetometer_data_dict['line4_data_read'] = True
        elif line_index == 4:
            magnetometer_data_dict['line 5'] = sensors
            magnetometer_data_dict['line5_data_read'] = True
        elif line_index == 5:
            magnetometer_data_dict['line 6'] = sensors
            magnetometer_data_dict['line6_data_read'] = True
        elif line_index == 6:
            magnetometer_data_dict['line 7'] = sensors
            magnetometer_data_dict['line7_data_read'] = True
        else:
            print(str(line_index)+" Warning: Data Read ERROR!")
        
        # with all_lines_data_read.get_lock():
        #     all_lines_data_read.value += 1


def readArray(arrayPort=None):
    print(arrayPort)
    magnetometerArrayVar = mangnetometerArray(
        arrayPort=arrayPort, arrayBaudrate=BAUDRATE)
    while True:
        magnetometerArrayVar.readWriteMeasurement()

def rigid_transform_3D(A, B):
    assert A.shape == B.shape

    num_rows, num_cols = A.shape
    if num_rows != 3:
        raise Exception(f"matrix A is not 3xN, it is {num_rows}x{num_cols}")

    num_rows, num_cols = B.shape
    if num_rows != 3:
        raise Exception(f"matrix B is not 3xN, it is {num_rows}x{num_cols}")

    # find mean column wise
    centroid_A = np.mean(A, axis=1)
    centroid_B = np.mean(B, axis=1)

    # ensure centroids are 3x1
    centroid_A = centroid_A.reshape(-1, 1)
    centroid_B = centroid_B.reshape(-1, 1)

    # subtract mean
    Am = A - centroid_A
    Bm = B - centroid_B

    H = Am @ np.transpose(Bm)

    # sanity check
    #if linalg.matrix_rank(H) < 3:
    #    raise ValueError("rank of H = {}, expecting 3".format(linalg.matrix_rank(H)))

    # find rotation
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # special reflection case
    if np.linalg.det(R) < 0:
        print("det(R) < R, reflection detected!, correcting for it ...")
        Vt[2,:] *= -1
        R = Vt.T @ U.T

    t = -R @ centroid_A + centroid_B

    return R, t

def vive_angular_correction(vive_coordinates_data, vive_angular_data, vive_init_angles, vive_magdesk_tracking_diff):

    pitch_radians = math.radians(vive_angular_data[0] -  90) 
    yaw_radians = math.radians(vive_angular_data[1]) 

    x = float(vive_coordinates_data[0] + (vive_magdesk_tracking_diff * math.cos(pitch_radians) * math.sin(yaw_radians)))
    y = float(vive_coordinates_data[1] - (vive_magdesk_tracking_diff * math.cos(pitch_radians) * math.cos(yaw_radians)))
    z = float(vive_coordinates_data[2] - (vive_magdesk_tracking_diff * math.sin(pitch_radians)))
    return x, y, z

def viveCalib():
    global rigid_transform_rotation
    global rigid_transform_translation
    global vive_init_angles
    vive_calib_data = np.empty([3, 3])
    vive_calib_data_test = np.empty([3,1])
    # vive_ground_truth_data = np.array([[-86, 86, 86],[- 14.7, - 14.7, 42],[9.0, 9.0, 9.0]])
    # vive_ground_truth_data = np.array([[-86, 86, 86],[0, 0, 78.3],[9.0, 9.0, 9.0]])
    # vive_ground_truth_data = np.array([[-86, 86, 86],[0, 0, 59.8],[9.0, 9.0, 9.0]])
    vive_ground_truth_data = np.array([[-86, 86, 86],[19, 19, 79.2],[6.0, 6.0, 6.0]])
    vive_ground_truth_data = vive_ground_truth_data * 1e-2
    # vive_ground_truth_test = np.array([[-69.5], [92], [6]])
    vive_ground_truth_test = np.array([[-69], [92.5], [5]])
    vive_ground_truth_test = vive_ground_truth_test * 1e-2

    vive_calib_data_angles = np.empty([3,3])
    vive_calib_data_angles_test = np.empty([3])

    input("Press Enter when you have kept the controller at the first static location")
    txt = ""
    for each in v.devices["controller_1"].get_pose_euler():
        txt += "%.4f" % each
        txt += " "
    x, y, z, roll, yaw, pitch = txt.split(" ")[:6]

    vive_calib_data[0,0] = float(x)
    vive_calib_data[1,0] = float(y)
    vive_calib_data[2,0] = float(z)
    vive_calib_data_angles[0,0] = float(pitch)
    vive_calib_data_angles[0,1] = float(yaw)
    vive_calib_data_angles[0,2] = float(roll)

    input("Press Enter when you have kept the controller at the second static location")
    txt = ""
    for each in v.devices["controller_1"].get_pose_euler():
        txt += "%.4f" % each
        txt += " "
    x, y, z, roll, yaw, pitch = txt.split(" ")[:6]
    vive_calib_data[0,1] = float(x)
    vive_calib_data[1,1] = float(y)
    vive_calib_data[2,1] = float(z)
    vive_calib_data_angles[1,0] = float(pitch)
    vive_calib_data_angles[1,1] = float(yaw)
    vive_calib_data_angles[1,2] = float(roll)

    input("Press Enter when you have kept the controller at the third static location")
    txt = ""
    for each in v.devices["controller_1"].get_pose_euler():
        txt += "%.4f" % each
        txt += " "
    x, y, z, roll, yaw, pitch = txt.split(" ")[:6]
    vive_calib_data[0,2] = float(x)
    vive_calib_data[1,2] = float(y)
    vive_calib_data[2,2] = float(z)
    vive_calib_data_angles[2,0] = float(pitch)
    vive_calib_data_angles[2,1] = float(yaw)
    vive_calib_data_angles[2,2] = float(roll)
    print(f'Angles: {vive_calib_data_angles}')
    
    input("Press Enter when you have kept the controller at the fourth static test location")
    txt = ""
    for each in v.devices["controller_1"].get_pose_euler():
        txt += "%.4f" % each
        txt += " "
    x, y, z, roll, yaw, pitch = txt.split(" ")[:6]
    vive_calib_data_test[0] = float(x)
    vive_calib_data_test[1] = float(y)
    vive_calib_data_test[2] = float(z)
    vive_calib_data_angles_test[0] = pitch
    vive_calib_data_angles_test[1] = yaw
    vive_calib_data_angles_test[2] = roll

    vive_init_angles = (np.mean(vive_calib_data_angles, axis = 0))
    rigid_transform_rotation, rigid_transform_translation = rigid_transform_3D(vive_calib_data, vive_ground_truth_data)

    position_rotation_matrix = v.devices["controller_1"].get_pose_matrix()
    position_rotation_matrix = np.asmatrix(list(position_rotation_matrix))
    rotation_matrix = position_rotation_matrix[:, 0:3]

    corrected_rotation_matrix = rigid_transform_rotation @ rotation_matrix
    position_rotation_matrix[:,0:3] = corrected_rotation_matrix

    yaw = 180 / math.pi * math.atan2(corrected_rotation_matrix[1,0], corrected_rotation_matrix[0, 0])
    roll = 180 / math.pi * math.atan2(corrected_rotation_matrix[2,0], corrected_rotation_matrix[0, 0])
    pitch = 180 / math.pi * math.atan2(corrected_rotation_matrix[2, 1], corrected_rotation_matrix[2, 2])

    # print(f"Vive Pose: Yaw:{yaw}, Pitch: {pitch}, Roll:{roll}", end = "\r")
    
    vive_axes_corrected_data_test = (rigid_transform_rotation@vive_calib_data_test) + rigid_transform_translation
    x, y, z = vive_angular_correction(vive_axes_corrected_data_test, np.array([float(pitch), float(yaw), float(roll)]), vive_init_angles, vive_magdesk_tracking_point_diff)
    
    # RMSE for raw data
    err = (vive_calib_data_test - vive_ground_truth_test) * 100
    err = err * err
    err = np.sum(err)
    rmse_test_raw_data = np.sqrt(err)

    # RMSE for axes corrected dataoutput_data_i = [time.time_ns(),x,y,z, str(float(yaw) - vive_init_angles[1]), str(float(pitch) - vive_init_angles[0]), str(float(roll) - vive_init_angles[2])]
    rmse_test_axes_corrected = np.sqrt(err)
        
    # RMSE for angular corrected data
    err = (np.array([[x], [y], [z]]) - vive_ground_truth_test) * 100
    err = err * err
    err = np.sum(err)
    rmse_test_angle_corrected = np.sqrt(err)

    print(f'Test datapoint ground truth:\n{vive_ground_truth_test * 100}')
    print(f'Test datapoint vive raw data:\n{vive_calib_data_test * 100}, RMSE: {rmse_test_raw_data}')
    print(f'Test datapoint vive raw angle data:\n{vive_calib_data_angles_test}')
    print(f'Test datapoint vive axes corrected:\n{vive_axes_corrected_data_test * 100}, RMSE: {rmse_test_axes_corrected}')
    print(f'Test datapoint vive tracker angle corrected:\n{x* 100, y* 100, z* 100}, RMSE: {rmse_test_angle_corrected}')

    np.savez('vive_calib', rotation=rigid_transform_rotation, translation=rigid_transform_translation, init_angles=vive_init_angles)

def read_vive_data():
    global rigid_transform_rotation
    global rigid_transform_translation
    global vive_magdesk_tracking_point_diff
    global vive_init_angles
    while True:
        txt = ""
        for each in v.devices["controller_1"].get_pose_euler():
            txt += "%.4f" % each
            txt += " "
        x, y, z, roll, yaw, pitch = txt.split(" ")[:6]

        position_rotation_matrix = v.devices["controller_1"].get_pose_matrix()
        position_rotation_matrix = np.asmatrix(list(position_rotation_matrix))
        rotation_matrix = position_rotation_matrix[:, 0:3]

        corrected_rotation_matrix = rigid_transform_rotation @ rotation_matrix
        position_rotation_matrix[:,0:3] = corrected_rotation_matrix

        yaw = 180 / math.pi * math.atan2(corrected_rotation_matrix[1,0], corrected_rotation_matrix[0, 0])
        roll = 180 / math.pi * math.atan2(corrected_rotation_matrix[2,0], corrected_rotation_matrix[0, 0])
        pitch = 180 / math.pi * math.atan2(corrected_rotation_matrix[2, 1], corrected_rotation_matrix[2, 2])

        vive_coordinates_data = np.array([[float(x)], [float(y)], [float(z)]])
        vive_coordinates_data_axes_corrected = (rigid_transform_rotation@vive_coordinates_data) + rigid_transform_translation
        x, y, z = vive_angular_correction(vive_coordinates_data_axes_corrected , np.array([float(pitch), float(yaw), float(roll)]), vive_init_angles, vive_magdesk_tracking_point_diff)

        vive_data_dict['x'] = x
        vive_data_dict['y'] = y
        vive_data_dict['z'] = z
        vive_data_dict['yaw'] = yaw
        vive_data_dict['pitch'] = pitch
        vive_data_dict['roll'] = roll

def main(magcount=1):
    """
    This is the main entry point for the program
    """
    if(not VIVE_CALIB_DONE):
        viveCalib()
        
    input("Press Enter when you are ready to collect data!")

    ports = serial.tools.list_ports.grep('ttyUSB')

    header_row = []
    header_row.append('Time Stamp')
    header_row.append('x')
    header_row.append('y')
    header_row.append('z')
    header_row.append('yaw')
    header_row.append('pitch')
    header_row.append('roll')
    for i in range(1, 16*7+1):
        header_row.append('Sensor '+ str(i))
    dataset_writer.writerow(header_row)

    background_noise_header_row = []
    background_noise_header_row.append('Time Stamp')
    for i in range(1, 16*7+1):
        header_row.append('Sensor '+ str(i))
    background_noise_dataset_writer.writerow(background_noise_header_row)
    
    # Run the tasks
    with Timer(text="\nTotal elapsed time: {:.1f}"):
        multiprocessing.Process(
            target=read_vive_data, args=()).start()
        processes = list()
        for count, value in enumerate(ports):
            # print(count)
            processes.append(multiprocessing.Process(
                target=readArray, args=(value.device,)))
        processes.append(multiprocessing.Process(target=dataWriter))
        for iter in range(len(processes)):
            # print(iter)
            processes[iter].start()

        for iter in range(len(processes)):
            # print(iter)
            processes[iter].join()

if __name__ == '__main__':

    main(1)
