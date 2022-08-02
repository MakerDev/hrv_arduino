#! /usr/bin/python3
#import Jetson.GPIO as GPIO

from matplotlib import pyplot as plt
from matplotlib import animation as ani
from scipy.signal import filtfilt, butter

import os
import datetime
import threading
import numpy as np
import pandas as pd
import serial
import serial.tools.list_ports as portlist
##################################

##################################
stopEvent = threading.Event()

def PortConnect():
    port_list = portlist.comports()

    for portlist_idx in range(0, len(port_list)):
        print("Available COM ports #"+ str(portlist_idx) + ": " + str(port_list[portlist_idx].device))

    print('Enter a COM port number: ', end='')
    PortNum = input()

    return serial.Serial('COM'+str(PortNum), 9600, timeout=1)
##################################

##################################
wr_data_buf = []
wr_save_dir = './Recorded_data/'

if not os.path.exists(wr_save_dir):
    os.mkdir(wr_save_dir)

def WriteRecordedData(data, file_name):
    print('Writing... ' , file_name + '.csv ,', np.shape(data))
    pd.DataFrame(data).to_csv(wr_save_dir + file_name+'.csv', header=False)


plt.rcParams["figure.figsize"] = (15, 5)
max_pt = 4000
fig    = plt.figure()
ax     = plt.axes(xlim=(0, max_pt), ylim=(450,600))#xlim=(0,max_pt), ylim=(0,255))
line,  = ax.plot([], [], lw=1)
y_buf  = [100] * max_pt

plt.tight_layout()
ax.grid(True)

def HandleExit(event):
    print("Figure close")
    stopEvent.set()
    global wr_data_buf
    f_name = str(wr_data_buf[0][0]).split('.')[0].replace(':', '-')
    WriteRecordedData(wr_data_buf, f_name)
    # WRD_Thread = threading.Thread(target=WriteRecordedData, args=(wr_data_buf, f_name, ))
    # WRD_Thread.start()
    # wr_data_buf = []

fig.canvas.mpl_connect('close_event', HandleExit)

def RecordData(y_buf):
    global wr_data_buf

    print("Connecting COM Port..")
    conn = PortConnect()

    if conn.readable():
        print('The connection through a COM port is now readable.')
        while(True):

            if stopEvent.is_set():
                print("Stop reading")
                return

            readline = conn.readline()
            try:
                value = readline[:-2].decode('utf-8')
                wr_data_buf.append([datetime.datetime.now(), value])

                if value != '!' and len(value) != 0:
                    y_buf.pop(0)
                    y_buf.append(int(value))

            except:
                continue

            # if len(wr_data_buf) == 1000:
            #     f_name = str(wr_data_buf[0][0]).split('.')[0].replace(':', '-')
            #     WRD_Thread = threading.Thread(target=WriteRecordedData, args=(wr_data_buf, f_name, ))
            #     WRD_Thread.start()
            #     wr_data_buf = []

##################################

##################################
b, a = butter(5, 0.1)

def AniFunc(i, line, y_buf):
    filtered_y = filtfilt(b, a, y_buf)

    line.set_data(list(range(1,len(filtered_y)+1)), filtered_y)
    #line.set_data(list(range(1,len(y_buf)+1)), y_buf)
    return line,
##################################

##################################
if __name__ == '__main__':
    print("==Program started==")
    Anim = ani.FuncAnimation(fig, func=AniFunc, fargs=(line, y_buf), interval=1, blit=False)

    RecordThread = threading.Thread(target=RecordData, args=(y_buf,))
    RecordThread.start()

    plt.show()