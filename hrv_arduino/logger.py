import csv
import serial
from datetime import datetime

baud = 9600
port = 'COM7'
ser = serial.Serial(port, baud)
i = 0
n_samples = 100000
logs = [["timestamp", "raw-value", 'bpm']]

while i < n_samples:    
    try:
        data = str(ser.readline().decode('utf-8')).strip()
        value, bpm = data.split(',')
        dt = datetime.now()
        logs.append([dt, value, bpm])
        #print(dt, value, bpm)
        i+=1
    except:
        pass
with open('log.csv', 'w') as f:
    writer = csv.writer(f)
