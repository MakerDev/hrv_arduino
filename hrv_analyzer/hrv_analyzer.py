import matplotlib.pyplot as plt
import csv
import numpy as np

if __name__ == "__main__":
    readings = []
    hrvs = []

    with open('hrv_analyzer/raw_readings.csv') as f:
        rdr = csv.reader(f)
        for line in rdr:
            timestamp, reading, hrv = line
            readings.append(int(reading))
            hrvs.append(0 if 'inf' in hrv else float(hrv))
    

    t = np.arange(0., len(readings), 1)
    hrvs = np.asarray(hrvs)
    readings = np.asarray(readings)

    plt.plot(t, hrvs, 'r--', t, readings, 'g')
    plt.ylim([0, 200])
    plt.show()