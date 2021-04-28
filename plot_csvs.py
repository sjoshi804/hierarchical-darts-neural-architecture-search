from csv import reader
import os 
import sys
import numpy as np
import matplotlib.pyplot as plt

os.chdir(sys.argv[1])
for filename in os.listdir(os.getcwd()):
    if "png" in filename:
        print("Plot found, skipping....", filename)
    with open(filename, 'r') as read_obj:        
        csv_reader = reader(read_obj)        
        data = np.transpose(np.array([[int(num) for num in row] for row in list(csv_reader)]))
        plt.rcParams["figure.figsize"] = (15,5)
        plt.title(filename)
        plt.xlabel("Epochs")
        plt.ylabel("Chosen Op")
        plt.plot(data[0], data[1])
        plt.savefig(filename.replace("csv", "png"))
        plt.close()