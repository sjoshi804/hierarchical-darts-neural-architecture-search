import numpy as np
import matplotlib.pyplot as plt
from urllib.request import urlopen

from numpy.core.shape_base import block


def wall_time_in_mins(time):
  base_time = time[0]
  return [(t - base_time)/60 for t in time]

def wall_time_in_hours(time):
  base_time = time[0]
  return [(t - base_time)/3600 for t in time]

def smoothen(data, window):
  alpha = 2 /(window + 1.0)
  alpha_rev = 1-alpha
  n = data.shape[0]

  pows = alpha_rev**(np.arange(n+1))

  scale_arr = 1/pows[:-1]
  offset = data[0]*pows[1:]
  pw0 = alpha*alpha_rev**(n-1)

  mult = data*pw0*scale_arr
  cumsums = mult.cumsum()
  out = offset + cumsums*scale_arr[::-1]
  return out



def createGraph(
  arrayOfPlots,
  title="HDARTS v/s DARTS",
  xlabel="GPU Time (minutes)", 
  ylabel="Top 1 Training Accuracy (%)",
  transparency_factor=0.8, 
  smoothening_factor=20
  ):
  """
    Args:
      transparency_factor = 0.2# Lower is more transparent
      smoothening_factor=20 # Higher is smoother
  """

  for plotInfo in arrayOfPlots:
    data = plotInfo['data']
    color = plotInfo['plot_color']
    label = plotInfo['plot_label']
    timeData = plotInfo['time']

    plt.plot(timeData, data[:,2] * 100, label=label, color=color, alpha=transparency_factor)


  plt.legend()
  plt.xlabel(xlabel)
  plt.yticks(np.arange(0, 110, step=20))
  plt.ylabel(ylabel)
  plt.title(title)
  plt.show()


def processTBCSVData(csv_url, plot_label, plot_color='red'):
  # Load csv
  csvData = np.loadtxt(urlopen(csv_url), delimiter=",", skiprows=1)
  csvTimeinMin = wall_time_in_mins(csvData[:,0])

  return {'data': csvData, 'time': csvTimeinMin, 'plot_color': plot_color, 
          'plot_label': plot_label}

# Upload csv to github and link raw file here
train_loss_alpha_01_hdarts_url = 'https://raw.githubusercontent.com/sjoshi804/hierarchical-darts-neural-architecture-search/main/results/hdarts_alpha_lr_.01/run-Alpha_LR_01_13-12-2020--18-25-39-tag-train_loss.csv'
train_loss_01_alpha_hdarts = processTBCSVData(csv_url=train_loss_alpha_01_hdarts_url, plot_label='HDarts Alpha LR=0.01', plot_color='green')


train_loss_alpha_05_hdarts_url = 'https://raw.githubusercontent.com/sjoshi804/hierarchical-darts-neural-architecture-search/main/results/hdarts_alpha_lr_.05/run-Alpha_LR_05_13-12-2020--19-00-09-tag-train_loss.csv'
train_loss_05_alpha_hdarts = processTBCSVData(csv_url=train_loss_alpha_05_hdarts_url, plot_label='HDarts Alpha LR=0.05', plot_color='purple')


train_loss_alpha_1_hdarts_url = 'https://raw.githubusercontent.com/sjoshi804/hierarchical-darts-neural-architecture-search/main/results/hdarts_alpha_lr_.1/run-Alpha_LR_%20.1_13-12-2020--19-47-16-tag-train_loss.csv'
train_loss_1_alpha_hdarts = processTBCSVData(csv_url=train_loss_alpha_05_hdarts_url, plot_label='HDarts Alpha LR=0.1', plot_color='red')


createGraph(arrayOfPlots=[train_loss_01_alpha_hdarts, train_loss_05_alpha_hdarts, train_loss_1_alpha_hdarts])

