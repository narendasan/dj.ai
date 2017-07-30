import sys
import torch
from sound_data import sound_data
from scipy import signal
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# Returns Frequency, Time, Signal
def get_spectrogram(data, seg_size, overlap):
  return signal.spectrogram(data, 
                            nperseg = 8 * 1024,
                            noverlap = 8 * (1024-512)) 

def main():
  dset = sound_data()
  loader = torch.utils.data.DataLoader(dset, num_workers = 8)



main()
