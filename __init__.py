import sys
import torch
from sound_data import sound_data
from scipy import signal
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# Returns Frequency, Time, Signal
def get_spectrogram(data, seg_size = 8 * 1024, overlap = 8 *(1024-512)):
  return signal.spectrogram(data, 
                            nperseg = seg_size,
                            noverlap = overlap) 

def main():
  dset = sound_data()
  loader = torch.utils.data.DataLoader(dset, num_workers = 8)
  # data_1 = next(iter(loader))[0][0][0].numpy()
  # f, t, Sxx = get_spectrogram(data_1)
  # fig = plt.pcolormesh(t, f, Sxx)
  # plt.savefig('spectro_1.png')


main()
