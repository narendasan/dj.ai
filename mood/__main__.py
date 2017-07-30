import sys
import torch.utils.data as D
import AudioLoader
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
    loader = D.DataLoader(AudioLoader.AudioLoader(), num_workers = 1)
    data_1 = next(iter(loader))[0][0][0].numpy()
    f, t, Sxx = get_spectrogram(data_1)
    fig = plt.pcolormesh(t, f, Sxx)
    plt.savefig('spectra.png')

if __name__ == '__main__':
    main()
