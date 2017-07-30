from pydub import AudioSegment
import numpy as np
import array

import torch
import torch.utils.data

import os

class sound_data(torch.utils.data.Dataset):
  def __init__(self):
    self.data_files = os.listdir("/vault/max/mood/result/")
    self.data_files = ["/vault/max/mood/result/" + f for f in self.data_files]

  def __getitem__(self, idx):
    return self.get_data(self.data_files[idx])

  def __len__(self):
    return len(self.data_files)

  def get_data(self, name, norm = False):
    TIME = 30 * 1000
    sound = AudioSegment.from_mp3(name)
    if(norm):
      sound = AudioSegment.normalize(sound)
    start = sound[:TIME]
    end = sound[-TIME:]
    start_split = start.split_to_mono()
    end_split = end.split_to_mono()
    for i in range(start.sample_width):
      start_split[i] = np.array(start_split[i].get_array_of_samples()).astype('float64')
      end_split[i] = np.array(end_split[i].get_array_of_samples()).astype('float64')
    return np.array([np.array(start_split), np.array(end_split)]).astype('float64')
