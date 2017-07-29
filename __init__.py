import sys
import torch
from sound_data import sound_data

def main():
  dset = sound_data()
  loader = torch.utils.data.DataLoader(dset, num_workers = 8)
  print(len(next(iter(loader))[0][0][0]))



main()
