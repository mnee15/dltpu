import os
import numpy as np
import torch
from torch.utils.data import Dataset


# (phase, fom)
class PhaseDataset(Dataset):
    def __init__(self, dataset_path, mode='train', low=True, fom=True):
        super().__init__()
        self.mode = mode
        self.low = low # if low=True, use high / low ph // else use high / eq ph
        self.fom = fom # if fom=True, gt: fom // else gt:uph
        data_dir = os.path.join(dataset_path, mode)

        # input
        self.phase_high_dir = os.path.join(data_dir, "high")
        self.phase_low_dir = os.path.join(data_dir, "low")
        self.phase_eq_dir = os.path.join(data_dir, "ph_eq")

        # gt
        self.fom_dir = os.path.join(data_dir, "fom")
        self.uph_dir = os.path.join(data_dir, "uph")

        # mask
        self.mask_dir = os.path.join(data_dir, "mask")


    def __getitem__(self, index):
        phase_high = np.loadtxt(os.path.join(self.phase_high_dir, str(index).zfill(4) + '.csv'), delimiter=',').astype(np.float32)
        if self.low:
            phase_2 = np.loadtxt(os.path.join(self.phase_low_dir, str(index).zfill(4) + '.csv'), delimiter=',').astype(np.float32)
        else:
            phase_2 = np.loadtxt(os.path.join(self.phase_eq_dir, str(index).zfill(4) + '.csv'), delimiter=',').astype(np.float32)
        
        if self.fom:
            gt = np.loadtxt(os.path.join(self.fom_dir, str(index).zfill(4) + '.csv'), delimiter=',').astype(np.float32)
        else:
            gt = np.loadtxt(os.path.join(self.uph_dir, str(index).zfill(4) + '.csv'), delimiter=',').astype(np.float32)
        
        phase_high, phase_2 = torch.from_numpy(phase_high), torch.from_numpy(phase_2)
        gt = torch.from_numpy(gt)

        phase_high, phase_2 = torch.unsqueeze(phase_high, 0), torch.unsqueeze(phase_2, 0) 
        phase = torch.cat((phase_high, phase_2), 0)

        mask = np.loadtxt(os.path.join(self.mask_dir, str(index).zfill(4) + '.csv'), delimiter=',').astype(np.float32)
        mask = torch.from_numpy(mask)
        
        return phase, gt, mask

    def __len__(self):
        assert len(os.listdir(self.phase_high_dir)) == len(os.listdir(self.fom_dir))
        return len(os.listdir(self.fom_dir))