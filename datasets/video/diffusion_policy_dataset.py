from typing import Sequence
import math
import torch
import numpy as np
from omegaconf import DictConfig
from tqdm import tqdm
from .base_video_dataset import BaseVideoDataset
import random


class DiffusionPolicyDataset(BaseVideoDataset):
    """
    Optical Flow dataset
    """

    def __init__(self, cfg: DictConfig, split: str = "training"):
        if split == "test":
            split = "validation"
        super().__init__(cfg, split)
        self.max_flow = self.get_max_flow(split)
        self.external_conditions = cfg.external_conditions

    def get_data_paths(self, split):
        data_dir = self.save_dir / split
        paths = sorted(list(data_dir.glob("**/*.npz")), key=lambda x: x.name)
        return paths

    def get_data_lengths(self, split):
        lengths = [np.load(p)["rgb_video"].shape[0] for p in self.get_data_paths(split)]
        return lengths
    
    def get_max_flow(self, split):
        max_flow = [np.max(np.abs(np.load(p)["traj_flow"])) for p in self.get_data_paths(split)][0]
        return max_flow

    def download_dataset(self):
        return

    def __len__(self):
        # HACK: set length of dataset to be big to ensure checkpointing happens
        return 490000

    def __getitem__(self, idx):
        idx = self.idx_remap[idx] % np.sum(self.get_data_lengths("training"))
        file_idx, frame_idx = self.split_idx(idx)
        data_path = self.data_paths[file_idx]
        data = np.load(data_path)

        # #HACK: have frame_idx be 10 or 29
        # frame_idx = 10 if idx % 2 == 0 else 29
        
        # for now, have t = 1 (self.n_frames = 1)
        # get each external condition and concatenate it to the video in the channel dim
        video = data["rgb_video"][
            frame_idx : frame_idx + self.n_frames
        ].copy()  # (t, h, w, 3)

        du = data["du"][
            frame_idx : frame_idx + self.n_frames
        ].copy()  # (t, 2, h, w)

        pad_len = self.n_frames - len(video)

        nonterminal = np.ones(self.n_frames)
        if len(video) < self.n_frames:
            video = np.pad(video, ((0, pad_len), (0, 0), (0, 0), (0, 0)))
            flow = np.pad(flow, ((0, pad_len),))
            nonterminal[-pad_len:] = 0

        video = torch.from_numpy(video / 255.0).float().permute(0, 3, 1, 2)
        for e in self.external_conditions:
            if e == "rgb_video":
                continue
            external_condition = torch.from_numpy(data[e][frame_idx : frame_idx + self.n_frames].copy()).float()
            video = torch.cat([video, external_condition], axis=1)
        
        video = video.contiguous()

        # # add timestep dimension to video
        if "timestep" in self.external_conditions:
            timestep_channel = torch.ones((self.n_frames, 1, video.shape[-2], video.shape[-1])) * torch.arange(frame_idx, frame_idx + self.n_frames).view(self.n_frames, 1, 1, 1).float() / len_video
            video = torch.cat([video, timestep_channel], dim=1)

        return (
            du[:: self.frame_skip],
            video[:: self.frame_skip],
            nonterminal[:: self.frame_skip],
        )


if __name__ == "__main__":
    import torch
    from unittest.mock import MagicMock
    import tqdm

    cfg = MagicMock()
    cfg.resolution = 256
    cfg.external_cond_dim = 0
    cfg.n_frames = 1
    cfg.frame_skip = 1
    cfg.save_dir = "/home/iyu/scene-jacobian-discovery/diff-force/diffusion-forcing/data/rod_flow"
    cfg.validation_multiplier = 1

    dataset = DiffusionPolicyDataset(cfg, "training")
    print(len(dataset))
    print(dataset.get_data_lengths("training"))

    vid, flow, term = dataset[0]
    print("dataset item shapes", vid.shape, flow.shape, term.shape)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=True, num_workers=16
    )

    for i in tqdm.tqdm(range(49)):
        flow, vid, term = dataset[i]
        # print the timestep channel
        print(vid[0, -1, 0, 0])
