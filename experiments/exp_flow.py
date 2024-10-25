from datasets.video import (
    MinecraftVideoDataset,
    DmlabVideoDataset,
    OpticalFlowVideoDataset
)
from algorithms.diffusion_forcing import DiffusionForcingVideo, DiffusionForcingFlow
from .exp_base import BaseLightningExperiment


class FlowPredictionExperiment(BaseLightningExperiment):
    """
    A video prediction experiment
    """

    compatible_algorithms = dict(
        df_video=DiffusionForcingVideo,
        df_flow=DiffusionForcingFlow,
    )

    compatible_datasets = dict(
        # video datasets
        video_minecraft=MinecraftVideoDataset,
        video_dmlab=DmlabVideoDataset,
        video_optical_flow=OpticalFlowVideoDataset
    )
