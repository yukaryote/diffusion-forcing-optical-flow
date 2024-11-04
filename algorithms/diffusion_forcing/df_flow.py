from omegaconf import DictConfig
import torch
import torch.nn.functional as F
import numpy as np
from lightning.pytorch.utilities.types import STEP_OUTPUT
from algorithms.common.metrics import (
    FrechetInceptionDistance,
    LearnedPerceptualImagePatchSimilarity,
    FrechetVideoDistance,
)
from .df_base import DiffusionForcingBase
from .models.diffusion import Diffusion
from utils.logging_utils import get_validation_metrics_for_videos, log_flow_video
from torchvision.utils import flow_to_image
from einops import rearrange
from tqdm import tqdm
from utils.logging_utils import get_sanity_metrics


class DiffusionForcingFlow(DiffusionForcingBase):
    """
    An optical flow prediction algorithm using Diffusion Forcing.
    """

    def __init__(self, cfg: DictConfig):
        self.metrics = cfg.metrics
        self.n_tokens = (
            cfg.n_frames // cfg.frame_stack
        )  # number of max tokens for the model
        super().__init__(cfg)
        self.external_cond_dim = 3

    def _build_model(self):
        # diffusion model with 2 output channels
        self.diffusion_model = Diffusion(
            x_shape=self.x_stacked_shape,
            external_cond_dim=self.external_cond_dim,
            is_causal=self.causal,
            cfg=self.cfg.diffusion,
            out_channels=2,
        )
        self.register_data_mean_std(self.cfg.data_mean, self.cfg.data_std)

        self.validation_fid_model = (
            FrechetInceptionDistance(feature=64) if "fid" in self.metrics else None
        )
        self.validation_lpips_model = (
            LearnedPerceptualImagePatchSimilarity() if "lpips" in self.metrics else None
        )
        self.validation_fvd_model = (
            [FrechetVideoDistance()] if "fvd" in self.metrics else None
        )

    def _preprocess_batch(self, batch):
        xs = batch[0]
        batch_size, n_frames = xs.shape[:2]

        if n_frames % self.frame_stack != 0:
            raise ValueError("Number of frames must be divisible by frame stack size")
        if self.context_frames % self.frame_stack != 0:
            raise ValueError(
                "Number of context frames must be divisible by frame stack size"
            )

        masks = torch.ones(n_frames, batch_size).to(xs.device)
        n_frames = n_frames // self.frame_stack
        if self.external_cond_dim:
            conditions = batch[1]
            # conditions = torch.cat([torch.zeros_like(conditions[:, :1]), conditions[:, 1:]], 1)
            conditions = rearrange(
                conditions, "b (t fs) c ... -> t b (fs c) ...", fs=self.frame_stack
            ).contiguous()
        else:
            conditions = [None for _ in range(n_frames)]

        xs = self._normalize_x(xs)
        xs = rearrange(
            xs, "b (t fs) c ... -> t b (fs c) ...", fs=self.frame_stack
        ).contiguous()

        return xs, conditions, masks

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        xs, conditions, masks = self._preprocess_batch(batch)

        xs_pred, loss, noised_x = self.diffusion_model(
            xs, conditions, noise_levels=self._generate_noise_levels(xs)
        )
        loss = self.reweight_loss(loss, masks)

        # log the loss
        if batch_idx % 20 == 0:
            self.log("training/loss", loss)
            for k, v in get_sanity_metrics(
                {"xs": xs, "conditions": conditions}
            ).items():
                self.log(f"sanity/input_{k}", v)

        xs = self._unstack_and_unnormalize(xs)
        xs_pred = self._unstack_and_unnormalize(xs_pred)  # (t, b, 2, h, w)
        noised_x = self._unstack_and_unnormalize(noised_x)

        output_dict = {
            "loss": loss,
            "xs_pred": xs_pred,
            "xs": xs,
        }

        # log the video
        if batch_idx % 5000 == 0 and self.logger:
            # since xs_pred has shape (t, b, 2, h, w) and flow_to_image expects (b, 2, h, w)
            # we need to iterate over the first dimension
            pred_flow = []
            gt_flow = []
            noised_x_flow = []
            for i in range(len(xs_pred)):
                H, W = xs.shape[-2], xs.shape[-1]
                pred_img = flow_to_image(xs_pred[i])
                pred_img = torch.nn.functional.interpolate(pred_img, (H, W))

                gt_img = flow_to_image(xs[i])
                gt_img = torch.nn.functional.interpolate(gt_img, (H, W))

                noised_img = flow_to_image(noised_x[i])
                noised_img = torch.nn.functional.interpolate(noised_img, (H, W))

                pred_flow.append(pred_img)
                gt_flow.append(gt_img)
                noised_x_flow.append(noised_img)

            pred_flow = torch.stack(pred_flow, 0)
            gt_flow = torch.stack(gt_flow, 0)
            noised_x_flow = torch.stack(noised_x_flow, 0)

            log_flow_video(
                conditions,
                pred_flow,
                gt_flow,
                noised_x_flow,
                step=self.global_step,
                namespace="training_vis",
                logger=self.logger.experiment,
            )
        return output_dict

    @torch.no_grad()
    def validation_step(self, batch, batch_idx, namespace="validation") -> STEP_OUTPUT:

        xs, conditions, masks = self._preprocess_batch(batch)

        n_frames, batch_size, *_ = xs.shape
        xs_pred = []
        curr_frame = 0

        # context
        n_context_frames = self.context_frames // self.frame_stack
        xs_pred = xs[:n_context_frames].clone()
        curr_frame += n_context_frames

        xs_pred_noise = xs_pred.clone()

        pbar = tqdm(total=n_frames, initial=curr_frame, desc="Sampling")
        while curr_frame < n_frames:
            if self.chunk_size > 0:
                horizon = min(n_frames - curr_frame, self.chunk_size)
            else:
                horizon = n_frames - curr_frame
            assert horizon <= self.n_tokens, "horizon exceeds the number of tokens."
            scheduling_matrix = self._generate_scheduling_matrix(horizon)

            chunk = torch.randn(
                (horizon, batch_size, *self.x_stacked_shape), device=self.device
            )
            chunk = torch.clamp(chunk, -self.clip_noise, self.clip_noise)
            xs_pred = torch.cat([xs_pred, chunk], 0)
            xs_pred_noise = xs_pred.clone()

            # sliding window: only input the last n_tokens frames
            start_frame = max(0, curr_frame + horizon - self.n_tokens)

            pbar.set_postfix(
                {
                    "start": start_frame,
                    "end": curr_frame + horizon,
                }
            )

            for m in range(scheduling_matrix.shape[0] - 1):
                from_noise_levels = np.concatenate(
                    (np.zeros((curr_frame,), dtype=np.int64), scheduling_matrix[m])
                )[:, None].repeat(batch_size, axis=1)
                to_noise_levels = np.concatenate(
                    (
                        np.zeros((curr_frame,), dtype=np.int64),
                        scheduling_matrix[m + 1],
                    )
                )[:, None].repeat(batch_size, axis=1)

                from_noise_levels = torch.from_numpy(from_noise_levels).to(self.device)
                to_noise_levels = torch.from_numpy(to_noise_levels).to(self.device)

                # import pdb; pdb.set_trace()

                # update xs_pred by DDIM or DDPM sampling
                # input frames within the sliding window
                xs_pred[start_frame:] = self.diffusion_model.sample_step(
                    xs_pred[start_frame:],
                    conditions[start_frame : curr_frame + horizon],
                    from_noise_levels[start_frame:],
                    to_noise_levels[start_frame:],
                )

            curr_frame += horizon
            pbar.update(horizon)

        # FIXME: loss
        loss = F.mse_loss(xs_pred, xs, reduction="none")
        loss = self.reweight_loss(loss, masks)

        xs = self._unstack_and_unnormalize(xs)
        xs_pred = self._unstack_and_unnormalize(xs_pred)
        xs_pred_noise = self._unstack_and_unnormalize(xs_pred_noise)

        self.validation_step_outputs.append(
            (
                xs_pred.detach().cpu(),
                xs.detach().cpu(),
                conditions.detach().cpu(),
                xs_pred_noise.detach().cpu(),
            )
        )

        return loss

    def on_validation_epoch_end(self, namespace="validation") -> None:
        if not self.validation_step_outputs:
            return
        xs_pred = []
        xs = []
        cond = []
        noisy = []
        for pred, gt, rgb_img, noisy_input in self.validation_step_outputs:
            H, W = rgb_img.shape[-2], rgb_img.shape[-1]

            # turn pred flow into image
            pred_flow = []
            gt_imgs = []
            noisy_imgs = []
            for i in range(len(pred)):
                pred_img = flow_to_image(pred[i])
                pred_img = torch.nn.functional.interpolate(pred_img, (H, W))
                pred_flow.append(pred_img)

                gt_img = flow_to_image(gt[i])
                gt_img = torch.nn.functional.interpolate(gt_img, (H, W))
                gt_imgs.append(gt_img)

                noisy_img = flow_to_image(noisy_input[i])
                noisy_img = torch.nn.functional.interpolate(noisy_img, (H, W))
                noisy_imgs.append(noisy_img)

            xs_pred.append(torch.stack(pred_flow, 0))
            xs.append(torch.stack(gt_imgs, 0))
            cond.append(rgb_img)
            noisy.append(torch.stack(noisy_imgs, 0))
        xs_pred = torch.cat(xs_pred, 1)
        xs = torch.cat(xs, 1)
        cond = torch.cat(cond, 1)
        noisy = torch.cat(noisy, 1)

        if self.logger:
            log_flow_video(
                cond,
                xs_pred,
                xs,
                noisy,
                step=None if namespace == "test" else self.global_step,
                namespace=namespace + "_vis",
                context_frames=self.context_frames,
                logger=self.logger.experiment,
            )

        metric_dict = get_validation_metrics_for_videos(
            xs_pred[self.context_frames :].float(),
            xs[self.context_frames :].float(),
            lpips_model=self.validation_lpips_model,
            fid_model=self.validation_fid_model,
            fvd_model=(
                self.validation_fvd_model[0] if self.validation_fvd_model else None
            ),
        )
        self.log_dict(
            {f"{namespace}/{k}": v for k, v in metric_dict.items()},
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        self.validation_step_outputs.clear()

    def forward(self, input):
        xs, conditions, masks = self._preprocess_batch(input)

        xs_pred, loss = self.diffusion_model(
            xs, conditions, noise_levels=self._generate_noise_levels(xs)
        )

        xs = self._unstack_and_unnormalize(xs)
        xs_pred = self._unstack_and_unnormalize(xs_pred)  # (t, b, 2, h, w)

        output_dict = {
            "loss": loss,
            "xs_pred": xs_pred,
            "xs": xs,
        }

        return output_dict
