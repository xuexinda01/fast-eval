"""
FastWAM Navigation Agent for VLN evaluation (Multi-Frame + Overhead Architecture).

This agent wraps the FastWAM model for use in the InternNav evaluation framework.
It maintains a history of recent observations, predicts continuous trajectories
using flow-matching with multi-frame conditioning + overhead view, and converts
them to discrete actions for Habitat execution.

New Architecture (2026-05-10):
- Input: 9 frames 0deg (8 history + current) + 1 frame 30deg (overhead) + text
- Condition: 9 frames → 3 latent frames (frozen)
- Overhead: encoded separately, concat along channel dim
- Output: 32 action waypoints (x, y, theta, moving_flag)

Stop Head (2026-05-11, optional):
- Standalone lightweight classifier (VAE-only, no DiT needed)
- Predicts stop probability from visual history + overhead + text
- Can be used standalone or as ensemble with moving_flag from main model
- Enable by passing stop_head_checkpoint_path to FastWAMNavAgent
"""

import hashlib
import os
import sys
from typing import Optional, List

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image

# Add FastWAM source to path
FASTWAM_SRC = "/apdcephfs_tj5/share_302528826/xxd/FastWAM/src"
FASTWAM_QY2_SCRIPTS = "/apdcephfs_qy2/share_303214315/hunyuan/xxd/FastWAM/scripts"
sys.path.insert(0, FASTWAM_SRC)
sys.path.insert(0, FASTWAM_QY2_SCRIPTS)

from traj_utils import STOP, fastwam_traj_to_actions, fastwam_traj_to_waypoint


class FastWAMNavAgent:
    """
    FastWAM-based navigation agent with multi-frame history + overhead conditioning.

    Maintains a buffer of recent RGB observations and uses them as condition frames
    for the flow-matching action prediction model.

    Optional standalone stop head: pass stop_head_checkpoint_path to enable.
    Stop decision logic (when stop head is loaded):
        final_stop = (moving_flag_stop) OR (stop_head_prob > stop_head_threshold)
    """

    def __init__(
        self,
        checkpoint_path: str,
        model_config_path: str = "/apdcephfs_tj5/share_302528826/xxd/FastWAM/configs/model/fastwam_nav.yaml",
        text_embedding_cache_dir: str = "/apdcephfs_qy2/share_303214315/hunyuan/xxd/FastWAM/text_embeds_cache/nav_vln",
        context_len: int = 256,
        action_horizon: int = 32,
        num_inference_steps: int = 20,
        re_infer_interval: int = 1,
        n_history_frames: int = 8,
        device: str = "cuda:0",
        seed: int = 42,
        # Trajectory discretization params
        step_size: float = 0.25,
        turn_angle_deg: float = 15,
        lookahead: int = 4,
        # Inference mode: False = discrete actions, True = continuous polar waypoint
        waypoint_mode: bool = False,
        # Optional standalone stop head
        stop_head_checkpoint_path: str = None,
        stop_head_vae_path: str = "/tmp/fastwam_checkpoints",
        stop_head_threshold: float = 0.5,
        stop_head_ensemble: bool = True,  # True = OR(moving_flag, stop_head); False = stop_head only
    ):
        self.checkpoint_path = checkpoint_path
        self.text_embedding_cache_dir = text_embedding_cache_dir
        self.context_len = context_len
        self.action_horizon = action_horizon
        self.num_inference_steps = num_inference_steps
        self.re_infer_interval = re_infer_interval
        self.n_history_frames = n_history_frames
        self.device = torch.device(device)
        self.seed = seed
        self.step_size = step_size
        self.turn_angle_deg = turn_angle_deg
        self.lookahead = lookahead
        self.waypoint_mode = waypoint_mode
        self.stop_head_threshold = stop_head_threshold
        self.stop_head_ensemble = stop_head_ensemble

        # Load main model
        self._load_model(model_config_path)

        # Load optional standalone stop head
        self.stop_head = None
        if stop_head_checkpoint_path and os.path.exists(stop_head_checkpoint_path):
            self._load_stop_head(stop_head_checkpoint_path, stop_head_vae_path)
        elif stop_head_checkpoint_path:
            print(f"[StopHead] WARNING: checkpoint not found at {stop_head_checkpoint_path}, stop head disabled.")

        # Runtime state
        self.episode_step = 0
        self.action_queue = []
        self.current_instruction = None
        self.frame_history: List[torch.Tensor] = []  # List of [3, 224, 224] tensors in [0,1]

    def _load_model(self, model_config_path: str):
        """Load FastWAM model from config and checkpoint."""
        from omegaconf import OmegaConf
        from fastwam.runtime import create_fastwam

        # Set DIFFSYNTH_MODEL_BASE_PATH so loader finds local checkpoints
        fastwam_project_dir = "/apdcephfs_tj5/share_302528826/xxd/FastWAM"
        os.environ["DIFFSYNTH_MODEL_BASE_PATH"] = "/apdcephfs_tj5/share_302528826/xxd/fastwam_checkpoints"
        os.environ["DIFFSYNTH_SKIP_DOWNLOAD"] = "true"

        # Load model config
        model_cfg = OmegaConf.load(model_config_path)

        # Resolve interpolation references manually (since we're outside hydra)
        OmegaConf.update(model_cfg, "video_dit_config.use_gradient_checkpointing", False)
        OmegaConf.update(model_cfg, "action_dit_config.use_gradient_checkpointing", False)

        model_cfg = OmegaConf.to_container(model_cfg, resolve=True)

        # Remove _target_ key (not needed for direct call)
        model_cfg.pop('_target_', None)
        model_cfg.pop('mot_checkpoint_mixed_attn', None)

        # Fix action_dit_pretrained_path if it points to non-existent location
        adt_path = model_cfg.get('action_dit_pretrained_path', '')
        if adt_path and not os.path.exists(adt_path):
            fallback = "/apdcephfs_tj5/share_302528826/xxd/fastwam_checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt"
            if os.path.exists(fallback):
                print(f"action_dit_pretrained_path not found at {adt_path}, using fallback: {fallback}")
                model_cfg['action_dit_pretrained_path'] = fallback

        # Create model
        self.model = create_fastwam(
            model_dtype=torch.bfloat16,
            device=str(self.device),
            **model_cfg,
        )

        # Load trained checkpoint
        if self.checkpoint_path and os.path.exists(self.checkpoint_path):
            print(f"Loading checkpoint: {self.checkpoint_path}")
            self.model.load_checkpoint(self.checkpoint_path)
        else:
            print(f"WARNING: No checkpoint loaded! Using pretrained weights only.")

        self.model.eval()
        print(f"FastWAM model loaded on {self.device}")

    def _load_stop_head(self, checkpoint_path: str, vae_path: str = None):
        """
        Load the standalone stop head checkpoint.
        Imports StopPredictor from the qy2 FastWAM scripts dir.
        Reuses the VAE already loaded by the main model (self.model.vae) instead
        of calling load_vae() a second time — both use the same frozen Wan VAE.
        vae_path is kept for API compatibility but is no longer used.
        """
        try:
            from train_stop_head import StopPredictor
        except ImportError:
            print(f"[StopHead] ERROR: Cannot import from train_stop_head. "
                  f"Ensure {FASTWAM_QY2_SCRIPTS} is in sys.path.")
            return

        print(f"[StopHead] Loading checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        args_dict = ckpt.get("args", {})
        video_feat_dim = args_dict.get("video_feat_dim", 512)
        overhead_feat_dim = args_dict.get("overhead_feat_dim", 256)
        hidden_dim = args_dict.get("hidden_dim", 256)
        dropout = args_dict.get("dropout", 0.1)

        # Reuse the VAE already loaded by the main model — no second load_vae() call needed.
        # Both models use the same frozen pretrained Wan VAE, so sharing is safe.
        vae = self.model.vae
        model = StopPredictor(
            vae=vae,
            text_dim=4096,
            vae_latent_dim=48,
            video_feat_dim=video_feat_dim,
            overhead_feat_dim=overhead_feat_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
        model_state = ckpt["model_state_dict"]
        model.video_pool_proj.load_state_dict(model_state["video_pool_proj"])
        model.overhead_pool_proj.load_state_dict(model_state["overhead_pool_proj"])
        model.stop_head.load_state_dict(model_state["stop_head"])

        model = model.to(self.device)
        model.eval()
        self.stop_head = model

        metrics = ckpt.get("metrics", {})
        print(f"[StopHead] Loaded successfully. "
              f"threshold={self.stop_head_threshold} (stop if prob > {self.stop_head_threshold}) "
              f"ensemble={self.stop_head_ensemble} "
              f"Metrics: {metrics}")

    @torch.no_grad()
    def _predict_stop_head(
        self,
        current_frame: torch.Tensor,  # [3, 224, 224] in [0,1]
        overhead: Optional[torch.Tensor],  # [3, 224, 224] in [0,1] or None
        instruction: str,
    ) -> float:
        """
        Run standalone stop head inference.
        Returns stop probability in [0, 1].
        """
        if self.stop_head is None:
            return 0.0

        # Build 9-frame video tensor in [-1, 1]
        if len(self.frame_history) == 0:
            history = [current_frame.clone() for _ in range(self.n_history_frames)]
        elif len(self.frame_history) < self.n_history_frames:
            indices = np.linspace(0, len(self.frame_history) - 1, self.n_history_frames).astype(int)
            history = [self.frame_history[i] for i in indices]
        else:
            indices = np.linspace(0, len(self.frame_history) - 1, self.n_history_frames).astype(int)
            history = [self.frame_history[i] for i in indices]

        frames = history + [current_frame]  # 9 frames, each [3, 224, 224] in [0,1]
        video = torch.stack(frames, dim=1) * 2.0 - 1.0  # [3, 9, 224, 224] in [-1,1]
        video = video.unsqueeze(0).to(device=self.device, dtype=torch.bfloat16)  # [1, 3, 9, 224, 224]

        # Overhead
        if overhead is not None:
            ovhd = overhead * 2.0 - 1.0  # [3, 224, 224]
        else:
            ovhd = current_frame.clone() * 2.0 - 1.0
        ovhd = ovhd.unsqueeze(0).to(device=self.device, dtype=torch.bfloat16)  # [1, 3, 224, 224]

        # Text context
        context, context_mask = self._get_text_context(instruction)
        context = context.to(device=self.device, dtype=torch.bfloat16)
        context_mask = context_mask.to(device=self.device)

        # Batch dims
        logits = self.stop_head(video, ovhd, context, context_mask)  # [1, 1]
        prob = torch.sigmoid(logits).item()
        return prob

    def reset(self):
        """Reset agent state for a new episode."""
        self.episode_step = 0
        self.action_queue = []
        self.current_instruction = None
        self.frame_history = []

    def _get_text_context(self, instruction: str):
        """
        Get text embedding for instruction.
        First tries cached embeddings, falls back to model's text encoder.
        """
        prompt = f"A video recorded from a navigation agent's point of view executing the following instruction: {instruction}"

        # Try cache first
        if self.text_embedding_cache_dir:
            hashed = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
            cache_path = os.path.join(
                self.text_embedding_cache_dir,
                f"{hashed}.t5_len{self.context_len}.wan22ti2v5b.pt"
            )
            if os.path.exists(cache_path):
                payload = torch.load(cache_path, map_location="cpu")
                context = payload["context"].unsqueeze(0)  # [1, L, D]
                context_mask = payload["mask"].bool().unsqueeze(0)  # [1, L]
                return context, context_mask

        # Fallback: use model's text encoder
        if self.model.text_encoder is not None:
            context, context_mask = self.model.encode_prompt(prompt)
            return context, context_mask

        # Last resort: zeros
        print(f"WARNING: No text embedding available for: {instruction[:50]}...")
        context = torch.zeros(1, self.context_len, 4096)
        context_mask = torch.ones(1, self.context_len, dtype=torch.bool)
        return context, context_mask

    def _preprocess_frame(self, rgb: np.ndarray) -> torch.Tensor:
        """
        Preprocess a single RGB frame to [3, 224, 224] tensor in [0, 1].

        Args:
            rgb: [H, W, 3] uint8 RGB image.

        Returns:
            tensor: [3, 224, 224] float tensor in [0, 1].
        """
        tensor = TF.to_tensor(Image.fromarray(rgb).convert("RGB"))
        tensor = TF.resize(
            tensor, [224, 224],
            interpolation=TF.InterpolationMode.BILINEAR,
            antialias=True,
        )
        return tensor  # [3, 224, 224] in [0, 1]

    def _build_condition_video(self, current_frame: torch.Tensor) -> torch.Tensor:
        """
        Build 9-frame condition video from history + current frame.

        If fewer than 8 history frames are available, duplicate the earliest frame.

        Args:
            current_frame: [3, 224, 224] tensor in [0, 1]

        Returns:
            video: [1, 3, 9, 224, 224] tensor in [-1, 1] (for VAE encoding)
        """
        # Build history: 8 uniformly sampled frames from frame_history
        if len(self.frame_history) == 0:
            # No history yet: duplicate current frame 8 times
            history = [current_frame.clone() for _ in range(self.n_history_frames)]
        elif len(self.frame_history) < self.n_history_frames:
            # Fewer than 8 history frames: uniformly sample with repetition
            indices = np.linspace(0, len(self.frame_history) - 1, self.n_history_frames).astype(int)
            history = [self.frame_history[i] for i in indices]
        else:
            # Full history: uniformly sample 8 frames
            indices = np.linspace(0, len(self.frame_history) - 1, self.n_history_frames).astype(int)
            history = [self.frame_history[i] for i in indices]

        # Condition frames: history(8) + current(1) = 9 frames
        condition_frames = history + [current_frame]  # 9 frames, each [3, 224, 224]

        # For inference, we only need condition frames (no future frames to generate)
        # But model expects T%4==1, so 9 frames satisfies this (9%4==1 ✓)
        video = torch.stack(condition_frames, dim=0)  # [9, 3, 224, 224]
        video = video * 2.0 - 1.0  # [0,1] → [-1,1]
        video = video.permute(1, 0, 2, 3)  # [3, 9, 224, 224]
        return video.unsqueeze(0)  # [1, 3, 9, 224, 224]

    @torch.no_grad()
    def predict_trajectory(self, rgb: np.ndarray, instruction: str, rgb_down: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Run FastWAM inference to predict continuous trajectory.

        Uses multi-frame conditioning (9 frames) + overhead view.

        Args:
            rgb: [H, W, 3] uint8 forward-facing RGB (current observation).
            instruction: Navigation instruction text.
            rgb_down: [H, W, 3] uint8 downward-facing RGB (30° overhead).

        Returns:
            trajectory: [action_horizon, 3] numpy array of (x, y, theta).
        """
        # Preprocess current frame
        current_frame = self._preprocess_frame(rgb)  # [3, 224, 224] in [0, 1]

        # Build condition video: [1, 3, 9, 224, 224] in [-1, 1]
        condition_video = self._build_condition_video(current_frame).to(self.device)

        # Preprocess overhead frame
        if rgb_down is not None:
            overhead = self._preprocess_frame(rgb_down)  # [3, 224, 224] in [0, 1]
        else:
            overhead = current_frame.clone()  # Duplicate forward if no overhead
        overhead = overhead * 2.0 - 1.0  # [-1, 1]
        overhead = overhead.unsqueeze(0).to(self.device)  # [1, 3, 224, 224]

        # Prepare text context
        context, context_mask = self._get_text_context(instruction)
        context = context.to(self.device)
        context_mask = context_mask.to(self.device)

        # --- Run inference using model internals ---
        # Since infer_action doesn't support multi-frame + overhead yet,
        # we call the model's internal methods directly

        # 1. Encode condition video through VAE
        condition_video_bf16 = condition_video.to(dtype=self.model.torch_dtype)
        input_latents = self.model._encode_video_latents(condition_video_bf16, tiled=False)
        # input_latents: [1, z_dim, T_lat, H', W'] where T_lat = (9+3)//4 = 3

        n_cond_latent_frames = 3  # (9+3)//4 = 3
        condition_latents = input_latents[:, :, :n_cond_latent_frames].clone()

        # 2. Encode overhead through VAE
        overhead_bf16 = overhead.to(dtype=self.model.torch_dtype)
        overhead_video = overhead_bf16.unsqueeze(2)  # [1, 3, 1, 224, 224]
        overhead_latent = self.model._encode_video_latents(overhead_video, tiled=False)
        # overhead_latent: [1, z_dim, 1, H', W']

        # 3. Prepare for video expert: concat overhead along channel dim
        # Broadcast overhead to match temporal extent of condition
        overhead_broadcast = overhead_latent.expand(-1, -1, n_cond_latent_frames, -1, -1)
        # Concat: [1, 2*z_dim, 3, H', W']
        latents_with_overhead = torch.cat([condition_latents, overhead_broadcast], dim=1)

        # 4. Run video expert pre_dit to get cached video tokens
        fuse_flag = bool(getattr(self.model.video_expert, "fuse_vae_embedding_in_latents", False))

        context_bf16 = context.to(dtype=self.model.torch_dtype)
        context_mask_bool = context_mask.to(dtype=torch.bool)

        # Set timestep=0 for all condition frames (they're frozen/known)
        timestep_video = torch.zeros(
            (1,), dtype=self.model.torch_dtype, device=self.device
        )

        video_pre = self.model.video_expert.pre_dit(
            x=latents_with_overhead,
            timestep=timestep_video,
            context=context_bf16,
            context_mask=context_mask_bool,
            action=None,
            fuse_vae_embedding_in_latents=fuse_flag,
            n_cond_latent_frames=n_cond_latent_frames,
        )

        video_seq_len = int(video_pre["tokens"].shape[1])

        # 5. Initialize action latents (noise)
        generator = torch.Generator(device="cpu").manual_seed(self.seed)
        latents_action = torch.randn(
            (1, self.action_horizon, self.model.action_expert.action_dim),
            generator=generator,
            device="cpu",
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.model.torch_dtype)

        # 6. Build attention mask
        attention_mask = self.model._build_mot_attention_mask(
            video_seq_len=video_seq_len,
            action_seq_len=latents_action.shape[1],
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=video_pre["tokens"].device,
            n_cond_latent_frames=n_cond_latent_frames,
        )

        # 7. Prefill video KV cache
        video_kv_cache = self.model.mot.prefill_video_cache(
            video_tokens=video_pre["tokens"],
            video_freqs=video_pre["freqs"],
            video_t_mod=video_pre["t_mod"],
            video_context_payload={
                "context": video_pre["context"],
                "mask": video_pre["context_mask"],
            },
            video_attention_mask=attention_mask[:video_seq_len, :video_seq_len],
        )

        # 8. Diffusion loop for action prediction
        infer_timesteps, infer_deltas = self.model.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=self.num_inference_steps,
            device=self.device,
            dtype=latents_action.dtype,
        )

        for step_t, step_delta in zip(infer_timesteps, infer_deltas):
            timestep_action = step_t.unsqueeze(0).to(dtype=latents_action.dtype, device=self.device)

            pred_action = self.model._predict_action_noise_with_cache(
                latents_action=latents_action,
                timestep_action=timestep_action,
                context=context_bf16,
                context_mask=context_mask_bool,
                video_kv_cache=video_kv_cache,
                attention_mask=attention_mask,
                video_seq_len=video_seq_len,
            )

            latents_action = self.model.infer_action_scheduler.step(pred_action, step_delta, latents_action)

        # 9. Return trajectory
        trajectory = latents_action[0].detach().to(device="cpu", dtype=torch.float32).numpy()

        # Add current frame to history for next step
        self.frame_history.append(current_frame)

        return trajectory  # [32, 4] (x, y, theta, moving_flag)

    def step(self, obs: dict) -> dict:
        """
        Agent step function compatible with InternNav evaluation framework.

        Args:
            obs: dict with keys 'rgb', 'rgb_down', 'depth', 'instruction'.
                 rgb: [H, W, 3] uint8 (forward camera)
                 rgb_down: [H, W, 3] uint8 (30° downward camera, optional)
                 depth: [H, W] or [H, W, 1] float
                 instruction: str

        Returns:
            dict with 'action': list of int, 'trajectory': list, optional 'stop_prob': float.
        """
        rgb = obs['rgb']
        rgb_down = obs.get('rgb_down', None)
        instruction = obs.get('instruction', self.current_instruction or '')
        self.current_instruction = instruction

        # Preprocess current frame (needed for stop head before predict_trajectory updates history)
        current_frame = self._preprocess_frame(rgb)  # [3, 224, 224] in [0, 1]
        overhead_frame = self._preprocess_frame(rgb_down) if rgb_down is not None else None

        # Run stop head (uses history BEFORE this step, same as training convention)
        stop_prob = 0.0
        stop_head_says_stop = False
        if self.stop_head is not None:
            stop_prob = self._predict_stop_head(current_frame, overhead_frame, instruction)
            stop_head_says_stop = stop_prob > self.stop_head_threshold
            print(f"[StopHead] step={self.episode_step} stop_prob={stop_prob:.3f} "
                  f"threshold={self.stop_head_threshold} "
                  f"stop={'YES' if stop_head_says_stop else 'no'}")

        # Run main trajectory inference (also appends current_frame to frame_history)
        trajectory = self.predict_trajectory(rgb, instruction, rgb_down=rgb_down)

        # Convert trajectory to action(s)
        if self.waypoint_mode:
            # --- Waypoint mode: continuous polar-coordinate target ---
            # Returns np.array([x, y]) in agent-local frame, or None for STOP.
            waypoint = fastwam_traj_to_waypoint(trajectory)
            if waypoint is None:
                action = STOP
            else:
                action = waypoint  # caller receives a continuous (x, y) target
        else:
            # --- Discrete action mode: LEFT / RIGHT / FORWARD / STOP ---
            all_actions = fastwam_traj_to_actions(
                trajectory,
                step_size=self.step_size,
                turn_angle_deg=self.turn_angle_deg,
                lookahead=self.lookahead,
            )
            action = all_actions[0] if all_actions else STOP

        # Ensemble stop decision:
        # - stop_head_ensemble=True:  STOP if moving_flag triggers OR stop head triggers
        # - stop_head_ensemble=False: STOP only from stop head (overrides moving_flag logic)
        if self.stop_head is not None:
            if self.stop_head_ensemble:
                # moving_flag already handled inside traj conversion (returns STOP / None)
                # additionally, if stop head says stop, force STOP
                if stop_head_says_stop:
                    action = STOP
            else:
                # Stop head is the sole stop signal; ignore moving_flag stop
                action = STOP if stop_head_says_stop else action

        self.episode_step += 1

        # Build result dict — action format depends on mode:
        #   discrete mode:  {'action': [int]}
        #   waypoint mode:  {'action': np.array([x, y])}  or  {'action': [STOP]}
        if self.waypoint_mode:
            if isinstance(action, np.ndarray):
                result = {'action': action, 'trajectory': trajectory.tolist()}
            else:
                # STOP signal (int 0)
                result = {'action': [STOP], 'trajectory': trajectory.tolist()}
        else:
            result = {'action': [action], 'trajectory': trajectory.tolist()}
        if self.stop_head is not None:
            result['stop_prob'] = stop_prob
        return result
