"""
FastWAM VLN Evaluation Configuration.

Modify checkpoint_path to point to your trained FastWAM navigation model.
"""

# Eval configuration dict (standalone, no InternNav dependency required)
eval_cfg = {
    "agent": {
        "model_name": "fastwam_nav",
        "model_settings": {
            # Path to trained FastWAM checkpoint
            "checkpoint_path": "/apdcephfs_qy2/share_303214315/hunyuan/xxd/FastWAM/runs/nav_vln_1e-4/debug2/checkpoints/weights/step_XXXXX.pt",

            # Model config
            "model_config_path": "/apdcephfs_qy2/share_303214315/hunyuan/xxd/FastWAM/configs/model/fastwam_nav.yaml",

            # Text embedding cache
            "text_embedding_cache_dir": "/apdcephfs_qy2/share_303214315/hunyuan/xxd/FastWAM/text_embeds_cache/nav_vln",
            "context_len": 256,

            # Inference params
            "action_horizon": 32,
            "num_inference_steps": 20,
            "re_infer_interval": 8,
            "seed": 42,
            "device": "cuda:0",

            # Trajectory discretization params
            "step_size": 0.25,
            "turn_angle_deg": 15,
            "lookahead": 4,
        },
    },
    "env": {
        "env_type": "habitat",
        "env_settings": {
            "config_path": "configs/vln_r2r.yaml",
        },
    },
    "eval_type": "habitat_vln",
    "eval_settings": {
        "output_path": "./logs/fastwam_nav_eval",
        "save_video": False,
        "epoch": 0,
        "max_steps_per_episode": 500,
    },
}

# ─── Server args (used by fastwam_server.py) ──────────────────────────────────
# When running server/client mode, the server auto-detects the latest stop head
# checkpoint from:
#   /apdcephfs_tj5/share_302528826/xxd/nav_vln_1e-4/stop_head/
#
# You can override by passing --stop_head_checkpoint explicitly:
#   python fastwam_server.py \
#       --checkpoint <main_ckpt> \
#       --stop_head_checkpoint /path/to/specific_stop_head.pt \
#       --stop_head_threshold 0.5 \
#       --stop_head_ensemble \
#       --port 9527
#
# Or just launch without --stop_head_checkpoint to use auto-detection:
#   python fastwam_server.py --checkpoint <ckpt> --port 9527
