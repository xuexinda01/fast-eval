"""
FastWAM VLN Evaluation Script.

Standalone evaluation entry point. Can be used:
1. Standalone mode: directly loads model and runs inference on images
2. Habitat mode: integrates with InternNav evaluation framework

Usage:
    # Standalone test (no Habitat needed):
    python eval.py --mode standalone --image /path/to/image.jpg --instruction "Go to the kitchen"

    # Habitat evaluation (requires InternNav + Habitat):
    python eval.py --mode habitat --config configs/fastwam_nav_cfg.py
"""

import argparse
import importlib.util
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def parse_args():
    parser = argparse.ArgumentParser(description="FastWAM VLN Evaluation")
    parser.add_argument(
        "--mode",
        type=str,
        default="standalone",
        choices=["standalone", "habitat"],
        help="Evaluation mode: standalone (single image test) or habitat (full VLN eval)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/fastwam_nav_cfg.py",
        help="Config file path (for habitat mode)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint path override",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Image path (for standalone mode)",
    )
    parser.add_argument(
        "--instruction",
        type=str,
        default="Walk forward and turn left into the hallway.",
        help="Navigation instruction (for standalone mode)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to run inference on",
    )
    return parser.parse_args()


def load_config(config_path):
    """Load evaluation config from Python file."""
    spec = importlib.util.spec_from_file_location("eval_config", config_path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    return config_module.eval_cfg


def run_standalone(args):
    """
    Standalone mode: Load model, run inference on a single image, print results.
    No Habitat dependency required.
    """
    from fastwam_agent import FastWAMNavAgent
    from traj_utils import STOP, FORWARD, LEFT, RIGHT

    ACTION_NAMES = {STOP: "STOP", FORWARD: "FORWARD", LEFT: "LEFT", RIGHT: "RIGHT"}

    # Load config
    cfg = load_config(args.config)
    model_settings = cfg["agent"]["model_settings"]

    # Override with command line args
    if args.checkpoint:
        model_settings["checkpoint_path"] = args.checkpoint
    if args.device:
        model_settings["device"] = args.device

    # Create agent
    print("=" * 60)
    print("FastWAM VLN Agent - Standalone Test")
    print("=" * 60)
    agent = FastWAMNavAgent(**model_settings)
    agent.reset()

    # Prepare input
    if args.image and os.path.exists(args.image):
        from PIL import Image
        img = np.array(Image.open(args.image).convert("RGB"))
        print(f"Loaded image: {args.image} ({img.shape})")
    else:
        # Generate dummy image for testing
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        print("Using random dummy image (480x640)")

    instruction = args.instruction
    print(f"Instruction: {instruction}")
    print()

    # Run inference
    print("Running FastWAM inference...")
    trajectory = agent.predict_trajectory(img, instruction)
    print(f"Predicted trajectory shape: {trajectory.shape}")
    print(f"Trajectory (first 5 steps):")
    for i in range(min(5, len(trajectory))):
        print(f"  step {i}: x={trajectory[i, 0]:.4f}, y={trajectory[i, 1]:.4f}, theta={trajectory[i, 2]:.4f}")
    print(f"  ...")
    print(f"  step {len(trajectory)-1}: x={trajectory[-1, 0]:.4f}, y={trajectory[-1, 1]:.4f}, theta={trajectory[-1, 2]:.4f}")
    print()

    # Convert to discrete actions
    from traj_utils import fastwam_traj_to_actions
    actions = fastwam_traj_to_actions(trajectory)
    action_str = " → ".join([ACTION_NAMES[a] for a in actions[:20]])
    if len(actions) > 20:
        action_str += f" ... ({len(actions)} total)"
    print(f"Discrete actions ({len(actions)} steps): {action_str}")
    print()
    print("=" * 60)
    print("Standalone test complete!")
    print("=" * 60)


def run_habitat(args):
    """
    Habitat mode: Full VLN evaluation using InternNav framework.
    Requires Habitat + InternNav installation.
    """
    # Add InternNav to path
    INTERNNAV_PATH = "/apdcephfs_tj5/share_302528826/zhenye/code/evaluation/InternNav"
    sys.path.insert(0, INTERNNAV_PATH)

    from fastwam_agent import FastWAMNavAgent

    cfg = load_config(args.config)
    model_settings = cfg["agent"]["model_settings"]

    if args.checkpoint:
        model_settings["checkpoint_path"] = args.checkpoint

    # Create agent
    agent = FastWAMNavAgent(**model_settings)

    # Import and run habitat evaluator
    try:
        from internnav.evaluator import Evaluator
        from internnav.configs.evaluator import EvalCfg, EnvCfg
        from internnav.configs.agent import AgentCfg

        # TODO: Register FastWAM agent with InternNav framework
        # and run full Habitat evaluation
        print("Habitat evaluation mode - TODO: integrate with InternNav evaluator")
        print("For now, use standalone mode to verify model inference.")
    except ImportError as e:
        print(f"ERROR: Cannot import InternNav/Habitat: {e}")
        print("Please ensure InternNav and Habitat are installed.")
        print("Falling back to standalone mode...")
        run_standalone(args)


def main():
    args = parse_args()

    if args.mode == "standalone":
        run_standalone(args)
    elif args.mode == "habitat":
        run_habitat(args)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
