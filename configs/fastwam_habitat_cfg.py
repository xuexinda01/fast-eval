"""
FastWAM VLN Evaluation Config for Habitat.

Uses client-server architecture:
- Server (fastwam env): runs FastWAM model inference
- Client (internnaveval env): runs Habitat + InternNav evaluation framework
"""
import os
import sys
sys.path.insert(0, '/apdcephfs_tj5/share_302528826/xxd/fastwam_vln_eval')

# Register GoTowardPoint action + Hydra ConfigStore entry (must come before habitat env init)
import habitat_extensions.actions  # noqa: F401

# Import to register the agent
import fastwam_client_agent  # noqa: F401

from internnav.configs.agent import AgentCfg
from internnav.configs.evaluator import EnvCfg, EvalCfg

# Allow dynamic output_path from environment variable
output_path = os.environ.get(
    "FASTWAM_EVAL_OUTPUT_PATH",
    "/apdcephfs_tj5/share_302528826/xxd/fastwam_nav_eval_results"
)

eval_cfg = EvalCfg(
    agent=AgentCfg(
        model_name='fastwam_nav',
        model_settings={
            "server_host": "127.0.0.1",
            "server_port": 9527,
            "max_retries": 60,
        },
    ),
    env=EnvCfg(
        env_type='habitat',
        env_settings={
            'config_path': 'scripts/eval/configs/vln_r2r_xxd.yaml',
        },
    ),
    eval_type='habitat_vln',
    eval_settings={
        "output_path": output_path,
        "save_video": True,
        "epoch": 0,
        "max_steps_per_episode": 500,
        "port": "2345",
        "dist_url": "env://",
    },
)
