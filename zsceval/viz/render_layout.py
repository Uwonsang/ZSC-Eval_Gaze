import sys
import numpy as np
import pygame
from zsceval.config import get_config
from PIL import Image

from zsceval.envs.overcooked.Overcooked_Env import Overcooked
from zsceval.envs.overcooked_new.Overcooked_Env import Overcooked as Overcooked_new
from zsceval.overcooked_config import get_overcooked_args

def parse_args(args, parser):
    parser = get_overcooked_args(parser)
    parser.add_argument(
        "--use_phi",
        default=False,
        action="store_true",
        help="While existing other agent like planning or human model, use an index to fix the main RL-policy agent.",
    )

    parser.add_argument("--test_policy_name", type=str, default="fcp", choices=["fcp", "mep", "traj", "hsp", "sp",
                                                                                "e3t", "cole"])
    parser.add_argument("--model_seed", type=int, default=1, choices=[1, 2, 3])
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--epsilon", type=float, default=0.0, help="stochastic eval epsilon")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--is_cam", type=str, default="False", choices=["ArgMax", "Whole", "Attention", "False"], help="Whether to use CAM")
    parser.add_argument("--cam_alpha", type=float, default=0.8)
    parser.add_argument("--cam_layers", type=str, default="2", help="'0, 1 ,2' or 'all'")
    # parse_args 안
    parser.add_argument("--win_path_fix", action="store_true",
                        help="Windows에서 PosixPath 들어간 pickle을 안전하게 로드")


    all_args = parser.parse_args(args)
    if all_args.layout_name in ["random0", "random0_medium", "random1", "random3", "small_corridor", "unident_s"]:
        all_args.old_dynamics = True
    else:
        all_args.old_dynamics = False
    return all_args



def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)
    out_path = "test.png"

    if all_args.layout_name in ["random0", "random0_medium", "random1",
                                "random3", "small_corridor", "unident_s",
                                "random3_large"]:
        env = Overcooked(all_args, run_dir=None)
    else:
        env = Overcooked_new(all_args, run_dir=None)

    both_agents_ob, share_obs, available_actions = env.reset()
    image_bgr = env.play_render()
    image_rgb = image_bgr[..., ::-1]
    Image.fromarray(image_rgb).save(out_path)
    print("saved:", out_path)


if __name__ == "__main__":
    main(sys.argv[1:])