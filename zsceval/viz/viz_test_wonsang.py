import os
import sys
import pygame
import time
from zsceval.config import get_config
from zsceval.overcooked_config import get_overcooked_args, OLD_LAYOUTS

from zsceval.envs.overcooked.Overcooked_Env import Overcooked
from zsceval.envs.overcooked_new.Overcooked_Env import Overcooked as Overcooked_new

from collections import deque
import numpy as np
path = "../policy_pool"
os.environ["POLICY_POOL"] = path

from zsceval.envs.overcooked.overcooked_ai_py.mdp.actions import Action, Direction
from zsceval.envs.overcooked.overcooked_ai_py.agents.agent import (
    GreedyHumanModel,
    CoupledPlanningAgent,
    EmbeddedPlanningAgent,
)


def parse_args(args, parser):
    parser = get_overcooked_args(parser)
    parser.add_argument(
        "--use_phi",
        default=False,
        action="store_true",
        help="While existing other agent like planning or human model, use an index to fix the main RL-policy agent.",
    )
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--epsilon", type=float, default=0.0, help="stochastic eval epsilon")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--script_agent", type=str, default="greedy", choices=["greedy", "coupled", "embedded"])
    # parse_args 안
    parser.add_argument("--win_path_fix", action="store_true",
                        help="Windows에서 PosixPath 들어간 pickle을 안전하게 로드")

    all_args = parser.parse_args(args)
    if all_args.layout_name in OLD_LAYOUTS:
        all_args.old_dynamics = True
    else:
        all_args.old_dynamics = False
    return all_args

def _build_script_agent(agent_name, mlp):

    if agent_name == "greedy":
        agent = GreedyHumanModel(mlp)
    elif agent_name == "coupled": # 동작안됨
        agent = CoupledPlanningAgent(mlp)
    elif agent_name == "embedded": # 동작안됨
        agent = EmbeddedPlanningAgent(mlp)

    return agent


def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)

    if all_args.layout_name in ["random0", "random0_medium", "random1", "random3", "small_corridor", "unident_s",
                                "random3_large", "random3_large_n"]:
        env = Overcooked(all_args, run_dir=None)
    else:
        env = Overcooked_new(all_args, run_dir=None)

    both_agents_ob, share_obs, available_actions = env.reset()

    script_human = _build_script_agent(all_args.script_agent, env.mlp)
    # script_human = GreedyHumanModel(env.mlp)

    start_time = time.time()
    clock = pygame.time.Clock()
    epi_done = False
    human_action_queue = deque(maxlen=32)

    try:
        image = env.play_render()
        screen = pygame.display.set_mode((image.shape[1], image.shape[0]))
        screen.blit(pygame.surfarray.make_surface(np.rot90(np.flip(image[..., ::-1], 1))), (0, 0))
        pygame.display.flip()

        while not epi_done:

            clock.tick(6.67)
            # enqueue keydown events
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP:
                        human_action_queue.append(Direction.NORTH)
                    elif event.key == pygame.K_DOWN:
                        human_action_queue.append(Direction.SOUTH)
                    elif event.key == pygame.K_LEFT:
                        human_action_queue.append(Direction.WEST)
                    elif event.key == pygame.K_RIGHT:
                        human_action_queue.append(Direction.EAST)
                    elif event.key == pygame.K_SPACE:
                        human_action_queue.append(Action.INTERACT)

            # a0 = random.randint(0, 5)
            script_human.set_agent_index(agent_index=0)
            a0_raw = script_human.action(env.base_env.state)
            a0 = Action.ACTION_TO_INDEX[a0_raw]

            a1_action = human_action_queue.popleft() if human_action_queue else Action.STAY
            a1 = Action.ACTION_TO_INDEX[a1_action]

            joint_action = np.array([[int(a0)], [int(a1)]])

            both_agents_ob, share_obs, reward, done, info, available_actions = env.step(joint_action)
            agent_position = env.base_env.state.players[0].position

            # render
            image = env.play_render()

            screen.blit(pygame.surfarray.make_surface(np.rot90(np.flip(image[..., ::-1], 1))), (0, 0))
            pygame.display.flip()

            end_time = time.time()
            game_time = end_time - start_time

        print('finish_time : ', game_time)
    finally:
        pygame.quit()


if __name__ == "__main__":
    main(sys.argv[1:])
