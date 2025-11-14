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
from zsceval.envs.overcooked.overcooked_ai_py.agents.agent import GreedyHumanModel


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
    # parse_args 안
    parser.add_argument("--win_path_fix", action="store_true",
                        help="Windows에서 PosixPath 들어간 pickle을 안전하게 로드")

    all_args = parser.parse_args(args)
    if all_args.layout_name in OLD_LAYOUTS:
        all_args.old_dynamics = True
    else:
        all_args.old_dynamics = False
    return all_args

def extract_intention(env, agent, state):

    readable_goal = {"target_pos": {}, "direction": {}}

    player = state.players[agent.agent_index]
    holding = player.get_object().name if player.has_object() else None

    possible_motion_goals = agent.ml_action(state)
    start_pos_and_or = state.players_pos_and_or[agent.agent_index]

    proximal_goal, _ = agent.choose_motion_goal(start_pos_and_or, possible_motion_goals)

    gx, gy = proximal_goal[0]
    face_pos = (gx + player.orientation[0], gy + player.orientation[1])
    terrain = env.base_env.mdp.get_terrain_type_at_pos(face_pos)
    
    terrain_map = {'O': "onion", 'T': "tomato", 'D': "dish", 'X': "counter", 'P': "pot", 'S': "serving"}
    pos_text = terrain_map.get(terrain, f"({gx}, {gy})")
    
    readable_goal["target_pos"] = pos_text
    readable_goal["direction"] = Direction.DIRECTION_TO_NAME[proximal_goal[1]]

    terrain_to_distal = {'O': "pickup_onion", 'T': "pickup_tomato", 'D': "pickup_dish", 'X': "pickup_counter_item"}
    holding_to_distal = {"onion": "put_onion_in_pot", "tomato": "put_tomato_in_pot", "dish": "pickup_soup", "soup": "deliver_soup"}
    
    if holding is None:
        distal = terrain_to_distal.get(terrain, "move")
    else:
        distal = holding_to_distal.get(holding, "move")

    return distal, readable_goal


def choose_lowest_cost_goal(env, player, ml_actions):
    mp = env.mlp.mp
    start = player.pos_and_or

    best_cost = float('inf')
    best_goal = None

    for g in ml_actions:
        if mp.is_valid_motion_start_goal_pair(start, g):
            _, _, cost = mp.get_plan(start, g)  # fast lookup
            if cost < best_cost:
                best_cost = cost
                best_goal = g
    return best_goal


def get_counter_item_type(state, pos):
    objects = state.objects
    for obj in objects:
        if obj.position == pos:
            return obj.name
    return None


def infer_future_distal_intention(env, agent, state):
    """
    Returns one of:
        get_onion
        put_onion_in_pot
        wait_for_cook
        pick_up_soup
        deliver_soup
        get_dish
        put_item_on_counter
    """

    p = state.players[agent.agent_index]
    holding = p.get_object().name if p.has_object() else None

    # ---- 1) candidate medium-level goals ----
    ml_actions = env.mlp.ml_action_manager.get_medium_level_actions(state, p)
    if len(ml_actions) == 0:
        return "idle"

    # ---- 2) choose goal like GreedyHumanModel ----
    best_goal = choose_lowest_cost_goal(env, p, ml_actions)
    if best_goal is None:
        return "idle"

    goal_pos, goal_orient = best_goal

    # Feature the agent will face when reaching goal
    face_x = goal_pos[0] + goal_orient[0]
    face_y = goal_pos[1] + goal_orient[1]
    feature = env.base_env.mdp.get_terrain_type_at_pos((face_x, face_y))

    # --- pot interaction
    if feature == 'P':  # Pot
        pot_state = state.pot_states[0]  # assume 1 pot

        if holding == "onion":
            return "put_onion_in_pot"

        if pot_state.cooking and not pot_state.ready:
            return "wait_for_cook"

        if holding == "dish" and pot_state.ready:
            return "pick_up_soup"

        return "wait_for_cook"

    # --- ingredient dispensers
    if feature == 'O':  # onion dispenser
        return "get_onion"

    # --- dish dispenser
    if feature == 'D':
        return "get_dish"

    # --- serving station
    if feature == 'S':
        if holding == "soup":
            return "deliver_soup"
        return "move_to_serving"

    # --- counter
    if feature == 'X':
        counter_item = get_counter_item_type(state, (face_x, face_y))

        if counter_item == "onion":
            return "get_onion"
        if counter_item == "dish":
            return "get_dish"
        if counter_item == "soup":
            return "pick_up_soup"

        # holding something → likely placing it on counter
        if holding is not None:
            return "put_item_on_counter"

        return "move"

    # fallback
    return "move"


def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)

    if all_args.layout_name in ["random0", "random0_medium", "random1", "random3", "small_corridor", "unident_s",
                                "random3_large", "random3_large_n"]:
        env = Overcooked(all_args, run_dir=None)
    else:
        env = Overcooked_new(all_args, run_dir=None)

    both_agents_ob, share_obs, available_actions = env.reset()
    script_human = GreedyHumanModel(env.mlp)
    script_human.set_agent_index(agent_index=0)

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

            distal, proximal = extract_intention(env, script_human, env.base_env.state)
            print(f"distal: {distal} | proximal: {proximal}")
            # intent = infer_future_distal_intention(env, script_human, env.base_env.state)
            # print("inferred future distal intention:", intent)
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