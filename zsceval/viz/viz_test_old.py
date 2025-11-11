import os
import sys
import numpy as np
import pygame
import random
import time
from typing import Optional, Dict, Any, List

from zsceval.config import get_config
from zsceval.overcooked_config import get_overcooked_args, OLD_LAYOUTS

from zsceval.envs.overcooked.Overcooked_Env import Overcooked
from zsceval.envs.overcooked_new.Overcooked_Env import Overcooked as Overcooked_new
import yaml
import pickle, pathlib
import torch
path = "../policy_pool"
os.environ["POLICY_POOL"] = path

from zsceval.algorithms.population.policy_pool import add_path_prefix
from zsceval.runner.shared.base_runner import make_trainer_policy_cls

from zsceval.viz.gradcam import GradCAM
import cv2
import torch.nn as nn
from zsceval.envs.overcooked.overcooked_ai_py.mdp.actions import Action, Direction
from collections import deque
# from topdown_posterior_fusion import AttentionFuser
from onlineViz import *
from utils import *


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
    parser.add_argument("--is_cam", type=str, default="False",
                        choices=["ArgMax", "Whole", "Attention", "Auto", "Fusion", "False"],
                        help="Whether to use CAM / and how to visualize")
    parser.add_argument("--cam_alpha", type=float, default=0.8)
    parser.add_argument("--cam_layers", type=str, default="2", help="'0, 1 ,2' or 'all'")
    # parse_args 안
    parser.add_argument("--win_path_fix", action="store_true",
                        help="Windows에서 PosixPath 들어간 pickle을 안전하게 로드")

    parser.add_argument("--cam_T_low", type=float, default=0.35)
    parser.add_argument("--cam_T_high", type=float, default=0.65)
    parser.add_argument("--cam_hold", type=int, default=8, help="min frames to hold a mode before switching")
    parser.add_argument("--cam_switch_req", type=int, default=3, help="consecutive frames required to switch")

    all_args = parser.parse_args(args)
    if all_args.layout_name in OLD_LAYOUTS:
        all_args.old_dynamics = True
    else:
        all_args.old_dynamics = False
    return all_args

# --- 렌더링 헬퍼: heatmap 블렌딩 ---
def overlay_heatmap(image, prob_map, map_W, map_H, pad_top, alpha_scale=0.8):
    M = normalize_prob(prob_map)
    cam_resized = cv2.resize(M, (map_W, map_H), interpolation=cv2.INTER_LINEAR)
    cam_resized = np.pad(cam_resized, ((pad_top, 0), (0, 0)), mode='constant', constant_values=0)
    heat_u8 = np.clip(cam_resized * 255, 0, 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)[:, :, ::-1].astype(np.float32)
    cam_soft = cv2.GaussianBlur(cam_resized.astype(np.float32), (0, 0), sigmaX=3, sigmaY=3)
    cam_soft = np.power(np.clip(cam_soft, 0.0, 1.0), 0.8)
    alpha = (cam_soft * alpha_scale)[..., None]
    img_f = image.astype(np.float32)
    blended = img_f * (1.0 - alpha) + heatmap_color * alpha
    return blended.clip(0, 255).astype(np.uint8)

# --- 렌더링 헬퍼: 점선 경로 & 점 트레일 ---
def overlay_trail(image, trail_points, color=(255, 200, 80), base_alpha=0.35, fade=0.88, blur_sigma=2, point_radius=50):
    H, W = image.shape[:2]
    trail_mask = np.zeros((H, W), dtype=np.float32)
    if len(trail_points) >= 2:
        pts = np.array(trail_points, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(trail_mask, [pts], isClosed=False, color=1.0, thickness=2, lineType=cv2.LINE_AA)
        trail_mask = cv2.GaussianBlur(trail_mask, (0, 0), sigmaX=blur_sigma, sigmaY=blur_sigma)
    for i, (tx, ty) in enumerate(reversed(trail_points)):
        a = base_alpha * (fade ** i)
        if a < 0.02: break
        cv2.circle(trail_mask, (tx, ty), point_radius, a, thickness=-1, lineType=cv2.LINE_AA)
    imagef = image.astype(np.float32)
    imagef = imagef * (1.0 - trail_mask[..., None]) + np.array(color, dtype=np.float32) * trail_mask[..., None]
    return imagef.clip(0, 255).astype(np.uint8)

from dataclasses import dataclass

CELL_OFFSET_X = 76
CELL_OFFSET_Y = 76

@dataclass
class GridDims:
    map_W: int
    map_H: int
    pad_top: int
    grid_W: int
    grid_H: int

def grid_to_px(dims: GridDims, c: int, r: int) -> tuple[int, int]:
    """그리드 (col,row) → 픽셀 좌표"""
    x = int(c * dims.grid_W) + CELL_OFFSET_X
    y = int(r * dims.grid_H) + dims.pad_top + CELL_OFFSET_Y
    return x, y
    
class EvalPolicy_Play:
    def __init__(self, population_yaml_path, layout_name, test_policy_name, deterministic=True, epsilon=0.5, win_path_fix=False):
        self.population_yaml_path = population_yaml_path
        self.layout_name = layout_name
        self.test_policy_name = test_policy_name
        self.deterministic = deterministic
        self.epsilon = epsilon
        self.population_config = yaml.load(open(self.population_yaml_path, encoding="utf-8"), yaml.Loader)

        policy_config_path = os.path.join("../policy_pool",
                                          self.population_config[self.test_policy_name]["policy_config_path"])
        # policy_config = list(pickle.load(open(policy_config_path, "rb")))
        try:
            if win_path_fix and sys.platform.startswith("win"):
                policy_config = list(load_pickle_with_path_fix(policy_config_path))
            else:
                with open(policy_config_path, "rb") as f:
                    policy_config = list(pickle.load(f))
        except NotImplementedError:
            # 윈도우에서 PosixPath로 터질 때 자동 폴백
            policy_config = list(load_pickle_with_path_fix(policy_config_path))
        
        
        self.policy_args = policy_config[0]
        _, policy_cls = make_trainer_policy_cls(self.policy_args.algorithm_name)  # ex) rmappo
        model_path = add_path_prefix("../policy_pool", self.population_config[self.test_policy_name]["model_path"])
        self.policy = policy_cls(*policy_config, device=torch.device("cpu"))
        self.policy.load_checkpoint(model_path)

    def init_mask_rnn_state(self):
        masks = np.ones((1, 1), dtype=np.float32)
        rnn_states = np.zeros((self.policy_args.recurrent_N, self.policy_args.hidden_size), dtype=np.float32)

        return masks, rnn_states

    def step(self, obs, masks, rnn_states, available_actions, deterministic=False):
        action, actions_prob, rnn_states = self.policy.act(obs, rnn_states, masks, available_actions=available_actions,
                                                           deterministic=deterministic, action_probs=True)
        return action, actions_prob, rnn_states

    @torch.no_grad()
    def get_action(self, obs, available_actions, masks, rnn_states):
        self.policy.prep_rollout()
        epsilon = random.random()
        if not self.deterministic or epsilon < self.epsilon:
            return self.step(obs, masks,
                             rnn_states,
                             available_actions,
                             deterministic=False)
        else:
            return self.step(obs, masks,
                             rnn_states,
                             available_actions,
                             deterministic=True)

    def init_cam(self, cam_layers: str):

        all_conv_layers = [module for module in self.policy.actor.base.cnn if isinstance(module, nn.Conv2d)]

        target_layers = []
        if cam_layers.lower() == "all":
            target_layers = all_conv_layers
        else:
            indices = [int(x.strip()) for x in cam_layers.split(",") if x.strip() != ""]
            for idx in indices:
                if 0 <= idx < len(all_conv_layers):
                    target_layers.append(all_conv_layers[idx])

        cam = GradCAM(model=self.policy.actor, target_layer=target_layers)

        return cam


def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)

    if all_args.layout_name in ["random0", "random0_medium", "random1", "random3", "small_corridor", "unident_s"]:
        env = Overcooked(all_args, run_dir=None)
    else:
        env = Overcooked_new(all_args, run_dir=None)

    population_yaml_path = os.path.join("./config", all_args.layout_name + "_benchmark.yml")
    test_policy_name = all_args.test_policy_name + str(all_args.model_seed)
    agent0_play = EvalPolicy_Play(population_yaml_path, all_args.layout_name, test_policy_name=test_policy_name, win_path_fix=all_args.win_path_fix)
    masks, rnn_states = agent0_play.init_mask_rnn_state()
    
    if all_args.is_cam != "False":
        cam = agent0_play.init_cam(all_args.cam_layers)

    both_agents_ob, share_obs, available_actions = env.reset()

    mode_ctrl = None
    if all_args.is_cam in ("Auto", "Fusion"):
        mode_ctrl = CamModeController(
            low=all_args.cam_T_low, high=all_args.cam_T_high,
            hold=all_args.cam_hold, req=all_args.cam_switch_req
        )

    start_time = time.time()
    clock = pygame.time.Clock()
    epi_done = False
    human_action_queue = deque(maxlen=32)
    trail = deque(maxlen=3)     # 최근 3 프레임 추적
    work_memory = WorkMemory(capacity=5)
    
    scorer = PredictionScorer(horizon=6, gamma=0.9, sigma_tol=0.9)
    cands = [
        {"sigma":1.0, "eta_stm":0.15, "mom":1},
        {"sigma":1.4, "eta_stm":0.20, "mom":1},
        {"sigma":1.8, "eta_stm":0.25, "mom":1},
        {"sigma":1.4, "eta_stm":0.25, "mom":2},
        {"sigma":1.0, "eta_stm":0.30, "mom":2},
    ]
    bandit = OnlineBandit(candidates=cands, tau=0.8, lr=0.25, update_every=60)

    agent_attention_fuser = AttentionFuser(
        shape=(8,5),
        fusion="log",
        sigma=1.8, ior_sigma=1.0, ior_strength=0.05, ior_k=2,
        momentum=True, momentum_scale=1,
        eta_min=0.35, eta_max=0.75,
        stm_capacity=4, stm_tau=5.0, stm_sigma=1.2, eta_stm=0.2,
        scorer=scorer,            # ★ 추가
        bandit=bandit,            # ★ 추가
    )

    
    try:
        image = env.play_render()
        
        image_W = image.shape[1]
        image_H = image.shape[0]        
        
        map_W = image_W
        map_H = int((image_W / 8 ) * 5)
        pad_top = image_H - map_H
        
        grid_W = map_W // 8
        grid_H = map_H // 5
        
        screen = pygame.display.set_mode((image_W, image_H))
        # screen = pygame.display.set_mode((image.shape[1], image.shape[0]))
        screen.blit(pygame.surfarray.make_surface(np.rot90(np.flip(image[..., ::-1], 1))), (0, 0))
        pygame.display.flip()

        while not epi_done:
            
            clock.tick(6.67)
            # clock.tick(15)
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
                    

            a0, a0_prob, rnn_states = agent0_play.get_action(np.expand_dims(both_agents_ob[0], axis=0),
                                                    available_actions, masks,
                                                    rnn_states)
            
            a1_action = human_action_queue.popleft() if human_action_queue else Action.STAY
            a1 = Action.ACTION_TO_INDEX[a1_action]

            joint_action = np.array([[int(a0)], [int(a1)]])

            if all_args.is_cam:
                cam_heatmap = cam(np.expand_dims(both_agents_ob[0], axis=0),
                                  available_actions, rnn_states, masks, target_action=int(a0))
                
                A_t, fix_rc, prior_pi, eta_used = agent_attention_fuser.step(cam_heatmap, hit=None, use_adaptive_eta=True)
                work_memory.push(cam_heatmap)
            both_agents_ob, share_obs, reward, done, info, available_actions = env.step(joint_action)
            agent_position = env.base_env.state.players[0].position
            r,c = agent_position
            agent_attention_fuser.note_position((c,r))
            # agent_attention_fuser.note_position(agent_position)  # 점수 산출 + 주기적 파라미터 업데이트
            hit_k = topk_near_hit(A_t, (r,c), k=3)
            p_hit = p_hit_near(A_t, (r,c), agg="sum")
            print(f"Top5-hit={hit_k}, Probablity={p_hit} Entropy={entropy(A_t):.2f}")
            next_pos = nextPosition(agent_position, a0)

            # epi_done = done[0]

            # render
            image = env.play_render(action_probs=a0_prob)
            
            # --- Auto / Fusion 모드 처리 ---
            if all_args.is_cam in ("Auto", "Fusion"):
                # 지표 계산 및 모드 결정
                mode, Hn, pmax, tk3 = mode_ctrl.update(A_t)

                # Whole 투명도는 엔트로피 기반으로 자동 조절
                alpha_whole = float(all_args.cam_alpha) * (0.4 + 0.6 * (1.0 - Hn))
                alpha_whole = float(np.clip(alpha_whole, 0.15, 0.95))

                if all_args.is_cam == "Auto":
                    # 1) 단일 모드 선택
                    if mode == "ArgMax":
                        # ArgMax: 최고 셀만 강조 + 다음 이동 방향 트레일
                        peak_map = np.zeros_like(A_t, dtype=np.float32)
                        rr, cc = np.unravel_index(np.argmax(A_t), A_t.shape)
                        peak_map[rr, cc] = A_t[rr, cc]
                        image = overlay_heatmap(image, peak_map, map_W, map_H, pad_top, alpha_scale=all_args.cam_alpha)

                        # next_pos 기반 트레일 유지 (기존 코드 로직 그대로)
                        c_next, r_next = next_pos
                        cx = int(c_next * grid_W) + 76
                        cy = int(r_next * grid_H) + pad_top + 76
                        trail.append((cx, cy))
                        image = overlay_trail(image, list(trail), point_radius=50)

                    elif mode == "Whole":
                        # Whole: A_t 히트맵 + 예측 fixation 점선 경로
                        image = overlay_heatmap(image, A_t, map_W, map_H, pad_top, alpha_scale=alpha_whole)
                        # fix_rc를 점으로 누적 (trail 사용)
                        c_fix, r_fix = fix_rc
                        cx = int(c_fix * grid_W) + 76
                        cy = int(r_fix * grid_H) + pad_top + 76
                        trail.append((cx, cy))
                        image = overlay_trail(image, list(trail), point_radius=50)

                    else:  # "Attention"
                        # WorkMemory 기반 트레일 (기존 구현 유지)
                        worked_vector = work_memory.get_top()
                        if worked_vector is not None:
                            (r_cell, c_cell) = worked_vector['index']      # (row, col)
                            streak_worked = worked_vector['streak']
                            cx = int(c_cell * grid_W) + 76
                            cy = int(r_cell * grid_H) + pad_top + 76
                            # streak -> radius 매핑
                            base_rad, gain_rad = 18, 6
                            radius = int(np.clip(base_rad + gain_rad * streak_worked, 8, 80))
                            # trail에 반경 저장
                            if len(trail) and np.hypot(trail[-1][0]-cx, trail[-1][1]-cy) < 1e-3:
                                trail[-1] = (cx, cy)  # 같은 셀 반복 시 덮어쓰기
                            else:
                                trail.append((cx, cy))
                            image = overlay_trail(image, list(trail), point_radius=radius)

                    # 화면 좌측 상단에 상태 텍스트(선택)
                    cv2.putText(image, f"Auto:{mode} H={Hn:.2f} pmax={pmax:.2f} tk3={tk3:.2f}",
                                (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 30, 30), 2, cv2.LINE_AA)

                else:
                    # Fusion: Whole을 옅게 + ArgMax 점 + Attention 트레일을 동시에
                    image = overlay_heatmap(image, A_t, map_W, map_H, pad_top, alpha_scale=alpha_whole * 0.7)

                    # ArgMax 점(작게)
                    r_peak, c_peak = np.unravel_index(np.argmax(A_t), A_t.shape)
                    cx = int(c_peak * grid_W) + 76
                    cy = int(r_peak * grid_H) + pad_top + 76
                    cv2.circle(image, (cx, cy), 10, (255, 220, 120), thickness=-1, lineType=cv2.LINE_AA)

                    # Attention 트레일(약하게)
                    worked_vector = work_memory.get_top()
                    if worked_vector is not None:
                        (r_cell, c_cell) = worked_vector['index']
                        cx2 = int(c_cell * grid_W) + 76
                        cy2 = int(r_cell * grid_H) + pad_top + 76
                        trail.append((cx2, cy2))
                        image = overlay_trail(image, list(trail),
                                            color=(255, 200, 80), base_alpha=0.20, fade=0.90, point_radius=30)

                    cv2.putText(image, f"Fusion H={Hn:.2f}", (10, 24),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 30, 30), 2, cv2.LINE_AA)

                        
            
            if all_args.is_cam == "ArgMax":
                # filter max heatmap
                # ArgMax: 다음 움직일 곳에 trail 표시
                max_idx = np.argmax(cam_heatmap)
                max_row, max_col = np.unravel_index(max_idx, cam_heatmap.shape)
                cam_filtered = np.zeros_like(cam_heatmap)
                cam_filtered[max_row, max_col] = cam_heatmap[max_row, max_col]

                cam_resized = cv2.resize(cam_filtered, (map_W, map_H), interpolation=cv2.INTER_LINEAR)
                cam_resized = np.pad(cam_resized, ((pad_top, 0), (0, 0)), mode='constant', constant_values=0)
                
                heat_u8 = (cam_resized * 255).astype(np.uint8)
                heatmap_color = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)[:, :, ::-1].astype(np.float32)
                
                # smoothing
                cam_soft = cv2.GaussianBlur(cam_resized.astype(np.float32), (0, 0), sigmaX=3, sigmaY=3)
                cam_soft = np.power(np.clip(cam_soft, 0.0, 1.0), 0.8)
                alpha = (cam_soft * all_args.cam_alpha)[..., None]  # (H, W, 1)

                img_f = image.astype(np.float32)
                blended = img_f * (1.0 - alpha) + heatmap_color * alpha
                image = blended.clip(0, 255).astype(np.uint8)

                c, r = next_pos
                cx = int(c*grid_W) + 76         
                cy = int(r*grid_H)+pad_top + 76               
                
                trail.append((cx, cy))
                trail_mask = np.zeros((image_H, image_W), dtype=np.float32)
                
                # (A) 연속 선 + 가우시안 블러 (연속감↑)
                if len(trail) >= 2:
                    pts = np.array(trail, dtype=np.int32).reshape(-1, 1, 2)
                    # 먼저 얇은 폴리라인으로 1.0 intensity를 그린 뒤
                    cv2.polylines(trail_mask, [pts], isClosed=False, color=1.0,
                                thickness=2, lineType=cv2.LINE_AA)
                    # 부드럽게 퍼지게 블러
                    trail_mask = cv2.GaussianBlur(trail_mask, (0, 0), sigmaX=2, sigmaY=2)

                # (B) 페이딩 점(원) 추가 (최근 점은 진하게, 오래된 점은 옅게)
                for i, (tx, ty) in enumerate(reversed(trail)):
                    a = 0.35 * (0.88 ** i)
                    if a < 0.02:
                        break
                    r = max(2, int(2 * 0.9))  # 점 반지름
                    cv2.circle(trail_mask, (tx, ty), 50, a, thickness=-1, lineType=cv2.LINE_AA)

                # 3) 트레일 색상으로 수동 블렌딩 (RGB)
                imagef = image.astype(np.float32)
                imagef = imagef * (1.0 - trail_mask[..., None]) + np.array([255, 200, 80], dtype=np.float32) * trail_mask[..., None]
                image = imagef.clip(0, 255).astype(np.uint8)
                
                
            elif all_args.is_cam == "Whole":
                # filter max heatmap
                # Whole: AttentionFuser 결과 -> 다음 fixaton 예측하여 trail 표시

                cam_resized = cv2.resize(A_t, (map_W, map_H), interpolation=cv2.INTER_LINEAR)
                cam_resized = np.pad(cam_resized, ((pad_top, 0), (0, 0)), mode='constant', constant_values=0)
                heat_u8 = (cam_resized * 255).astype(np.uint8)
                heatmap_color = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)[:, :, ::-1].astype(np.float32)
                # smoothing
                cam_soft = cv2.GaussianBlur(cam_resized.astype(np.float32), (0, 0), sigmaX=3, sigmaY=3)
                cam_soft = np.power(np.clip(cam_soft, 0.0, 1.0), 0.8)
                alpha = (cam_soft * all_args.cam_alpha)[..., None]  # (H, W, 1)

                img_f = image.astype(np.float32)
                blended = img_f * (1.0 - alpha) + heatmap_color * alpha
                image = blended.clip(0, 255).astype(np.uint8)
                
                Hh, Wh = cam_heatmap.shape[:2]          # 예: 8x5
                Hi, Wi = image.shape[:2]               # 원본 이미지 크기
                c, r = fix_rc

                cx = int(c*grid_W) + 76
                cy = int(r*grid_H)+pad_top + 76

                trail.append((cx, cy))
                trail_mask = np.zeros((image_H, image_W), dtype=np.float32)

                # (A) 연속 선 + 가우시안 블러 (연속감↑)
                if len(trail) >= 2:
                    pts = np.array(trail, dtype=np.int32).reshape(-1, 1, 2)
                    # 먼저 얇은 폴리라인으로 1.0 intensity를 그린 뒤
                    cv2.polylines(trail_mask, [pts], isClosed=False, color=1.0,
                                thickness=2, lineType=cv2.LINE_AA)
                    # 부드럽게 퍼지게 블러
                    trail_mask = cv2.GaussianBlur(trail_mask, (0, 0), sigmaX=2, sigmaY=2)

                # (B) 페이딩 점(원) 추가 (최근 점은 진하게, 오래된 점은 옅게)
                for i, (tx, ty) in enumerate(reversed(trail)):
                    a = 0.35 * (0.88 ** i)
                    if a < 0.02:
                        break
                    r = max(2, int(2 * 0.9))  # 점 반지름
                    cv2.circle(trail_mask, (tx, ty), 50, a, thickness=-1, lineType=cv2.LINE_AA)

                # 3) 트레일 색상으로 수동 블렌딩 (RGB)
                imagef = image.astype(np.float32)
                imagef = imagef * (1.0 - trail_mask[..., None]) + np.array([255, 200, 80], dtype=np.float32) * trail_mask[..., None]
                image = imagef.clip(0, 255).astype(np.uint8)

                mv, ml = agent_attention_fuser.get_prediction_metrics(k=50)
                txt_log = f"Pred@H={scorer.horizon} | visit={mv:.3f} | log={ml:.3f}"
                # print(txt_log)

            elif all_args.is_cam == "Attention":
                # filter max heatmap
                # Attention: WorkMemory + streak 기반으로 연속되면 radius 증가 트레일 표시
                
                worked_vector = work_memory.get_top()
                index_worked = worked_vector['index']      # (row, col)
                streak_worked = worked_vector['streak']    # 연속 카운트

                c_cell, r_cell = index_worked  # (row, col) 순서로 명확히

                cx = int(c_cell * grid_W) + 76
                cy = int(r_cell * grid_H) + pad_top + 76

                # --- streak을 radius로 매핑 ---
                # 필요에 맞게 파라미터 조정: base, gain, clamp
                base_rad = 18
                gain_rad = 6
                min_rad, max_rad = 8, 80
                radius = int(np.clip(base_rad + gain_rad * streak_worked, min_rad, max_rad))

                # --- trail 업데이트: 같은 셀 반복이면 덮어쓰기 ---
                if streak_worked > 1:
                    px, py, prad = trail[-1]
                    alpha = 1.0  # 필요 시 강도도 보관 가능
                    # 좌표를 살짝 부드럽게(선택): 0.5 가중 EMA
                    cx_smooth = int(0.5 * px + 0.5 * cx)
                    cy_smooth = int(0.5 * py + 0.5 * cy)
                    trail[-1] = (cx_smooth, cy_smooth, max(prad, radius))
                else:
                    trail.append((cx, cy, radius))

                # 2) trail 마스크 만들기
                trail_mask = np.zeros((image_H, image_W), dtype=np.float32)

                # (A) 연속 선: 서로 다른 연속 포인트들만 선으로 이어지게 좌표만 추출
                if len(trail) >= 2:
                    pts = np.array([(x, y) for (x, y, _) in trail], dtype=np.int32).reshape(-1, 1, 2)
                    cv2.polylines(trail_mask, [pts], isClosed=False, color=1.0,
                                thickness=2, lineType=cv2.LINE_AA)
                    trail_mask = cv2.GaussianBlur(trail_mask, (0, 0), sigmaX=2, sigmaY=2)

                # (B) 페이딩 점(원) 추가: 최근 점은 진하게, 오래된 점은 옅게
                #    각 점의 radius는 streak 기반으로 이미 저장됨
                base_alpha = 0.35
                fade = 0.88
                for i, (tx, ty, rad) in enumerate(reversed(trail)):
                    a = base_alpha * (fade ** i)
                    if a < 0.02:
                        break
                    cv2.circle(trail_mask, (tx, ty), int(rad), a, thickness=-1, lineType=cv2.LINE_AA)

                # 3) 수동 블렌딩
                imagef = image.astype(np.float32)
                imagef = imagef * (1.0 - trail_mask[..., None]) + np.array([255, 200, 80], dtype=np.float32) * trail_mask[..., None]
                image = imagef.clip(0, 255).astype(np.uint8)



            elif all_args.is_cam == "False":
                pass

            screen.blit(pygame.surfarray.make_surface(np.rot90(np.flip(image[..., ::-1], 1))), (0, 0))
            pygame.display.flip()

            end_time = time.time()
            game_time = end_time - start_time    

        print('finish_time : ', game_time)

    finally:
        if cam is not None:
            cam.remove_hooks()
        pygame.quit()


if __name__ == "__main__":
    main(sys.argv[1:])
