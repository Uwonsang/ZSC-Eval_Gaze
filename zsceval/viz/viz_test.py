# -*- coding: utf-8 -*-
import os
import sys
import time
import random
import pickle
import pathlib
from collections import deque
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple


# ---------------------------------------------------------------------
# 경로/환경 준비
# ---------------------------------------------------------------------
os.environ["POLICY_POOL"] = "../policy_pool"


import numpy as np
import pygame
import yaml
import torch
import torch.nn as nn
import cv2

from zsceval.config import get_config
from zsceval.overcooked_config import get_overcooked_args, OLD_LAYOUTS

from zsceval.envs.overcooked.Overcooked_Env import Overcooked
from zsceval.envs.overcooked_new.Overcooked_Env import Overcooked as Overcooked_new

from zsceval.algorithms.population.policy_pool import add_path_prefix
from zsceval.runner.shared.base_runner import make_trainer_policy_cls

from zsceval.viz.gradcam import GradCAM
from zsceval.envs.overcooked.overcooked_ai_py.mdp.actions import Action, Direction

# onlineViz: 사용자 정의 모듈 (기존 코드 유지)
from onlineViz import AttentionFuser, PredictionScorer, OnlineBandit



# ---------------------------------------------------------------------
# 유틸: pickle 로더 (윈도 호환)
# ---------------------------------------------------------------------
class _PathFixUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if sys.platform.startswith("win") and module == "pathlib" and name == "PosixPath":
            return pathlib.WindowsPath
        return super().find_class(module, name)

def load_pickle_with_path_fix(path):
    with open(path, "rb") as f:
        return _PathFixUnpickler(f).load()


# ---------------------------------------------------------------------
# 수학/확률 유틸
# ---------------------------------------------------------------------
def normalize_prob(A: np.ndarray) -> np.ndarray:
    A = A.astype(np.float32)
    s = float(A.sum())
    if s <= 0:
        return np.full_like(A, 1.0 / A.size, dtype=np.float32)
    return A / s

def entropy(P: np.ndarray) -> float:
    P = normalize_prob(P)
    return float(-(P * np.log(P + 1e-8)).sum())

def topk_near_hit(A: np.ndarray, rc: Tuple[int, int], k=5, radius=1) -> float:
    R, C = A.shape
    k = max(1, min(k, R*C))
    flat = A.ravel().astype(float)
    idxs = np.argpartition(-flat, k-1)[:k]
    rows, cols = np.unravel_index(idxs, (R, C))
    topk = set(zip(rows.tolist(), cols.tolist()))
    r, c = rc
    r0, r1 = max(0, r-radius), min(R-1, r+radius)
    c0, c1 = max(0, c-radius), min(C-1, c+radius)
    for rr in range(r0, r1+1):
        for cc in range(c0, c1+1):
            if (rr, cc) in topk:
                return 1.0
    return 0.0

def p_hit_near(A: np.ndarray, rc: Tuple[int, int], radius=1, agg="sum") -> float:
    R, C = A.shape
    P = normalize_prob(A)
    r, c = rc
    r0, r1 = max(0, r-radius), min(R-1, r+radius)
    c0, c1 = max(0, c-radius), min(C-1, c+radius)
    patch = P[r0:r1+1, c0:c1+1]
    return float(patch.sum() if agg == "sum" else patch.max())


# ---------------------------------------------------------------------
# WorkMemory: 최근 맵들의 가중합 + top 인덱스/스트릭
# ---------------------------------------------------------------------
class WorkMemory:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)
        self._last_index: Optional[Tuple[int, ...]] = None
        self._streak: int = 0

    def push(self, item: np.ndarray):
        self.memory.append(item)

    def _weighted_sum(self) -> Optional[np.ndarray]:
        if not self.memory:
            return None
        weights = np.exp(-0.5 * np.arange(len(self.memory))[::-1])
        weights = weights / np.sum(weights)
        return np.sum([w * m for w, m in zip(weights, self.memory)], axis=0)

    def get_memory(self) -> Optional[np.ndarray]:
        return self._weighted_sum()

    def get_top(self, prefer_not_last_on_tie: bool = True) -> Optional[Dict[str, Any]]:
        vec = self._weighted_sum()
        if vec is None:
            return None
        max_val = float(np.max(vec))
        tie_coords = [tuple(idx) for idx in np.argwhere(vec == max_val)]
        if len(tie_coords) == 1:
            chosen = tie_coords[0]
        else:
            if prefer_not_last_on_tie:
                unseen = [c for c in tie_coords if c != self._last_index]
                chosen = unseen[0] if unseen else tie_coords[0]
            else:
                chosen = tie_coords[0]

        if self._last_index is None or self._last_index != chosen:
            self._streak = 1
        else:
            self._streak += 1
        self._last_index = chosen
        flat_index = np.ravel_multi_index(chosen, vec.shape)
        return {
            "index": chosen, "flat_index": int(flat_index),
            "value": max_val, "streak": self._streak,
            "tie_candidates": tie_coords, "vector": vec,
        }

    def reset_history(self):
        self._last_index = None
        self._streak = 0

    def __len__(self):
        return len(self.memory)


# ---------------------------------------------------------------------
# 모드 게이팅(엔트로피/확률 + 히스테리시스)
# ---------------------------------------------------------------------
class CamModeController:
    def __init__(self, low=0.35, high=0.65, hold=8, req=3):
        self.low, self.high = low, high
        self.hold, self.req = int(hold), int(req)
        self.mode = "Whole"
        self.hold_count = 0
        self.cnt = {"ArgMax": 0, "Whole": 0, "Attention": 0}

    def _decide(self, A: np.ndarray) -> str:
        Hn = entropy(A) / np.log(A.size)
        pmax = normalize_prob(A).max()
        if Hn <= self.low and pmax >= 0.35:
            return "ArgMax"
        elif Hn >= self.high:
            return "Attention"
        else:
            return "Whole"

    def update(self, A: np.ndarray):
        Hn = entropy(A) / np.log(A.size)
        P = normalize_prob(A)
        pmax = float(P.max())
        flat = P.ravel()
        tk3 = float(np.partition(flat, -3)[-3:].sum())

        desired = self._decide(A)
        if self.hold_count < self.hold:
            self.hold_count += 1
            return self.mode, Hn, pmax, tk3

        for k in self.cnt.keys():
            self.cnt[k] = self.cnt[k] + 1 if k == desired else 0
        if self.cnt[desired] >= self.req:
            self.mode = desired
            self.hold_count = 0
            for k in self.cnt.keys():
                self.cnt[k] = 0
        return self.mode, Hn, pmax, tk3


# ---------------------------------------------------------------------
# 그리드/좌표 & 렌더 헬퍼
# ---------------------------------------------------------------------
CELL_OFFSET_X = 76
CELL_OFFSET_Y = 76

@dataclass
class GridDims:
    map_W: int
    map_H: int
    pad_top: int
    grid_W: int
    grid_H: int

def grid_to_px(dims: GridDims, c: int, r: int) -> Tuple[int, int]:
    """그리드 (col, row) -> 픽셀 좌표"""
    x = int(c * dims.grid_W) + CELL_OFFSET_X
    y = int(r * dims.grid_H) + dims.pad_top + CELL_OFFSET_Y
    return x, y

def overlay_heatmap(image, prob_map, map_W, map_H, pad_top, alpha_scale=0.8):
    M = normalize_prob(prob_map)
    cam_resized = cv2.resize(M, (map_W, map_H), interpolation=cv2.INTER_LINEAR)
    cam_resized = np.pad(cam_resized, ((pad_top, 0), (0, 0)), mode="constant", constant_values=0)
    heat_u8 = np.clip(cam_resized * 255, 0, 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)[:, :, ::-1].astype(np.float32)
    cam_soft = cv2.GaussianBlur(cam_resized.astype(np.float32), (0, 0), sigmaX=3, sigmaY=3)
    cam_soft = np.power(np.clip(cam_soft, 0.0, 1.0), 0.8)
    alpha = (cam_soft * alpha_scale)[..., None]
    img_f = image.astype(np.float32)
    blended = img_f * (1.0 - alpha) + heatmap_color * alpha
    return blended.clip(0, 255).astype(np.uint8)

def overlay_trail(image, trail_points, color=(255, 200, 80), base_alpha=0.35, fade=0.88, blur_sigma=2, point_radius=50):
    H, W = image.shape[:2]
    trail_mask = np.zeros((H, W), dtype=np.float32)
    if len(trail_points) >= 2:
        pts = np.array(trail_points, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(trail_mask, [pts], isClosed=False, color=1.0, thickness=2, lineType=cv2.LINE_AA)
        trail_mask = cv2.GaussianBlur(trail_mask, (0, 0), sigmaX=blur_sigma, sigmaY=blur_sigma)
    for i, (tx, ty) in enumerate(reversed(trail_points)):
        a = base_alpha * (fade ** i)
        if a < 0.02:
            break
        cv2.circle(trail_mask, (tx, ty), point_radius, a, thickness=-1, lineType=cv2.LINE_AA)
    imagef = image.astype(np.float32)
    imagef = imagef * (1.0 - trail_mask[..., None]) + np.array(color, dtype=np.float32) * trail_mask[..., None]
    return imagef.clip(0, 255).astype(np.uint8)


# ---------------------------------------------------------------------
# 계획/좌표 헬퍼
# ---------------------------------------------------------------------
def nextPosition(position: Tuple[int, int], action: int) -> Tuple[int, int]:
    r, c = position
    if action == 0:      # North
        return (r - 1, c)
    elif action == 1:    # South
        return (r + 1, c)
    elif action == 2:    # East
        return (r, c + 1)
    elif action == 3:    # West
        return (r, c - 1)
    else:                # Stay or others
        return position


# ---------------------------------------------------------------------
# 시각화 엔진(단일 진입점): ArgMax/Whole/Attention/Auto/Fusion
# ---------------------------------------------------------------------
def render_cam(
    image: np.ndarray,
    *,
    mode: str,
    cam_alpha: float,
    dims: GridDims,
    cam_heatmap: Optional[np.ndarray],
    fused_map: Optional[np.ndarray],
    fix_rc: Optional[Tuple[int, int]],
    next_pos: Optional[Tuple[int, int]],
    work_memory: Optional[WorkMemory],
    trail: deque,
    controller: Optional[CamModeController] = None
) -> np.ndarray:
    """모든 CAM 시각화를 한 곳에서 수행."""
    if mode == "False":
        return image

    # Auto/Fusion은 fused_map(A_t) + controller 필요
    if mode in ("Auto", "Fusion"):
        if fused_map is None or controller is None:
            return image

    img = image

    def alpha_from_entropy(A: np.ndarray) -> Tuple[float, float]:
        Hn = float(entropy(normalize_prob(A)) / np.log(A.size))
        a = cam_alpha * (0.4 + 0.6 * (1.0 - Hn))
        return float(np.clip(a, 0.15, 0.95)), Hn

    def draw_argmax(A: np.ndarray, use_next_pos: bool = True) -> np.ndarray:
        peak = np.zeros_like(A, dtype=np.float32)
        rr, cc = np.unravel_index(np.argmax(A), A.shape)
        peak[rr, cc] = A[rr, cc]
        out = overlay_heatmap(img, peak, dims.map_W, dims.map_H, dims.pad_top, alpha_scale=cam_alpha)
        if use_next_pos and next_pos is not None:
            r_next, c_next = next_pos  # nextPosition은 (r,c)
            cx, cy = grid_to_px(dims, c_next, r_next)
            trail.append((cx, cy))
            out = overlay_trail(out, list(trail), point_radius=50)
        return out

    def draw_whole(A: np.ndarray, rc: Optional[Tuple[int, int]]) -> np.ndarray:
        a, Hn = alpha_from_entropy(A)
        out = overlay_heatmap(img, A, dims.map_W, dims.map_H, dims.pad_top, alpha_scale=a)
        if rc is not None:
            c_fix, r_fix = rc  # fix_rc는 (c,r)
            cx, cy = grid_to_px(dims, c_fix, r_fix)
            trail.append((cx, cy))
            out = overlay_trail(out, list(trail), point_radius=50)
        cv2.putText(out, f"H={Hn:.2f}", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 30, 30), 2, cv2.LINE_AA)
        return out

    def draw_attention() -> np.ndarray:
        if work_memory is None:
            return img
        wv = work_memory.get_top()
        if wv is None:
            return img
        r_cell, c_cell = wv["index"]  # (row, col)
        streak = int(wv["streak"])
        cx, cy = grid_to_px(dims, c_cell, r_cell)
        base_rad, gain_rad = 18, 6
        radius = int(np.clip(base_rad + gain_rad * streak, 8, 80))
        trail.append((cx, cy))
        return overlay_trail(img, list(trail), point_radius=radius)

    if mode in ("Auto", "Fusion"):
        m, Hn, pmax, tk3 = controller.update(fused_map)
        if mode == "Auto":
            if m == "ArgMax":
                out = draw_argmax(fused_map, use_next_pos=True)
            elif m == "Whole":
                out = draw_whole(fused_map, fix_rc)
            else:
                out = draw_attention()
            cv2.putText(out, f"Auto:{m} H={Hn:.2f} p={pmax:.2f} tk3={tk3:.2f}",
                        (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 30, 30), 2, cv2.LINE_AA)
            return out
        else:
            a, _ = alpha_from_entropy(fused_map)
            out = overlay_heatmap(img, fused_map, dims.map_W, dims.map_H, dims.pad_top, alpha_scale=a*0.7)
            r_peak, c_peak = np.unravel_index(np.argmax(fused_map), fused_map.shape)
            cx, cy = grid_to_px(dims, c_peak, r_peak)
            cv2.circle(out, (cx, cy), 10, (255, 220, 120), thickness=-1, lineType=cv2.LINE_AA)
            # attention 트레일(약하게)
            if work_memory is not None:
                wv = work_memory.get_top()
                if wv is not None:
                    r_cell, c_cell = wv["index"]
                    cx2, cy2 = grid_to_px(dims, c_cell, r_cell)
                    trail.append((cx2, cy2))
                    out = overlay_trail(out, list(trail),
                                        color=(255, 200, 80), base_alpha=0.20, fade=0.90, point_radius=30)
            cv2.putText(out, "Fusion", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 30, 30), 2, cv2.LINE_AA)
            return out

    # 수동 모드
    if mode == "ArgMax":
        A = fused_map if fused_map is not None else cam_heatmap
        return image if A is None else draw_argmax(A, use_next_pos=True)
    if mode == "Whole":
        A = fused_map if fused_map is not None else cam_heatmap
        return image if A is None else draw_whole(A, fix_rc)
    if mode == "Attention":
        return draw_attention()

    return image


# ---------------------------------------------------------------------
# 인자 파서
# ---------------------------------------------------------------------
def parse_args(args, parser):
    parser = get_overcooked_args(parser)
    parser.add_argument("--use_phi", default=False, action="store_true",
                        help="While existing other agent like planning or human model, use an index to fix the main RL-policy agent.")
    parser.add_argument("--test_policy_name", type=str, default="fcp",
                        choices=["fcp", "mep", "traj", "hsp", "sp", "e3t", "cole"])
    parser.add_argument("--model_seed", type=int, default=1, choices=[1, 2, 3])
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--epsilon", type=float, default=0.0, help="stochastic eval epsilon")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--is_cam", type=str, default="False",
                        choices=["ArgMax", "Whole", "Attention", "Auto", "Fusion", "False"],
                        help="Whether to use CAM / and how to visualize")
    parser.add_argument("--cam_alpha", type=float, default=0.8)
    parser.add_argument("--cam_layers", type=str, default="2", help="'0, 1, 2' or 'all'")
    parser.add_argument("--win_path_fix", action="store_true",
                        help="Windows에서 PosixPath 들어간 pickle을 안전하게 로드")
    parser.add_argument("--cam_T_low", type=float, default=0.35)
    parser.add_argument("--cam_T_high", type=float, default=0.65)
    parser.add_argument("--cam_hold", type=int, default=8, help="min frames to hold a mode before switching")
    parser.add_argument("--cam_switch_req", type=int, default=3, help="consecutive frames required to switch")

    all_args = parser.parse_args(args)
    all_args.old_dynamics = all_args.layout_name in OLD_LAYOUTS
    return all_args


# ---------------------------------------------------------------------
# 정책 래퍼
# ---------------------------------------------------------------------
class EvalPolicy_Play:
    def __init__(self, population_yaml_path, layout_name, test_policy_name,
                 deterministic=True, epsilon=0.5, win_path_fix=False):
        self.population_yaml_path = population_yaml_path
        self.layout_name = layout_name
        self.test_policy_name = test_policy_name
        self.deterministic = deterministic
        self.epsilon = epsilon

        self.population_config = yaml.load(open(self.population_yaml_path, encoding="utf-8"), yaml.Loader)
        policy_config_path = os.path.join("../policy_pool",
                                          self.population_config[self.test_policy_name]["policy_config_path"])
        try:
            if win_path_fix and sys.platform.startswith("win"):
                policy_config = list(load_pickle_with_path_fix(policy_config_path))
            else:
                with open(policy_config_path, "rb") as f:
                    policy_config = list(pickle.load(f))
        except NotImplementedError:
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
        action, actions_prob, rnn_states = self.policy.act(
            obs, rnn_states, masks, available_actions=available_actions,
            deterministic=deterministic, action_probs=True
        )
        return action, actions_prob, rnn_states

    @torch.no_grad()
    def get_action(self, obs, available_actions, masks, rnn_states):
        self.policy.prep_rollout()
        epsilon = random.random()
        if not self.deterministic or epsilon < self.epsilon:
            return self.step(obs, masks, rnn_states, available_actions, deterministic=False)
        else:
            return self.step(obs, masks, rnn_states, available_actions, deterministic=True)

    def init_cam(self, cam_layers: str):
        all_conv_layers = [m for m in self.policy.actor.base.cnn if isinstance(m, nn.Conv2d)]
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


# ---------------------------------------------------------------------
# 메인
# ---------------------------------------------------------------------
def main(args):
    pygame.init()
    parser = get_config()
    all_args = parse_args(args, parser)

    # 환경 선택
    if all_args.layout_name in ["random0", "random0_medium", "random1", "random3", "small_corridor", "unident_s"]:
        env = Overcooked(all_args, run_dir=None)
    else:
        env = Overcooked_new(all_args, run_dir=None)

    # 정책 로딩
    population_yaml_path = os.path.join("./config", all_args.layout_name + "_benchmark.yml")
    test_policy_name = all_args.test_policy_name + str(all_args.model_seed)
    agent0_play = EvalPolicy_Play(population_yaml_path, all_args.layout_name,
                                  test_policy_name=test_policy_name, win_path_fix=all_args.win_path_fix)
    masks, rnn_states = agent0_play.init_mask_rnn_state()

    # CAM on/off
    use_cam = (all_args.is_cam != "False")
    cam = None
    if use_cam:
        cam = agent0_play.init_cam(all_args.cam_layers)

    # 초기화
    both_agents_ob, share_obs, available_actions = env.reset()

    mode_ctrl = None
    if use_cam and all_args.is_cam in ("Auto", "Fusion"):
        mode_ctrl = CamModeController(low=all_args.cam_T_low, high=all_args.cam_T_high,
                                      hold=all_args.cam_hold, req=all_args.cam_switch_req)

    clock = pygame.time.Clock()
    epi_done = False
    human_action_queue = deque(maxlen=32)
    trail = deque(maxlen=3)          # 최근 좌표 트레일
    work_memory = WorkMemory(capacity=5)

    # 시선/주의 융합기(onlineViz)
    scorer = PredictionScorer(horizon=6, gamma=0.9, sigma_tol=0.9)
    cands = [
        {"sigma": 1.0, "eta_stm": 0.15, "mom": 1},
        {"sigma": 1.4, "eta_stm": 0.20, "mom": 1},
        {"sigma": 1.8, "eta_stm": 0.25, "mom": 1},
        {"sigma": 1.4, "eta_stm": 0.25, "mom": 2},
        {"sigma": 1.0, "eta_stm": 0.30, "mom": 2},
    ]
    bandit = OnlineBandit(candidates=cands, tau=0.8, lr=0.25, update_every=60)

    agent_attention_fuser = AttentionFuser(
        shape=(8, 5),
        fusion="log",
        sigma=1.8, ior_sigma=1.0, ior_strength=0.05, ior_k=2,
        momentum=True, momentum_scale=1,
        eta_min=0.35, eta_max=0.75,
        stm_capacity=4, stm_tau=5.0, stm_sigma=1.2, eta_stm=0.2,
        scorer=scorer, bandit=bandit,
    )

    # 첫 프레임
    image = env.play_render()
    image_H, image_W = image.shape[0], image.shape[1]
    map_W = image_W
    map_H = int((image_W / 8) * 5)
    pad_top = image_H - map_H
    grid_W = map_W // 8
    grid_H = map_H // 5
    dims = GridDims(map_W=map_W, map_H=map_H, pad_top=pad_top, grid_W=grid_W, grid_H=grid_H)

    screen = pygame.display.set_mode((image_W, image_H))
    screen.blit(pygame.surfarray.make_surface(np.rot90(np.flip(image[..., ::-1], 1))), (0, 0))
    pygame.display.flip()

    try:
        while not epi_done:
            clock.tick(all_args.fps)

            # 입력 처리
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    epi_done = True
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        epi_done = True
                    elif event.key == pygame.K_UP:
                        human_action_queue.append(Direction.NORTH)
                    elif event.key == pygame.K_DOWN:
                        human_action_queue.append(Direction.SOUTH)
                    elif event.key == pygame.K_LEFT:
                        human_action_queue.append(Direction.WEST)
                    elif event.key == pygame.K_RIGHT:
                        human_action_queue.append(Direction.EAST)
                    elif event.key == pygame.K_SPACE:
                        human_action_queue.append(Action.INTERACT)

            # 에이전트 액션
            a0, a0_prob, rnn_states = agent0_play.get_action(
                np.expand_dims(both_agents_ob[0], axis=0),
                available_actions, masks, rnn_states
            )
            a1_action = human_action_queue.popleft() if human_action_queue else Action.STAY
            a1 = Action.ACTION_TO_INDEX[a1_action]
            joint_action = np.array([[int(a0)], [int(a1)]])

            # CAM / 주의 맵 갱신
            if use_cam:
                cam_heatmap = cam(np.expand_dims(both_agents_ob[0], axis=0),
                                  available_actions, rnn_states, masks, target_action=int(a0))
                cam_heatmap = np.nan_to_num(cam_heatmap, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

                A_t, fix_rc, prior_pi, eta_used = agent_attention_fuser.step(
                    cam_heatmap, hit=None, use_adaptive_eta=True
                )
                A_t = np.nan_to_num(A_t, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
                work_memory.push(cam_heatmap)

            # 환경 스텝
            both_agents_ob, share_obs, reward, done, info, available_actions = env.step(joint_action)

            # 위치/로깅
            agent_position = env.base_env.state.players[0].position  # (r, c)
            r, c = agent_position
            if use_cam:
                agent_attention_fuser.note_position((c, r))  # fuser는 (c,r) 순서 사용
                hit_k = topk_near_hit(A_t, (r, c), k=3)
                p_hit = p_hit_near(A_t, (r, c), agg="sum")
                print(f"Top3-hit={hit_k:.1f}, Prob={p_hit:.3f}, Entropy={entropy(A_t):.2f}")

            next_pos = nextPosition(agent_position, int(a0))  # (r,c)

            # 종료 플래그
            try:
                epi_done = bool(np.any(done))
            except Exception:
                pass

            # 렌더
            image = env.play_render(action_probs=a0_prob)

            if use_cam:
                # CAM 시각화 한 줄
                image = render_cam(
                    image,
                    mode=all_args.is_cam,
                    cam_alpha=all_args.cam_alpha,
                    dims=dims,
                    cam_heatmap=cam_heatmap,
                    fused_map=A_t,
                    fix_rc=fix_rc,             # (c,r)
                    next_pos=next_pos,          # (r,c)
                    work_memory=work_memory,
                    trail=trail,
                    controller=mode_ctrl
                )

            # 화면 업데이트 (원본 코드의 회전/채널 변환 유지)
            screen.blit(pygame.surfarray.make_surface(np.rot90(np.flip(image[..., ::-1], 1))), (0, 0))
            pygame.display.flip()

    finally:
        try:
            if cam is not None:
                cam.remove_hooks()
        finally:
            pygame.quit()


if __name__ == "__main__":
    main(sys.argv[1:])
