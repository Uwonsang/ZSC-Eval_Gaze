from collections import deque
from typing import Optional, Dict, Any, Tuple, List
import numpy as np

class WorkMemory:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)

        # 상태
        self._last_index: Optional[Tuple[int, ...]] = None  # 직전 (multi-index)
        self._streak: int = 0

    def push(self, item: np.ndarray):
        # item shape: (5, 8) 등. 모두 동일 shape 가정
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
        """
        2D(또는 임의 차원) 가중합에서 최대값 위치를 선택.
        - 같은 좌표가 연속이면 streak += 1, 아니면 1로 리셋
        - 동률이면 이전에 뽑히지 않은 좌표 우선
        반환:
          {
            'index': Tuple[int, ...],   # (row, col) 등 다차원 인덱스
            'flat_index': int,          # 평탄화 인덱스
            'value': float,             # 최대값
            'streak': int,              # 연속 카운트
            'tie_candidates': List[Tuple[int, ...]],  # 동률 좌표들
            'vector': np.ndarray        # 가중합 배열
          }
        """
        vec = self._weighted_sum()
        if vec is None:
            return None

        max_val = float(np.max(vec))
        # 동률 후보 좌표들 (예: 5x8이면 (r, c) 튜플 리스트)
        tie_coords = [tuple(idx) for idx in np.argwhere(vec == max_val)]

        # 후보 선택
        if len(tie_coords) == 1:
            chosen = tie_coords[0]
        else:
            if prefer_not_last_on_tie:
                unseen = [c for c in tie_coords if c != self._last_index]
                chosen = unseen[0] if unseen else tie_coords[0]
            else:
                chosen = tie_coords[0]

        # streak 업데이트
        if self._last_index is None or self._last_index != chosen:
            self._streak = 1
        else:
            self._streak += 1

        self._last_index = chosen

        # 평탄화 인덱스도 함께 제공 (필요시)
        flat_index = np.ravel_multi_index(chosen, vec.shape)

        return {
            'index': chosen,
            'flat_index': int(flat_index),
            'value': max_val,
            'streak': self._streak,
            'tie_candidates': tie_coords,
            'vector': vec,
        }

    def reset_history(self):
        self._last_index = None
        self._streak = 0

    def __len__(self):
        return len(self.memory)
  
def topk_near_hit(A: np.ndarray, rc, k=5, radius=1) -> float:
    """(r,c) 주변 반경 radius(체비쇼프/8방) 안에 Top-k 셀이 하나라도 있으면 1.0"""
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

def p_hit_near(A: np.ndarray, rc, radius=1, agg="sum") -> float:
    """주변 반경 안 확률 집계. agg='sum'이면 합, 'max'면 최대."""
    R, C = A.shape
    P = A.astype(float); s = P.sum(); P = P/s if s>0 else np.full_like(P, 1.0/(R*C))
    r, c = rc
    r0, r1 = max(0, r-radius), min(R-1, r+radius)
    c0, c1 = max(0, c-radius), min(C-1, c+radius)
    patch = P[r0:r1+1, c0:c1+1]
    return float(patch.sum() if agg=="sum" else patch.max())
  
  

def nextPosition(position, action: int):
    if action == 0:  # North
        return (position[0] - 1, position[1])
    elif action == 1:  # South
        return (position[0] + 1, position[1])
    elif action == 2:  # East
        return (position[0], position[1] + 1)
    elif action == 3:  # West
        return (position[0], position[1] - 1)
    else:  # Stay
        return position

# --- CAM 게이팅 유틸 ---
def normalize_prob(A: np.ndarray) -> np.ndarray:
    A = A.astype(np.float32)
    s = float(A.sum())
    if s <= 0:
        return np.full_like(A, 1.0 / A.size, dtype=np.float32)
    return A / s

def norm_entropy(A: np.ndarray) -> float:
    P = normalize_prob(A)
    H = float(-(P * np.log(P + 1e-8)).sum())
    return H / np.log(P.size)

def topk_mass(A: np.ndarray, k: int = 3) -> float:
    P = normalize_prob(A)
    flat = P.ravel()
    idx = np.argpartition(flat, -k)[-k:]
    return float(flat[idx].sum())

class CamModeController:
    """엔트로피/확률 기반 모드 결정을 시간적으로 매끈하게."""
    def __init__(self, low=0.35, high=0.65, margin=0.0, hold=8, req=3):
        self.low, self.high, self.margin = low, high, margin
        self.hold, self.req = int(hold), int(req)
        self.mode = "Whole"         # 시작은 중간 모드
        self.hold_count = 0
        self.cnt = {"ArgMax": 0, "Whole": 0, "Attention": 0}

    def decide_desired(self, A: np.ndarray) -> str:
        Hn = norm_entropy(A)
        pmax = normalize_prob(A).max()
        if Hn <= self.low and pmax >= 0.35:
            return "ArgMax"
        elif Hn >= self.high:
            return "Attention"
        else:
            return "Whole"

    def update(self, A: np.ndarray):
        """모드와 주요 지표 반환: (mode, Hn, pmax, top3)."""
        Hn = norm_entropy(A)
        P = normalize_prob(A)
        pmax = float(P.max())
        tk3 = topk_mass(P, k=3)

        desired = self.decide_desired(A)

        # 최소 hold 프레임 동안은 유지
        if self.hold_count < self.hold:
            self.hold_count += 1
            return self.mode, Hn, pmax, tk3

        # 전환을 위해 연속 req 프레임 확인
        for k in self.cnt.keys():
            self.cnt[k] = self.cnt[k] + 1 if k == desired else 0

        if self.cnt[desired] >= self.req:
            self.mode = desired
            self.hold_count = 0
            # 카운터 리셋
            for k in self.cnt.keys():
                self.cnt[k] = 0

        return self.mode, Hn, pmax, tk3
    


class _PathFixUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        # 윈도우에서 리눅스/맥의 PosixPath를 WindowsPath로 매핑
        if sys.platform.startswith("win") and module == "pathlib" and name == "PosixPath":
            return pathlib.WindowsPath
        return super().find_class(module, name)

def load_pickle_with_path_fix(path):
    with open(path, "rb") as f:
        return _PathFixUnpickler(f).load()