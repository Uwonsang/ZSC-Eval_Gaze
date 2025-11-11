from collections import deque
import numpy as np
import math
from itertools import islice


EPS = 1e-9

def normalize(P: np.ndarray) -> np.ndarray:
    P = np.clip(P, EPS, None)
    s = P.sum()
    if s <= 0:
        return np.ones_like(P) / P.size
    return P / s

def entropy(P: np.ndarray) -> float:
    Pn = normalize(P).ravel()
    return float(-(Pn * np.log(Pn)).sum())

def gaussian_kernel2d(sigma: float, radius: int = None) -> np.ndarray:
    if sigma <= 0:
        return np.array([[1.0]], dtype=float)
    if radius is None:
        radius = max(1, int(3.0 * sigma))
    xs = np.arange(-radius, radius + 1, dtype=float)
    ys = np.arange(-radius, radius + 1, dtype=float)
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    K = np.exp(- (X**2 + Y**2) / (2 * sigma**2))
    K /= K.sum()
    return K

def convolve_same(P: np.ndarray, K: np.ndarray) -> np.ndarray:
    # zero-padding convolution, "same" output size
    R, C = P.shape
    kr, kc = K.shape
    pr, pc = kr // 2, kc // 2
    out = np.zeros_like(P, dtype=float)
    # pad
    Pp = np.pad(P, ((pr, pr), (pc, pc)), mode="edge")
    for i in range(R):
        for j in range(C):
            region = Pp[i:i+kr, j:j+kc]
            out[i, j] = float((region * K).sum())
    return out

def shift_with_wrap(P: np.ndarray, dr: int, dc: int) -> np.ndarray:
    # wrap-around shift; for clamped use edge padding + slice, but wrap is OK for grids
    return np.roll(np.roll(P, dr, axis=0), dc, axis=1)

def build_ior_mask(shape, fixations, sigma=1.0, strength=0.10, k_last=3):
    if not fixations:
        return np.ones(shape, dtype=float)
    R, C = shape
    rr, cc = np.meshgrid(np.arange(R), np.arange(C), indexing="ij")
    field = np.zeros(shape, dtype=float)
    for (r, c) in fixations[-k_last:]:
        field += np.exp(-((rr - r) ** 2 + (cc - c) ** 2) / (2 * sigma ** 2))
    if field.max() > 0:
        field = field / field.max()
    # multiplicative soft mask
    mask = 1.0 - strength * field
    return np.clip(mask, 0.0, 1.0)

def predict_prior(A_prev: np.ndarray,
                  sigma: float = 1.0,
                  drift: tuple = (0, 0),
                  ior_mask: np.ndarray = None) -> np.ndarray:
    """One-step predicted prior π_t from previous posterior A_{t-1}."""
    K = gaussian_kernel2d(sigma=sigma)
    P = convolve_same(A_prev, K)
    if drift != (0, 0):
        P = shift_with_wrap(P, int(drift[0]), int(drift[1]))
    if ior_mask is not None:
        P = P * ior_mask
    return normalize(P)

def fuse_log(A_post_hat: np.ndarray, prior_pi: np.ndarray, eta: float = 0.4) -> np.ndarray:
    """Log-weighted fusion to avoid double counting and keep stability."""
    A_post_hat = normalize(A_post_hat)
    prior_pi   = normalize(prior_pi)
    logA = (1 - eta) * np.log(A_post_hat + EPS) + eta * np.log(prior_pi + EPS)
    A = np.exp(logA - logA.max())
    return normalize(A)

def fuse_dirichlet(A_post_hat: np.ndarray, prior_pi: np.ndarray, kappa: float = 4.0, N: float = 1.0) -> np.ndarray:
    """Pseudo-count (Dirichlet-like) averaging."""
    A_post_hat = normalize(A_post_hat)
    prior_pi   = normalize(prior_pi)
    return normalize(N * A_post_hat + kappa * prior_pi)

def neighborhood_kernel(radius: int, metric: str = "chebyshev", normalize=True) -> np.ndarray:
    """
    radius=1이면 3x3. metric='chebyshev'는 8방(□), 'euclid'는 원(●) 근사.
    """
    r = int(max(0, radius))
    xs = np.arange(-r, r+1)
    ys = np.arange(-r, r+1)
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    if metric == "euclid":
        K = ((X**2 + Y**2) <= r*r).astype(float)
    else:  # chebyshev
        K = (np.maximum(np.abs(X), np.abs(Y)) <= r).astype(float)
    if normalize and K.sum() > 0:
        K /= K.sum()
    return K

class PredictionScorer:
    """예측 분포(A_t)와 미래 H프레임 실제 위치를 비교해 점수화."""
    def __init__(self, horizon=6, gamma=0.9, sigma_tol=0.9):
        self.horizon = horizon
        self.gamma = gamma
        self.sigma_tol = sigma_tol
        self.queue = deque()               # [(frame_t, A_pred)]
        self.path = deque(maxlen=4096)     # 실제 위치 로그
        self.logs = []                     # (t0, r_visit, r_log)
        self.nb_kernel = neighborhood_kernel(radius=1, metric="chebyshev", normalize=True)

    def push_pred(self, A_pred, frame):
        self.queue.append((frame, A_pred.copy()))

    def push_pos(self, rc):
        self.path.append(rc)

    def _visit_reward(self, A_pred, future_positions):
        R, C = A_pred.shape
        P = A_pred.astype(float); P = P / (P.sum() + 1e-12)
        rr, cc = np.meshgrid(np.arange(R), np.arange(C), indexing="ij")
        V = np.zeros_like(P)
        kr, kc = self.nb_kernel.shape
        pr, pc = kr//2, kc//2
        for d, (r, c) in enumerate(future_positions, start=1):
            # 커널을 (r,c) 중심으로 놓되 경계는 잘라붙임
            r0, r1 = r - pr, r + (kr - pr)
            c0, c1 = c - pc, c + (kc - pc)
            # 겹치는 부분만 더함
            rr0, rr1 = max(0, r0), min(R, r1)
            cc0, cc1 = max(0, c0), min(C, c1)
            kr0, kr1 = rr0 - r0, kr - (r1 - rr1)
            kc0, kc1 = cc0 - c0, kc - (c1 - cc1)
            V[rr0:rr1, cc0:cc1] += (self.gamma**(d-1)) * self.nb_kernel[kr0:kr1, kc0:kc1]
        V = V / (V.sum() + 1e-12)
        return float((P * V).sum())

    def _log_reward(self, A_pred, future_positions):
        R, C = A_pred.shape
        P = A_pred.astype(float); P = P / (P.sum() + 1e-12)
        kr, kc = self.nb_kernel.shape
        pr, pc = kr//2, kc//2
        best = -1e9
        for d, (r, c) in enumerate(future_positions, start=1):
            r0, r1 = r - pr, r + (kr - pr)
            c0, c1 = c - pc, c + (kc - pc)
            rr0, rr1 = max(0, r0), min(R, r1)
            cc0, cc1 = max(0, c0), min(C, c1)
            kr0, kr1 = rr0 - r0, kr - (r1 - rr1)
            kc0, kc1 = cc0 - c0, kc - (c1 - cc1)
            # 근방 확률질량 합산
            p_near = float((P[rr0:rr1, cc0:cc1] * self.nb_kernel[kr0:kr1, kc0:kc1]).sum())
            best = max(best, (self.gamma**(d-1)) * np.log(p_near + 1e-12))
        return best

    def pop_and_score(self, now_frame):
        """horizon이 지난 예측들을 꺼내 점수화."""
        scored = []
        while self.queue and (now_frame - self.queue[0][0]) >= self.horizon:
            t0, Ap = self.queue.popleft()

            # t0 이후 horizon 프레임의 실제 경로 시작 인덱스 계산
            # push_pos를 매 프레임 한 번씩 호출한다고 가정
            need = self.horizon
            # t0 이후 now_frame까지 기록된 위치 수
            recorded = min(len(self.path), now_frame - t0)
            if recorded <= 0:
                continue

            start = len(self.path) - recorded
            start = max(0, start)
            stop  = min(len(self.path), start + need)

            # deque는 슬라이싱 안 되므로 islice 사용
            future = list(islice(self.path, start, stop))
            if not future:
                continue

            r_visit = self._visit_reward(Ap, future)
            r_log   = self._log_reward(Ap, future)
            self.logs.append((t0, r_visit, r_log))
            scored.append((t0, r_visit, r_log))
        return scored

    def running_means(self, k=50):
        if not self.logs:
            return 0.0, 0.0
        arr = np.array(self.logs[-k:])
        return float(arr[:,1].mean()), float(arr[:,2].mean())


class OnlineBandit:
    """후보 파라미터 집합에서 소프트맥스 밴딧으로 온라인 적응."""
    def __init__(self, candidates, tau=0.8, lr=0.25, update_every=60):
        """
        candidates: [{'sigma':1.4,'eta_stm':0.2,'mom':1}, ...]
        tau: 소프트맥스 온도(높을수록 탐색↑)
        lr : 가중치 업데이트 속도
        update_every: 몇 프레임마다 업데이트할지(대략 6~10초에 한 번 권장)
        """
        self.cands = candidates
        self.tau = tau
        self.lr = lr
        self.update_every = update_every
        self.w = np.zeros(len(candidates), dtype=float)
        self.buf = []           # (idx, reward)
        self.cur_idx = 0
        self._last_update_at = 0

    def softmax(self):
        z = self.w / max(1e-6, self.tau)
        z -= z.max()
        p = np.exp(z)
        return p / p.sum()

    def sample(self):
        p = self.softmax()
        self.cur_idx = int(np.random.choice(len(self.cands), p=p))
        return self.cands[self.cur_idx]

    def record(self, r):
        self.buf.append((self.cur_idx, float(r)))

    def maybe_update(self, frame_now):
        if (frame_now - self._last_update_at) < self.update_every:
            return None
        if not self.buf:
            self._last_update_at = frame_now
            return None

        n = len(self.cands)
        avg = np.zeros(n)
        cnt = np.zeros(n)
        for i, r in self.buf:
            avg[i] += r
            cnt[i] += 1
        self.buf.clear()
        baseline = np.dot(self.softmax(), np.where(cnt>0, avg/np.clip(cnt,1,None), 0.0))
        for i in range(n):
            if cnt[i] > 0:
                avg_i = avg[i] / cnt[i]
                self.w[i] += self.lr * (avg_i - baseline)

        self._last_update_at = frame_now
        return self.sample()  # 새 후보 반환
    
class AttentionFuser:
    def __init__(self,
                 shape=(8,5),
                 fusion="log",
                 eta=0.4,
                 kappa=4.0, N=1.0,
                 sigma=1.0,
                 ior_sigma=1.0,
                 ior_strength=0.10,
                 ior_k=3,
                 momentum=True,
                 momentum_scale=1,
                 eta_min=0.05, eta_max=0.9,
                 # STM
                 stm_capacity=4,
                 stm_tau=5.0,
                 stm_sigma=1.2,
                 eta_stm=0.2,
                 # --- 온라인 적응 옵션 ---
                 scorer=None,         # PredictionScorer | None
                 bandit=None          # OnlineBandit     | None
                 ):
        self.shape = shape
        self.fusion = fusion
        self.eta = eta
        self.kappa = kappa
        self.N = N
        self.sigma = sigma
        self.ior_sigma = ior_sigma
        self.ior_strength = ior_strength
        self.ior_k = ior_k
        self.momentum = momentum
        self.momentum_scale = momentum_scale
        self.eta_min, self.eta_max = eta_min, eta_max

        # states
        self.A_prev = np.ones(shape, dtype=float) / (shape[0] * shape[1])
        self.fixations = []  # list of (r,c)
        self.hitrate_ma = 0.0
        self.last_fix = None
        self.prev_fix = None

        # STM
        self.stm = []  # list of dicts: {"p":(r,c), "s":strength, "t":frame_idx}
        self.stm_capacity = stm_capacity
        self.stm_tau = stm_tau
        self.stm_sigma = stm_sigma
        self.eta_stm = eta_stm
        self._frame = 0

        # momentum EMA
        self.vr, self.vc = 0.0, 0.0
        self.beta = 0.8

        # 온라인 적응 구성요소
        self.scorer = scorer
        self.bandit = bandit

    # --------- (기존 STM/보조 함수 동일) ---------

    def _stm_boost(self) -> np.ndarray:
        R, C = self.shape
        if len(self.stm) == 0:
            return np.ones((R, C), dtype=float)
        rr, cc = np.meshgrid(np.arange(R), np.arange(C), indexing="ij")
        B = np.zeros((R, C), dtype=float)
        for it in self.stm:
            age = max(0, self._frame - it["t"])
            s_eff = it["s"] * np.exp(-age / max(1e-6, self.stm_tau))
            if s_eff <= 1e-6:
                continue
            pr, pc = it["p"]
            spatial = np.exp(-((rr - pr)**2 + (cc - pc)**2) / (2 * self.stm_sigma**2))
            B += s_eff * spatial
        if B.max() > 0:
            B = 1.0 + (B / (B.max() + 1e-9))
        else:
            B = np.ones((R, C), dtype=float)
        return B

    def _stm_encode(self, fixation_rc, boost=1.0):
        r, c = fixation_rc
        for it in self.stm:
            pr, pc = it["p"]
            if abs(pr - r) + abs(pc - c) <= 1:
                it["s"] = min(1.0, it["s"] + 0.3 * boost)
                it["t"] = self._frame
                break
        else:
            self.stm.append({"p": (r, c), "s": 0.6 * boost, "t": self._frame})
            if len(self.stm) > self.stm_capacity:
                self.stm.sort(key=lambda z: z["s"])
                self.stm.pop(0)
        # prune
        cleaned = []
        for it in self.stm:
            age = max(0, self._frame - it["t"])
            s_eff = it["s"] * np.exp(-age / max(1e-6, self.stm_tau))
            if s_eff > 1e-3:
                cleaned.append(it)
        self.stm = cleaned

    def _drift_from_momentum(self):
        if len(self.fixations) < 2 or not self.momentum:
            return (0, 0)
        (r1, c1), (r0, c0) = self.fixations[-1], self.fixations[-2]
        dr, dc = (r1 - r0), (c1 - c0)
        self.vr = self.beta * self.vr + (1 - self.beta) * dr
        self.vc = self.beta * self.vc + (1 - self.beta) * dc
        return (int(np.sign(self.vr)) * self.momentum_scale,
                int(np.sign(self.vc)) * self.momentum_scale)

    def adapt_eta(self, A_post_hat: np.ndarray):
        H = entropy(A_post_hat)
        Hmax = math.log(A_post_hat.size + EPS)
        x = 1.0 - (H / (Hmax + EPS))   # confidence proxy
        eta = self.eta_min + (self.eta_max - self.eta_min) * (1.0 - x)
        return float(np.clip(eta, self.eta_min, self.eta_max))

    # ---------- 온라인 적응에서 쓰는 편의 메서드 ----------
    def set_params(self, sigma=None, eta_stm=None, momentum_scale=None):
        if sigma is not None:
            self.sigma = float(sigma)
        if eta_stm is not None:
            self.eta_stm = float(eta_stm)
        if momentum_scale is not None:
            self.momentum_scale = int(momentum_scale)

    def note_position(self, rc):
        """
        외부(env)에서 실제 위치(rc)를 알려줌 -> scorer가 horizon 지난 예측을 점수화.
        점수가 나오면 밴딧에 기록하고, 주기적으로 파라미터 갱신.
        """
        if self.scorer is None:
            return []
        self.scorer.push_pos(rc)
        scored = self.scorer.pop_and_score(self._frame)  # [(t0, r_visit, r_log), ...]

        if self.bandit and scored:
            for (_, r_visit, r_log) in scored:
                r = 0.85 * r_visit + 0.15 * max(-5.0, min(0.0, r_log)) / 5.0
                self.bandit.record(r)
            maybe_cfg = self.bandit.maybe_update(self._frame)
            if maybe_cfg is not None:
                self.set_params(sigma=maybe_cfg["sigma"],
                                eta_stm=maybe_cfg["eta_stm"],
                                momentum_scale=maybe_cfg["mom"])
        return scored

    def get_prediction_metrics(self, k=50):
        if self.scorer is None:
            return 0.0, 0.0
        return self.scorer.running_means(k=k)

    # --------- 메인 스텝 ----------
    def step(self, A_post_hat: np.ndarray, hit: float = None, use_adaptive_eta=False):
        assert A_post_hat.shape == self.shape

        # IOR & drift
        ior_mask = build_ior_mask(self.shape, self.fixations,
                                  sigma=self.ior_sigma,
                                  strength=self.ior_strength,
                                  k_last=self.ior_k)
        drift = self._drift_from_momentum()

        # prior
        prior_pi = predict_prior(self.A_prev, sigma=self.sigma, drift=drift, ior_mask=ior_mask)

        # STM
        B_t = self._stm_boost()

        A_post_hat = normalize(np.clip(A_post_hat, 0, None) ** 0.85)

        # fuse
        eta_used = self.adapt_eta(A_post_hat) if use_adaptive_eta else self.eta
        if self.fusion == "log":
            A_post_hat = normalize(A_post_hat)
            prior_pi   = normalize(prior_pi)
            logA = (1 - eta_used) * np.log(A_post_hat + EPS) + eta_used * np.log(prior_pi + EPS) \
                   + self.eta_stm * np.log(B_t + EPS)
            A_t = np.exp(logA - logA.max())
            A_t = normalize(A_t)
        else:
            A_t = fuse_dirichlet(A_post_hat, prior_pi, kappa=self.kappa, N=self.N)
            A_t = normalize(A_t * (B_t ** self.eta_stm))

        # --- 온라인 적응: 예측 분포를 scorer에 즉시 기록
        if self.scorer is not None:
            self.scorer.push_pred(A_t, self._frame)

        # sample fixation
        flat = A_t.ravel()
        idx = np.random.choice(flat.size, p=flat)
        r, c = np.unravel_index(idx, self.shape)

        # traces
        self.fixations.append((r, c))
        if len(self.fixations) > 32:
            self.fixations.pop(0)
        self.prev_fix = self.last_fix
        self.last_fix = (r, c)
        if hit is not None:
            self.hitrate_ma = 0.9 * self.hitrate_ma + 0.1 * hit

        # STM encode
        self._stm_encode((r, c), boost=1.0)

        # roll
        self.A_prev = A_t.copy()
        self._frame += 1

        return A_t, (r, c), prior_pi, eta_used

