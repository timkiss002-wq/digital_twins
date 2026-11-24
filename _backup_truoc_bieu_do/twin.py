"""
twin.py
-------
Lõi DIGITAL TWIN của nhà xe: vòng kín Thực -> Ảo -> Thực.
Thuần Python (không phụ thuộc YOLO/OpenCV/FastAPI) để kiểm thử riêng.

Ba thành phần
=============
1. PhysicalGates  - HỆ CỔNG "THỰC".
   Vì đang dùng video thay thế (không có cổng thật để điều khiển), cổng được
   mô phỏng theo kiểu trace-driven: DÒNG XE ĐẾN là dữ liệu THẬT (thời điểm từng
   xe vào vùng chờ, đo bằng YOLO trên video); phần PHỤC VỤ tại cổng là mô phỏng
   (FIFO, c cổng Vào, thời gian phục vụ ~ mũ với tốc độ μ). Lệnh mở/đóng cổng
   từ bản ảo tác động lên hệ này, nên tác động của lệnh đo được.
   -> Khi có cổng thật, thay lớp này bằng dữ liệu cảm biến/barrier thật.

2. VirtualModel   - BẢN ẢO: mô hình động học của hàng đợi
       - ρ = λ/(cμ) < 1 : q hồi dần về L = λ·W(M/M/c) với hằng số thời gian
                          τ = max(1/μ, 1/(cμ - λ))  (gần quá tải -> hồi chậm)
       - ρ >= 1         : dq/dt = λ - cμ (hàng chờ tăng tuyến tính)
   - Đồng bộ (Thực -> Ảo) mỗi `sync_s` giây: gán q_ảo = q_thực, cập nhật λ
     (đo từ video) và μ (hiệu chỉnh từ thời gian chờ đo được, M/M/c).
   - Ngay trước mỗi lần đồng bộ, so q_ảo (đã chạy tự do từ lần đồng bộ trước)
     với q_thực -> sai số đồng bộ -> MAE = ĐỘ KHỚP của bản ảo.
   - Dự báo q sau `horizon_s` giây, lưu lại và chấm điểm khi đến hạn.

3. DecisionEngine - Ảo -> Thực: luật ngưỡng trên ρ = λ/(cμ) và hàng chờ dự báo.
   Đề xuất +1 / -1 cổng Vào; người vận hành chấp nhận / bỏ qua; lệnh được thực
   thi sau `actuation_delay_s` (thời gian đổi barrier). Tác động được đo bằng
   ĐỐI CHỨNG: tại thời điểm thực thi, tạo bản sao hệ cổng giữ số cổng cũ; hai hệ
   nhận cùng dòng xe đến và cùng thời gian phục vụ (common random numbers). Sau
   `effect_window_s` giây so TG chờ TB của xe đến trong cửa sổ: có lệnh vs không.

Mọi thời gian tính bằng GIÂY VIDEO (tăng đơn điệu, kể cả khi video lặp).
"""
from __future__ import annotations

import bisect
import random
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import copy
import math

from analytics import calibrate_mu, mmc_time_in_system

TOTAL_GATES = 4          # tổng số cổng (Vào + Ra)
MIN_ENTRY = 1
MAX_ENTRY = 3            # luôn giữ ít nhất 1 cổng Ra


@dataclass
class TwinConfig:
    baseline_gates: int = 2
    sync_s: float = 2.0             # chu kỳ đồng bộ Thực -> Ảo
    horizon_s: float = 30.0         # tầm dự báo
    dt: float = 0.25                # bước tích phân mô hình ảo
    rho_high: float = 0.75          # ρ vượt ngưỡng -> đề xuất mở thêm cổng
    rho_low: float = 0.50           # ρ (với c-1 cổng) dưới ngưỡng -> đề xuất bớt cổng
    queue_high: float = 8.0         # hàng chờ dự báo vượt ngưỡng -> mở thêm cổng
    cooldown_s: float = 20.0        # khoảng nghỉ giữa hai đề xuất
    actuation_delay_s: float = 3.0  # thời gian thực thi lệnh ở cổng
    effect_window_s: float = 30.0   # cửa sổ đo tác động (có lệnh vs đối chứng)
    min_samples: int = 5            # số mẫu TG chờ tối thiểu để hiệu chỉnh μ
    default_mu: float = 0.5         # μ mặc định (xe/giây/cổng) khi chưa đủ dữ liệu
    history_s: float = 120.0        # độ dài lịch sử giữ cho biểu đồ
    seed: int = 7


# ============================================================
# 1. HỆ CỔNG "THỰC" (trace-driven)
# ============================================================
class PhysicalGates:
    def __init__(self, gates: int, mu: float, seed: int = 7):
        self.gates = gates
        self.mu = mu
        self._rng = random.Random(seed)
        self._free: dict = {g: 0.0 for g in range(1, gates + 1)}   # cổng số g -> thời điểm rảnh
        self._departs: list[float] = []             # thời điểm rời của xe còn trong hệ (đã sắp xếp)
        self._jobs: deque = deque()                 # xe còn trong hệ: (t_đến, t_bắt_đầu, t_rời, cổng, số xe)
        self.done: deque = deque(maxlen=5000)       # (t_đến, t_rời, thời gian trong hệ)
        self.arrived = 0

    def set_gates(self, n: int, t: float) -> None:
        """Cổng Vào được đánh số 1..n (khớp C1..Cn trên giao diện).
        Thêm cổng: cổng n+1 sẵn sàng từ t. Bớt cổng: cổng số lớn nhất ngừng nhận xe mới,
        xe đang được phục vụ ở cổng đó vẫn hoàn tất."""
        while len(self._free) < n:
            self._free[len(self._free) + 1] = t
        while len(self._free) > n:
            self._free.pop(max(self._free))
        self.gates = n

    def draw_service(self) -> float:
        return self._rng.expovariate(self.mu)

    def arrive(self, t: float, service: Optional[float] = None) -> float:
        """Xe đến lúc t; trả về thời gian phục vụ đã dùng (để hệ đối chứng dùng lại)."""
        s = self.draw_service() if service is None else service
        g = min(self._free, key=lambda k: (self._free[k], k))   # cổng rảnh sớm nhất (FIFO chung)
        start = max(t, self._free[g])
        dep = start + s
        self._free[g] = dep
        bisect.insort(self._departs, dep)
        self.arrived += 1
        self._jobs.append((t, start, dep, g, self.arrived))
        self.done.append((t, dep, dep - t))
        return s

    def state(self, t: float) -> dict:
        """Ảnh chụp hệ cổng tại t để hiển thị: xe đang xếp hàng và xe đang qua từng cổng."""
        while self._jobs and self._jobs[0][2] <= t and self._jobs[0][1] <= t:
            self._jobs.popleft()
        waiting, serving = [], {g: None for g in self._free}
        for arr, start, dep, g, vid in self._jobs:
            if arr > t or dep <= t:
                continue
            if start > t:
                waiting.append({"id": vid, "waited_s": round(t - arr, 1)})
            else:
                serving[g] = {"id": vid, "progress": round(min(1.0, (t - start) / max(1e-6, dep - start)), 2),
                              "remain_s": round(dep - t, 1)}
        return {"waiting": waiting, "serving": [{"gate": g, "car": v} for g, v in sorted(serving.items())],
                "served_total": sum(1 for d in self.done if d[1] <= t)}

    def queue(self, t: float) -> int:
        """Số xe đang trong hệ (chờ + đang được phục vụ) tại t."""
        k = bisect.bisect_right(self._departs, t)
        if k:
            del self._departs[:k]
        return len(self._departs)

    def mean_wait(self, t0: float, t1: float, by: str = "depart") -> Optional[float]:
        """TG trong hệ TB của xe rời (by='depart') hoặc đến (by='arrive') trong (t0, t1]."""
        k = 1 if by == "depart" else 0
        v = [rec[2] for rec in self.done if t0 < rec[k] <= t1]
        return sum(v) / len(v) if v else None


# ============================================================
# 2. BẢN ẢO (mô hình fluid)
# ============================================================
def fluid_step(q: float, lam: float, mu: float, c: int, dt: float) -> float:
    """Một bước của mô hình ảo (xem mô tả đầu file)."""
    if mu <= 0:
        return q
    if lam < c * mu:
        L = lam * (mmc_time_in_system(lam, mu, c) or 0.0)
        tau = max(1.0 / mu, 1.0 / (c * mu - lam))
        return q + (L - q) * (1.0 - math.exp(-dt / tau))
    return max(0.0, q + (lam - c * mu) * dt)


def fluid_forecast(q0: float, lam: float, mu: float, c: int, horizon: float, dt: float) -> float:
    q = q0
    for _ in range(int(round(horizon / dt))):
        q = fluid_step(q, lam, mu, c, dt)
    return q


# ============================================================
# 3. DIGITAL TWIN (vòng kín)
# ============================================================
@dataclass
class Recommendation:
    id: int
    t: float
    from_gates: int
    to_gates: int
    reason: str
    forecast_now: float
    forecast_new: float
    status: str = "pending"        # pending | accepted | rejected | applied
    apply_t: Optional[float] = None
    manual: bool = False           # True: lệnh do người vận hành tự điều chỉnh


def _r(v, nd=2):
    return None if v is None else round(v, nd)


class DigitalTwin:
    def __init__(self, config: Optional[TwinConfig] = None):
        self.cfg = config or TwinConfig()
        c = self.cfg
        self.mu = c.default_mu
        self.mu_calibrated = False
        self.lam = 0.0
        self.physical = PhysicalGates(c.baseline_gates, self.mu, c.seed)
        self.virtual_gates = c.baseline_gates
        self.q_virtual = 0.0
        self.last_sync_t: Optional[float] = None
        self.sync_count = 0
        self.sync_errors: deque = deque(maxlen=60)       # |q_ảo - q_thực| trước mỗi lần đồng bộ
        self.forecasts: deque = deque()                  # (t_đến_hạn, q_dự_báo, t_lập)
        self.forecast_errors: deque = deque(maxlen=60)
        self.history: deque = deque()                    # (t, q_thực, q_ảo trước đồng bộ)
        self.events: deque = deque(maxlen=200)           # nhật ký hai chiều
        self.recommendation: Optional[Recommendation] = None
        self._rec_seq = 0
        self._last_decision_t = -1e9
        self.commands: list = []                         # các mốc đã đổi cổng (t, từ, sang)
        self._pending_effect: list = []                  # (t_áp_dụng, rec)
        self.t = 0.0

    # ---------------- nhật ký ----------------
    def _log(self, t: float, direction: str, text: str) -> None:
        self.events.append({"t": round(t, 2), "dir": direction, "text": text})

    # ---------------- vòng lặp chính ----------------
    def step(self, t: float, arrivals: list, lam_obs: Optional[float],
             w_obs: Optional[float], n_w_samples: int) -> None:
        """Gọi mỗi khung hình. arrivals: thời điểm các xe qua vạch 2, tức tới cổng (giây video)."""
        c = self.cfg
        dt_frame = 0.0 if self.t is None else max(0.0, t - self.t)
        self.t = t
        for a in arrivals:
            s = self.physical.arrive(a)                   # dữ liệu thật đi vào hệ cổng
            for _, _, cf in self._pending_effect:         # hệ đối chứng: cùng xe, cùng TG phục vụ
                cf.arrive(a, s)
        self._apply_due_commands(t)
        # bản ảo chạy tự do giữa hai lần đồng bộ
        if dt_frame > 0:
            n = max(1, int(dt_frame / c.dt))
            for _ in range(n):
                self.q_virtual = fluid_step(self.q_virtual, self.lam, self.mu,
                                            self.virtual_gates, dt_frame / n)
        if self.last_sync_t is None or t - self.last_sync_t >= c.sync_s:
            self._sync(t, lam_obs, w_obs, n_w_samples)
        self._score_forecasts(t)
        self._measure_effects(t)

    # ---------------- Thực -> Ảo ----------------
    def _sync(self, t, lam_obs, w_obs, n_w_samples):
        c = self.cfg
        q_real = self.physical.queue(t)
        q_pre = self.q_virtual                       # giá trị ảo chạy tự do, trước khi đồng bộ
        if self.last_sync_t is not None:
            self.sync_errors.append(abs(q_pre - q_real))
        if lam_obs is not None:
            self.lam = lam_obs
        if (w_obs and n_w_samples >= c.min_samples and self.lam > 0):
            # μ hiệu chỉnh từ video với số cổng gốc (bố cục khi quay video)
            self.mu = calibrate_mu(self.lam, w_obs, c.baseline_gates)
            self.mu_calibrated = True
            self.physical.mu = self.mu
        self.q_virtual = float(q_real)
        self.last_sync_t = t
        self.sync_count += 1
        fc = fluid_forecast(q_real, self.lam, self.mu, self.virtual_gates, c.horizon_s, c.dt)
        self.forecasts.append((t + c.horizon_s, fc, t))
        self.history.append((t, q_real, round(q_pre, 2)))
        while self.history and self.history[0][0] < t - c.history_s:
            self.history.popleft()
        if self.sync_count == 1 or self.sync_count % 5 == 0:
            self._log(t, "T2A", f"Đồng bộ: {q_real} xe tại cổng, λ={self.lam*60:.0f} xe/phút, "
                                f"μ={self.mu*60:.0f} xe/phút/cổng")
        self._decide(t, q_real, fc)

    def _score_forecasts(self, t):
        q_real = None
        while self.forecasts and self.forecasts[0][0] <= t:
            _, fc, _ = self.forecasts.popleft()
            if q_real is None:
                q_real = self.physical.queue(t)
            self.forecast_errors.append(abs(fc - q_real))

    # ---------------- Ảo -> Thực ----------------
    def rho(self, gates: Optional[int] = None) -> Optional[float]:
        g = gates or self.virtual_gates
        if self.mu <= 0:
            return None
        return self.lam / (g * self.mu)

    def _decide(self, t, q_real, fc):
        c = self.cfg
        if self.recommendation and self.recommendation.status in ("pending", "accepted"):
            return
        if t - self._last_decision_t < c.cooldown_s or not self.mu_calibrated:
            return
        g = self.virtual_gates
        rho = self.rho(g)
        target, reason = None, None
        if g < MAX_ENTRY and (rho >= c.rho_high or fc >= c.queue_high):
            target = g + 1
            reason = (f"ρ={rho:.2f} ≥ {c.rho_high}" if rho >= c.rho_high
                      else f"dự báo {fc:.0f} xe sau {c.horizon_s:.0f}s ≥ {c.queue_high:.0f}")
        elif g > MIN_ENTRY and g > c.baseline_gates and self.rho(g - 1) < c.rho_low \
                and fc < c.queue_high / 2:
            target = g - 1
            reason = f"tải thấp: ρ với {g-1} cổng = {self.rho(g-1):.2f} < {c.rho_low}"
        if target is None:
            return
        self._rec_seq += 1
        fc_new = fluid_forecast(q_real, self.lam, self.mu, target, c.horizon_s, c.dt)
        self.recommendation = Recommendation(self._rec_seq, t, g, target, reason,
                                             round(fc, 1), round(fc_new, 1))
        self._last_decision_t = t
        verb = "Mở thêm" if target > g else "Đóng bớt"
        self._log(t, "A2T", f"Đề xuất: {verb} cổng Vào ({g}→{target}). Lý do: {reason}. "
                            f"Dự báo {c.horizon_s:.0f}s: {fc:.1f} → {fc_new:.1f} xe")

    def manual_set(self, target: int) -> tuple:
        """Người vận hành tự mở / đóng cổng Vào (Ảo -> Thực, thủ công).
        Trả về (ok, thông báo). Đề xuất đang chờ (nếu có) bị thay thế."""
        c = self.cfg
        if not MIN_ENTRY <= target <= MAX_ENTRY:
            return False, f"Số cổng Vào phải từ {MIN_ENTRY} đến {MAX_ENTRY} (luôn giữ ít nhất 1 cổng Ra)"
        r = self.recommendation
        if r is not None and r.status == "accepted":
            return False, "Đang có lệnh chuyển cổng chưa hoàn tất, vui lòng chờ"
        g = self.virtual_gates
        if target == g:
            return False, f"Đang có sẵn {g} cổng Vào"
        if r is not None and r.status == "pending":
            r.status = "rejected"
            self._log(self.t, "A2T", "Đề xuất tự động bị thay bằng lệnh thủ công")
        q = self.physical.queue(self.t)
        fc = fluid_forecast(q, self.lam, self.mu, g, c.horizon_s, c.dt)
        fc_new = fluid_forecast(q, self.lam, self.mu, target, c.horizon_s, c.dt)
        self._rec_seq += 1
        self.recommendation = Recommendation(
            self._rec_seq, self.t, g, target, "Người vận hành điều chỉnh thủ công",
            round(fc, 1), round(fc_new, 1), status="accepted",
            apply_t=self.t + c.actuation_delay_s, manual=True)
        self._last_decision_t = self.t
        verb = "Mở thêm" if target > g else "Đóng bớt"
        self._log(self.t, "A2T", f"Lệnh thủ công: {verb} cổng Vào ({g}→{target}) → gửi tới cổng. "
                                 f"Dự báo {c.horizon_s:.0f}s: {fc:.1f} → {fc_new:.1f} xe")
        return True, "ok"

    def respond(self, rec_id: int, accept: bool) -> bool:
        r = self.recommendation
        if r is None or r.id != rec_id or r.status != "pending":
            return False
        if accept:
            r.status = "accepted"
            r.apply_t = self.t + self.cfg.actuation_delay_s
            self._log(self.t, "A2T", f"Đã chấp nhận → gửi lệnh đổi {r.from_gates}→{r.to_gates} cổng Vào")
        else:
            r.status = "rejected"
            self._log(self.t, "A2T", "Người vận hành bỏ qua đề xuất")
        self._last_decision_t = self.t
        return True

    def _apply_due_commands(self, t):
        r = self.recommendation
        if r and r.status == "accepted" and r.apply_t is not None and t >= r.apply_t:
            counterfactual = copy.deepcopy(self.physical)     # giữ số cổng cũ
            self.physical.set_gates(r.to_gates, t)
            self.virtual_gates = r.to_gates
            r.status = "applied"
            self.commands.append((round(t, 2), r.from_gates, r.to_gates))
            self._pending_effect.append((t, r, counterfactual))
            self._log(t, "A2T", f"Cổng đã chuyển: {r.to_gates} cổng Vào đang hoạt động")

    def _measure_effects(self, t):
        w = self.cfg.effect_window_s
        keep = []
        for item in self._pending_effect:
            t_apply, r, cf = item
            # chờ thêm để phần lớn xe đến trong cửa sổ đã rời hệ
            if t - t_apply < w * 1.5:
                keep.append(item)
                continue
            got = self.physical.mean_wait(t_apply, t_apply + w, by="arrive")
            base = cf.mean_wait(t_apply, t_apply + w, by="arrive")
            fmt = lambda v: "—" if v is None else f"{v:.1f}s"
            pct = f" ({(got / base - 1) * 100:+.0f}%)" if got and base else ""
            self._log(t, "T2A", f"Tác động lệnh {r.from_gates}→{r.to_gates}: TG chờ TB {fmt(got)} "
                                f"so với {fmt(base)} nếu giữ {r.from_gates} cổng{pct}")
        self._pending_effect = keep

    # ---------------- truy vấn ----------------
    def trajectory(self, q0: float, gates: int, step_s: float = 2.0) -> list:
        """Quỹ đạo dự báo [(t, q)] từ hiện tại tới t + horizon với số cổng cho trước."""
        c = self.cfg
        out, q = [[round(self.t, 2), round(q0, 2)]], float(q0)
        for k in range(1, int(c.horizon_s / step_s) + 1):
            q = fluid_forecast(q, self.lam, self.mu, gates, step_s, c.dt)
            out.append([round(self.t + k * step_s, 2), round(q, 2)])
        return out

    def snapshot(self) -> dict:
        r = self.recommendation
        mae = (sum(self.sync_errors) / len(self.sync_errors)) if self.sync_errors else None
        fmae = (sum(self.forecast_errors) / len(self.forecast_errors)) if self.forecast_errors else None
        q_real = self.physical.queue(self.t)
        rho = self.rho()
        return {
            "gates": self.virtual_gates,
            "q_real": q_real,
            "q_virtual": round(self.q_virtual, 2),
            "lambda_per_min": round(self.lam * 60, 1),
            "mu_per_min": round(self.mu * 60, 1),
            "mu_calibrated": self.mu_calibrated,
            "rho": round(rho, 3) if rho is not None else None,
            "sync_mae": round(mae, 2) if mae is not None else None,
            "forecast_mae": round(fmae, 2) if fmae is not None else None,
            "sync_s": self.cfg.sync_s,
            "horizon_s": self.cfg.horizon_s,
            "last_sync_t": self.last_sync_t,
            "gate_state": self.physical.state(self.t),
            "wait_recent_s": _r(self.physical.mean_wait(self.t - self.cfg.effect_window_s, self.t)),
            "forecast": self.trajectory(q_real, self.virtual_gates),
            "forecast_alt": (self.trajectory(q_real, r.to_gates)
                             if r is not None and r.status == "pending" else None),
            "recommendation": None if r is None or r.status in ("rejected", "applied") else {
                "id": r.id, "from": r.from_gates, "to": r.to_gates, "reason": r.reason,
                "forecast_now": r.forecast_now, "forecast_new": r.forecast_new, "status": r.status,
                "manual": r.manual},
            "history": [list(h) for h in self.history],
            "commands": list(self.commands),
            "events": list(self.events)[-12:][::-1],
        }
