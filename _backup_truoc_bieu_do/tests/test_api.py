"""
Test API + worker với VIDEO GIẢ LẬP và DETECTOR GIẢ LẬP (không chạy YOLO).

Giới hạn: kiểm tra luồng dữ liệu, trạng thái worker, chuyển nguồn và tên/kiểu
KPI. Không kiểm tra chất lượng phát hiện của YOLO.

Kịch bản giả lập (khung 400x200, xe đi từ trên xuống):
  - Vạch 1 ở y = 50, vạch 2 ở y = 150.
  - Xe "đỗ" (track 1000) đứng yên trong vùng ở y = 100 suốt video.
  - Cứ 10 khung có 1 xe mới xuất hiện ở y = 0 và đi xuống 6 px/khung,
    nên lần lượt vào vùng, rời qua vạch 2 rồi ra khỏi khung hình.
"""
import csv
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

import app as appmod
from queue_zone import Detection

H, W, FPS = 200, 400, 10.0


class FakeCapture:
    def __init__(self, path, n_frames=10_000):
        self.path = path
        self.n = n_frames
        self.idx = 0
        self.released = False

    def isOpened(self):
        return True

    def read(self):
        if self.idx >= self.n:
            return False, None
        self.idx += 1
        return True, np.zeros((H, W, 3), dtype=np.uint8)

    def get(self, prop):
        import cv2
        if prop == cv2.CAP_PROP_FPS:
            return FPS
        if prop == cv2.CAP_PROP_POS_FRAMES:
            return float(self.idx)
        return 0.0

    def set(self, prop, val):
        self.idx = int(val)

    def release(self):
        self.released = True


class FakeDetector:
    instances = []

    def __init__(self, profile):
        self.profile = profile
        self.frame = 0
        FakeDetector.instances.append(self)

    def __call__(self, frame, reset=False):
        if reset:
            self.frame = 0
        self.frame += 1
        f = self.frame
        dets = [Detection(1000, (180, 70, 220, 100), 2)]   # xe đứng yên trong vùng
        for k in range(0, f, 10):
            y = (f - k) * 6
            if y <= H + 40:
                tid = k + 1
                dets.append(Detection(tid, (50 + (k % 30) * 10, y - 30, 90 + (k % 30) * 10, min(y, H - 1)), 2))
        return dets


TEST_PROFILES = {
    "test": {"label": "fake A", "path": "fake_a.mp4",
             "entry_line": ((0, 0.25), (1, 0.25)), "exit_line": ((0, 0.75), (1, 0.75)),
             "max_seconds": None, "loop": True},
    "test2": {"label": "fake B", "path": "fake_b.mp4",
              "entry_line": ((0, 0.25), (1, 0.25)), "exit_line": ((0, 0.75), (1, 0.75)),
              "max_seconds": None, "loop": True},
}


def wait_until(cond, timeout=5.0, step=0.02):
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return True
        time.sleep(step)
    return False


@pytest.fixture
def client(monkeypatch, tmp_path):
    FakeDetector.instances.clear()
    monkeypatch.setattr(appmod, "VIDEO_PROFILES", TEST_PROFILES)
    monkeypatch.setattr(appmod, "DEFAULT_PROFILE", "test")
    monkeypatch.setattr(appmod, "make_detector", FakeDetector)
    monkeypatch.setattr(appmod, "open_capture", lambda path: FakeCapture(path))
    monkeypatch.setattr(appmod, "REAL_CSV_PATH", str(tmp_path / "real.csv"))
    monkeypatch.setattr(appmod, "SLOWDOWN", 0.05)     # 5 ms / khung
    monkeypatch.setattr(appmod, "STATE", appmod.AppState())
    with TestClient(appmod.app) as c:
        yield c


def status(c):
    return c.get("/api/status").json()


def frames(c):
    return status(c)["worker"]["frames_processed"]


# ---------------------------------------------------------------- 7
def test_worker_starts_paused_and_does_not_process_frames(client):
    assert wait_until(lambda: status(client)["worker"]["state"] == "paused")
    time.sleep(0.2)
    assert frames(client) == 0
    s = client.get("/api/stats").json()
    assert s["system_running"] is False
    assert s["worker"]["state"] == "paused"


def test_pause_and_resume_actually_control_the_worker(client):
    r = client.post("/api/control/start").json()
    assert r["system_running"] is True
    assert wait_until(lambda: status(client)["worker"]["state"] == "running")
    assert wait_until(lambda: frames(client) > 20)

    r = client.post("/api/control/pause").json()
    assert r["system_running"] is False
    assert wait_until(lambda: status(client)["worker"]["state"] == "paused")
    n1 = frames(client)
    time.sleep(0.4)
    n2 = frames(client)
    assert n2 == n1, "worker vẫn xử lý khung hình sau khi tạm dừng"
    assert client.get("/api/stats").json()["worker"]["state"] == "paused"

    client.post("/api/control/start")
    assert wait_until(lambda: frames(client) > n2 + 10)
    assert status(client)["worker"]["state"] == "running"


def test_invalid_control_action(client):
    assert client.post("/api/control/explode").status_code == 400


def test_worker_error_is_reported_and_restart_recovers(client, monkeypatch):
    def broken(profile):
        raise RuntimeError("không tải được model")
    monkeypatch.setattr(appmod, "make_detector", broken)
    client.post("/api/video/test2")
    assert wait_until(lambda: status(client)["worker"]["state"] == "error")
    assert "model" in status(client)["worker"]["error"]
    monkeypatch.setattr(appmod, "make_detector", FakeDetector)
    client.post("/api/control/start")      # khởi động lại worker lỗi
    assert wait_until(lambda: status(client)["worker"]["state"] == "running")


# ---------------------------------------------------------------- 8
def test_switch_video_stops_old_worker_and_starts_fresh(client):
    client.post("/api/control/start")
    assert wait_until(lambda: frames(client) > 30)
    old_worker = appmod.STATE.worker
    old_gen = old_worker.generation

    r = client.post("/api/video/test2").json()
    assert r["profile"] == "test2"
    assert not old_worker.is_alive(), "worker cũ vẫn còn chạy"
    assert old_worker.status()["state"] == "stopped"
    st = status(client)["worker"]
    assert st["generation"] == old_gen + 1
    assert st["profile"] == "test2"
    # worker mới bắt đầu từ đầu, không mang KPI của worker cũ
    s = client.get("/api/stats").json()
    assert s["worker"]["generation"] == old_gen + 1
    assert wait_until(lambda: frames(client) > 5)
    assert client.post("/api/video/nope").status_code == 400


# ---------------------------------------------------------------- 9
def test_stats_kpi_names_and_types(client):
    client.post("/api/control/start")
    # đủ lâu để vài xe đi qua vạch 2 (mỗi xe cần ~25 khung)
    assert wait_until(lambda: (client.get("/api/stats").json()["kpi"]["departed_total"] or 0) >= 3, timeout=10)
    s = client.get("/api/stats").json()
    assert "total_vehicles" not in s, "tên KPI cũ gây hiểu nhầm phải được loại bỏ"
    k = s["kpi"]
    for name in ("queue_count", "queue_occluded", "total_detected", "entered_total",
                 "departed_total", "left_other_total", "lost_in_zone_total",
                 "wait_samples_in_window", "window_s"):
        assert isinstance(k[name], int), name
    for name in ("avg_wait_s", "arrival_rate_per_min"):
        assert k[name] is None or isinstance(k[name], (int, float)), name
    # Xe đứng yên luôn trong vùng -> ít nhất 1 xe đang chờ
    assert k["queue_count"] >= 1
    # Số xe đang chờ KHÔNG phải tổng xe đã ghi nhận / đã đi qua
    assert k["queue_count"] < k["total_detected"]
    assert k["queue_count"] <= k["entered_total"] - k["departed_total"] - k["left_other_total"] + 1
    assert s["twin"]["gates"] == 2 and isinstance(s["twin"]["history"], list)


def test_real_csv_contains_wait_samples(client):
    client.post("/api/control/start")
    assert wait_until(lambda: (client.get("/api/stats").json()["kpi"]["departed_total"] or 0) >= 2, timeout=10)
    client.post("/api/control/pause")
    with open(appmod.REAL_CSV_PATH) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) >= 2
    for r in rows:
        assert float(r["wait_in_zone_s"]) == pytest.approx(float(r["exit_video_t"]) - float(r["enter_video_t"]), abs=1e-2)


# ---------------------------------------------------------------- 10
def _write_sim(n, wait, spacing=1.5):
    with open(appmod.SIM_CSV_PATH, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timestamp", "travel_time"])
        now = time.time()
        for i in range(n):
            w.writerow([now - spacing * (n - i), wait])


def test_twin_is_synced_from_video_and_exposed(client):
    client.post("/api/control/start")
    assert wait_until(lambda: ((client.get("/api/stats").json()["twin"] or {}).get("last_sync_t") or 0) > 3,
                      timeout=10)
    tw = client.get("/api/stats").json()["twin"]
    assert tw["gates"] == appmod.BASELINE_ENTRANCE_GATES
    assert tw["history"] and len(tw["forecast"]) > 1
    assert any(e["dir"] == "T2A" for e in tw["events"])


def test_recommendation_endpoint_validation(client):
    assert client.post("/api/twin/recommendation/1/explode").status_code == 400
    assert client.post("/api/twin/recommendation/99/accept").status_code == 409


def test_recommendation_accept_reaches_physical_gates(client):
    client.post("/api/control/start")
    assert wait_until(lambda: client.get("/api/stats").json()["twin"]["sync_mae"] is not None, timeout=10)
    w = appmod.STATE.worker
    with w.lock:                       # ép bản ảo thấy quá tải để sinh đề xuất
        w.twin.mu, w.twin.mu_calibrated, w.twin.lam = 0.1, True, 0.5
        w.twin._last_decision_t = -1e9
    assert wait_until(lambda: client.get("/api/stats").json()["twin"]["recommendation"] is not None, timeout=10)
    rec = client.get("/api/stats").json()["twin"]["recommendation"]
    assert client.post(f"/api/twin/recommendation/{rec['id']}/accept").status_code == 200
    assert wait_until(lambda: client.get("/api/stats").json()["twin"]["gates"] == rec["to"], timeout=10)
    assert w.twin.physical.gates == rec["to"]


def test_manual_gate_endpoint(client):
    client.post("/api/control/start")
    assert wait_until(lambda: client.get("/api/stats").json()["twin"]["last_sync_t"] is not None, timeout=10)
    assert client.post("/api/twin/gates/4").status_code == 409
    assert client.post("/api/twin/gates/3").status_code == 200
    assert wait_until(lambda: client.get("/api/stats").json()["twin"]["gates"] == 3, timeout=10)
    assert appmod.STATE.worker.twin.physical.gates == 3
