import cv2
import csv
import os
import time
import threading
from collections import deque
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from analytics import rolling_mean, rolling_rate, rolling_values
from queue_zone import CounterConfig, QueueCounter, ZoneGeometry
from twin import DigitalTwin, TwinConfig
from vision import draw_overlay   # vision chỉ import ultralytics khi tạo YoloDetector

# ============================================================
# CẤU HÌNH CHUNG
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))   # thư mục chứa app.py
def p(name):
    return os.path.join(BASE_DIR, name)

REAL_CSV_PATH = p("traffic_real.csv")    # Nguồn CHÍNH: thời gian chờ đo từ video (bằng chứng)
WEIGHTS_PATH = p("yolov8n.pt")
ROLLING_WINDOW_SECONDS = 30           # Cửa sổ tính "TG chờ TB" và tốc độ xe đến (theo đồng hồ của nguồn)
BASELINE_ENTRANCE_GATES = 2           # Bố cục gốc khi quay video: 2 cổng Vào / 2 cổng Ra

# Hệ số làm chậm TỐC ĐỘ PHÁT video. 1.0 = tốc độ gốc; 3.0 = chậm gấp 3 lần.
# LƯU Ý: chỉ ảnh hưởng tốc độ phát/hiển thị. Mọi thời gian đo (thời gian chờ,
# tốc độ xe đến) đều tính theo GIÂY VIDEO GỐC (số frame / fps), KHÔNG nhân SLOWDOWN,
# để số liệu phản ánh đúng những gì xảy ra trong video.
SLOWDOWN = 3.0

# ------------------------------------------------------------
# PROFILE VIDEO + HAI VẠCH ẢO
# Toạ độ vạch chuẩn hoá 0..1 theo (rộng, cao) khung hình.
#   entry_line = Vạch 1: bắt đầu vùng chờ (phía xe đi tới, xa cổng)
#   exit_line  = Vạch 2: kết thúc vùng chờ (phía cổng)
# Trong cả hai video, xe di chuyển từ TRÊN xuống DƯỚI khung hình
# (đã kiểm tra bằng YOLO tracking), nên Vạch 1 nằm trên, Vạch 2 nằm dưới.
# ------------------------------------------------------------
VIDEO_PROFILES = {
    "hanoi": {
        "label": "Đường đô thị ùn tắc (videoplayback.mp4)",
        "path": "videoplayback.mp4",
        "entry_line": ((0.20, 0.60), (0.90, 0.60)),
        "exit_line": ((0.10, 0.89), (1.00, 0.89)),
        "max_seconds": 21,     # chỉ phát đến giây 21 (thời gian video gốc), rồi lặp lại
        "loop": True,
        # YOLOv8n (COCO) bỏ sót phần lớn xe máy trong cảnh ùn tắc dày đặc. Không cần
        # huấn luyện lại, ta: (1) chỉ xử lý nửa dưới khung (roi_top) ở imgsz 1280 để xe máy
        # "to" hơn; (2) hạ ngưỡng tin cậy; (3) phát hiện thêm người lái và gộp với xe máy
        # (riders, xem fusion.py). Đo trên 35 khung: 0,4 -> 13,8 xe máy/khung trong vùng chờ,
        # thời gian phát hiện ~45 -> ~90 ms/khung trên CPU.
        "detector": {"imgsz": 1280, "conf": 0.15, "roi_top": 0.45, "riders": True,
                     "tracker": "bytetrack_dense.yaml"},
    },
    "sample2": {
        "label": "Đường cao tốc thông thoáng (Vehicle Dataset Sample 2)",
        "path": "Vehicle Dataset Sample 2 [JqhdBCCUVyQ].mp4",
        "entry_line": ((0.30, 0.30), (0.76, 0.30)),
        "exit_line": ((0.06, 0.80), (0.98, 0.80)),
        "max_seconds": None,
        "loop": True,
        "detector": {"imgsz": 640, "conf": 0.25},
    },
}
DEFAULT_PROFILE = "hanoi"


def counter_config_for(height: int) -> CounterConfig:
    """Tham số đếm tỉ lệ theo chiều cao khung hình (để dùng chung cho 720p/1080p)."""
    return CounterConfig(
        margin_px=0.011 * height,
        enter_frames=3,
        exit_frames=3,
        confirm_frames=3,
        lost_grace_s=1.5,
        reid_retain_s=5.0,
        reid_max_dist_px=0.06 * height,
    )


# Factory: test có thể thay bằng detector / capture giả lập
def make_detector(profile):
    from vision import YoloDetector
    return YoloDetector(WEIGHTS_PATH, **profile.get("detector", {}))


def open_capture(path):
    return cv2.VideoCapture(path)


# ============================================================
# PHẦN 1 — WORKER XỬ LÝ VIDEO (YOLO + tracker + đếm xe chờ)
# Trạng thái worker (state) do CHÍNH LUỒNG WORKER cập nhật, nên giao diện
# hiển thị đúng việc worker đang thực sự chạy hay đã dừng.
#   starting -> paused <-> running -> stopped | finished | error
# ============================================================
class VideoWorker:
    def __init__(self, profile_name, profile, generation, detector_factory, capture_factory,
                 csv_path, slowdown=SLOWDOWN):
        self.profile_name = profile_name
        self.profile = profile
        self.generation = generation
        self._detector_factory = detector_factory
        self._capture_factory = capture_factory
        self._csv_path = csv_path
        self._slowdown = slowdown

        self.lock = threading.Lock()
        self._run_event = threading.Event()
        self._stop_event = threading.Event()
        self._thread = None

        self.state = "starting"
        self.error = None
        self.frames_processed = 0
        self.video_t = 0.0
        self.loops = 0
        self.kpi = None
        self.wait_samples = deque(maxlen=5000)   # (exit_video_t, wait_s, wall_ts)
        self.entry_times = deque(maxlen=20000)   # video_t của từng lượt vào vùng
        self.latest_jpeg = None
        self.scene = []                          # vị trí xe (toạ độ vùng chờ) cho bản ảo
        # Digital Twin: hệ cổng (trace-driven) + bản ảo + bộ ra quyết định
        self.twin = DigitalTwin(TwinConfig(baseline_gates=BASELINE_ENTRANCE_GATES))

    # ---------- điều khiển (gọi từ luồng API) ----------
    def start(self, running):
        if running:
            self._run_event.set()
        self._thread = threading.Thread(target=self._run, name=f"video-worker-{self.generation}",
                                        daemon=True)
        self._thread.start()

    def resume(self):
        self._run_event.set()

    def pause(self):
        self._run_event.clear()

    def stop(self, timeout=10.0):
        self._stop_event.set()
        self._run_event.set()      # đánh thức nếu đang chờ ở trạng thái paused
        if self._thread is not None:
            self._thread.join(timeout)
        return not self.is_alive()

    def is_alive(self):
        return self._thread is not None and self._thread.is_alive()

    def status(self):
        with self.lock:
            return {
                "state": self.state,
                "requested": "running" if self._run_event.is_set() and not self._stop_event.is_set() else "paused",
                "thread_alive": self.is_alive(),
                "profile": self.profile_name,
                "generation": self.generation,
                "frames_processed": self.frames_processed,
                "video_time_s": round(self.video_t, 2),
                "loops": self.loops,
                "error": self.error,
            }

    def data_snapshot(self):
        with self.lock:
            return {
                "kpi": dict(self.kpi) if self.kpi else None,
                "wait_samples": list(self.wait_samples),
                "entry_times": list(self.entry_times),
                "video_t": self.video_t,
                "scene": list(self.scene),
            }

    def twin_snapshot(self):
        with self.lock:
            return self.twin.snapshot()

    def twin_respond(self, rec_id, accept):
        with self.lock:
            return self.twin.respond(rec_id, accept)

    # ---------- luồng worker ----------
    def _set_state(self, s):
        with self.lock:
            self.state = s

    def _run(self):
        try:
            self._loop()
        except Exception as e:      # báo lỗi thật lên API, không im lặng
            print(f"[VIDEO WORKER ERROR] {e}")
            with self.lock:
                self.state = "error"
                self.error = str(e)

    def _loop(self):
        detector = self._detector_factory(self.profile)
        path = self.profile["path"]
        if not os.path.isabs(path):
            path = p(path)
        cap = self._capture_factory(path)
        if not cap.isOpened():
            raise RuntimeError(f"Không thể mở video: {path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        max_s = self.profile.get("max_seconds")
        counter = None
        frame_idx = 0
        reset_tracker = True

        real_f = open(self._csv_path, "w", newline="", encoding="utf-8")
        writer = csv.writer(real_f)
        writer.writerow(["timestamp", "wait_in_zone_s", "enter_video_t", "exit_video_t",
                         "vehicle_id", "profile"])
        real_f.flush()

        try:
            while not self._stop_event.is_set():
                if not self._run_event.is_set():
                    self._set_state("paused")
                    self._run_event.wait(0.2)
                    continue
                self._set_state("running")
                loop_start = time.time()

                ret, frame = cap.read()
                past_limit = (max_s is not None
                              and cap.get(cv2.CAP_PROP_POS_FRAMES) / fps > max_s)
                if not ret or past_limit:
                    if not self.profile.get("loop", True):
                        self._set_state("finished")
                        return
                    # Lặp lại video: xoá track đang theo dõi (không tính là lost/departed)
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    if counter is not None:
                        counter.reset_tracks()
                    reset_tracker = True
                    with self.lock:
                        self.loops += 1
                    continue

                frame_idx += 1
                video_t = frame_idx / fps    # giây video gốc, tăng đơn điệu kể cả khi lặp

                if counter is None:
                    h, w = frame.shape[:2]
                    zone = ZoneGeometry.from_normalized(self.profile["entry_line"],
                                                        self.profile["exit_line"], w, h)
                    counter = QueueCounter(zone, counter_config_for(h))

                detections = detector(frame, reset_tracker)
                reset_tracker = False
                samples, events = counter.update(video_t, detections)

                jpg = None
                try:
                    draw_overlay(frame, counter, detections, video_t)
                    ok, buf = cv2.imencode('.jpg', frame)
                    jpg = buf.tobytes() if ok else None
                except Exception as e:
                    print(f"[OVERLAY ERROR] {e}")

                now = time.time()
                with self.lock:
                    self.frames_processed += 1
                    self.video_t = video_t
                    self.kpi = counter.snapshot()
                    for ev in events:
                        if ev.kind == "enter":
                            self.entry_times.append(ev.t)
                    for s in samples:
                        self.wait_samples.append((s.exit_t, s.wait_s, now))
                    # Thực -> Ảo: đưa số đo của camera vào Digital Twin
                    W = ROLLING_WINDOW_SECONDS
                    ws = [(t_, v_) for t_, v_, _ in self.wait_samples]
                    self.twin.step(video_t,
                                   [ev.t for ev in events if ev.kind == "depart"],   # xe qua vạch 2 -> tới cổng
                                   rolling_rate(self.entry_times, video_t, W, start_t=0.0),
                                   rolling_mean(ws, video_t, W),
                                   len(rolling_values(ws, video_t, W)))
                    self.scene = counter.scene()
                    if jpg is not None:
                        self.latest_jpeg = jpg
                for s in samples:
                    writer.writerow([now, round(s.wait_s, 3), round(s.enter_t, 3),
                                     round(s.exit_t, 3), s.vid, self.profile_name])
                if samples:
                    real_f.flush()

                # Làm chậm tốc độ phát: mỗi frame kéo dài SLOWDOWN/fps giây.
                remain = (self._slowdown / fps) - (time.time() - loop_start)
                if remain > 0:
                    self._stop_event.wait(remain)
        finally:
            cap.release()
            real_f.close()
            with self.lock:
                if self.state not in ("error", "finished"):
                    self.state = "stopped"


# ============================================================
# PHẦN 2 — TRẠNG THÁI ỨNG DỤNG (mọi đọc/ghi đều qua STATE.lock)
# ============================================================
class AppState:
    def __init__(self):
        self.lock = threading.RLock()
        self.system_running = False         # người dùng đã bấm Bắt đầu hay chưa
        self.profile_name = DEFAULT_PROFILE
        self.worker = None
        self.generation = 0

    def apply_run_state(self):
        if self.worker is not None:
            if self.system_running:
                self.worker.resume()
            else:
                self.worker.pause()

    def replace_worker(self, profile_name):
        """Dừng hẳn worker cũ (join) rồi mới tạo worker mới -> không có 2 worker cùng ghi."""
        old = self.worker
        self.worker = None
        if old is not None:
            if not old.stop():
                print(f"[WARN] Worker thế hệ {old.generation} chưa dừng kịp; dữ liệu của nó bị bỏ qua.")
        self.generation += 1
        self.profile_name = profile_name
        self.worker = VideoWorker(profile_name, VIDEO_PROFILES[profile_name], self.generation,
                                  make_detector, open_capture, REAL_CSV_PATH, slowdown=SLOWDOWN)
        self.worker.start(self.system_running)

    def shutdown(self):
        if self.worker is not None:
            self.worker.stop()


STATE = AppState()


@asynccontextmanager
async def lifespan(_app):
    with STATE.lock:
        STATE.replace_worker(STATE.profile_name)   # worker khởi động ở trạng thái paused
    yield
    with STATE.lock:
        STATE.shutdown()


app = FastAPI(lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])


def mjpeg_generator():
    last = None
    while True:
        with STATE.lock:
            w = STATE.worker
        jpg = None
        if w is not None:
            with w.lock:
                jpg = w.latest_jpeg
        if jpg is None or jpg is last:
            time.sleep(0.02)
            continue
        last = jpg
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpg + b'\r\n')


@app.get("/video_feed")
def video_feed():
    return StreamingResponse(mjpeg_generator(), media_type="multipart/x-mixed-replace; boundary=frame")


def _status_payload():
    with STATE.lock:
        return {
            "system_running": STATE.system_running,
            "profile": STATE.profile_name,
            "worker": STATE.worker.status() if STATE.worker else None,
        }


@app.get("/api/status")
def get_status():
    return JSONResponse(_status_payload())


@app.post("/api/control/{action}")
def control(action: str):
    if action not in ("start", "pause"):
        return JSONResponse({"error": "Hành động không hợp lệ (start hoặc pause)"}, status_code=400)
    with STATE.lock:
        STATE.system_running = (action == "start")
        if STATE.worker is not None and STATE.worker.status()["state"] in ("error", "stopped", "finished") \
                and STATE.system_running:
            STATE.replace_worker(STATE.profile_name)     # khởi động lại worker đã chết
        STATE.apply_run_state()
    return JSONResponse(_status_payload())


@app.get("/api/videos")
def list_videos():
    return JSONResponse({
        "active": STATE.profile_name,
        "profiles": {k: {"label": v["label"], "path": v["path"],
                         "exists": os.path.exists(p(v["path"]))}
                     for k, v in VIDEO_PROFILES.items()},
    })


@app.post("/api/video/{name}")
def set_video(name: str):
    if name not in VIDEO_PROFILES:
        return JSONResponse({"error": f"Profile video không tồn tại: {name}"}, status_code=400)
    with STATE.lock:
        STATE.replace_worker(name)
    return JSONResponse(_status_payload())


# ============================================================
# PHẦN 3 — SỐ LIỆU + DIGITAL TWIN
# ============================================================
@app.get("/api/stats")
def get_stats():
    try:
        with STATE.lock:
            return JSONResponse(_build_stats())
    except Exception as e:
        print(f"[API ERROR] {e}")
        return JSONResponse({"error": f"Lỗi tính thống kê: {e}"}, status_code=500)


def _build_stats():
    worker = STATE.worker
    snap = worker.data_snapshot() if worker else {"kpi": None, "wait_samples": [],
                                                  "entry_times": [], "video_t": 0.0}
    now, W = snap["video_t"], ROLLING_WINDOW_SECONDS
    rows = [(t, v) for t, v, _ in snap["wait_samples"]]
    avg_wait = rolling_mean(rows, now, W)
    lam = rolling_rate(snap["entry_times"], now, W, start_t=0.0)
    kq = snap["kpi"] or {}
    kpi = {
        # Số xe ĐANG ở trong vùng chờ (giữa 2 vạch) — đo bằng camera
        "queue_count": kq.get("queue_count"),
        "queue_occluded": kq.get("queue_occluded"),
        # Các bộ đếm tích luỹ — KHÔNG phải số xe đang chờ
        "total_detected": kq.get("total_detected"),
        "entered_total": kq.get("entered_total"),
        "departed_total": kq.get("departed_total"),
        "left_other_total": kq.get("left_other_total"),
        "lost_in_zone_total": kq.get("lost_in_zone_total"),
        "avg_wait_s": round(avg_wait, 2) if avg_wait is not None else None,
        "wait_samples_in_window": len(rolling_values(rows, now, W)),
        "arrival_rate_per_min": round(lam * 60, 2) if lam is not None else None,
        "window_s": W,
    }
    return {
        "system_running": STATE.system_running,
        "profile": STATE.profile_name,
        "video_t": round(now, 2),
        "worker": worker.status() if worker else None,
        "kpi": kpi,
        "twin": worker.twin_snapshot() if worker else None,
        "scene": snap.get("scene", []),
    }


@app.post("/api/twin/gates/{n}")
def set_entry_gates(n: int):
    """Ảo -> Thực (thủ công): người vận hành tự đặt số cổng Vào (1-3)."""
    with STATE.lock:
        w = STATE.worker
        if w is None:
            return JSONResponse({"error": "Worker chưa sẵn sàng"}, status_code=409)
        with w.lock:
            ok, msg = w.twin.manual_set(n)
    if not ok:
        return JSONResponse({"error": msg}, status_code=409)
    return JSONResponse({"ok": True, "twin": w.twin_snapshot()})


@app.post("/api/twin/recommendation/{rec_id}/{action}")
def respond_recommendation(rec_id: int, action: str):
    """Ảo -> Thực: người vận hành chấp nhận / bỏ qua đề xuất của bản ảo."""
    if action not in ("accept", "reject"):
        return JSONResponse({"error": "Hành động không hợp lệ (accept hoặc reject)"}, status_code=400)
    with STATE.lock:
        w = STATE.worker
        ok = w is not None and w.twin_respond(rec_id, action == "accept")
    if not ok:
        return JSONResponse({"error": "Đề xuất không còn hiệu lực"}, status_code=409)
    return JSONResponse({"ok": True, "twin": w.twin_snapshot()})


@app.get("/", response_class=HTMLResponse)
def index():
    with open(p("new_2.html"), "r", encoding="utf-8") as f:
        return f.read()


if __name__ == "__main__":
    import uvicorn
    # reload=False: tránh việc reloader tạo 2 tiến trình -> 2 luồng YOLO cùng chạy
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=False)
