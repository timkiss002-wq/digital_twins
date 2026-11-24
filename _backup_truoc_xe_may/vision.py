"""
vision.py
---------
Phần phụ thuộc YOLO / OpenCV: bộ phát hiện + theo dõi xe và vẽ overlay.
Tách riêng để app.py và công cụ kiểm tra offline dùng chung, và để test có
thể thay bằng detector giả lập.
"""
from __future__ import annotations

import cv2

from queue_zone import IN, SUSPENDED, Detection, QueueCounter

VEHICLE_CLASSES = {2: 'Car', 3: 'Motorcycle', 5: 'Bus', 7: 'Truck'}


class YoloDetector:
    """Gọi YOLO + tracker (ByteTrack mặc định của Ultralytics).
    __call__(frame, reset) -> list[Detection]. reset=True: khởi tạo lại tracker."""

    def __init__(self, weights_path: str, classes=None, conf: float = 0.25, imgsz: int = 640):
        from ultralytics import YOLO  # import trễ để test không cần ultralytics
        self.model = YOLO(weights_path)
        self.classes = list(classes or VEHICLE_CLASSES.keys())
        self.conf = conf
        self.imgsz = imgsz

    def __call__(self, frame, reset: bool = False):
        results = self.model.track(frame, persist=not reset, classes=self.classes,
                                   conf=self.conf, imgsz=self.imgsz, verbose=False)
        b = results[0].boxes
        if b is None or b.id is None:
            return []
        boxes = b.xyxy.cpu().numpy().tolist()
        ids = b.id.int().cpu().tolist()
        cls = b.cls.int().cpu().tolist()
        return [Detection(tid, tuple(box), c) for box, tid, c in zip(boxes, ids, cls)]


def draw_overlay(frame, counter: QueueCounter, detections, video_t: float):
    """Vẽ vùng chờ, 2 vạch, bbox (màu theo trạng thái) và KPI lên frame (in-place)."""
    zone = counter.zone
    pts = [(int(x), int(y)) for x, y in zone.polygon]
    overlay = frame.copy()
    import numpy as np
    cv2.fillPoly(overlay, [np.array(pts, dtype=np.int32)], (0, 200, 255))
    cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)
    (e1, e2), (x1, x2) = zone.entry_line, zone.exit_line
    cv2.line(frame, tuple(map(int, e1)), tuple(map(int, e2)), (255, 160, 0), 3)
    cv2.line(frame, tuple(map(int, x1)), tuple(map(int, x2)), (0, 0, 255), 3)
    cv2.putText(frame, "VACH 1 (vao vung cho)", (int(e1[0]) + 8, int(e1[1]) - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 160, 0), 2)
    cv2.putText(frame, "VACH 2 (cong)", (int(x1[0]) + 8, int(x1[1]) - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    for det in detections:
        rec = counter.record_for_track(det.track_id)
        state = rec.state if rec else "OUT"
        if state == IN:
            color = (0, 255, 255)   # vàng: đang chờ
        else:
            color = (0, 200, 0)     # xanh: ngoài vùng
        bx1, by1, bx2, by2 = map(int, det.box)
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), color, 2)
        label = f"V{rec.vid}" if rec else f"T{det.track_id}"
        if rec and rec.state == IN and rec.enter_t is not None:
            label += f" {video_t - rec.enter_t:.1f}s"
        cv2.putText(frame, label, (bx1, by1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        ax, ay = det.anchor
        cv2.circle(frame, (int(ax), int(ay)), 4, color, -1)

    # Xe đang chờ nhưng tạm mất dấu: vẽ vòng tròn tại vị trí cuối
    for rec in counter._records.values():
        if rec.state == IN and not rec.seen_now:
            cv2.circle(frame, tuple(map(int, rec.last_anchor)), 10, (0, 165, 255), 2)
        elif rec.state == SUSPENDED:
            cv2.circle(frame, tuple(map(int, rec.last_anchor)), 10, (128, 128, 128), 1)

    s = counter.snapshot()
    txt = (f"Dang cho: {s['queue_count']} (bi che: {s['queue_occluded']})  "
           f"Da qua vach 2: {s['departed_total']}  Ghi nhan: {s['total_detected']}")
    # Dải KPI đặt ở đáy khung (góc trên dashboard đã có nhãn camera)
    H = frame.shape[0]
    cv2.rectangle(frame, (0, H - 34), (frame.shape[1], H), (0, 0, 0), -1)
    cv2.putText(frame, txt, (10, H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return frame
