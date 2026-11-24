# Hệ thống quản lý nhà xe NEU

Hệ thống **Digital Twin** cho cổng nhà xe, vận hành theo vòng kín:

- **Thực → Ảo:** YOLOv8 + ByteTrack đếm xe chờ giữa **hai vạch ảo**, đo dòng xe đến (λ) và thời gian chờ. Cứ mỗi 2 giây, bản ảo được đồng bộ từ các số đo này.
- **Ảo → Thực:** bản ảo dự báo hàng chờ và tính mức tải ρ. Khi quá tải, nó đề xuất đổi số cổng Vào. Người vận hành chấp nhận thì lệnh được thực thi, và tác động được đo so với đối chứng.

## 1. Cấu trúc thư mục

```text
 yolo/
 ├── app.py                  # Backend FastAPI: worker video, trạng thái, API
 ├── queue_zone.py           # Logic đếm xe chờ 2 vạch (thuần Python, có test)
 ├── analytics.py            # Cửa sổ trượt + công thức M/M/c (Erlang C)
 ├── twin.py                 # Digital Twin: hệ cổng, bản ảo, ra quyết định (có test)
 ├── vision.py               # YOLO detector + vẽ overlay
 ├── fusion.py               # Gộp người lái + xe máy thành 1 xe 2 bánh (thuần Python, có test)
 ├── bytetrack_dense.yaml    # Cấu hình ByteTrack ngưỡng thấp cho cảnh dày đặc
 ├── verify_queue_video.py   # Kiểm tra offline trên video (xuất video + CSV)
 ├── new_2.html              # Giao diện dashboard
 ├── tests/                  # pytest
 ├── yolov8n.pt              # Model YOLOv8
 ├── videoplayback.mp4       # Video 1: đường đô thị ùn tắc (profile "hanoi")
 ├── Vehicle Dataset Sample 2 [JqhdBCCUVyQ].mp4   # Video 2: cao tốc (profile "sample2")
 └── traffic_real.csv        # Thời gian chờ từng xe đo từ video (app ghi lại)
```

## 2. Cài đặt

```bash
pip install fastapi uvicorn ultralytics opencv-python lapx
# để chạy test:
pip install pytest httpx
```

`lapx` là thư viện mà tracker của Ultralytics cần; nếu thiếu, `model.track()` báo lỗi `No module named 'lap'`.

## 3. Chạy ứng dụng

```bash
cd "D:\D DOWNLOAD\yolo123\yolo"
python app.py
```

Mở `http://127.0.0.1:8000`, nhấn **Bắt đầu**. Worker video khởi động ở trạng thái **tạm dừng** và chỉ xử lý khi bấm Bắt đầu. Nút **Tạm dừng** dừng thật worker trên server (số khung đã xử lý đứng yên). Dòng "Worker: …" trên giao diện hiển thị trạng thái do server báo về.

Đổi video bằng ô chọn **Video** trên giao diện (hoặc `POST /api/video/{profile}`).

## 4. Đếm xe chờ bằng hai vạch

### Định nghĩa

- **Vạch 1** (xanh dương): ranh giới bắt đầu vùng chờ, phía xe đi tới.
- **Vạch 2** (đỏ): ranh giới kết thúc vùng chờ, phía cổng.
- **Vùng chờ** = tứ giác nối hai vạch (vạch có thể xiên).
- **Điểm neo** của xe = tâm cạnh dưới bounding box (điểm chạm mặt đường).
- **Số xe đang chờ** = số xe có điểm neo đang nằm trong vùng chờ **tại thời điểm hiện tại**. Số này được tính lại mỗi khung hình từ trạng thái từng xe, **không cộng dồn** số lượt cắt vạch.

### Quy tắc chống đếm sai

| Tình huống | Cách xử lý |
|---|---|
| Xe đứng yên nhiều khung | Mỗi xe là một bản ghi theo ID; không đếm lặp |
| Xe dao động quanh vạch | Dải trễ ±`margin_px` (≈1,1% chiều cao khung) + cần 3 khung liên tiếp để đổi trạng thái |
| Bị che ngắn (≤ 1,5 s video) | Vẫn tính là đang chờ, báo riêng là "bị che (tạm giữ)" |
| Bị che lâu (> 1,5 s) | Loại khỏi số xe chờ; nếu xuất hiện lại trong 5 s thì khôi phục, giữ nguyên thời điểm vào vùng |
| Tracker đổi ID trong vùng | ID mới ở gần vị trí xe vừa mất được ghép vào xe cũ (không đếm thêm) |
| Mất hẳn (> 5 s) | Tính vào "Mất dấu trong vùng", **không** tính là đã qua cổng |
| Lùi ra qua vạch 1 / ra mép bên | Tính là "rời theo hướng khác", không lấy mẫu thời gian chờ |
| Video lặp lại từ đầu | Xoá các track đang theo dõi; các tổng tích luỹ giữ nguyên |

### Cấu hình vạch

Toạ độ chuẩn hoá 0–1 trong `VIDEO_PROFILES` của `app.py`:

```python
"hanoi": {
    "path": "videoplayback.mp4",
    "entry_line": ((0.20, 0.60), (0.90, 0.60)),   # Vạch 1
    "exit_line":  ((0.10, 0.89), (1.00, 0.89)),   # Vạch 2
    "max_seconds": 21,
    "detector": {"imgsz": 1280, "conf": 0.15, "roi_top": 0.45, "riders": True,
                 "tracker": "bytetrack_dense.yaml"},
},
```

Muốn thêm video: thêm một profile mới với `path`, hai vạch và tham số detector. Tham số đếm (margin, số khung, thời gian giữ) nằm trong `counter_config_for()`.

## 5. KPI (`GET /api/stats` → `kpi`)

| Trường | Ý nghĩa |
|---|---|
| `queue_count` | Số xe **đang** trong vùng chờ (gồm cả xe bị che tạm thời) |
| `queue_occluded` | Trong số trên, số xe đang bị che nhưng vẫn được giữ lại |
| `total_detected` | Tổng số xe khác nhau đã ghi nhận trong khung hình (tích luỹ) |
| `entered_total` | Số lượt xe đi vào vùng chờ (tích luỹ) |
| `departed_total` | Số xe rời vùng qua vạch 2 (tích luỹ) |
| `left_other_total` | Số xe rời vùng theo hướng khác |
| `lost_in_zone_total` | Số xe mất dấu trong vùng quá thời gian chờ xuất hiện lại |
| `avg_wait_s` | TB thời gian ở trong vùng của các xe đã rời qua vạch 2, trong cửa sổ 30 s (giây video) |
| `arrival_rate_per_min` | Tốc độ xe vào vùng (xe/phút video), đầu vào của Digital Twin |


**Thời gian** đo theo giây video gốc (số frame / fps). `SLOWDOWN` chỉ làm chậm tốc độ phát; từ bản này, thời gian đo **không còn** bị nhân với `SLOWDOWN` nữa.

## 6. Digital Twin (`twin.py`)

| Thành phần | Vai trò |
|---|---|
| Sơ đồ vị trí (T→A) | Mỗi khung hình, điểm neo của từng xe được chiếu sang toạ độ vùng chờ (u: trái→phải, v: vạch 1→vạch 2; `ZoneGeometry.to_zone_coords`) và vẽ thành chấm tròn trên bản ảo. Số chấm vàng = `queue_count`. |
| `PhysicalGates` | Hệ cổng "thực". Dùng video thay thế nên cổng được mô phỏng theo kiểu **trace-driven**: dòng xe đến cổng là thời điểm xe **qua vạch 2**, đo thật bằng YOLO; phần phục vụ (FIFO, c cổng Vào, thời gian phục vụ phân phối mũ) là mô phỏng. Khi có cổng thật thì thay lớp này bằng dữ liệu barrier/cảm biến. |
| Bản ảo | Mô hình động học hàng đợi. Khi ρ < 1, q hồi về L = λ·W(M/M/c) với hằng số thời gian τ = max(1/μ, 1/(cμ−λ)). Khi ρ ≥ 1, q tăng với tốc độ λ − cμ. μ được hiệu chỉnh từ thời gian chờ đo trên video (bố cục gốc 2 cổng). |
| Đồng bộ (T→A) | Mỗi `sync_s` = 2 s: gán q_ảo = q_thực, cập nhật λ, μ. Ngay trước khi gán, lấy sai số \|q_ảo − q_thực\| để tính **độ khớp (MAE)**. |
| Dự báo | Mỗi lần đồng bộ, dự báo q sau `horizon_s` = 30 s. Đến hạn thì so với thực tế để tính **sai số dự báo**. |
| Quyết định (A→T) | ρ ≥ 0,75 hoặc hàng chờ dự báo ≥ 8 xe → đề xuất +1 cổng Vào (tối đa 3, luôn giữ 1 cổng Ra). Khi ρ với c−1 cổng < 0,5 → đề xuất bớt cổng (không thấp hơn bố cục gốc). Giữa hai đề xuất nghỉ 20 s. |
| Thực thi | Chấp nhận → sau 3 s (thời gian chuyển barrier) cổng đổi trạng thái. |
| Điều khiển thủ công | Nút − / + hoặc bấm vào ô cổng (`POST /api/twin/gates/{n}`, n = 1–3). Lệnh thủ công đi qua cùng quy trình: ghi nhật ký, thực thi sau 3 s, đo tác động có đối chứng; đề xuất tự động đang chờ (nếu có) bị thay thế. |
| Đo tác động | Lúc thực thi, hệ tạo một **bản sao đối chứng** giữ số cổng cũ. Hai hệ nhận cùng dòng xe và cùng thời gian phục vụ (common random numbers). Sau 30 s, hệ so TG chờ TB của các xe đến: có lệnh so với không có lệnh. |

Kiểm tra trên dữ liệu verify của video Hà Nội: ρ ≈ 0,77 kích hoạt đề xuất 2→3 cổng; TG chờ giảm khoảng 45% so với đối chứng; MAE đồng bộ ≈ 1,1 xe. Với video cao tốc, ρ ≈ 0,55 nên hệ không đề xuất gì.

**Giới hạn:** phần phục vụ tại cổng là mô phỏng. Vì vậy con số tác động là kết quả của mô hình trên dòng xe đến thật, chưa phải số đo tại cổng thật.

## 7. API

| Method | Endpoint | Mục đích |
|---|---|---|
| `GET` | `/` | Dashboard |
| `GET` | `/video_feed` | Luồng MJPEG có overlay vùng chờ |
| `GET` | `/api/stats` | KPI camera, trạng thái worker, toàn bộ trạng thái Digital Twin (`twin`) |
| `GET` | `/api/status` | Trạng thái worker |
| `POST` | `/api/control/start` · `/api/control/pause` | Chạy / tạm dừng worker thật |
| `GET` | `/api/videos` | Danh sách profile video |
| `POST` | `/api/video/{profile}` | Đổi video (dừng và join worker cũ, tạo twin mới) |
| `POST` | `/api/twin/recommendation/{id}/accept` · `/reject` | Người vận hành chấp nhận / bỏ qua đề xuất |
| `POST` | `/api/twin/gates/{n}` | Người vận hành tự đặt số cổng Vào (1–3) |

## 8. Kiểm thử

```bash
python -m pytest -q tests
```

- `tests/test_queue_counter.py`: logic đếm, chạy trên dữ liệu track giả lập (xe vào/ra, đứng yên, nhiều xe, dao động quanh vạch, bị che, đổi ID, mất hẳn).
- `tests/test_analytics.py`: cửa sổ trượt, Erlang C, what-if.
- `tests/test_fusion.py`: gộp người lái + xe máy (người lái và người ngồi sau tính 1 xe, xe máy bị che dựng từ người lái, bỏ người trong ô tô/bus, đổi toạ độ vùng cắt).
- `tests/test_twin.py`: hệ cổng, mô hình ảo, đồng bộ và độ khớp, đề xuất, chấp nhận/bỏ qua, đo tác động có đối chứng.
- `tests/test_api.py`: API và worker với video/detector giả lập (tạm dừng/tiếp tục, đổi video, tên và kiểu KPI, CSV, đồng bộ twin, chấp nhận đề xuất tới cổng).

Các test này **không** đánh giá độ chính xác của YOLO. Để kiểm tra trên video thật:

```bash
python verify_queue_video.py --profile hanoi --out verify_out --snapshots 150,400
```

Lệnh trên tạo `verify_out/hanoi_annotated.mp4` (vẽ vùng chờ; xe vàng = đang chờ, xe xanh = ngoài vùng) và `hanoi_queue.csv` (số xe chờ theo từng frame). Để đối chiếu: dừng video ở một frame, đếm bằng mắt số chấm neo nằm giữa hai vạch, rồi so với cột `queue_count`.

## 9. Giới hạn đã biết

- **Nhận diện xe máy.** YOLOv8n (COCO) ở cấu hình gốc (640 px, ngưỡng 0,25) gần như không thấy xe máy trong cảnh ùn tắc của `videoplayback.mp4` (0,4 xe máy/khung trong vùng chờ). Profile `hanoi` vì vậy bật ba tuỳ chọn, **không cần huấn luyện lại**: chỉ xử lý từ 45% chiều cao khung trở xuống ở 1280 px (`roi_top`), hạ ngưỡng tin cậy xuống 0,15 kèm `bytetrack_dense.yaml`, và gộp người lái với xe máy (`riders`, xem `fusion.py`). Trên 21 giây video, số xe đang chờ trung bình tăng từ 3,5 lên 24,1; thời gian phát hiện ~100 ms/khung trên CPU (gấp ~2 lần). Mô hình vẫn bỏ sót xe bị che kín; người đi bộ trong vùng chờ bị tính là xe 2 bánh; số lần ghép lại định danh tăng nhiều (133 lần/21 s) nên thời gian chờ từng xe kém tin cậy hơn. Muốn đếm gần đủ cần mô hình được huấn luyện trên giao thông Việt Nam.
- Mọi xe nằm trong vùng đều được tính là "đang chờ"; hệ thống chưa lọc theo vận tốc. Với video cao tốc (`sample2`), xe chạy qua vùng trong khoảng 0,5 s.
- Khi xe đứng sát nhau, việc ghép ID mới với xe vừa mất có thể ghép nhầm. `total_detected` có thể đếm dư nếu tracker đổi ID ở ngoài vùng.
- Xe đã có sẵn trong vùng ở khung đầu tiên (hoặc sau khi video lặp lại) được tính là một lượt "vào vùng", nên tốc độ đến bị cao trong vài giây đầu.

## 10. Xử lý lỗi thường gặp

- **Không mở được video / không tìm thấy model:** dòng "Worker" trên giao diện hiển thị trạng thái `LỖI` kèm thông báo. Kiểm tra file nằm cùng thư mục với `app.py`, rồi bấm Bắt đầu để thử lại.
- **Cổng 8000 đã được sử dụng:** dừng tiến trình cũ hoặc đổi cổng trong `uvicorn.run`.
- **Không thấy khuyến nghị:** bản ảo cần ít nhất 5 mẫu thời gian chờ để hiệu chỉnh μ, và chỉ đề xuất khi ρ ≥ 0,75. Video cao tốc có tải thấp nên không có đề xuất là đúng.
