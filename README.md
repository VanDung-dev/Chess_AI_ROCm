# ChessAI MCTS – Hệ thống AI chơi cờ vua sử dụng Deep Learning và MCTS

ChessAI MCTS là một hệ thống trí tuệ nhân tạo kết hợp giữa học sâu (Deep Learning) và thuật toán tìm kiếm cây Monte Carlo (MCTS) nhằm phát triển một tác nhân có khả năng chơi cờ vua chiến lược, tự học từ dữ liệu, và điều chỉnh chiến thuật theo tiến trình trận đấu.

---

## Giới thiệu

Dự án được phát triển với mục tiêu mô phỏng khả năng tư duy của con người trong cờ vua, dựa trên:

- Mạng nơ-ron `ChessNet` để dự đoán nước đi và đánh giá thế trận.
- MCTS tùy biến theo giai đoạn trận đấu để tối ưu quyết định chiến lược.
- Cơ chế tự học thông qua dữ liệu PGN và self-play.
- Huấn luyện theo phương pháp supervised learning từ các ván cờ thực tế.

---

## Cấu hình phát triển

Dự án này được xây dựng và thử nghiệm trên hệ thống có cấu hình phần cứng sau:

- GPU: **AMD Radeon™ RX 7600S**
- Framework: **ROCm 6.4**
- Thư viện học sâu: **PyTorch 2.8.0+rocm6.4**

Tuy nhiên, mã nguồn **không phụ thuộc vào phần cứng cụ thể**. Người dùng hoàn toàn có thể triển khai dự án này trên các hệ thống sử dụng **GPU NVIDIA với CUDA**, hoặc CPU nếu cần thiết (tốc độ sẽ chậm hơn).

---

## Yêu cầu môi trường

- Python 3.12 trở lên
- PyTorch (phiên bản phù hợp với phần cứng):
  - Với GPU AMD: `torch==2.8.0+rocm6.4`
  - Với GPU NVIDIA: `torch>=2.1.0` (CUDA 11.8/12.1 tùy môi trường)
- Các thư viện phụ thuộc:
  - `python-chess`
  - `colorlog`
  - `numpy`
  - `tqdm`
  - `torchvision` (nếu mở rộng)
  - `maturin` (để build Rust extension)

Cài đặt các thư viện bằng:

```bash
pip install -r requirements.txt
```

Nếu bạn dùng GPU NVIDIA (ví dụ RTX 30xx, 40xx), có thể dùng bản PyTorch mặc định từ PyPI:

```bash
pip install torch torchvision
```

---

## Khởi động chương trình

```bash
python main.py
```

Tùy chọn sẽ xuất hiện:

- `1`: Huấn luyện mô hình từ dữ liệu PGN
- `2`: Chơi với AI hoặc để AI tự chơi

Các file dữ liệu sẽ được lưu vào thư mục `data/`, và mô hình sau huấn luyện nằm trong `model/`.

---

## Tính năng chính

- Dự đoán nước đi với độ chính xác cao qua `ChessNet`, mạng nơ-ron tích hợp CNN + Residual Blocks + Transformer.
- Sử dụng heuristic đánh giá thế trận và giai đoạn (khai cuộc / trung cuộc / tàn cuộc).
- Tự chơi để tạo thêm dữ liệu huấn luyện mà không cần người dùng can thiệp.
- Ghi lại toàn bộ ván đấu dưới dạng PGN để phân tích và cải thiện.
- Sử dụng Rust extension để tăng tốc độ MCTS.

---

## Huấn luyện mô hình

Dữ liệu cần thiết là các ván đấu định dạng `.pgn` được lưu trong thư mục `data/`.

Bạn có thể huấn luyện bằng cách chạy:

```bash
python main.py  # chọn chế độ 1
```

Mô hình sẽ được lưu tự động sau mỗi phiên huấn luyện thành công.

---

## Rust Extension

Dự án sử dụng extension viết bằng Rust để tăng tốc độ của thuật toán MCTS. Để build và sử dụng extension này:

1. Cài đặt Rust toolchain từ https://rustup.rs/
2. Chạy lệnh build: `maturin develop`

Hệ thống sẽ sử dụng phiên bản Rust trực tiếp thay vì phiên bản Python.

---

## Ghi chú triển khai

- Với hệ thống AMD, cần đảm bảo **ROCm đã được cài đặt và cấu hình đúng**.
- Dự án có sử dụng `torch.compile()` để tối ưu tốc độ tính toán nếu được hỗ trợ.
- Logging được thiết kế rõ ràng và có màu sắc giúp theo dõi quá trình huấn luyện dễ dàng.

---

## Giấy phép

Dự án được phát hành theo [Giấy phép MIT](LICENSE). Bạn có thể sử dụng, sao chép, tùy chỉnh, và phân phối với điều kiện giữ nguyên thông tin tác giả.

---

## Liên hệ và đóng góp

Mọi ý kiến phản hồi, đóng góp cải tiến, hoặc báo lỗi đều được hoan nghênh. Hãy sử dụng GitHub Issues hoặc gửi pull request để tham gia cùng phát triển dự án này.