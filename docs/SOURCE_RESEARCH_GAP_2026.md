# Deep research: Research gap cho IMAVIS với Jetson và Raspberry Pi 5 + Hailo

**Ngày chốt nguồn:** 24-08-2026  
**Journal:** *Image and Vision Computing* (IMAVIS), Elsevier  
**Thiết bị mục tiêu:** Jetson Nano, Jetson Xavier NX, Jetson AGX Xavier, Raspberry Pi 5 + Hailo  
**Kết luận ngắn:** Hướng có xác suất tạo đóng góp tốt nhất là **semantic segmentation thích nghi theo ngân sách, có hiệu chuẩn độ tin cậy trong điều kiện bất lợi, và được đồng thiết kế cho nhiều compiler/accelerator**. Không nên làm thêm một biến thể “YOLO + attention + pruning” hoặc chỉ báo cáo FLOPs/FPS.

> **Lưu ý về mức độ khẳng định:** “Gap” dưới đây là kết luận tổng hợp từ phạm vi journal, các bài IMAVIS gần đây, literature liên quan và tài liệu phần cứng. Đây không phải tuyên bố tuyệt đối rằng chưa từng có bài nào kết hợp hai thành phần bất kỳ. Trước khi dùng cụm “the first” trong manuscript, cần hoàn tất systematic search với Scopus/Web of Science và cập nhật đến ngày nộp.

## 1. Khuyến nghị chủ lực

### Working title

**PACE-Seg: Platform-Aware Calibrated Elastic Semantic Segmentation for Reliable Edge Vision under Adverse Conditions**

Tên tiếng Việt: **Phân đoạn ngữ nghĩa co giãn theo nền tảng, có hiệu chuẩn độ tin cậy, cho thị giác biên trong điều kiện bất lợi**.

### Một câu mô tả công trình

Huấn luyện một *elastic supernet* duy nhất; trích xuất 3–4 subnet tĩnh INT8 tương thích compiler; dùng bộ định tuyến độ khó đã hiệu chuẩn để chọn subnet theo ảnh/cửa sổ thời gian; tối ưu bằng **p95 latency và joule/frame đo trên phần cứng thật**, rồi kiểm chứng trên Jetson Nano, Xavier NX, AGX Xavier và Pi 5 + Hailo dưới ngày, đêm, mưa, sương mù và tuyết.

### Vì sao đây là lựa chọn tốt nhất

1. **Đúng IMAVIS:** đóng góp chính vẫn là phương pháp computer vision—elastic representation, hardware-in-the-loop search và risk-aware routing—chứ không phải một bài benchmark hệ thống thuần túy.
2. **Tận dụng trọn bộ phần cứng:** bốn thiết bị tạo ra ba lớp backend khác nhau: TensorRT GPU, Xavier DLA và Hailo dataflow NPU.
3. **Gap có thể bảo vệ:** nhiều công trình tối ưu một trong các trục *accuracy–FLOPs–average FPS*; ít công trình cùng lúc xử lý compiler portability, INT8, tail latency, năng lượng toàn hệ thống, nhiệt và calibrated reliability dưới domain shift.
4. **Đúng thời điểm:** special issue **Complex Environment Vision** của IMAVIS có hạn 15-02-2027; special issue **Visual Perception enabling Autonomous Navigation (VP-NAV)** có hạn 31-12-2026. Hướng trên khớp cả hai, nhưng mốc tháng 2 thực tế hơn ([Complex Environment Vision](https://www.sciencedirect.com/special-issue/335145/complex-environment-vision), [VP-NAV](https://www.sciencedirect.com/special-issue/335697/visual-perception-enabling-autonomous-navigation-vp-nav)).

## 2. Journal fit và tín hiệu từ IMAVIS

IMAVIS ưu tiên nghiên cứu lý thuyết hoặc ứng dụng có chất lượng cao, mang tính nền tảng đối với diễn giải ảnh và thị giác máy tính, với phương pháp mới hoặc ứng dụng vào cảnh thực ([Guide for Authors](https://www.sciencedirect.com/journal/image-and-vision-computing/publish/guide-for-authors)). Tại thời điểm khảo sát, trang journal hiển thị **CiteScore 7.7** và **Impact Factor 5.0** ([IMAVIS](https://www.sciencedirect.com/journal/image-and-vision-computing), [Insights](https://www.sciencedirect.com/journal/image-and-vision-computing/about/insights)).

Điều này dẫn tới một nguyên tắc quan trọng: **edge deployment là bằng chứng cho đóng góp CV, không thể là toàn bộ đóng góp**. Một manuscript chỉ so FPS của các model có sẵn trên bốn board sẽ gần bài systems/benchmark hơn là IMAVIS. Ngược lại, một phương pháp mới có giả thuyết rõ ràng, được đồng thiết kế với compiler và xác nhận trên phần cứng thật là phù hợp.

### Các tín hiệu literature trực tiếp

| Tín hiệu | Điều đã có | Hàm ý cho novelty |
|---|---|---|
| Review về lightweight CNN trong IMAVIS (2024) | Tổng quan kiến trúc nhẹ, quantization/binarization và hệ thống hạn chế tài nguyên ([bài review](https://www.sciencedirect.com/science/article/pii/S0262885624001410)) | “Lightweight CNN” tự thân đã là chủ đề trưởng thành. |
| YOLIC, IMAVIS 2024 | Localization/classification trên edge; tác giả báo cáo trên 30 FPS bằng CPU Raspberry Pi 4B ([bài báo](https://www.sciencedirect.com/science/article/pii/S0262885624001999), [project](https://kai3316.github.io/yolic.github.io/)) | Edge deployment và Raspberry Pi không còn là novelty độc lập. |
| Light-SEF, IMAVIS 2025 | Depth completion nhẹ; báo cáo giảm khoảng 53% tham số, 50% FLOPs/MACs và 36% thời gian chạy ([bài báo](https://www.sciencedirect.com/science/article/pii/S0262885624004402), [abstract của tác giả](https://pure.ecnu.edu.cn/en/publications/a-lightweight-depth-completion-network-with-spatial-efficient-fus/)) | Kiểu câu chuyện “ít tham số/FLOPs hơn” đã đông; cần metric triển khai sâu hơn. |
| UCPNet, IMAVIS 2026 | Ultra-lightweight real-time semantic segmentation ([bài báo](https://www.sciencedirect.com/science/article/pii/S0262885626001162)) | Một segmentation network nhẹ mới phải có trục đóng góp khác biệt, không chỉ module fusion/attention mới. |
| Survey dynamic neural networks, IMAVIS 2026 | Hệ thống hóa early exits, adaptive graph/input và dynamic computation ([bài báo](https://www.sciencedirect.com/science/article/pii/S0262885626000879), [preprint](https://arxiv.org/abs/2501.07451)) | Adaptive compute là hướng hợp journal, nhưng “dynamic network” chung chung không còn mới. |
| HARD, ACCV 2024 | Năm biến thể semantic segmentation hardware-aware từ edge đến GPU ([CVF Open Access](https://openaccess.thecvf.com/content/ACCV2024/html/Kwon_HARD__Hardware-Aware_lightweight_Real-time_semantic_segmentation_model_Deployable_from_ACCV_2024_paper.html)) | “Hardware-aware segmentation” riêng lẻ cũng đã có; phải vượt qua bằng đa-backend + reliability + measured energy. |
| Cloud-edge CTTA, CVPR 2026 | Continual test-time adaptation giữa kiến trúc cloud và edge ([paper](https://openaccess.thecvf.com/content/CVPR2026/html/Xu_Cross-Architecture_Adaptation_Cloud-Edge_Continual_Test-Time_Adaptation_with_Dynamic_Sampling_and_CVPR_2026_paper.html)) | Continual adaptation rất cạnh tranh và khó hoàn thiện trong thời gian ngắn. |
| Open-vocabulary real-time | YOLOE đã đưa open-vocabulary detection/segmentation về real-time ([ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Wang_YOLOE_Real-Time_Seeing_Anything_ICCV_2025_paper.html)); IMAVIS đã có APOVIS dựa trên VLM/foundation segmentation ([APOVIS](https://www.sciencedirect.com/science/article/abs/pii/S026288562400489X)) | Novelty cao nhưng compute/training/toolchain risk rất cao đối với Xavier/Hailo. |

### Khoảng trống tổng hợp có thể bảo vệ

Các mảnh riêng lẻ đều đã tồn tại. Khoảng trống khả thi nằm ở **giao điểm bốn trục**:

- **G1 — Proxy gap:** tham số và FLOPs không dự báo đáng tin cậy latency/năng lượng trên các kiến trúc khác nhau. Chính Hailo lưu ý tốc độ không thể suy ra đáng tin cậy chỉ từ FLOPs hoặc tham số và cung cấp benchmark phần cứng thật ([Hailo Software Suite](https://hailo.ai/products/hailo-software/hailo-ai-software-suite/)).
- **G2 — Compiler gap:** một graph chạy tốt trên TensorRT GPU chưa chắc chạy trọn vẹn trên Xavier DLA hoặc biên dịch được sang Hailo HEF. DLA không hỗ trợ dynamic dimensions; Xavier DLA không hỗ trợ softmax; GPU fallback có thể che giấu lớp không chạy trên DLA ([DLA restrictions](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/dla-layer-restrictions.html), [GPU fallback](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/dla-runtime-configuration.html)).
- **G3 — Reliability gap:** accuracy trung bình trên ảnh sạch không phản ánh rủi ro khi ảnh tối, mưa, tuyết hoặc sương mù. ACDC cung cấp cả vùng ngữ nghĩa không chắc chắn, tạo điều kiện đánh giá risk-aware segmentation thay vì chỉ mIoU ([ACDC](https://acdc.vision.ee.ethz.ch/), [paper](https://arxiv.org/abs/2104.13395)).
- **G4 — Static budget gap:** một model tĩnh cho mỗi thiết bị làm tăng chi phí huấn luyện/bảo trì; một model duy nhất lại không khai thác được ngân sách thay đổi theo thiết bị, power mode, nhiệt và độ khó của ảnh.

**Research gap đề xuất:** chưa được giải quyết thuyết phục trong nhóm công trình đã khảo sát là một phương pháp segmentation duy nhất có thể sinh các engine tĩnh tương thích nhiều accelerator, chọn compute theo calibrated visual risk, và tối ưu trên p95 latency + năng lượng đo thật dưới adverse domain shift.

## 3. Đọc đúng năng lực phần cứng

| Thiết bị | Năng lực chính thức | Ràng buộc cần biến thành biến nghiên cứu | Vai trò trong paper |
|---|---|---|---|
| Jetson Nano | 472 GFLOPS, 4 GB LPDDR4, 5–10 W, không có DLA ([NVIDIA](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-nano/product-development/)) | Memory nhỏ; Maxwell; JetPack 4 đã EOL từ 11-2024 ([Jetson FAQ](https://developer.nvidia.com/embedded/faq)) | “Worst-case/legacy target”; chứng minh subnet nhỏ nhất và tính tái lập trên stack cũ. |
| Jetson Xavier NX | 14 TOPS ở 10 W, tối đa 21 TOPS ở 15 W; 48 Tensor Cores và 2 DLA ([NVIDIA](https://developer.nvidia.com/blog/jetson-xavier-nx-the-worlds-smallest-ai-supercomputer/)) | GPU/DLA có hành vi khác nhau; fixed shape; INT8/FP16; giới hạn operator | Thiết bị trung gian; so GPU với DLA và hai power modes. |
| Jetson AGX Xavier | Tối đa 32 TOPS; power profiles 10/15/30 W; 2 DLA ([NVIDIA](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-agx-xavier/)) | Peak TOPS không đồng nghĩa end-to-end performance; cần kiểm soát fan/clocks | Reference edge mạnh; subnet lớn nhất; profile hardware-in-the-loop. |
| Pi 5 + Hailo-8 | Hailo-8: 26 TOPS, mức tiêu thụ điển hình của chip 2.5 W, dataflow architecture, HEF/HailoRT ([Hailo-8](https://hailo.ai/products/ai-accelerators/hailo-8-ai-accelerator/)) | INT8, offline compilation, resource allocation; host preprocessing/postprocessing vẫn tốn điện | NPU khác kiến trúc; phép thử mạnh nhất cho compiler portability. |

**Không so trực tiếp TOPS giữa các hãng.** TOPS phụ thuộc precision, sparsity, toán tử và cách công bố. Kết luận paper phải dựa trên accuracy-matched p95 latency, energy/frame và system power.

### Cần xác minh ngay model Hailo

Tài liệu Raspberry Pi phân biệt:

- **Raspberry Pi AI Kit cũ:** Hailo-8L, 13 TOPS.
- **Raspberry Pi AI HAT+:** có bản Hailo-8L 13 TOPS hoặc Hailo-8 26 TOPS.

Vì vậy, mô tả “AI Kit 26 TOPS” nhiều khả năng là AI HAT+/M.2 Hailo-8, không phải AI Kit cũ ([Raspberry Pi AI documentation](https://www.raspberrypi.com/documentation/computers/ai.html)). Chạy:

```bash
hailortcli fw-control identify
```

Ghi lại chip, firmware, HailoRT, Model Zoo và Dataflow Compiler. Đây là biến kiểm soát bắt buộc trong bài.

### Training và deployment

Bốn board này nên được xem là **deployment/profiling targets**. AGX Xavier có thể huấn luyện mô hình nhỏ, nhưng một supernet segmentation với QAT và nhiều seed sẽ rất chậm. Kế hoạch dưới đây giả định có GPU huấn luyện rời; nếu không có GPU ít nhất khoảng 16–24 GB VRAM, cần giảm input resolution, số subnet và số baseline.

## 4. Xếp hạng các hướng nghiên cứu

Điểm 1–5 dưới đây là đánh giá chuyên gia dựa trên landscape hiện tại, không phải thống kê acceptance của journal.

| Hướng | Fit IMAVIS | Novelty còn lại | Tận dụng hardware | Khả thi trước 15-02-2027 | Kết luận |
|---|---:|---:|---:|---:|---|
| **PACE-Seg: calibrated elastic adverse segmentation đa-accelerator** | 5.0 | 4.5 | 5.0 | 4.0 | **Chọn** |
| Budget-aware multi-task navigation: drivable area + lane + object/semantic | 5.0 | 4.0 | 5.0 | 3.0 | Phương án B nếu có dataset/ứng dụng xe tự hành rõ |
| Tiny-adapter continual/test-time adaptation trên edge | 5.0 | 4.5 | 3.5 | 2.5 | Rủi ro cao; Hailo chỉ inference, competitor 2026 mạnh |
| INT8 open-vocabulary segmentation với cached text prompts | 4.5 | 5.0 | 4.0 | 2.0 | Novelty cao nhưng training/compiler risk rất cao |
| Task-aware dehazing/restoration để cải thiện downstream perception | 4.5 | 3.0 | 4.5 | 3.5 | Có thể làm, nhưng all-in-one/continual restoration đang đông |
| Benchmark năng lượng đa-board thuần túy | 3.0 | 2.5 | 5.0 | 5.0 | Dùng làm phần thực nghiệm, không nên là contribution chính |
| YOLO phiên bản mới + attention/pruning cho small objects | 4.0 | 1.5 | 5.0 | 5.0 | **Tránh**; literature quá đông, reviewer dễ xem là incremental |

## 5. Đề cương PACE-Seg

### 5.1 Research questions và giả thuyết

| RQ | Câu hỏi | Giả thuyết kiểm chứng được |
|---|---|---|
| RQ1 | FLOPs-aware hay measured-hardware-aware search tạo Pareto frontier tốt hơn? | Với cùng mIoU, objective dùng p95 latency/J-frame thật giảm ít nhất 15–20% cost trên phần lớn thiết bị so với FLOPs-constrained search. |
| RQ2 | Một supernet có thể thay nhiều model huấn luyện độc lập không? | Subnet trích từ supernet đạt trong khoảng 1.0–1.5 mIoU của model train riêng, đồng thời giảm chi phí huấn luyện/bảo trì. |
| RQ3 | Calibrated difficulty routing có giúp dưới adverse shift? | Routing đạt risk/accuracy target với năng lượng thấp hơn static-large, và có AURC/UIoU tốt hơn routing chỉ dùng entropy thô. |
| RQ4 | Compiler-constrained search space có tăng portability thật không? | Tăng tỷ lệ compile thành công/accelerator coverage, giảm fallback và variance latency so với search space unconstrained. |

### 5.2 Bốn đóng góp dự kiến

1. **Compiler-safe elastic segmentation supernet.** Co giãn theo width, depth và input resolution; giới hạn search space vào các operator ứng viên tương thích nhiều backend. Đây mới là *candidate intersection* và phải smoke-test trên từng compiler trong tuần 1–2; không giả định mọi Conv/Resize đều có behavior giống nhau.
2. **Hardware-in-the-loop Pareto search.** Xây latency/energy lookup table hoặc surrogate cho từng backend từ đo thật, rồi chọn subnet theo multi-objective cost, thay vì dùng FLOPs làm proxy.
3. **Calibrated visual-risk router.** Một quality/risk probe nhỏ dự đoán expected segmentation error từ ảnh downsampled; chọn một trong các engine tĩnh theo risk target và energy budget. Calibration được fit trên validation set, không dùng test labels.
4. **Cross-platform reliability benchmark protocol.** Đánh giá clean/adverse accuracy, uncertainty, p50/p95/p99 latency, J/frame, thermal stability và compiler coverage trên bốn nền tảng với artifact/version disclosure.

### 5.3 Thiết kế phương pháp khả thi

#### A. Supernet

- Backbone ưu tiên CNN/hybrid đơn giản với các block Conv–BN–activation, depthwise/pointwise Conv, pooling, add/concat và resize tĩnh.
- Ba hoặc bốn mức: `tiny`, `small`, `medium`, `large`.
- Co giãn theo channel multiplier, số block và resolution; tránh dynamic tensor shape ở runtime.
- Sandwich rule + in-place distillation để huấn luyện shared weights.
- Boundary-aware auxiliary loss để bảo vệ pedestrian, pole, sign và ranh giới đường khi subnet nhỏ.

#### B. Quantization

- FP32 teacher → shared supernet → QAT INT8.
- Calibration set phải đại diện đủ ngày/đêm/mưa/sương/tuyết.
- Báo cáo riêng FP32, FP16 và INT8; không gộp accuracy trước/sau quantization.
- Hailo biên dịch ONNX/TensorFlow sang representation nội bộ, quantize/allocate resource rồi tạo HEF; toolchain có emulator và profiler ([Hailo Software Suite](https://hailo.ai/products/hailo-software/hailo-ai-software-suite/)). Kết quả chính vẫn phải đo trên chip, không dùng profiler estimate thay thế.

#### C. Engine tĩnh thay vì dynamic graph

- Compile từng subnet thành TensorRT engine cho GPU, DLA-compatible engine cho Xavier và HEF cho Hailo.
- Router chạy trên host hoặc accelerator nhỏ, chọn engine theo từng đoạn 8–32 frame để giảm switching overhead.
- Cách này phù hợp giới hạn fixed-shape của DLA và thực tế Hailo hỗ trợ pipeline/model scheduling; không cần control flow động bên trong graph.

#### D. Objective

Với subnet `a`, nền tảng `d`, precision `q` và điều kiện ảnh `c`, có thể dùng:

\[
\min_{a,\theta} \; \mathcal{L}_{seg} + \alpha\,\mathcal{L}_{distill} +
\beta\,\mathcal{L}_{cal} + \lambda\,\widehat{L}_{p95}(a,d,q) +
\mu\,\widehat{E}(a,d,q)
\]

với ràng buộc compile thành công, memory budget, accelerator coverage và risk target. `L` và `E` là surrogate được fit từ p95 latency và energy/frame đo thật, không phải FLOPs.

## 6. Dữ liệu và protocol thực nghiệm

### 6.1 Datasets

| Vai trò | Dataset | Cách dùng |
|---|---|---|
| Clean/source | **Cityscapes**: 5.000 ảnh fine, 20.000 coarse, 50 thành phố ([official](https://www.cityscapes-dataset.com/)) | Train nền tảng, benchmark clean mIoU và khả năng real-time. |
| Adverse chính | **ACDC**: 4.006 ảnh adverse chia đều fog/night/rain/snow, có ảnh normal tương ứng và uncertainty masks ([official](https://acdc.vision.ee.ethz.ch/), [paper](https://arxiv.org/abs/2104.13395)) | Train/validation/test theo split chính thức; đánh giá từng condition và uncertainty-aware segmentation. |
| External shift | **Dark Zurich**: day/twilight/night và uncertainty-aware evaluation ([challenge](https://codalab.lisn.upsaclay.fr/competitions/3783), [paper](https://arxiv.org/abs/1901.05946)) | Zero-shot hoặc adaptation-free external test; không tune threshold trên test. |
| Optional diversity | **BDD100K**: 100.000 video, đa thời tiết/thời gian ([Berkeley](https://bair.berkeley.edu/blog/2018/05/30/bdd/)) | Chỉ thêm nếu đủ compute; ưu tiên drivable-area/multi-task cho phương án B. |

**Scope tối thiểu nên giữ:** Cityscapes + ACDC. Dark Zurich là external validation mạnh. Không nên thêm quá nhiều dataset rồi giảm độ sâu ablation.

### 6.2 Baselines bắt buộc

- Fast-SCNN.
- BiSeNetV2.
- PIDNet-S hoặc DDRNet-23-slim.
- SegFormer-B0.
- MobileNetV3-Large + DeepLabV3.
- HARD variant gần cùng ngân sách nếu code/checkpoint tái lập được.
- UCPNet nếu code/checkpoint được công bố kịp thời.

Baselines thiết kế thí nghiệm:

1. Model train riêng cho từng budget vs shared supernet.
2. FLOPs-aware selection vs latency-aware vs latency+energy-aware.
3. Static-small, static-large, oracle router, entropy router và calibrated risk router.
4. FP32/FP16, INT8-PTQ và INT8-QAT.
5. Unconstrained operator space vs compiler-constrained space.

### 6.3 Metrics

**Vision quality**

- mIoU tổng và theo từng condition.
- Per-class IoU; đặc biệt person/rider, traffic sign/light, pole và road boundary.
- Boundary IoU hoặc boundary F-score.
- Mức giảm accuracy từ FP32 → FP16/INT8.

**Reliability**

- Expected Calibration Error (ECE), NLL/Brier.
- Risk–coverage curve và AURC.
- UIoU/uncertainty-aware metric trên ACDC hoặc Dark Zurich.
- Selective mIoU tại các coverage/risk target cố định.

**Deployment**

- End-to-end p50/p95/p99 latency, batch size 1.
- Throughput ở single-stream; multi-stream chỉ là phụ.
- Joule/frame, images/J, idle/active system power.
- Peak host RAM/device memory.
- Nhiệt độ, clocks và throttling trong run kéo dài.
- Compile success, số operator fallback, tỷ lệ layer/compute thực sự trên accelerator khi backend cho phép quan sát.
- Preprocess, transfer, inference và postprocess breakdown.

### 6.4 Power/latency protocol

MLPerf coi năng lượng là metric hạng nhất; phương pháp MLPerf Power đo ở mức toàn hệ thống “at the wall”, gồm host, accelerator, memory và fan, đồng thời yêu cầu power và performance đến từ cùng một run ([MLPerf Power paper](https://arxiv.org/abs/2410.12032), [measurement rules](https://github.com/mlcommons/inference_policies/blob/master/power_measurement.adoc), [MLPerf Edge](https://mlcommons.org/benchmarks/inference-edge/)). Paper không cần tuyên bố tuân thủ MLPerf chính thức, nhưng nên mượn các nguyên tắc:

1. Batch 1; cùng resolution, preprocessing và postprocessing.
2. Warm-up tối thiểu 200 inference; đo tối thiểu 5.000 frame hoặc đủ dài để CI ổn định.
3. Ba run độc lập cho mỗi cấu hình; randomize thứ tự model để giảm bias do nhiệt.
4. Một run sustained 30–60 phút cho từng model đại diện.
5. Khóa và công bố `nvpmodel`, clocks, fan mode, ambient temperature, OS, JetPack, CUDA, TensorRT, HailoRT, DFC, firmware.
6. Đo end-to-end bằng external calibrated meter; `tegrastats`/telemetry dùng để giải thích, không thay thế số toàn hệ thống.
7. Báo cáo kernel-only và end-to-end riêng; không so kernel-only của Jetson với pipeline end-to-end của Pi.
8. Bootstrap 95% CI cho latency/energy; ba seed huấn luyện cho kết quả accuracy quan trọng.

## 7. Ablation plan

| Ablation | Câu hỏi trả lời | Kết quả cần thấy |
|---|---|---|
| Shared supernet vs independent training | Weight sharing có làm mất quality? | Khoảng cách nhỏ, đổi lại giảm training/storage. |
| FLOPs vs measured latency objective | FLOPs có sai thứ hạng model? | Có các cặp cùng FLOPs nhưng latency khác, và Pareto measured-cost tốt hơn. |
| Latency-only vs latency+energy | Model nhanh nhất có luôn tiết kiệm năng lượng? | Không nhất thiết; chứng minh bằng J/frame. |
| Unconstrained vs compiler-safe operators | Portability đến từ đâu? | Compile/fallback/latency variance cải thiện rõ. |
| PTQ vs QAT | INT8 accuracy có giữ được? | QAT giảm degradation, nhất là night/fog và class nhỏ. |
| Không distillation vs distillation | Subnet nhỏ có học được context? | Cải thiện mIoU/boundary ở cùng cost. |
| Static large/small vs router | Dynamic allocation có đáng overhead? | Router đạt target risk với energy thấp hơn large. |
| Entropy vs calibrated risk router | Calibration có giá trị thật? | AURC/UIoU tốt hơn, threshold ổn định hơn qua condition. |
| Per-frame vs temporal-window routing | Switching overhead và flicker? | Window routing giảm overhead/oscillation mà không tăng risk nhiều. |
| Power modes | Kết luận có bền qua 5/10/15/30 W? | Pareto frontier thay đổi; policy thích nghi vẫn hợp lý. |

## 8. Tiêu chí go/no-go

Đặt trước tiêu chí giúp tránh “cherry-pick” sau thí nghiệm.

### Go

- Tất cả bốn thiết bị chạy được ít nhất hai subnet INT8 tĩnh; NX/AGX có kết quả GPU và ít nhất một DLA path được audit rõ.
- INT8-QAT giảm không quá khoảng 1.0–1.5 mIoU so với FP32 cho mỗi subnet quan trọng.
- Ở cùng mức quality/risk, phương pháp giảm ít nhất 20% p95 latency **hoặc** J/frame trên ít nhất ba trong bốn thiết bị so với baseline phù hợp.
- Calibrated router cải thiện AURC/UIoU hoặc risk-at-coverage có ý nghĩa, không chỉ tiết kiệm compute.
- Router + switching overhead dưới khoảng 5% end-to-end cost.
- Kết quả giữ được trong sustained run, không chỉ burst ngắn trước throttling.

### No-go hoặc đổi scope

- Hailo không compile được decoder/resize chính sau hai tuần: thu hẹp search space hoặc dùng encoder trên Hailo + host postprocess; phải công bố partition, không che giấu.
- DLA fallback quá nhiều: coi DLA là ablation phụ, giữ TensorRT GPU + Hailo là hai backend chính.
- Calibration không cải thiện external shift: bỏ claim “reliable”, chuyển thành budget-aware deployment; không dùng ngôn ngữ “safety”.
- Supernet thua model train riêng trên 2 mIoU ở nhiều budget: giảm số trục elasticity, chỉ giữ width + resolution.
- Chỉ đạt speedup do giảm resolution: contribution quá yếu; cần hardware-aware selection hoặc calibrated routing tạo lợi ích độc lập.

## 9. Những phản biện reviewer có thể đặt ra

| Phản biện | Cách thiết kế trước để trả lời |
|---|---|
| “Chỉ là once-for-all network áp dụng cho segmentation.” | Chứng minh compiler-constrained search, multi-backend measured cost và calibrated risk routing đều có ablation độc lập. |
| “Đây là benchmark systems, không phải CV method.” | Đặt elastic representation + risk model là contributions 1–2; phần cứng là kiểm chứng cơ chế. |
| “Hardware-aware segmentation đã có HARD.” | So trực tiếp; nêu khác biệt: multiple compilers, INT8/DLA/HEF, adverse calibration, energy/tail latency. |
| “Dynamic model không chạy trên DLA/Hailo.” | Dùng router chọn giữa các engine tĩnh đã compile; đo switching overhead. |
| “FLOPs đã đủ cho model nhẹ.” | Cho rank-correlation giữa FLOPs và p95/J-frame theo từng device; Hailo cũng tuyên bố benchmark thật cần thiết. |
| “ECE thấp không có nghĩa an toàn.” | Không tuyên bố safety; báo cáo AURC, risk-at-coverage, UIoU và failure cases. |
| “Bốn board đều cũ/khác thế hệ.” | Đóng khung là heterogeneous legacy-to-NPU deployment; công bố stack EOL và giới hạn ngoại suy. Nếu có thể, mượn thêm một Orin làm external hardware validation. |

## 10. Lộ trình đến hạn nộp

### Mục tiêu chính: Complex Environment Vision — 15-02-2027

| Tuần | Deliverable | Gate |
|---|---|---|
| 1–2 | Inventory phần cứng/toolchain; compile Fast-SCNN/SegFormer-B0; xác minh Hailo-8 vs 8L | Có ít nhất một segmentation model chạy end-to-end trên cả bốn board |
| 3–4 | Benchmark harness, external power setup, baseline clean/adverse | Có p95/J-frame tái lập; không trộn kernel-only với end-to-end |
| 5–7 | Elastic supernet v1, independent-budget baselines | Subnet gap ≤2 mIoU ban đầu |
| 8–10 | Latency/energy LUT và hardware-in-the-loop Pareto selection | Measured objective thắng FLOPs objective trên ≥2 backend |
| 11–13 | QAT + distillation + compiler-safe refinement | INT8 gap tiến về ≤1.5 mIoU; HEF/TRT engines ổn định |
| 14–15 | Calibrated risk router + temporal policy | Router overhead <10% ở bản đầu |
| 16–18 | Full Cityscapes/ACDC experiments, 3 seeds | Có kết quả per-condition và reliability |
| 19–20 | Ablations, sustained thermal/power, failure cases | Đạt tiêu chí go/no-go chính |
| 21–22 | Viết manuscript, artifacts, reproducibility appendix | Internal review vòng 1 |
| 23–24 | Sửa, kiểm tra thống kê/hình/bảng, similarity/language | Submission-ready trước hạn 1–2 tuần |

**VP-NAV — 31-12-2026:** chỉ nên nhắm nếu đến giữa tháng 9 đã có baseline chạy trên bốn thiết bị và cuối tháng 10 có method ổn định. Nếu không, nộp tháng 2 để có đủ ablation và external validation.

## 11. Cấu trúc manuscript đề xuất

1. **Introduction:** real-world adverse vision + heterogeneous accelerator; nêu bốn gap, không mở đầu bằng “deep learning has developed rapidly”.
2. **Related work:** real-time segmentation; hardware-aware/dynamic networks; quantized cross-platform deployment; uncertainty/calibration under domain shift.
3. **Method:** compiler-safe search space, supernet training, measured-cost surrogate, calibrated router.
4. **Deployment protocol:** export/compile, engine audit, measurement boundary và reproducibility.
5. **Experiments:** datasets/baselines/metrics, main Pareto results, adverse reliability, cross-platform results.
6. **Ablations and failure analysis:** compiler failures, fallback, thermal, routing errors, rare classes.
7. **Limitations:** Xavier/Nano legacy stacks; calibration không phải chứng nhận safety; result phụ thuộc compiler version.
8. **Conclusion.**

### Cách diễn đạt contribution an toàn

Không viết “the first” ở bản đầu. Viết:

> We study the underexplored intersection of compiler-portable elastic segmentation, hardware-measured resource optimization, and calibrated risk-aware inference under adverse visual conditions.

Sau systematic review, chỉ nâng thành “to the best of our knowledge” nếu có bảng đối chiếu đủ rộng.

## 12. Kế hoạch 14 ngày đầu

1. Chạy `hailortcli fw-control identify`; lưu output và version manifest.
2. Chốt JetPack/TensorRT trên từng Jetson; ghi rõ Nano dùng JetPack 4 EOL, Xavier dùng bản JetPack được hỗ trợ cho platform.
3. Export cùng một Fast-SCNN hoặc BiSeNetV2 sang ONNX static shape, batch 1.
4. Build TensorRT FP16/INT8 cho Nano/NX/AGX; build DLA path trên NX/AGX với fallback tắt trước để phát hiện operator không hỗ trợ.
5. Parse–optimize–compile HEF trên Hailo; đo accuracy trước/sau quantization.
6. Chạy 5.000 ảnh hoặc replay video; log p50/p95/p99, memory, nhiệt và clocks.
7. Đo power toàn hệ thống; tách idle baseline; tính J/frame từ cùng performance run.
8. Lập bảng `operator × backend × status × latency`; bảng này sẽ định nghĩa search space thật.
9. Đánh giá một baseline trên Cityscapes và ACDC; xác nhận pipeline label/evaluation trước khi train model mới.
10. Chỉ bắt đầu supernet sau khi ít nhất một graph đi qua được cả TensorRT và Hailo.

## 13. Bốn thông tin cần khóa để chuyển sang implementation

1. **Ứng dụng cuối:** road/autonomous navigation, công nghiệp, nông nghiệp hay surveillance. Mặc định tốt nhất hiện tại là road-scene adverse segmentation.
2. **GPU training thực tế:** model GPU, VRAM, số lượng và thời gian được phép chạy.
3. **Hailo chính xác:** Hailo-8 26 TOPS hay Hailo-8L 13 TOPS; loại HAT/M.2; phiên bản software.
4. **Thiết bị đo điện/camera:** có external power analyzer và video/camera domain riêng hay chỉ dùng public datasets.

## Kết luận quyết định

Nếu mục tiêu là một bài IMAVIS có đóng góp đủ sâu, hãy xây **PACE-Seg** và xem bốn board là một “phòng thí nghiệm dị thể” để chứng minh phương pháp, không phải mục tiêu benchmark đơn thuần. Đóng góp trung tâm nên là:

> **Một model co giãn, sinh các engine tĩnh tương thích đa-compiler, phân bổ compute theo calibrated visual risk, và được tối ưu bằng latency/năng lượng đo thật dưới adverse domain shift.**

Hướng này khác đáng kể với ba lựa chọn yếu hơn: model nhẹ chỉ dựa FLOPs, một biến thể YOLO có thêm module, hoặc bảng so FPS của model có sẵn. Nó cũng đủ hẹp để hoàn thành trước 15-02-2027 nếu hai tuần đầu vượt qua compiler smoke test.

## Nguồn trọng yếu

- [IMAVIS journal page](https://www.sciencedirect.com/journal/image-and-vision-computing)
- [IMAVIS Guide for Authors](https://www.sciencedirect.com/journal/image-and-vision-computing/publish/guide-for-authors)
- [Complex Environment Vision call](https://www.sciencedirect.com/special-issue/335145/complex-environment-vision)
- [VP-NAV call](https://www.sciencedirect.com/special-issue/335697/visual-perception-enabling-autonomous-navigation-vp-nav)
- [Dynamic Neural Networks survey, IMAVIS 2026](https://www.sciencedirect.com/science/article/pii/S0262885626000879)
- [UCPNet, IMAVIS 2026](https://www.sciencedirect.com/science/article/pii/S0262885626001162)
- [YOLIC, IMAVIS 2024](https://www.sciencedirect.com/science/article/pii/S0262885624001999)
- [HARD, ACCV 2024](https://openaccess.thecvf.com/content/ACCV2024/html/Kwon_HARD__Hardware-Aware_lightweight_Real-time_semantic_segmentation_model_Deployable_from_ACCV_2024_paper.html)
- [ACDC dataset and benchmark](https://acdc.vision.ee.ethz.ch/)
- [NVIDIA DLA restrictions](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/dla-layer-restrictions.html)
- [NVIDIA Jetson FAQ](https://developer.nvidia.com/embedded/faq)
- [Hailo-8 official specifications](https://hailo.ai/products/ai-accelerators/hailo-8-ai-accelerator/)
- [Raspberry Pi AI software/hardware documentation](https://www.raspberrypi.com/documentation/computers/ai.html)
- [MLPerf Power methodology](https://arxiv.org/abs/2410.12032)
- [MLPerf power measurement rules](https://github.com/mlcommons/inference_policies/blob/master/power_measurement.adoc)
