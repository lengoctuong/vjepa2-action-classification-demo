import os

# Chỉ dùng GPU index 4
os.environ["CUDA_VISIBLE_DEVICES"] = "4"                                

import sys
sys.path.append("/home/dev/tuongln2/vjepa2")


#%%
import torch
import numpy as np
import os
import json
import subprocess
import pandas as pd
from decord import VideoReader, cpu
from transformers import AutoVideoProcessor, AutoModel # <--- Đổi ở đây

# Import module Classifier từ src (Vẫn bắt buộc vì HF không có class này)
from evals.action_anticipation_frozen.models import AttentiveClassifier

EMB_VIDEO_SIZE = 384

# ==========================================
# 1. CẤU HÌNH & TẢI TÀI NGUYÊN PHỤ
# ==========================================

# HuggingFace Model ID (Encoder ViT-Giant)
# Lưu ý: Chọn đúng model tương ứng với Classifier (ViT-g/16)
HF_MODEL_NAME = f"facebook/vjepa2-vitg-fpc64-{EMB_VIDEO_SIZE}" 

# Classifier Weights (Vẫn phải tải thủ công vì đây là downstream task)
CLASSIFIER_URL = "https://dl.fbaipublicfiles.com/vjepa2/evals/ek100-vitg-384.pt"
classifier_path = "checkpoints/ek100-vitg-384.pt"

# Tải tài nguyên phụ
os.makedirs("checkpoints", exist_ok=True)
def download_if_not_exists(url, path):
    if not os.path.exists(path):
        print(f"Downloading {path}...")
        subprocess.run(["wget", url, "-O", path, "-q"])

download_if_not_exists(CLASSIFIER_URL, classifier_path)
# (Giả sử bạn đã có file csv verb/noun classes như bài trước)

# Load Class Mappings
verbs_df = pd.read_csv("EPIC_100_verb_classes.csv")
nouns_df = pd.read_csv("EPIC_100_noun_classes.csv")
id2verb = dict(zip(verbs_df.id, verbs_df.key))
id2noun = dict(zip(nouns_df.id, nouns_df.key))

# ==========================================
# 2. LOAD MODEL (CÁCH DÙNG HUGGINGFACE)
# ==========================================
device = "cuda" if torch.cuda.is_available() else "cpu"

print(">>> Loading Encoder from HuggingFace...")
# Tự động tải weights và config
encoder_hf = AutoModel.from_pretrained(HF_MODEL_NAME).to(device).eval()

try:
    processor = AutoVideoProcessor.from_pretrained(HF_MODEL_NAME)
    print("✅ Đã load được AutoVideoProcessor từ HuggingFace.")
except OSError:
    print("❌ Lỗi: Repo trên HuggingFace thiếu file 'preprocessor_config.json'.")
    print("👉 Đang chuyển sang dùng Manual Transform (Code thủ công)...")
    # Fallback về Manual Transform nếu load thất bại (Code ở dưới)
    processor = None


#%%
print(">>> Loading Classifier (Manual)...")
# Cấu hình Classifier khớp với ViT-Giant (Embed dim 1408)
dummy_verb_classes = {i: str(i) for i in range(97)}
dummy_noun_classes = {i: str(i) for i in range(289)}
dummy_action_classes = {i: str(i) for i in range(3568)}

classifier = AttentiveClassifier(
    embed_dim=1408, # ViT-Giant output dim
    num_heads=16,
    depth=4,
    verb_classes=dummy_verb_classes,
    noun_classes=dummy_noun_classes,
    action_classes=dummy_action_classes,
    use_activation_checkpointing=False
).to(device).eval()

# Load weights cho classifier
# Lưu ý: Cần xử lý key như cũ vì file .pt này là của Meta, không phải HF
ckpt = torch.load(classifier_path, map_location="cpu")
state_dict = ckpt["classifiers"][0] if "classifiers" in ckpt else ckpt
state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
classifier.load_state_dict(state_dict, strict=False)


#%%
import torch
import numpy as np
from decord import VideoReader, cpu

def prepare_vjepa_input(
    video_path, 
    device, 
    start_sec=0.0, 
    target_fps=8, 
    context_frames=32
):
    """
    Chuẩn bị dữ liệu đầu vào cho V-JEPA Action Anticipation.
    
    Args:
        video_path (str): Đường dẫn video.
        processor: HuggingFace AutoVideoProcessor (hoặc tương đương).
        device: 'cuda' hoặc 'cpu'.
        start_sec (float): Thời điểm bắt đầu lấy mẫu (giây). Default: 0.0
        anticipation_gap (float): Khoảng thời gian muốn dự đoán về tương lai (giây). Default: 1.0
        target_fps (int): FPS mục tiêu muốn lấy mẫu. Default: 8 (cho V-JEPA 2).
        context_frames (int): Số lượng frame ngữ cảnh. Default: 32 (32/8 = 4 giây).
        
    Returns:
        pixel_values: Tensor [1, 32, 3, H, W] đã chuẩn hóa.
        anticipation_tensor: Tensor [1] chứa giá trị anticipation_gap.
    """
    
    # 1. Load Video Info
    vr = VideoReader(video_path, ctx=cpu(0))
    native_fps = vr.get_avg_fps()
    total_frames = len(vr)
    video_duration = total_frames / native_fps
    
    print(f"Video info: {native_fps:.2f} FPS | Duration: {video_duration:.2f}s")
    
    # 2. Tính toán Indices để lấy mẫu (Resampling)
    # Công thức: frame_idx = round(time * native_fps)
    # Ta cần lấy các mốc thời gian: t0, t0 + 1/8, t0 + 2/8, ...
    
    time_stamps = [start_sec + i * (1.0 / target_fps) for i in range(context_frames)]
    frame_indices = np.array([int(t * native_fps) for t in time_stamps])
    
    # Xử lý biên (nếu start_sec quá trễ, video không đủ frame)
    # Clip indices để không vượt quá tổng số frame
    frame_indices = np.clip(frame_indices, 0, total_frames - 1)
    
    # Kiểm tra log
    actual_duration = (frame_indices[-1] - frame_indices[0]) / native_fps
    print(f"Sampling: {context_frames} frames @ {target_fps}fps")
    print(f"Context window: {start_sec}s -> {start_sec + actual_duration:.2f}s (Indices: {frame_indices[0]}-{frame_indices[-1]})")
    
    # 3. Lấy dữ liệu ảnh
    return vr.get_batch(frame_indices).asnumpy() # (T, H, W, C)
    
def process_video(buffer, processor, device, context_frames=32, anticipation_gap=1.0, ):
    # 4. Preprocessing (Qua Processor)
    inputs = processor(videos=list(buffer), return_tensors="pt")
    pixel_values = inputs.pixel_values_videos.to(device)
    
    # 5. Fix Shape: Đảm bảo (Batch, Time, Channels, Height, Width)
    if pixel_values.ndim == 4: # Nếu bị gộp (B*T, C, H, W)
        _, c, h, w = pixel_values.shape
        pixel_values = pixel_values.view(1, context_frames, c, h, w)
    elif pixel_values.ndim == 5:
        # Đôi khi HF trả về (B, C, T, H, W), V-JEPA cần (B, T, C, H, W)
        if pixel_values.shape[1] == 3: # Nếu channels đứng trước time
             pixel_values = pixel_values.permute(0, 2, 1, 3, 4)
             
    # 6. Tạo Anticipation Tensor
    # V-JEPA nhận vào tensor chứa thời gian muốn dự đoán (đơn vị giây)
    anticipation_tensor = torch.tensor([anticipation_gap], dtype=torch.float32).to(device)

    return pixel_values, anticipation_tensor

# ==========================================
# CÁCH SỬ DỤNG
# ==========================================

# Giả sử bạn đã có:
# - video_path
# - processor (hoặc val_transform pipeline thủ công)
# - device

video_path = "P01_101.mp4"

# Cấu hình mong muốn
START_SECOND = 8   # Bắt đầu từ giây số 0
ANTICIPATION = 1.0   # Dự đoán 1 giây sau đó
TARGET_FPS = 8       # Downsample về 8fps
CONTEXT_LEN = 32     # Lấy 32 frames (4 giây)

buffer = prepare_vjepa_input(
    video_path=video_path,
    device=device,
    start_sec=START_SECOND,
    target_fps=TARGET_FPS,
    context_frames=CONTEXT_LEN
)

print(f"Buffer Shape: {buffer.shape}")


#%%
pixel_values, anticipation_times = process_video(buffer, 
              processor=processor, # Truyền cái val_transform thủ công vào đây nếu dùng manual)
              device=device,
              context_frames=CONTEXT_LEN,
              anticipation_gap=ANTICIPATION,)

print(f"Final Input Shape: {pixel_values.shape}")
print(f"Anticipation Gap: {anticipation_times.item()}s")


#%%
# Run Inference
with torch.inference_mode():
    # 1. Encoder (HuggingFace)
    # Output của HF model thường là object, ta cần lấy `last_hidden_state`
    outputs_hf = encoder_hf(pixel_values_videos=pixel_values)
    patch_features = outputs_hf.last_hidden_state 
    
    # *** QUAN TRỌNG ***
    # V-JEPA HF output shape: (Batch, Num_Tokens, Dim)
    # Token của V-JEPA bao gồm cả thời gian và không gian đã được làm phẳng.
    
    # 2. Classifier (Custom)
    predictions = classifier(patch_features)

# Xử lý kết quả (Giống bài trước)
top_verbs = predictions['verb'][0].softmax(dim=-1).topk(5)
top_nouns = predictions['noun'][0].softmax(dim=-1).topk(5)
top_actions = predictions['action'][0].softmax(dim=-1).topk(5)

print("\n========== KẾT QUẢ (HF BACKBONE) ==========")
print("TOP 5 ĐỘNG TỪ:")
for score, idx in zip(top_verbs.values, top_verbs.indices):
    print(f"  - {id2verb.get(idx.item(), 'Unknown'):<15} ({score.item()*100:.2f}%)")

print("\nTOP 5 DANH TỪ:")
for score, idx in zip(top_nouns.values, top_nouns.indices):
    print(f"  - {id2noun.get(idx.item(), 'Unknown'):<15} ({score.item()*100:.2f}%)")

print("\nTOP 5 HẠNG DỘNG:")
for score, idx in zip(top_actions.values, top_actions.indices):
    print(f"  - {idx.item():<5} ({score.item()*100:.2f}%)")