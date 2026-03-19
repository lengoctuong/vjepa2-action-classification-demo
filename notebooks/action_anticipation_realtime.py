import os

# Chỉ dùng GPU index 4
os.environ["CUDA_VISIBLE_DEVICES"] = "4"                                

import sys
sys.path.append("/home/dev/tuongln2/vjepa2")

import subprocess
import cv2
import torch
import numpy as np
import pandas as pd
from collections import deque
from tqdm import tqdm

class VJEPA_Anticipator:
    def __init__(self, 
                 encoder, 
                 classifier, 
                 processor, 
                 device, 
                 verb_map, 
                 noun_map, 
                 verb_threshold=0.05,   # > 5%
                 noun_threshold=0.02,   # > 2%
                 ):
        
        self.encoder = encoder
        self.classifier = classifier
        self.processor = processor
        self.device = device
        
        self.verb_map = verb_map
        self.noun_map = noun_map
        self.verb_threshold = verb_threshold
        self.noun_threshold = noun_threshold
        
        # Cấu hình cứng của V-JEPA
        self.context_frames = 32  # 32 frames @ 8fps = 4 giây
        self.buffer = deque(maxlen=self.context_frames)
        
        self.current_text = ""
        self.current_color = (0, 255, 255) # Yellow default

    def predict(self, buffer_frames):
        """
        Thực hiện dự đoán từ buffer frames (đã là 8fps)
        """
        # 1. Preprocess qua Processor
        # inputs.pixel_values_videos shape: [1, 32, 3, 384, 384] (nếu manual)
        inputs = self.processor(videos=[buffer_frames], return_tensors="pt")
        pixel_values = inputs.pixel_values_videos.to(self.device)
        
        # Fix shape nếu cần (HF đôi khi trả về Channel trước Time hoặc gộp Batch)
        if pixel_values.ndim == 4:
             _, c, h, w = pixel_values.shape
             pixel_values = pixel_values.view(1, self.context_frames, c, h, w)
        elif pixel_values.ndim == 5 and pixel_values.shape[1] == 3:
             pixel_values = pixel_values.permute(0, 2, 1, 3, 4)

        # 2. Inference
        with torch.inference_mode():
            # Encoder (nhận tham số pixel_values_videos theo yêu cầu)
            outputs = self.encoder(pixel_values_videos=pixel_values) # HF model arguments
            patch_features = outputs.last_hidden_state
            
            # Classifier
            preds = self.classifier(patch_features)
            
            # Softmax & Get Top 1
            verb_prob = preds['verb'][0].softmax(dim=-1)
            noun_prob = preds['noun'][0].softmax(dim=-1)
            
            v_score, v_idx = verb_prob.max(dim=-1)
            n_score, n_idx = noun_prob.max(dim=-1)
            
            return (v_idx.item(), v_score.item()), (n_idx.item(), n_score.item())

    def process_video(self, input_path, output_path, inference_fps=1.0):
        # --- BƯỚC 1: CONVERT VIDEO SANG 8 FPS ---
        temp_8fps_path = "temp_8fps.mp4"
        print(f"⏳ Converting video to 8 FPS...")
        cmd = [
            "ffmpeg", "-y", "-v", "error",
            "-i", input_path,
            "-r", "8",  # Force 8 FPS
            temp_8fps_path
        ]
        subprocess.run(cmd, check=True)
        
        # --- BƯỚC 2: XỬ LÝ TRÊN VIDEO 8 FPS ---
        cap = cv2.VideoCapture(temp_8fps_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Tính toán bước nhảy inference
        # Video 8fps, muốn infer 1 lần/s => step = 8 frames
        # Video 8fps, muốn infer 2 lần/s => step = 4 frames
        inference_step = int(8 / inference_fps)
        total_inferences = (total_frames - self.context_frames) // inference_step
        
        print(f"🎬 Processing: {total_frames} frames @ 8FPS.")
        print(f"🧠 Total Inferences: ~{total_inferences} steps (Every {inference_step} frames).")
        
        writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 8, (width, height))
        
        # Tqdm dựa trên số lần infer thực sự
        pbar = tqdm(total=total_inferences, unit="infer")
        
        frame_idx = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            # Convert BGR -> RGB cho model
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.buffer.append(frame_rgb)
            
            # Điều kiện chạy Inference:
            # 1. Buffer đầy (đủ 32 frames)
            # 2. Đúng nhịp (frame_idx chia hết cho step)
            if len(self.buffer) == self.context_frames and (frame_idx % inference_step == 0):
                
                # Convert deque to list for processor
                buffer_list = list(self.buffer)
                
                # Predict
                verb_res, noun_res = self.predict(buffer_list)
                
                v_id, v_score = verb_res
                n_id, n_score = noun_res
                v_name = self.verb_map.get(v_id, "???")
                n_name = self.noun_map.get(n_id, "???")
                
                # Update Tqdm & Terminal Log
                pbar.update(1)
                tqdm.write(
                    f"[Time {frame_idx/8:.1f}s] "
                    f"Pred: {v_name} ({v_score:.1%}) {n_name} ({n_score:.1%})"
                )
                
                # Update UI State (Dựa trên Threshold mới)
                if v_score > self.verb_threshold and n_score > self.noun_threshold:
                    self.current_text = f"NEXT: {v_name} ({v_score:.1%}) {n_name} ({n_score:.1%})"
                    
                    # Logic màu sắc: Xanh nếu rất tự tin (>20%), Vàng nếu vừa vừa
                    if v_score > 0.18:
                        self.current_color = (0, 255, 0) # Green
                    else:
                        self.current_color = (0, 255, 255) # Yellow
                else:
                    self.current_text = "" # Dưới ngưỡng thì ẩn
            
            # Vẽ UI (Overlay)
            if self.current_text:
                # Vẽ nền đen mờ
                overlay = frame.copy()
                cv2.rectangle(overlay, (20, 20), (700, 80), (0, 0, 0), -1)
                frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
                
                # Vẽ chữ
                cv2.putText(frame, self.current_text, (30, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, self.current_color, 2)
            
            writer.write(frame)
            frame_idx += 1
            
        cap.release()
        writer.release()
        pbar.close()

        # --- THÊM ĐOẠN NÀY ĐỂ FIX LỖI XEM VIDEO ---
        print("🔄 Re-encoding video for compatibility...")
        temp_output = output_path.replace(".mp4", "_temp.mp4")
        os.rename(output_path, temp_output)
        
        subprocess.run([
            "ffmpeg", "-y", "-v", "error",
            "-i", temp_output,
            "-c:v", "libx264", "-pix_fmt", "yuv420p", # Chuẩn H.264
            output_path
        ], check=True)
        
        if os.path.exists(temp_8fps_path): os.remove(temp_8fps_path)
        if os.path.exists(temp_output): os.remove(temp_output)
        
        print(f"✅ Done! Output saved to {output_path}")

# ==========================================
# CẤU HÌNH & CHẠY
# ==========================================
from transformers import AutoVideoProcessor, AutoModel # <--- Đổi ở đây
from evals.action_anticipation_frozen.models import AttentiveClassifier

device = "cuda:0"
EMB_VIDEO_SIZE = 384    
HF_MODEL_NAME = f"facebook/vjepa2-vitg-fpc64-{EMB_VIDEO_SIZE}" 
classifier_path = "checkpoints/ek100-vitg-384.pt"

processor = AutoVideoProcessor.from_pretrained(HF_MODEL_NAME)
encoder_hf = AutoModel.from_pretrained(HF_MODEL_NAME).to(device).eval()

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

ckpt = torch.load(classifier_path, map_location="cpu")
state_dict = ckpt["classifiers"][0] if "classifiers" in ckpt else ckpt
state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
classifier.load_state_dict(state_dict, strict=False)

verbs_df = pd.read_csv("EPIC_100_verb_classes.csv")
nouns_df = pd.read_csv("EPIC_100_noun_classes.csv")
id2verb = dict(zip(verbs_df.id, verbs_df.key))
id2noun = dict(zip(nouns_df.id, nouns_df.key))

anticipator = VJEPA_Anticipator(
    encoder=encoder_hf,
    classifier=classifier,
    processor=processor,
    device=device,
    verb_map=id2verb,
    noun_map=id2noun,
    verb_threshold=0.10,    # Ngưỡng > 5%
    noun_threshold=0.08,    # Ngưỡng > 2%
)

# 3. Chạy xử lý
# inference_fps=8: Check AI 8 lần mỗi giây (Rất mượt nhưng nặng)

import time
start = time.time()
anticipator.process_video("P01_102.mp4", "result_realtime_8fps.mp4",
                            inference_fps=1.0       # 1 giây dự đoán 1 lần
                        )
print("Total infer:", time.time() - start)