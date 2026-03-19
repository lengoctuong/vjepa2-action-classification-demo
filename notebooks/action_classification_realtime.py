import time
from datetime import datetime
import cv2
import json
import os
import subprocess
import numpy as np
import torch
import torch.nn.functional as F
import gradio as gr
from transformers import AutoVideoProcessor, AutoModel

# --- Import các module local ---
import src.datasets.utils.video.transforms as video_transforms
import src.datasets.utils.video.volume_transforms as volume_transforms
from src.models.attentive_pooler import AttentiveClassifier
from src.models.vision_transformer import vit_giant_xformers_rope

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

def load_pretrained_vjepa_classifier_weights(model, pretrained_weights):
    pretrained_dict = torch.load(pretrained_weights, weights_only=True, map_location="cpu")["classifiers"][0]
    pretrained_dict = {k.replace("module.", ""): v for k, v in pretrained_dict.items()}
    msg = model.load_state_dict(pretrained_dict, strict=False)
    print("Pretrained weights found at {} and loaded with msg: {}".format(pretrained_weights, msg))

# ==========================================
# 1. KHỞI TẠO VÀ TẢI MODEL 
# ==========================================
print("Đang tải dữ liệu và khởi tạo Model, vui lòng đợi...")

ssv2_classes_path = "ssv2_classes.json"
if not os.path.exists(ssv2_classes_path):
    command = [
        "wget",
        "https://huggingface.co/datasets/huggingface/label-files/resolve/d79675f2d50a7b1ecf98923d42c30526a51818e2/something-something-v2-id2label.json",
        "-O",
        "ssv2_classes.json",
    ]
    subprocess.run(command)

SOMETHING_SOMETHING_V2_CLASSES = json.load(open("ssv2_classes.json", "r"))

hf_model_name = "facebook/vjepa2-vitg-fpc64-384"
model_hf = AutoModel.from_pretrained(hf_model_name)
model_hf.cuda().eval()
hf_transform = AutoVideoProcessor.from_pretrained(hf_model_name)

classifier_model_path = "ssv2/ssv2-vitg-384-64x2x3.pt"
classifier = AttentiveClassifier(
    embed_dim=model_hf.config.hidden_size, 
    num_heads=16, 
    depth=4, 
    num_classes=174
).cuda().eval()

if os.path.exists(classifier_model_path):
    load_pretrained_vjepa_classifier_weights(classifier, classifier_model_path)
else:
    print(f"⚠️ Không tìm thấy file {classifier_model_path}. Code sẽ chạy với weights ngẫu nhiên!")

# ==========================================
# 2. HÀM XỬ LÝ TOÀN BỘ VIDEO TỪ CLIENT
# ==========================================
MODEL_SEQ_LEN = 64       

def process_video_chunk(video_path):
    if not video_path:
        return "❌ Lỗi: Không nhận được file video.", None

    start_infer_time = time.time()
    
    try:
        # 1. Đọc toàn bộ video file được gửi từ Client
        cap = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
        cap.release()
        
        total_frames = len(frames)
        if total_frames == 0:
            return "❌ Lỗi: Video trống hoặc không hợp lệ.", None

        # # 2. LOGIC TRÍCH XUẤT CHUẨN KINETICS-400 (SINGLE CLIP)
        # target_frames = 16
        # frame_step = 4
        
        # # Để lấy 16 frames với bước nhảy 4, cần một đoạn video dài tối thiểu 61 frames (khoảng ~2s ở 30fps)
        # # Công thức: (16 - 1) * 4 + 1 = 61
        # required_length = (target_frames - 1) * frame_step + 1 
        
        # if total_frames >= required_length:
        #     # Lấy 1 đoạn 61 frames nằm chính giữa video
        #     start_idx = (total_frames - required_length) // 2
        #     extract_indices = [start_idx + i * frame_step for i in range(target_frames)]
        # else:
        #     # Fallback: Nếu video quá ngắn (< 2s), tự động rải đều để gom đủ 16 frames
        #     extract_indices = np.linspace(0, total_frames - 1, target_frames).astype(int)

        # extracted_frames = [frames[i] for i in extract_indices]

        # 2. LOGIC TRÍCH XUẤT CHUẨN KINETICS-400 (SINGLE CLIP - LẤY CUỐI VIDEO)
        target_frames = 16
        frame_step = 4
        
        # Chiều dài tối thiểu cần thiết để lấy 16 frames với step 4 là 61 frames
        required_length = (target_frames - 1) * frame_step + 1 
        
        if total_frames >= required_length:
            # Dịch chuyển start_idx về sát cuối video thay vì chia đôi
            start_idx = total_frames - required_length
            extract_indices = [start_idx + i * frame_step for i in range(target_frames)]
        else:
            # Fallback: Rải đều nếu video ngắn hơn 61 frames
            extract_indices = np.linspace(0, total_frames - 1, target_frames).astype(int)

        extracted_frames = [frames[i] for i in extract_indices]

        # 3. Nội suy lên 64 frames (yêu cầu đầu vào của kiến trúc model hiện tại)
        model_indices = np.linspace(0, len(extracted_frames) - 1, MODEL_SEQ_LEN).astype(int)
        padded_frames = [extracted_frames[i] for i in model_indices]
        
        # 4. Chuyển đổi sang Tensor và đưa vào Model
        video_array = np.array(padded_frames) 
        video_tensor = torch.from_numpy(video_array).permute(0, 3, 1, 2)
        
        x_hf = hf_transform(video_tensor, return_tensors="pt")["pixel_values_videos"].to("cuda")
        
        with torch.inference_mode():
            out_patch_features_hf = model_hf.get_vision_features(x_hf)
            out_classifier = classifier(out_patch_features_hf)
            
        top3_indices = out_classifier.topk(3).indices[0]
        top3_probs = F.softmax(out_classifier.topk(3).values[0], dim=0) * 100.0
        
        infer_duration = time.time() - start_infer_time
        
        # 5. TẠO VIDEO TỪ CÁC FRAME ĐÃ TRÍCH XUẤT
        output_video_path = "temp_infer_video.mp4"
        height, width, _ = extracted_frames[0].shape
        
        # Khởi tạo VideoWriter (8 fps để quan sát mượt 16 frames trong 2 giây)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = 8 
        out_video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        
        for frame_rgb in extracted_frames:
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            out_video.write(frame_bgr)
            
        out_video.release()
        
        recv_time_str = datetime.now().strftime('%H:%M:%S')
        
        # 6. Chuẩn bị text kết quả
        result_text = f"⏱️ Thời gian xử lý: {infer_duration:.2f}s\n"
        result_text += f"📦 Phân tích {target_frames} frames (Step={frame_step})\n"
        result_text += f"✂️ Vị trí trích xuất: Frame {extract_indices[0]} đến {extract_indices[-1]} / Tổng {total_frames}\n"
        result_text += f"🕒 Nhận input lúc: {recv_time_str}\n\n"
        result_text += "🎯 Nhận diện (Kinetics Style):\n"
        
        i = 0
        for idx, prob in zip(top3_indices, top3_probs):
            class_name = SOMETHING_SOMETHING_V2_CLASSES.get(str(idx.item()), "Unknown")
            result_text += f"- Top {i+1} - {class_name}: {prob:.1f}%\n"
            i += 1
            
        return result_text, output_video_path

    except Exception as e:
        print(f"LỖI BACKEND: {str(e)}")
        return f"❌ Lỗi: {str(e)}", None

# ==========================================
# CẤU HÌNH GIAO DIỆN
# ==========================================
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎥 Nhận diện Hành động V-JEPA (Cấu hình K400 - Single Clip)")
    
    with gr.Row():
        with gr.Column(scale=1):
            video_input = gr.Video(sources=["webcam"], label="1. Quay hoặc Tải video lên")
            submit_btn = gr.Button("🚀 Gửi Video", variant="primary")
            
        with gr.Column(scale=1):
            val_video = gr.Video(label="2. Video AI đã nhận (16 Frames trích xuất)")
            
        with gr.Column(scale=1):
            result_output = gr.Textbox(label="3. Log & Kết quả", lines=10)

    # Không còn input từ Slider
    submit_btn.click(
        fn=process_video_chunk, 
        inputs=[video_input], 
        outputs=[result_output, val_video]
    )

demo.launch(share=True)