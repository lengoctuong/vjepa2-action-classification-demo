# V-JEPA 2: Real-time Action Classification Demo

**Original Project:** [V-JEPA 2 by Meta FAIR](https://github.com/facebookresearch/vjepa2)

**Paper:** [V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning](https://arxiv.org/abs/2506.09985)

> **Disclaimer:** This repository is a modified clone of the official V-JEPA 2 repository. It is intended solely for demonstration purposes. All core architecture and pre-trained weights belong to the original authors at Meta FAIR.

## 📌 About This Demo

This repository includes an interactive Gradio Web UI that performs **real-time action classification** on the Something-Something v2 (SSv2) dataset (174 classes).

It uses the frozen **V-JEPA 2 ViT-g384** (1B parameters) backbone to extract features from 16-frame video clips, which are then padded to 64 frames and passed through a pre-trained Attentive Classifier to predict complex physical interactions.

## ⚙️ Setup & Installation

**1. Create a Conda environment**
```bash
conda create -n vjepa2-demo python=3.10 -y
conda activate vjepa2-demo
```

**2. Install Core Dependencies**
Install PyTorch (CUDA recommended) along with other required libraries for the Web UI:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118  # Update CUDA version as needed
pip install transformers gradio opencv-python numpy
```

**3. Install the V-JEPA local package**
In the root directory of this repository, install the project in editable mode so the script can resolve the `src` module imports:
```bash
pip install -e .
```

## 📥 Download Pre-trained Weights

While the main ViT-g384 backbone (`facebook/vjepa2-vitg-fpc64-384`) and the class mappings (`ssv2_classes.json`) will be downloaded automatically by the script via HuggingFace, you **must manually download the specific SSv2 Classifier Probe** before running the app.

Create a folder named `ssv2` in the root directory and download the weights into it:
```bash
mkdir ssv2
wget https://dl.fbaipublicfiles.com/vjepa2/evals/ssv2-vitg-384-64x2x3.pt -P ssv2/
```
*(Ensure the file is located exactly at `ssv2/ssv2-vitg-384-64x2x3.pt`)*

## 🚀 Running the Demo

Once everything is installed and the weights are downloaded, launch the Gradio Web UI by running:
```bash
python notebooks/action_classification_realtime.py
```

A local URL (e.g., `http://127.0.0.1:7860`) and a public shareable link will be generated. Open the link in your browser to interact with the model via your webcam or video uploads.

Dưới đây là phần nội dung Markdown được thiết kế riêng cho tính năng **Action Anticipation** để bạn sao chép và dán bổ sung vào file `README.md` hiện tại (có thể đặt ngay dưới phần *Running the Demo* cũ).

## 🔮 Additional Demo: Action Anticipation (EPIC-KITCHENS-100)

This repository also includes a script demonstrating **Action Anticipation**. Instead of classifying an ongoing action, this script analyzes a 4-second context window (32 frames at 8 FPS) to predict what action (Verb and Noun) will happen **1 second in the future**.

**1. Download the Pre-trained Probe Weights**
You need the specific Attentive Classifier probe trained on the EPIC-KITCHENS-100 dataset.

```bash
mkdir -p checkpoints
wget https://dl.fbaipublicfiles.com/vjepa2/evals/ek100-vitg-384.pt -P checkpoints/
```

**2. Prepare Label Files**
Ensure you have the EPIC-KITCHENS-100 class mappings in your root directory:
* `EPIC_100_verb_classes.csv`
* `EPIC_100_noun_classes.csv`

**3. Run the Inference Script**

Open the script, update the `video_path` variable to point to your target video, and run:

```bash
python notebooks/action_anticipation_realtime.py
```
The output will display the Top 5 predicted Verbs, Nouns, and combined Actions in the terminal.

## 📖 Citation

If you find the underlying models or the original repository useful, please consider citing the original paper by Meta FAIR:

```bibtex
@article{assran2025vjepa2,
  title={V-JEPA~2: Self-Supervised Video Models Enable Understanding, Prediction and Planning},
  author={Assran, Mahmoud and Bardes, Adrien and Fan, David and Garrido, Quentin and Howes, Russell and
Komeili, Mojtaba and Muckley, Matthew and Rizvi, Ammar and Roberts, Claire and Sinha, Koustuv and Zholus, Artem and
Arnaud, Sergio and Gejji, Abha and Martin, Ada and Robert Hogan, Francois and Dugas, Daniel and
Bojanowski, Piotr and Khalidov, Vasil and Labatut, Patrick and Massa, Francisco and Szafraniec, Marc and
Krishnakumar, Kapil and Li, Yong and Ma, Xiaodong and Chandar, Sarath and Meier, Franziska and LeCun, Yann and
Rabbat, Michael and Ballas, Nicolas},
  journal={arXiv preprint arXiv:2506.09985},
  year={2025}
}
```
