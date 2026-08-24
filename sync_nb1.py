import json

base_experiments = [
    ("prelu", "baseline_prelu"),
    ("gelu", "baseline_gelu"),
    ("silu", "baseline_silu"),
    ("soft_clamp_relu", "baseline_soft_clamp_relu"),
    ("soft_clamp_silu", "baseline_soft_clamp_silu"),
    ("smooth_prelu", "baseline_smooth_prelu"),
    ("optimized_smooth_prelu", "baseline_optimized_smooth_prelu"),
]

nb1 = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# 🚀 Kaggle: Huấn luyện Baseline RIFE (Nạp dataset từ RAM siêu tốc)"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 1. KÉO CODE TỪ GITHUB\n",
                "!git clone https://github.com/NguyenMinhTri24072005/CT282-RIFE.git RIFE-Project\n",
                "%cd /kaggle/working/RIFE-Project/"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "#@title 📤 CẤU HÌNH GITHUB TOKEN TỪ KAGGLE SECRETS\n",
                "from kaggle_secrets import UserSecretsClient\n",
                "token = UserSecretsClient().get_secret(\"GITHUB_TOKEN\")\n",
                "\n",
                "GITHUB_USERNAME = \"NguyenMinhTri24072005\" #@param {type:\"string\"}\n",
                "GITHUB_EMAIL = \"Nguyenminhtri2475n@gmail.com\" #@param {type:\"string\"}\n",
                "REPO_NAME = \"CT282-RIFE\" #@param {type:\"string\"}\n",
                "\n",
                "GITHUB_TOKEN = token\n",
                "\n",
                "REPO_URL = f\"https://{GITHUB_TOKEN}@github.com/{GITHUB_USERNAME}/{REPO_NAME}.git\""
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "!git config --global user.email \"{GITHUB_EMAIL}\"\n",
                "!git config --global user.name \"{GITHUB_USERNAME}\""
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "!git remote set-url origin {REPO_URL}"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 2. CÀI ĐẶT THƯ VIỆN BỔ TRỢ\n",
                "!pip install -q datasets"
            ]
        }
    ],
    "metadata": {
        "accelerator": "GPU",
        "language_info": {
            "name": "python"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

for act, folder in base_experiments:
    train_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            f"# 3. HUẤN LUYỆN MODEL BASELINE ({act.upper()})\n",
            "!python train.py \\\n",
            "    --model_type original \\\n",
            "    --hf_dataset bijinc/vimeo-90k-mini \\\n",
            "    --batch_size 16 \\\n",
            "    --epoch 40 \\\n",
            f"    --act '{act}' \\\n",
            f"    --save_dir /kaggle/working/RIFE-Project/trained_model/{folder}"
        ]
    }
    push_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "!git add trained_model/\n",
            f'!git commit -m "Lưu kết quả train {folder} từ Kaggle [Auto Sync]"\n',
            "!git push {REPO_URL} main"
        ]
    }
    nb1["cells"].append(train_cell)
    nb1["cells"].append(push_cell)

target_file_1 = r"D:\GG_1\COMPUTER SCIENCE\NAM3_HK3_(2025-2026)\CT282_Deep Learning\PROJECT\RIFE-MinhTri\RIFE-Project\notebooks\kaggle\1_Train_Baseline_Kaggle.ipynb"
with open(target_file_1, "w", encoding="utf-8") as f:
    json.dump(nb1, f, indent=1, ensure_ascii=False)

print("Synchronized 1_Train_Baseline_Kaggle.ipynb successfully!")
