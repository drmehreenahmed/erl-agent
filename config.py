# config.py
import os
from dotenv import load_dotenv

load_dotenv()

# API settings
API_URL = os.getenv("DEEPSEEK_API_URL", "https://models.sjtu.edu.cn/api/v1/chat/completions")
API_MODEL = os.getenv("DEEPSEEK_API_MODEL", "qwen3.5-27b")
API_KEY = os.getenv("SJTU_API_KEY", "")

# Model paths
YOLO_MODEL_PATH = os.getenv("YOLO_MODEL_PATH", "best.pt")