# api_client.py
import base64
import requests
import cv2
import numpy as np
from config import API_URL, API_MODEL, API_KEY

def image_to_base64(image_np: np.ndarray) -> str:
    """Convert numpy image (BGR or RGB) to base64 string."""
    _, buffer = cv2.imencode('.png', image_np)
    return base64.b64encode(buffer).decode('utf-8')

def query_api(prompt_text: str, image_np: np.ndarray = None, system_msg: str = None) -> str:
    """
    Send request to the configured LLM API.
    Returns the assistant's reply or an error message.
    """
    if not API_KEY:
        return "ERROR: Missing API_KEY. Please set SJTU_API_KEY in .env file."

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }

    content = [{"type": "text", "text": prompt_text}]
    if image_np is not None:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{image_to_base64(image_np)}"}
        })

    messages = []
    if system_msg:
        messages.append({"role": "system", "content": system_msg})
    messages.append({"role": "user", "content": content})

    data = {
        "model": API_MODEL,
        "messages": messages,
        "temperature": 0.2,
        "max_tokens": 800
    }

    try:
        r = requests.post(API_URL, headers=headers, json=data, timeout=200)
        r.raise_for_status()
        return r.json()['choices'][0]['message']['content']
    except Exception as e:
        return f"ERROR: {str(e)}"