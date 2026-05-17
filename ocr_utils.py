# ocr_utils.py - Corrected version
import cv2
import re
import numpy as np
from typing import List, Tuple, Dict, Optional, Any
from cnocr import CnOcr
from paddleocr import PaddleOCR

class ScaleExtractor:
    def __init__(self):
        self.cn_ocr = CnOcr()
        self.paddle_ocr = PaddleOCR(use_angle_cls=True, lang='en')
    
    def normalize_units(self, text: str) -> str:
        """Enhanced normalization for unit representations"""
        if not text:
            return ""
        normalized = (
            text.strip()
            .replace('μm', 'µm')
            .replace('μ', 'µ')
            .replace('u', 'µ')
            .replace(' ', '')
            .lower()
        )
        if 'um' in normalized and not any(char.isdigit() for char in normalized.split('um')[0][-1:]):
            normalized = normalized.replace('um', 'µm')
        normalized = re.sub(r'[^\wµ]', '', normalized)
        return normalized

    def parse_scale_text(self, scale_text: str) -> Tuple[Optional[float], Optional[str]]:
        """Extract numerical value and unit from scale text"""
        if not scale_text:
            return None, None
        text = self.normalize_units(scale_text.lower())
        match = re.search(r'([\d.]+)\s*([a-zµμu]+)', text)
        if not match:
            return None, None
        try:
            value = float(match.group(1))
            unit = match.group(2)
            return value, unit
        except (ValueError, AttributeError):
            return None, None

    def run_cnocr(self, image: np.ndarray) -> Tuple[List[str], List[float], List[List[Tuple[int, int]]]]:
        """Run CnOCR on an image and return text, confidence, and boxes"""
        try:
            results = self.cn_ocr.ocr(image)
            box_texts = [res['text'] for res in results]
            box_confs = [res['score'] for res in results]
            boxes = [[(int(p[0]), int(p[1])) for p in res['position']] for res in results]
            return box_texts, box_confs, boxes
        except Exception as e:
            print(f"Error running CnOCR: {str(e)}")
            return [], [], []

    def run_paddleocr(self, image: np.ndarray) -> Tuple[List[str], List[float], List[List[Tuple[int, int]]]]:
        """Run PaddleOCR on an in-memory image"""
        try:
            result = self.paddle_ocr.ocr(image)
            texts = []
            confs = []
            boxes = []
            for page in result:
                if isinstance(page, dict) and 'rec_texts' in page:
                    for text, conf, box in zip(page['rec_texts'], page['rec_scores'], page['rec_polys']):
                        if text.strip():
                            texts.append(text)
                            confs.append(conf)
                            boxes.append(box)
                elif isinstance(page, list):
                    for line in page:
                        try:
                            box, (text, conf) = line
                            texts.append(text)
                            confs.append(conf)
                            boxes.append(box)
                        except Exception as e:
                            print(f"Skipping OCR line: {line}, error: {e}")
            return texts, confs, boxes
        except Exception as e:
            print(f"PaddleOCR error: {e}")
            return [], [], []

    def extract_scale_value(self, image: np.ndarray, scale_bar_bbox: List[Tuple[int, int]]) -> Dict[str, Any]:
        """Hybrid OCR approach to extract scale value near the scale bar"""
        try:
            x_coords = [p[0] for p in scale_bar_bbox]
            y_coords = [p[1] for p in scale_bar_bbox]
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)
            
            x_bar, y_bar = (x_min + x_max) / 2, (y_min + y_max) / 2
            
            unit_patterns = [r'cm', r'mm', r'[µμu]m', r'nm', r'pm']
            unit_regex = re.compile(r'\b(' + '|'.join(unit_patterns) + r')\b', re.IGNORECASE)
            candidates = []
            
            cn_texts, cn_confs, cn_boxes = self.run_cnocr(image)
            for text, conf, box in zip(cn_texts, cn_confs, cn_boxes):
                if conf < 0.3:
                    continue
                box_points = np.array(box)
                x_text, y_text = np.mean(box_points[:, 0]), np.mean(box_points[:, 1])
                if unit_regex.search(text) or unit_regex.search(self.normalize_units(text)):
                    distance = np.sqrt((x_bar - x_text)**2 + (y_bar - y_text)**2)
                    candidates.append({
                        'text': text,
                        'distance': distance,
                        'confidence': conf,
                        'position': (x_text, y_text),
                        'engine': 'CnOCR'
                    })
            
            if not candidates:
                paddle_texts, paddle_confs, paddle_boxes = self.run_paddleocr(image)
                for text, conf, box in zip(paddle_texts, paddle_confs, paddle_boxes):
                    if conf < 0.3:
                        continue
                    box_points = np.array(box)
                    x_text, y_text = np.mean(box_points[:, 0]), np.mean(box_points[:, 1])
                    if unit_regex.search(text) or unit_regex.search(self.normalize_units(text)):
                        distance = np.sqrt((x_bar - x_text)**2 + (y_bar - y_text)**2)
                        candidates.append({
                            'text': text,
                            'distance': distance,
                            'confidence': conf,
                            'position': (x_text, y_text),
                            'engine': 'PaddleOCR'
                        })
            
            if not candidates:
                for text, conf, box in zip(cn_texts, cn_confs, cn_boxes):
                    if conf < 0.3:
                        continue
                    box_points = np.array(box)
                    x_text, y_text = np.mean(box_points[:, 0]), np.mean(box_points[:, 1])
                    distance = np.sqrt((x_bar - x_text)**2 + (y_bar - y_text)**2)
                    candidates.append({
                        'text': text,
                        'distance': distance,
                        'confidence': conf,
                        'position': (x_text, y_text),
                        'engine': 'CnOCR'
                    })
            
            if not candidates:
                return {
                    "text": "No OCR candidates",
                    "confidence": 0.0,
                    "value": None,
                    "unit": None,
                    "engine": "None"
                }
            
            closest = min(candidates, key=lambda x: x['distance'])
            value, unit = self.parse_scale_text(closest['text'])
            
            return {
                "text": closest['text'],
                "confidence": closest['confidence'],
                "value": value,
                "unit": unit,
                "engine": closest['engine']
            }
        except Exception as e:
            print(f"Error in extract_scale_value: {e}")
            return {
                "text": f"Error: {str(e)}",
                "confidence": 0.0,
                "value": None,
                "unit": None,
                "engine": "Exception"
            }

def extract_scale_bar_value(extractor, image, polygon):
    """Wrapper function for the ScaleExtractor method"""
    return extractor.extract_scale_value(image, polygon)
# NEW: Agent-compatible OCR wrapper
def extract_scale_bar_value_from_path(image_path: str, polygon_str: str) -> str:
    """
    Extract scale bar text from image using a polygon string.
    Used by LLM agent.
    
    Args:
        image_path: Path to the image file.
        polygon_str: Stringified polygon like "[[x1,y1],[x2,y2],[x3,y3],[x4,y4]]"
        
    Returns:
        JSON-style string: {"text": ..., "confidence": ..., "value": ..., "unit": ...}
    """
    import json
    from PIL import Image

    try:
        polygon = json.loads(polygon_str)
        image = np.array(Image.open(image_path).convert("RGB"))
        extractor = ScaleExtractor()
        result = extractor.extract_scale_value(image, polygon)
        return str(result)
    except Exception as e:
        return str({"error": str(e)})
