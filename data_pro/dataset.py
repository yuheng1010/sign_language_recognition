import os
import cv2
import torch
import numpy as np
import pandas as pd
import json
import random
from torch.utils.data import Dataset
from torchvision import transforms
from scipy import ndimage

class WLASLDataset(Dataset):
    def __init__(self, json_file, root_dir, transform=None, mode='train', num_classes=2000,
                 num_frames=16, img_size=224, config=None):
        self.root_dir = root_dir
        self.transform = transform
        self.num_frames = num_frames
        self.img_size = img_size
        self.config = config or {}

        self._apply_augmentation = (mode == 'train' and
                                   self.config.get('augmentation', {}).get('enabled', False))
        
        with open(json_file, 'r') as f:
            content = json.load(f)

        self.samples = []

        if isinstance(content, dict):
            print(f"檢測到 NSLT 格式數據集")
            self._load_nslt_format(content, mode, num_classes)
        elif isinstance(content, list):
            print(f"檢測到 WLASL 格式數據集")
            self._load_wlasl_format(content, mode, num_classes)
        else:
            raise ValueError(f"不支持的數據格式: {type(content)}")

    def _load_wlasl_format(self, content, mode, num_classes):
        """載入 WLASL 格式數據"""
        self.label_map = {item['gloss']: idx for idx, item in enumerate(content) if idx < num_classes}

        for entry in content:
            gloss = entry['gloss']
            if gloss not in self.label_map:
                continue

            label = self.label_map[gloss]

            for instance in entry['instances']:
                if instance['split'] == mode:
                    video_id = instance['video_id']
                    video_path = os.path.join(self.root_dir, f"{video_id}.mp4")
                    if os.path.exists(video_path):
                        self.samples.append((video_path, label))

    def _load_nslt_format(self, content, mode, num_classes):

        for video_id, info in content.items():
            if info['subset'] != mode:
                continue

            class_id = info['action'][0]  

            if class_id >= num_classes:
                continue

            video_path = os.path.join(self.root_dir, f"{video_id}.mp4")
            if os.path.exists(video_path):
                self.samples.append((video_path, class_id))

        print(f"[{mode.upper()}] Loaded {len(self.samples)} samples.")

    def __len__(self):
        return len(self.samples)

    def _load_video(self, path):
        cap = cv2.VideoCapture(path)
        frames = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames <= 0:
            return np.zeros((self.num_frames, self.img_size, self.img_size, 3), dtype=np.uint8)

        indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
        
        for i in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break
            if i in indices:
                frame = cv2.resize(frame, (self.img_size, self.img_size))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
        
        cap.release()
        
        while len(frames) < self.num_frames:
            frames.append(frames[-1] if frames else np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8))

        frames = np.array(frames)[:self.num_frames]

        if hasattr(self, '_apply_augmentation') and self._apply_augmentation:
            frames = self._apply_video_augmentations(frames)

        return frames

    def _apply_video_augmentations(self, frames):
        if random.random() < 0.5:
            frames = frames[:, :, ::-1, :]  

        if random.random() < 0.3:
            angle = random.uniform(-15, 15)
            frames = self._rotate_frames(frames, angle)

        # 隨機縮放 (0.8-1.2倍)
        if random.random() < 0.3:
            scale = random.uniform(0.8, 1.2)
            frames = self._scale_frames(frames, scale)

        return frames

    def _rotate_frames(self, frames, angle):
        rotated_frames = []
        for frame in frames:
            frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0)
            rotated_tensor = transforms.functional.rotate(frame_tensor, angle)
            rotated_frame = rotated_tensor.squeeze(0).permute(1, 2, 0).numpy()
            rotated_frames.append(rotated_frame)
        return np.array(rotated_frames)

    def _scale_frames(self, frames, scale):
        scaled_frames = []
        for frame in frames:
            h, w = frame.shape[:2]
            new_h, new_w = int(h * scale), int(w * scale)
            scaled = cv2.resize(frame, (new_w, new_h))

            start_h = (new_h - h) // 2
            start_w = (new_w - w) // 2
            scaled = scaled[start_h:start_h+h, start_w:start_w+w]
            scaled_frames.append(scaled)
        return np.array(scaled_frames)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        buffer = self._load_video(path) 

        buffer = torch.from_numpy(buffer).float() / 255.0  
        buffer = buffer.permute(0, 3, 1, 2) 

        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        buffer = (buffer - mean) / std

        if self.transform:
            buffer = self.transform(buffer)

        return buffer, label