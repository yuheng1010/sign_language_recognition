import os
import torch
import numpy as np
import pandas as pd
import json
import random
from pathlib import Path
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
        import cv2
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
        import cv2
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


class WLASLSkeletonDataset(Dataset):
    """
    skeleton dataset for DSLNet dual stream architecture
    return: (skeleton_shape, skeleton_traj, label)
    - skeleton_shape: hand pose data (B, T, 21, 3) - Wrist-centric
    - skeleton_traj: wrist trajectory data (B, T, 3) - Facial-centric
    """
    def __init__(self, json_file, mode='train', config=None):
        self.config = config or {}
        self.mode = mode
        self.num_frames = self.config.get('input', {}).get('num_frames', 16)
        self.num_joints = self.config.get('model', {}).get('num_joints', 21)

        with open(json_file, 'r') as f:
            content = json.load(f)
        
        self.samples = []
        num_classes = self.config.get('model', {}).get('num_classes', 100)

        if isinstance(content, list):
            if content and 'gloss' in content[0] and 'instances' in content[0]:
                self.label_map = {item['gloss']: idx for idx, item in enumerate(content) if idx < num_classes}

                for entry in content:
                    gloss = entry['gloss']
                    if gloss not in self.label_map:
                        continue

                    label = self.label_map[gloss]

                    for instance in entry['instances']:
                        if instance.get('split') == mode or instance.get('subset') == mode:
                            has_skeleton = ('skeleton' in instance or 'pose' in instance or 'skeleton_path' in instance)

                            if has_skeleton:
                                self.samples.append({
                                    'video_id': instance['video_id'],
                                    'label': label,
                                    'skeleton_data': instance.get('skeleton', instance.get('pose', None)),
                                    'skeleton_path': instance.get('skeleton_path', None)
                                })
            else:
                self.label_map = {}
                unique_glosses = set()

                for sample in content:
                    unique_glosses.add(sample['gloss'])

                sorted_glosses = sorted(list(unique_glosses))
                self.label_map = {gloss: idx for idx, gloss in enumerate(sorted_glosses) if idx < num_classes}

                for sample in content:
                    if sample.get('subset') == mode:
                        gloss = sample['gloss']
                        if gloss in self.label_map:
                            label = self.label_map[gloss]
                            self.samples.append({
                                'video_id': sample['video_id'],
                                'label': label,
                                'skeleton_data': sample.get('skeleton_data', sample.get('skeleton', None)),
                                'skeleton_path': sample.get('skeleton_path', None)
                            })
        
        print(f"[{mode.upper()}] Loaded {len(self.samples)} skeleton samples.")

        real_samples = [s for s in self.samples if s.get('skeleton_path') or s.get('skeleton_data')]
        print(f"[{mode.upper()}] Real skeleton samples: {len(real_samples)}/{len(self.samples)}")

        if len(real_samples) == 0:
            print(f"{mode} 模式沒有真實skeleton data 從其他模式借...")

            all_samples = []
            if isinstance(content, list):
                for sample in content:
                    gloss = sample['gloss']
                    if gloss in self.label_map:
                        label = self.label_map[gloss]
                        all_samples.append({
                            'video_id': sample['video_id'],
                            'label': label,
                            'skeleton_data': sample.get('skeleton_data', sample.get('skeleton', None)),
                            'skeleton_path': sample.get('skeleton_path', None),
                            'original_subset': sample.get('subset', 'unknown')
                        })

            all_real_samples = [s for s in all_samples if s.get('skeleton_path') or s.get('skeleton_data')]

            if len(all_real_samples) > 0:
                import random
                random.seed(42)  

                target_val_size = max(50, min(200, len(all_real_samples) // 5))

                if mode == 'val':
                    selected_samples = random.sample(all_real_samples, min(target_val_size, len(all_real_samples)))
                    self.samples = selected_samples
                    print(f"{mode} 模式從全數據集中借用 {len(self.samples)} 個真實skeleton data")
                else:
                    val_video_ids = set(s['video_id'] for s in all_real_samples if s.get('original_subset') == 'val')
                    train_samples = [s for s in all_real_samples if s['video_id'] not in val_video_ids]
                    self.samples = train_samples
                    print(f"{mode} 模式使用 {len(self.samples)} 個真實skeleton data（排除驗證集）")

                self.use_mock_data = False
            else:
                print(f"沒有任何真實skeleton data，使用模擬數據")
                self.use_mock_data = True
                self.samples = []
                for i in range(min(100, len(self.label_map))):
                    for label_idx in range(min(num_classes, len(self.label_map))):
                        self.samples.append({
                            'video_id': f'mock_{i}_{label_idx}',
                            'label': label_idx,
                            'skeleton_data': None,
                            'skeleton_path': None
                        })
                        if len(self.samples) >= 100:
                            break
                    if len(self.samples) >= 100:
                        break
        else:
            print(f"{mode} 模式使用真實skeleton data ({len(real_samples)} 個樣本)")
            self.samples = real_samples
            self.use_mock_data = False
    
    def __len__(self):
        return len(self.samples)
    
    def _normalize_skeleton(self, skeleton_data):
        wrist_positions = skeleton_data[:, 0, :]  # (T, 3)
        wrist_centric = skeleton_data - wrist_positions[:, np.newaxis, :]  # (T, 21, 3)

        if skeleton_data.shape[1] >= 9:  
            diff = wrist_centric[:, 4, :] - wrist_centric[:, 8, :]  # (T, 3)
            scale_ref = np.linalg.norm(diff, axis=-1, keepdims=True)  # (T, 1)
            scale_ref = np.maximum(scale_ref, 1e-6)  

            scale_ref = scale_ref[:, :, np.newaxis]  # (T, 1, 1)
            wrist_centric = wrist_centric / scale_ref  # (T, 21, 3)

        wrist_centric = np.clip(wrist_centric, -2.0, 2.0)

        return wrist_centric
    
    def _extract_trajectory(self, skeleton_shape):
        return skeleton_shape[:, 0, :]  # (T, 3)
    
    def _generate_mock_skeleton(self):
        skeleton_shape = np.random.randn(self.num_frames, self.num_joints, 3).astype(np.float32)
        skeleton_shape = skeleton_shape * 0.3
        
        t = np.linspace(0, 2*np.pi, self.num_frames)
        skeleton_traj = np.stack([
            np.sin(t) * 0.5,
            np.cos(t) * 0.3,
            np.sin(t * 2) * 0.2
        ], axis=-1).astype(np.float32)
        
        return skeleton_shape, skeleton_traj
    
    def _apply_augmentation(self, skeleton_shape, skeleton_traj):
        aug_config = self.config.get('augmentation', {})

        if aug_config.get('random_rotate', False) and random.random() < 0.5:
            angle = random.uniform(-15, 15) * np.pi / 180
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            rotation_matrix = np.array([[cos_a, -sin_a, 0], 
                                       [sin_a, cos_a, 0], 
                                       [0, 0, 1]], dtype=np.float32)
            skeleton_shape = np.dot(skeleton_shape, rotation_matrix.T)
            skeleton_traj = np.dot(skeleton_traj, rotation_matrix.T)
        
        if aug_config.get('random_scale', False) and random.random() < 0.5:
            scale = random.uniform(0.9, 1.1)
            skeleton_shape = skeleton_shape * scale
            skeleton_traj = skeleton_traj * scale

        noise_std = aug_config.get('gaussian_noise', 0.0)
        if noise_std > 0:
            skeleton_shape += np.random.randn(*skeleton_shape.shape).astype(np.float32) * noise_std
            skeleton_traj += np.random.randn(*skeleton_traj.shape).astype(np.float32) * noise_std
        
        return skeleton_shape, skeleton_traj
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        label = sample['label']
        
        if self.use_mock_data:
            skeleton_shape, skeleton_traj = self._generate_mock_skeleton()
        else:
            skeleton_data = sample.get('skeleton_data', None)
            skeleton_path = sample.get('skeleton_path', None)
            skeleton_traj = None  

            if skeleton_data is not None:
                skeleton_shape = np.array(skeleton_data, dtype=np.float32)
            elif skeleton_path is not None:
                try:
                    json_dir = Path(self.config.get('data', {}).get('json_file', '')).parent
                    full_path = json_dir / skeleton_path

                    with open(full_path, 'r', encoding='utf-8') as f:
                        file_data = json.load(f)

                    skeleton_traj_data = file_data.get('skeleton_traj', [])
                    if skeleton_traj_data:
                        skeleton_traj = np.array(skeleton_traj_data, dtype=np.float32)
                        if skeleton_traj.shape[0] != self.num_frames:
                            indices = np.linspace(0, len(skeleton_traj) - 1, self.num_frames, dtype=int)
                            skeleton_traj = skeleton_traj[indices]
                    else:
                        skeleton_traj = None

                    skeleton_shape_data = file_data.get('skeleton_shape', [])
                    if skeleton_shape_data:
                        skeleton_shape = np.array(skeleton_shape_data, dtype=np.float32)
                    else:
                        raise ValueError("骨架文件格式錯誤")
                except Exception as e:
                    print(f"載入骨架文件失敗 {skeleton_path}: {e}")
                    skeleton_shape = None
                    skeleton_traj = None
            else:
                skeleton_shape = None

            if skeleton_shape is None:
                skeleton_shape, skeleton_traj = self._generate_mock_skeleton()
            else:
                if skeleton_shape.shape[0] != self.num_frames:
                    indices = np.linspace(0, len(skeleton_shape) - 1, self.num_frames, dtype=int)
                    skeleton_shape = skeleton_shape[indices]

                if skeleton_traj is None:
                    skeleton_traj = self._extract_trajectory(skeleton_shape)

                skeleton_shape = self._normalize_skeleton(skeleton_shape)
        
        if self.mode == 'train':
            skeleton_shape, skeleton_traj = self._apply_augmentation(skeleton_shape, skeleton_traj)
        
        skeleton_shape = torch.from_numpy(skeleton_shape).float()  # (T, 21, 3)
        skeleton_traj = torch.from_numpy(skeleton_traj).float()    # (T, 3)
        label = torch.tensor(label, dtype=torch.long)
        
        return skeleton_shape, skeleton_traj, label