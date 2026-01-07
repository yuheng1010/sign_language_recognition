# app.py
import cv2
import mediapipe as mp
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import time
from collections import deque
from labels import gloss_dict

from web_server import SignLanguageWebServer

# ================= 1. TensorRT 引擎類別 =================
class SignLanguageModel:
    def __init__(self, engine_path):
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)

        print(f"Loading Engine: {engine_path}")
        with open(engine_path, "rb") as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())

        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()

        self.inputs = []
        self.outputs = []
        self.allocations = []

        for i in range(self.engine.num_bindings):
            name = self.engine.get_binding_name(i)
            dtype = self.engine.get_binding_dtype(i)
            shape = self.engine.get_binding_shape(i)
            size = trt.volume(shape) * self.engine.max_batch_size
            dtype_np = trt.nptype(dtype)

            # Allocate Memory
            host_mem = cuda.pagelocked_empty(size, dtype_np)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            self.allocations.append(int(device_mem))

            binding = {
                'index': i, 'name': name, 'dtype': dtype_np,
                'shape': shape, 'host': host_mem, 'device': device_mem
            }

            if self.engine.binding_is_input(i):
                self.inputs.append(binding)
            else:
                self.outputs.append(binding)

    def infer(self, input_shape_data, input_traj_data):
        # 確保數據展平並複製到 Host Memory
        np.copyto(self.inputs[0]['host'], input_shape_data.ravel())
        np.copyto(self.inputs[1]['host'], input_traj_data.ravel())

        # Host -> Device
        for inp in self.inputs:
            cuda.memcpy_htod_async(inp['device'], inp['host'], self.stream)

        # Execute
        self.context.execute_async_v2(bindings=self.allocations, stream_handle=self.stream.handle)

        # Device -> Host
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out['host'], out['device'], self.stream)

        self.stream.synchronize()
        return self.outputs[0]['host']

# ================= 2. 數據處理與正規化 =================

def normalize_skeleton_wrist_centric(skeleton: np.ndarray) -> np.ndarray:
    """
    將骨架座標轉換為以手腕為中心，並根據手指長度進行縮放。
    Input: (T, 21, 3)
    Output: (T, 21, 3)
    """
    # 1. 以手腕 (index 0) 為中心
    wrist_positions = skeleton[:, 0, :]  # (T, 3)
    wrist_centric = skeleton - wrist_positions[:, None, :]  # (T, 21, 3)

    # 2. 縮放 (使用 index 4 拇指尖 和 index 8 食指尖 的距離作為參考)
    if skeleton.shape[1] >= 9:
        diff = wrist_centric[:, 4, :] - wrist_centric[:, 8, :]  # (T, 3)
        scale_ref = np.linalg.norm(diff, axis=-1, keepdims=True)  # (T, 1)
        scale_ref = np.maximum(scale_ref, 1e-6) # 防止除以 0
        wrist_centric = wrist_centric / scale_ref[:, :, None]  # (T, 21, 3)

    # 3. 數值截斷，保持在合理範圍
    wrist_centric = np.clip(wrist_centric, -2.0, 2.0)
    return wrist_centric.astype(np.float32)

def normalize_traj_delta_scaled(skeleton_raw: np.ndarray) -> np.ndarray:
    """
    計算手腕相對於第一幀的移動軌跡，並進行縮放。
    Input: (T, 21, 3) - 原始座標
    Output: (T, 3) - 軌跡向量
    """
    wrist = skeleton_raw[:, 0, :]  # (T,3) 每一幀的手腕座標
    base = wrist[0:1, :]           # (1,3) 第一幀的手腕座標 (基準點)
    delta = wrist - base           # (T,3) 相對位移

    # 計算整個序列的平均縮放比例
    if skeleton_raw.shape[1] >= 9:
        diff = skeleton_raw[:, 4, :] - skeleton_raw[:, 8, :]  # (T,3)
        scale = np.linalg.norm(diff, axis=-1, keepdims=True)  # (T,1)
        scale = np.maximum(scale, 1e-6)
        scale_mean = float(np.mean(scale))
    else:
        scale_mean = 1.0
    
    scale_mean = max(scale_mean, 1e-6)

    traj = delta / scale_mean
    traj = np.clip(traj, -2.0, 2.0)
    return traj.astype(np.float32)

def process_landmarks(results):
    """
    從 MediaPipe 結果提取原始座標 (不進行平均或預處理，保留原始數據給正規化函式用)。
    """
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        coords = []
        for lm in hand_landmarks.landmark:
            coords.append([lm.x, lm.y, lm.z])
        coords = np.array(coords, dtype=np.float32)
        return coords
    else:
        # 若沒偵測到手，回傳全 0
        return np.zeros((21, 3), dtype=np.float32)

def get_label_text(class_id):
    label = gloss_dict.get(class_id)
    if label:
        return f"{class_id}: {label}"
    else:
        return f"Unknown ID: {class_id}"

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

# ================= 3. 主程式邏輯 =================

def main():
    # --- 設定 ---
    ENGINE_PATH = "model.engine"
    WINDOW_SIZE = 16  # 必須與 export_onnx.py 的 num_frames 一致

    web_server = SignLanguageWebServer()
    web_server.start(port=5000)
    print("Web Server 已在背景啟動，OpenCV 視窗將在稍後出現...")

    trt_model = SignLanguageModel(ENGINE_PATH)

    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        max_num_hands=1,
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # [修改] 只使用一個 Buffer 儲存原始座標 (T, 21, 3)
    raw_buffer = deque(maxlen=WINDOW_SIZE)

    print("初始化緩衝區...")
    # 先填入零，避免剛啟動時 buffer 不足
    for _ in range(WINDOW_SIZE):
        raw_buffer.append(np.zeros((21, 3), dtype=np.float32))

    print("系統就緒! 按 'q' 離開")

    while cap.isOpened():
        start_time = time.time()
        ret, frame = cap.read()
        if not ret: break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        # 1. 取得原始座標 (21, 3)
        coords = process_landmarks(results)
        
        # 2. 加入 Buffer
        raw_buffer.append(coords)

        # 3. 在推論前進行正規化 (Normalization)
        # 將 deque 轉為 numpy array: (T, 21, 3)
        sk_raw = np.array(raw_buffer, dtype=np.float32)

        # 計算 shape: 以手腕為中心 + 縮放
        sk_norm = normalize_skeleton_wrist_centric(sk_raw)  # Output: (T, 21, 3)
        
        # 計算 trajectory: 相對第一幀的位移 + 縮放
        traj_norm = normalize_traj_delta_scaled(sk_raw)     # Output: (T, 3)

        # 4. 增加 Batch 維度 (B, T, ...)
        input_shape = sk_norm[np.newaxis, ...]  # (1, 16, 21, 3)
        input_traj = traj_norm[np.newaxis, ...] # (1, 16, 3)

        # --- TensorRT 推論 ---
        logits = trt_model.infer(input_shape, input_traj)
        
        # 套用 Softmax
        probs = softmax(logits)

        # 取得 Top 5
        top5_indices = np.argsort(probs)[::-1][:5]

        # --- 顯示與 Web 更新 ---
        fps = 1.0 / (time.time() - start_time)
        web_display_list = []

        if results.multi_hand_landmarks:
            for hand_lms in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)

        base_y = 50
        for i, idx in enumerate(top5_indices):
            score = probs[idx]
            label_text = get_label_text(idx)
            
            display_text = f"{i+1}. {label_text} ({score:.1%})"
            web_display_list.append(display_text)

            color = (0, 255, 0) if i == 0 else (0, 255, 255)
            font_scale = 0.8 if i == 0 else 0.6
            thickness = 2 if i == 0 else 1
            
            cv2.putText(frame, display_text, (10, base_y + i * 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        web_message = " <br> ".join(web_display_list)
        web_server.update_inference_result(web_message)

        cv2.imshow("Jetson Nano Sign Language", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()