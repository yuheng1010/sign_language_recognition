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

# 引入剛剛建立的網頁伺服器模組
from web_server import SignLanguageWebServer 

# ================= 1. TensorRT 引擎類別 (保持不變) =================
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
        # 複製數據到 Host Memory
        np.copyto(self.inputs[0]['host'], input_shape_data.ravel()) # skeleton_shape
        np.copyto(self.inputs[1]['host'], input_traj_data.ravel())  # skeleton_traj
        
        # Host -> GPU
        for inp in self.inputs:
            cuda.memcpy_htod_async(inp['device'], inp['host'], self.stream)
            
        # Run Inference
        self.context.execute_async_v2(bindings=self.allocations, stream_handle=self.stream.handle)
        
        # GPU -> Host
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out['host'], out['device'], self.stream)
            
        self.stream.synchronize()
        return self.outputs[0]['host']

# ================= 2. 輔助函式 =================

def process_landmarks(results):
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        coords = []
        for lm in hand_landmarks.landmark:
            coords.append([lm.x, lm.y, lm.z])
        coords = np.array(coords, dtype=np.float32)
        center = np.mean(coords, axis=0)
        return coords, center
    else:
        return np.zeros((21, 3), dtype=np.float32), np.zeros((3,), dtype=np.float32)

def get_label_text(class_id):
    # 先嘗試取得對應的文字
    label = gloss_dict.get(class_id)
    
    if label:
        # 如果有找到，回傳 "ID: 文字" 的格式
        return f"{class_id}: {label}"
    else:
        # 如果沒找到，回傳錯誤訊息
        return f"未知手語 ID: {class_id}"

# ================= 3. 主程式邏輯 =================

def main():
    # --- 設定 ---
    ENGINE_PATH = "model.engine"
    WINDOW_SIZE = 16
    
    # 1. 啟動網頁伺服器 (在背景執行)
    web_server = SignLanguageWebServer()
    web_server.start(port=5000)
    print("Web Server 已在背景啟動，OpenCV 視窗將在稍後出現...")

    # 2. 初始化模型
    trt_model = SignLanguageModel(ENGINE_PATH)
    
    # 3. 初始化 MediaPipe
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        max_num_hands=1,
        model_complexity=0, 
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    # 4. 初始化相機
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    shape_buffer = deque(maxlen=WINDOW_SIZE)
    traj_buffer = deque(maxlen=WINDOW_SIZE)
    
    print("初始化緩衝區...")
    for _ in range(WINDOW_SIZE):
        shape_buffer.append(np.zeros((21, 3), dtype=np.float32))
        traj_buffer.append(np.zeros((3,), dtype=np.float32))
    
    print("系統就緒! 按 'q' 離開")
    
    frame_count = 0 
    
    while cap.isOpened():
        start_time = time.time()
        ret, frame = cap.read()
        if not ret: break
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        
        coords, center = process_landmarks(results)
        
        shape_buffer.append(coords)
        traj_buffer.append(center)
        
        input_shape = np.array(shape_buffer, dtype=np.float32)[np.newaxis, ...]
        input_traj = np.array(traj_buffer, dtype=np.float32)[np.newaxis, ...]
        
        # --- TensorRT 推論 ---
        logits = trt_model.infer(input_shape, input_traj)
        
        class_id = np.argmax(logits)
        prob = logits[class_id] # 取得信心度
        
        # 轉換成文字
        label_text = get_label_text(class_id)
        
        # --- 【關鍵】上傳結果到網頁 ---
        # 為了避免太頻繁傳送導致網路延遲，可以設定每 N 幀傳一次，或像我在 web_server 裡寫的「有變動才傳」
        # 這裡直接呼叫，web_server 內部有防呆機制
        web_server.update_inference_result(label_text)
        
        # --- 顯示畫面 ---
        fps = 1.0 / (time.time() - start_time)
        
        if results.multi_hand_landmarks:
            for hand_lms in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)
        
        cv2.putText(frame, f"Sign: {label_text}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        cv2.imshow("Jetson Nano Sign Language", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
