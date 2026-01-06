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
        np.copyto(self.inputs[0]['host'], input_shape_data.ravel())
        np.copyto(self.inputs[1]['host'], input_traj_data.ravel())

        for inp in self.inputs:
            cuda.memcpy_htod_async(inp['device'], inp['host'], self.stream)

        self.context.execute_async_v2(bindings=self.allocations, stream_handle=self.stream.handle)

        for out in self.outputs:
            cuda.memcpy_dtoh_async(out['host'], out['device'], self.stream)

        self.stream.synchronize()
        return self.outputs[0]['host']

# ================= 2. 輔助函式 (新增 Softmax) =================

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
    label = gloss_dict.get(class_id)
    if label:
        return f"{class_id}: {label}"
    else:
        return f"Unknown ID: {class_id}"

# [新增] Softmax 函式：將 Logits 轉為機率
def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x)) # 減去最大值是為了數值穩定性，防止溢位
    return e_x / e_x.sum()

# ================= 3. 主程式邏輯 =================

def main():
    # --- 設定 ---
    ENGINE_PATH = "model.engine"
    WINDOW_SIZE = 16

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

    shape_buffer = deque(maxlen=WINDOW_SIZE)
    traj_buffer = deque(maxlen=WINDOW_SIZE)

    print("初始化緩衝區...")
    for _ in range(WINDOW_SIZE):
        shape_buffer.append(np.zeros((21, 3), dtype=np.float32))
        traj_buffer.append(np.zeros((3,), dtype=np.float32))

    print("系統就緒! 按 'q' 離開")

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
        
        # [修改] 套用 Softmax 取得機率分布
        probs = softmax(logits)

        # [修改] 取得 Top 5 的索引 (由大到小排序)
        # argsort 會回傳索引，[::-1] 反轉變成由大到小，[:5] 取前五個
        top5_indices = np.argsort(probs)[::-1][:5]

        # 準備要傳給 Web 的字串列表
        web_display_list = []

        # --- 顯示畫面與資料處理 ---
        fps = 1.0 / (time.time() - start_time)

        if results.multi_hand_landmarks:
            for hand_lms in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)

        # [修改] 迴圈印出前五名
        base_y = 50 # 第一行的 Y 座標
        for i, idx in enumerate(top5_indices):
            score = probs[idx]
            label_text = get_label_text(idx)
            
            # 格式化顯示文字 e.g., "1. 105: Hello (85.2%)"
            display_text = f"{i+1}. {label_text} ({score:.1%})"
            
            # 加入 Web 顯示列表
            web_display_list.append(display_text)

            # 根據排名改變顏色 (第一名綠色，其他黃色)
            color = (0, 255, 0) if i == 0 else (0, 255, 255)
            font_scale = 0.8 if i == 0 else 0.6
            thickness = 2 if i == 0 else 1
            
            cv2.putText(frame, display_text, (10, base_y + i * 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        # --- 上傳結果到網頁 ---
        # 將 Top 5 用 HTML 的換行符號 <br> 連接，或者用逗號連接，視你的網頁端如何顯示
        # 這裡示範用 " | " 連接成一行字串
        web_message = " <br> ".join(web_display_list)
        web_server.update_inference_result(web_message)

        cv2.imshow("Jetson Nano Sign Language", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
