# web_server.py
# -*- coding: utf-8 -*-
import threading
from flask import Flask, render_template, request
from flask_socketio import SocketIO
from flask_cors import CORS

class SignLanguageWebServer:
    def __init__(self):
        # 1. 初始化 Flask
        self.app = Flask(__name__)
        self.app.config['SECRET_KEY'] = 'secret!'
        CORS(self.app)

        # 2. 初始化 SocketIO
        # async_mode='threading' 對於整合 Jetson Nano 上的 OpenCV 迴圈至關重要
        self.socketio = SocketIO(self.app,
                                 cors_allowed_origins='*',
                                 async_mode='threading',
                                 ping_timeout=60,
                                 ping_interval=25,
                                 allow_upgrades=False)
        
        # 3. 註冊路由與事件
        self.register_routes()
        
        # 狀態變數
        self.current_sign = "等待辨識..."
        self.last_sent_sign = None

    def register_routes(self):
        # 設定網頁首頁
        @self.app.route('/')
        def index():
            # 確保您目錄下有 templates/index.html
            return render_template('index.html')

        # 處理連線事件
        @self.socketio.on('connect')
        def handle_connect():
            # 當網頁連上時，先傳送目前的狀態
            self.socketio.emit('update_sign', {'data': self.current_sign})
            print(f"[Web] 客戶端已連線 (SID: {request.sid})")

    def start(self, host='0.0.0.0', port=5000):
        """
        啟動伺服器 (會開啟一個獨立的 Thread)
        """
        print(f"[Web] 伺服器啟動中: http://{host}:{port}")
        
        # 將 socketio.run 包裝在 Thread 中執行，避免卡住主程式
        server_thread = threading.Thread(target=self._run_server, args=(host, port))
        server_thread.daemon = True  # 設定為守護執行緒，主程式關閉時此執行緒也會關閉
        server_thread.start()

    def _run_server(self, host, port):
        # 實際啟動 SocketIO 伺服器
        self.socketio.run(self.app, host=host, port=port, debug=False)

    def update_inference_result(self, sign_text):
        """
        給 app.py 呼叫的介面：更新辨識結果並推播
        """
        # 只有當結果改變時，才推送更新 (節省頻寬)
        if sign_text != self.current_sign:
            self.current_sign = sign_text
            try:
                self.socketio.emit('update_sign', {'data': self.current_sign})
                # print(f"[Web] 推播更新: {sign_text}") 
            except Exception as e:
                print(f"[Web] 推播失敗: {e}")
