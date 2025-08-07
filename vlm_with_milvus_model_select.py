#!/usr/bin/env python3
"""
VLM + Milvus 向量資料庫整合版本（可選模型）
- 支援 llava-phi3:latest 或 gemma3:1b 於 Ollama
- 影像直接儲存在 Milvus
- GUI 可選模型
"""
import cv2
import numpy as np
from PIL import Image, ImageTk
import requests
import base64
import io
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
from datetime import datetime
import threading
import time
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
from sentence_transformers import SentenceTransformer
import uuid

class VLMWithMilvusModelSelect:
    def __init__(self):
        self.cap = None
        self.current_frame = None
        self.captured_frame = None
        self.is_running = False
        self.ollama_url = "http://localhost:11434"
        self.live_video_label = None
        self.captured_image_label = None
        self.update_video_flag = True
        self.is_analyzing = False
        self.current_model = "llava-phi3:latest"
        self.available_models = ["llava-phi3:latest", "gemma3:1b"]
        self.setup_gui()
        self.init_vector_db()

    def init_vector_db(self):
        try:
            connections.connect(alias="default", host='localhost', port='19530')
            self.encoder = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            self.collection_name = "vlm_conversations_with_images"
            if utility.has_collection(self.collection_name):
                self.collection = Collection(self.collection_name)
            else:
                fields = [
                    FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=100, is_primary=True),
                    FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=384),
                    FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=5000),
                    FieldSchema(name="question", dtype=DataType.VARCHAR, max_length=1000),
                    FieldSchema(name="answer", dtype=DataType.VARCHAR, max_length=3000),
                    FieldSchema(name="image_base64", dtype=DataType.VARCHAR, max_length=65000),
                    FieldSchema(name="image_size", dtype=DataType.VARCHAR, max_length=50),
                    FieldSchema(name="image_format", dtype=DataType.VARCHAR, max_length=20),
                    FieldSchema(name="timestamp", dtype=DataType.VARCHAR, max_length=50),
                    FieldSchema(name="date", dtype=DataType.VARCHAR, max_length=20),
                    FieldSchema(name="time", dtype=DataType.VARCHAR, max_length=20),
                    FieldSchema(name="model", dtype=DataType.VARCHAR, max_length=100),
                    FieldSchema(name="temperature", dtype=DataType.DOUBLE),
                    FieldSchema(name="conversation_type", dtype=DataType.VARCHAR, max_length=50)
                ]
                schema = CollectionSchema(fields=fields, description="VLM conversations with embedded images")
                self.collection = Collection(name=self.collection_name, schema=schema)
            try:
                index_params = {"metric_type": "COSINE", "index_type": "IVF_FLAT", "params": {"nlist": 128}}
                self.collection.create_index(field_name="vector", index_params=index_params)
            except Exception as e:
                pass
            self.collection.load()
            self.vector_db_status = "✅ Milvus 向量資料庫已連接"
        except Exception as e:
            self.vector_db_status = f"❌ Milvus 向量資料庫連接失敗: {e}"
            self.collection = None

    def image_to_base64(self, frame, quality=85, max_size=800):
        try:
            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if max(pil_image.size) > max_size:
                ratio = max_size / max(pil_image.size)
                new_size = tuple(int(dim * ratio) for dim in pil_image.size)
                pil_image = pil_image.resize(new_size, Image.Resampling.LANCZOS)
            img_base64 = None
            current_quality = quality
            while current_quality >= 20:
                buffer = io.BytesIO()
                pil_image.save(buffer, format='JPEG', quality=current_quality)
                current_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
                if len(current_base64) <= 65000:
                    img_base64 = current_base64
                    break
                current_quality -= 10
            if img_base64 is None:
                return None, None
            return img_base64, pil_image.size
        except Exception:
            return None, None

    def base64_to_image(self, base64_str):
        try:
            image_data = base64.b64decode(base64_str)
            pil_image = Image.open(io.BytesIO(image_data))
            return pil_image
        except Exception:
            return None

    def save_conversation_to_vector_db(self, question, answer, frame_base64, image_size):
        if not self.collection:
            return
        try:
            timestamp = datetime.now()
            conversation_text = f"問題: {question}\n回答: {answer}"
            vector = self.encoder.encode([conversation_text])[0]
            data = [{
                "id": str(uuid.uuid4()),
                "vector": vector.tolist(),
                "content": conversation_text,
                "question": question,
                "answer": answer,
                "image_base64": frame_base64,
                "image_size": f"{image_size[0]}x{image_size[1]}" if image_size else "unknown",
                "image_format": "JPEG",
                "timestamp": timestamp.isoformat(),
                "date": timestamp.strftime("%Y-%m-%d"),
                "time": timestamp.strftime("%H:%M:%S"),
                "model": self.current_model,
                "temperature": self.temperature_var.get(),
                "conversation_type": "vlm_qa"
            }]
            self.collection.insert(data)
            self.collection.flush()
            self.log_message(f"💾 對話和影像已儲存到 Milvus 向量資料庫")
        except Exception as e:
            self.log_message(f"⚠️ Milvus 向量資料庫儲存失敗: {e}")

    def setup_gui(self):
        self.root = tk.Tk()
        self.root.title("VLM 系統 + Milvus 向量資料庫 (模型可選)")
        self.root.geometry("1200x800")
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        image_container = ttk.Frame(main_frame)
        image_container.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        live_frame = ttk.LabelFrame(image_container, text="📹 即時影像")
        live_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 5))
        self.live_video_label = ttk.Label(live_frame, text="即時相機畫面\n點擊「開始相機」啟動")
        self.live_video_label.pack(expand=True, padx=10, pady=10)
        captured_frame = ttk.LabelFrame(image_container, text="📸 正在分析的照片")
        captured_frame.pack(fill=tk.BOTH, expand=True, pady=(5, 0))
        self.captured_image_label = ttk.Label(captured_frame, text="尚未拍攝照片\n點擊「拍照分析」開始")
        self.captured_image_label.pack(expand=True, padx=10, pady=10)
        right_frame = ttk.Frame(main_frame, width=500)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(10, 0))
        right_frame.pack_propagate(False)
        status_frame = ttk.LabelFrame(right_frame, text="系統狀態")
        status_frame.pack(fill=tk.X, pady=(0, 10))
        self.ollama_status = ttk.Label(status_frame, text="Ollama 狀態: 檢查中...")
        self.ollama_status.pack(anchor=tk.W, padx=5, pady=3)
        self.vector_db_label = ttk.Label(status_frame, text=getattr(self, 'vector_db_status', '檢查中...'))
        self.vector_db_label.pack(anchor=tk.W, padx=5, pady=3)
        camera_frame = ttk.LabelFrame(right_frame, text="相機控制")
        camera_frame.pack(fill=tk.X, pady=(0, 10))
        camera_btn_frame = ttk.Frame(camera_frame)
        camera_btn_frame.pack(fill=tk.X, padx=5, pady=5)
        self.start_btn = ttk.Button(camera_btn_frame, text="開始相機", command=self.start_camera)
        self.start_btn.pack(side=tk.LEFT, padx=(0, 5))
        self.stop_btn = ttk.Button(camera_btn_frame, text="停止相機", command=self.stop_camera, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=(0, 5))
        self.capture_btn = ttk.Button(camera_btn_frame, text="拍照分析", command=self.capture_and_ask, state=tk.DISABLED)
        self.capture_btn.pack(side=tk.RIGHT)
        options_frame = ttk.LabelFrame(right_frame, text="模型參數")
        options_frame.pack(fill=tk.X, pady=(0, 10))

        # 第一行：模型選擇
        model_row = ttk.Frame(options_frame)
        model_row.pack(fill=tk.X, pady=(0, 2))
        model_label = ttk.Label(model_row, text="選擇模型:")
        model_label.pack(side=tk.LEFT, padx=(5,0))
        self.model_var = tk.StringVar()
        self.model_combobox = ttk.Combobox(model_row, textvariable=self.model_var, state="readonly", width=22)
        self.model_combobox.pack(side=tk.LEFT, padx=(5, 10))
        self.model_combobox['values'] = self.available_models
        self.model_var.set(self.current_model)
        self.model_combobox.bind("<<ComboboxSelected>>", self.on_model_selected)

        # 第二行：溫度調整
        temp_row = ttk.Frame(options_frame)
        temp_row.pack(fill=tk.X, pady=(0, 2))
        temp_label = ttk.Label(temp_row, text="Temperature (0.0-1.0):")
        temp_label.pack(side=tk.LEFT, padx=(5,0))
        self.temperature_var = tk.DoubleVar(value=0.2)
        self.temperature_scale = ttk.Scale(temp_row, from_=0.0, to=1.0, orient=tk.HORIZONTAL, variable=self.temperature_var, command=self.update_temp_label, length=120)
        self.temperature_scale.pack(side=tk.LEFT, padx=(5,0))
        self.temp_value_label = ttk.Label(temp_row, text=f"{self.temperature_var.get():.2f}", width=4)
        self.temp_value_label.pack(side=tk.LEFT, padx=(5,5))

        # 問題輸入區塊（高度調大，字體加大）
        question_frame = ttk.LabelFrame(right_frame, text="問題輸入")
        question_frame.pack(fill=tk.X, pady=(0, 10))
        self.question_entry = tk.Text(question_frame, height=4, wrap=tk.WORD, font=("Arial", 14))
        self.question_entry.pack(fill=tk.X, padx=5, pady=3)
        self.question_entry.insert(tk.END, "請描述這張圖片。")

        # Milvus 對話記錄與影像功能區塊
        vector_frame = ttk.LabelFrame(right_frame, text="📚 對話記錄與影像 (Milvus)")
        vector_frame.pack(fill=tk.X, pady=(0, 10))
        vector_btn_frame = ttk.Frame(vector_frame)
        vector_btn_frame.pack(fill=tk.X, padx=5, pady=5)
        self.search_btn = ttk.Button(vector_btn_frame, text="搜尋對話", command=self.search_conversations)
        self.search_btn.pack(side=tk.LEFT, padx=(0, 5))
        self.view_history_btn = ttk.Button(vector_btn_frame, text="查看歷史", command=self.view_conversation_history)
        self.view_history_btn.pack(side=tk.LEFT, padx=(0, 5))
        self.view_image_btn = ttk.Button(vector_btn_frame, text="檢視影像", command=self.view_image_from_history)
        self.view_image_btn.pack(side=tk.LEFT)
        search_frame = ttk.Frame(vector_frame)
        search_frame.pack(fill=tk.X, padx=5, pady=(0, 5))
        ttk.Label(search_frame, text="搜尋:").pack(side=tk.LEFT)
        self.search_entry = ttk.Entry(search_frame)
        self.search_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 0))
        self.search_entry.bind('<Return>', lambda e: self.search_conversations())
        result_frame = ttk.LabelFrame(right_frame, text="AI 回答")
        result_frame.pack(fill=tk.BOTH, expand=True)
        self.result_text = scrolledtext.ScrolledText(result_frame, height=12, wrap=tk.WORD)
        self.result_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.status_label = ttk.Label(self.root, text="準備就緒")
        self.status_label.pack(fill=tk.X, side=tk.BOTTOM, padx=10, pady=5)
        self.root.after(100, self.check_ollama_connection)

    def on_model_selected(self, event=None):
        self.current_model = self.model_var.get()
        self.log_message(f"已切換模型: {self.current_model}")

    def update_temp_label(self, value):
        self.temp_value_label.config(text=f"{float(value):.2f}")

    def check_ollama_connection(self):
        def check_in_background():
            try:
                response = requests.get(f"{self.ollama_url}/api/tags", timeout=3)
                if response.status_code == 200:
                    models = response.json().get("models", [])
                    model_names = [model.get("name", "") for model in models]
                    # 不篩選，全部顯示
                    filtered = model_names
                    if filtered:
                        self.available_models = filtered
                        self.root.after(0, lambda: self.model_combobox.config(values=filtered))
                        if self.current_model not in filtered:
                            self.current_model = filtered[0]
                            self.root.after(0, lambda: self.model_var.set(filtered[0]))
                        self.root.after(0, lambda: self.ollama_status.config(text=f"✅ Ollama 可用: {', '.join(filtered)}", foreground="green"))
                        self.root.after(0, lambda: self.capture_btn.config(state=tk.NORMAL))
                    else:
                        self.root.after(0, lambda: self.ollama_status.config(text="⚠️ 沒有可用模型，請先下載", foreground="orange"))
                else:
                    raise Exception("服務不可用")
            except Exception:
                self.root.after(0, lambda: self.ollama_status.config(text="❌ Ollama 未運行", foreground="red"))
        threading.Thread(target=check_in_background, daemon=True).start()

    def start_camera(self):
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                raise Exception("無法開啟相機")
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.is_running = True
            self.update_video_flag = True
            self.video_thread = threading.Thread(target=self.update_live_video)
            self.video_thread.daemon = True
            self.video_thread.start()
            self.start_btn.config(state=tk.DISABLED)
            self.stop_btn.config(state=tk.NORMAL)
            self.capture_btn.config(state=tk.NORMAL)
            self.status_label.config(text="相機已啟動")
            self.log_message("相機啟動成功")
        except Exception as e:
            messagebox.showerror("錯誤", f"相機啟動失敗: {str(e)}")

    def stop_camera(self):
        self.is_running = False
        self.update_video_flag = False
        if self.cap:
            self.cap.release()
        self.live_video_label.config(image="", text="相機已停止")
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.capture_btn.config(state=tk.DISABLED)
        self.status_label.config(text="相機已停止")
        self.log_message("相機已停止")

    def update_live_video(self):
        while self.is_running and self.update_video_flag:
            try:
                ret, frame = self.cap.read()
                if ret:
                    self.current_frame = frame.copy()
                    display_frame = cv2.resize(frame, (400, 300))
                    display_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(display_frame)
                    photo = ImageTk.PhotoImage(pil_image)
                    self.root.after(0, self._update_live_video_label, photo)
                    time.sleep(0.033)
            except Exception:
                break

    def _update_live_video_label(self, photo):
        if self.live_video_label and self.update_video_flag:
            self.live_video_label.config(image=photo, text="")
            self.live_video_label.image = photo

    def capture_and_ask(self):
        if self.current_frame is None:
            messagebox.showwarning("警告", "沒有可用的影像")
            return
        question = self.question_entry.get(1.0, tk.END).strip()
        if not question:
            question = "請描述這張圖片。"
        self.captured_frame = self.current_frame.copy()
        self.display_captured_image()
        self.is_analyzing = True
        self.status_label.config(text="AI 分析中...")
        self.log_message(f"問題: {question}")
        threading.Thread(target=self.process_with_ollama, args=(self.captured_frame.copy(), question), daemon=True).start()

    def display_captured_image(self):
        if self.captured_frame is not None:
            display_frame = cv2.resize(self.captured_frame, (400, 300))
            display_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(display_frame)
            photo = ImageTk.PhotoImage(pil_image)
            self.captured_image_label.config(image=photo, text="")
            self.captured_image_label.image = photo

    def process_with_ollama(self, frame, question):
        try:
            frame_base64, image_size = self.image_to_base64(frame, quality=85, max_size=800)
            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if max(pil_image.size) > 512:
                ratio = 512 / max(pil_image.size)
                new_size = tuple(int(dim * ratio) for dim in pil_image.size)
                pil_image = pil_image.resize(new_size, Image.Resampling.LANCZOS)
            buffer = io.BytesIO()
            pil_image.save(buffer, format='PNG')
            img_base64_api = base64.b64encode(buffer.getvalue()).decode()
            temperature = self.temperature_var.get()
            payload = {
                "model": self.current_model,
                "prompt": question,
                "images": [img_base64_api],
                "stream": False,
                "options": {"temperature": temperature}
            }
            self.log_message(f"正在呼叫 Ollama API (Temperature: {temperature:.2f}, Model: {self.current_model})...")
            response = requests.post(f"{self.ollama_url}/api/generate", json=payload, timeout=120)
            if response.status_code == 200:
                result = response.json()
                answer = result.get("response", "無法獲得回答")
                if answer and answer.strip():
                    self.log_message(f"AI 回答: {answer}")
                    if frame_base64:
                        self.save_conversation_to_vector_db(
                            question=question,
                            answer=answer,
                            frame_base64=frame_base64,
                            image_size=image_size
                        )
                else:
                    self.log_message("⚠️ AI 回答為空，請重試")
            else:
                self.log_message(f"API 錯誤，狀態碼: {response.status_code}")
                self.log_message(f"錯誤內容: {response.text}")
            self.is_analyzing = False
            self.root.after(0, lambda: self.status_label.config(text="分析完成"))
        except requests.exceptions.Timeout:
            self.log_message("⏰ 請求超時 - LLaVA-Phi3 首次推理需要較長時間")
            self.is_analyzing = False
            self.root.after(0, lambda: self.status_label.config(text="首次推理超時，請重試"))
        except Exception as e:
            self.log_message(f"處理錯誤: {str(e)}")
            self.is_analyzing = False
            self.root.after(0, lambda: self.status_label.config(text="處理失敗"))

    def log_message(self, message):
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_message = f"[{timestamp}] {message}\n"
        self.root.after(0, self._update_text, formatted_message)

    def _update_text(self, message):
        self.result_text.insert(tk.END, message)
        self.result_text.see(tk.END)

    def run(self):
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.mainloop()

    def on_closing(self):
        self.stop_camera()
        try:
            connections.disconnect("default")
        except:
            pass
        self.root.destroy()

    def search_conversations(self):
        """搜尋歷史對話"""
        if not self.collection:
            messagebox.showwarning("警告", "Milvus 向量資料庫未連接")
            return
        query = self.search_entry.get().strip()
        if not query:
            messagebox.showinfo("提示", "請輸入搜尋關鍵字")
            return
        try:
            query_vector = self.encoder.encode([query])[0]
            search_params = {
                "metric_type": "COSINE",
                "params": {"nprobe": 10}
            }
            results = self.collection.search(
                data=[query_vector.tolist()],
                anns_field="vector",
                param=search_params,
                limit=5,
                output_fields=[
                    "content", "question", "answer", "date", "time",
                    "model", "image_size", "timestamp"
                ]
            )
            if results and results[0]:
                search_results = f"\n🔍 搜尋結果: '{query}'\n" + "="*50 + "\n"
                for i, result in enumerate(results[0]):
                    entity = result.entity
                    search_results += f"\n結果 {i+1} (相似度: {result.distance:.3f}):\n"
                    search_results += f"時間: {entity.get('date', '')} {entity.get('time', '')}\n"
                    search_results += f"問題: {entity.get('question', '')}\n"
                    search_results += f"回答: {entity.get('answer', '')[:200]}...\n"
                    search_results += f"模型: {entity.get('model', '')}\n"
                    search_results += f"影像大小: {entity.get('image_size', 'unknown')}\n"
                    search_results += "-" * 30 + "\n"
                self.log_message(search_results)
            else:
                self.log_message(f"🔍 沒有找到與 '{query}' 相關的對話")
        except Exception as e:
            messagebox.showerror("錯誤", f"搜尋失敗: {e}")

    def view_conversation_history(self):
        """查看最近的對話歷史"""
        if not self.collection:
            messagebox.showwarning("警告", "Milvus 向量資料庫未連接")
            return
        try:
            total_conversations = self.collection.num_entities
            if total_conversations == 0:
                self.root.after(0, self.log_message, "📚 目前沒有對話記錄")
                return
            limit = min(total_conversations, 1000)
            conversations = self.collection.query(
                expr="",
                output_fields=[
                    "id", "content", "question", "answer", "date", "time",
                    "model", "image_size", "timestamp"
                ],
                limit=limit
            )
            if conversations:
                history_text = f"\n📚 最近的對話歷史 (共 {total_conversations} 個對話)\n" + "="*60 + "\n"
                conversations.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
                for i, conversation in enumerate(conversations[:10]):
                    history_text += f"\n對話 {i+1}:\n"
                    history_text += f"時間: {conversation.get('date', '')} {conversation.get('time', '')}\n"
                    history_text += f"問題: {conversation.get('question', '')}\n"
                    history_text += f"回答: {conversation.get('answer', '')[:150]}...\n"
                    history_text += f"模型: {conversation.get('model', '')}\n"
                    history_text += f"影像: {conversation.get('image_size', '')} JPEG\n"
                    history_text += f"ID: {conversation.get('id', '')}\n"
                    history_text += "-" * 40 + "\n"
                self.log_message(history_text)
            else:
                self.log_message("📚 目前沒有對話記錄")
        except Exception as e:
            messagebox.showerror("錯誤", f"查看歷史失敗: {e}")

    def view_image_from_history(self):
        """從歷史記錄中檢視影像"""
        if not self.collection:
            messagebox.showwarning("警告", "Milvus 向量資料庫未連接")
            return
        dialog = tk.Toplevel(self.root)
        dialog.title("檢視歷史影像")
        dialog.geometry("400x200")
        ttk.Label(dialog, text="請輸入要檢視的對話編號 (1-10):").pack(pady=10)
        entry = ttk.Entry(dialog, width=10)
        entry.pack(pady=5)
        def on_submit():
            try:
                index = int(entry.get()) - 1
                if index < 0:
                    raise ValueError("編號必須大於0")
                self._fetch_and_display_image(index)
            except ValueError:
                messagebox.showerror("錯誤", "請輸入有效的數字")
            except Exception as e:
                messagebox.showerror("錯誤", f"檢視影像失敗: {e}")
        ttk.Button(dialog, text="檢視影像", command=on_submit).pack(pady=10)
        ttk.Button(dialog, text="取消", command=dialog.destroy).pack()

    def _fetch_and_display_image(self, index):
        try:
            num_entities = self.collection.num_entities
            if num_entities == 0:
                self.root.after(0, lambda: messagebox.showerror("錯誤", "資料庫中沒有對話記錄"))
                return
            limit = min(num_entities, 1000)
            conversations = self.collection.query(
                expr="",
                output_fields=[
                    "id", "question", "answer", "date", "time",
                    "image_base64", "timestamp"
                ],
                limit=limit
            )
            if conversations and index < len(conversations):
                conversations.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
                selected_conversation = conversations[index]
                self.root.after(0, self._show_image_window, selected_conversation, index)
            else:
                self.root.after(0, lambda: messagebox.showerror("錯誤", "對話編號不存在"))
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("錯誤", f"檢視影像失敗: {e}"))

    def _show_image_window(self, conversation_data, index):
        image_base64 = conversation_data.get('image_base64')
        if image_base64:
            pil_image = self.base64_to_image(image_base64)
            if pil_image:
                img_window = tk.Toplevel(self.root)
                img_window.title(f"歷史影像 - 對話 {index + 1}")
                display_size = (600, 450)
                pil_image.thumbnail(display_size, Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(pil_image)
                img_label = ttk.Label(img_window, image=photo)
                img_label.pack(padx=10, pady=10)
                img_label.image = photo
                info_text = f"""時間: {conversation_data.get('date')} {conversation_data.get('time')}
問題: {conversation_data.get('question', '')}
回答: {conversation_data.get('answer', '')[:100]}..."""
                info_label = ttk.Label(img_window, text=info_text, wraplength=580)
                info_label.pack(padx=10, pady=5)
            else:
                messagebox.showerror("錯誤", "無法載入影像")
        else:
            messagebox.showwarning("警告", "該對話沒有影像資料")

def main():
    print("VLM 系統 + Milvus 向量資料庫 (模型可選)")
    print("=" * 50)
    print("✨ 支援 LLaVA-Phi3:latest 與 Gemma3:1b")
    print("🖼️ 影像直接儲存在 Milvus 向量資料庫中")
    print("🔍 支援對話搜尋和影像檢視")
    print("🌏 支援中英文問答")
    print("\n📦 需要的服務:")
    print("1. ollama pull llava-phi3:latest")
    print("2. ollama pull gemma3:1b")
    print("3. docker run -p 19530:19530 -p 9091:9091 milvusdb/milvus:latest")
    print("4. docker run -p 8000:3000 -e MILVUS_URL=localhost:19530 zilliz/attu:latest")
    print("=" * 50)
    try:
        app = VLMWithMilvusModelSelect()
        app.run()
    except KeyboardInterrupt:
        print("\n程式已停止")

if __name__ == "__main__":
    main() 