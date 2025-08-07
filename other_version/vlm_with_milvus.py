#!/usr/bin/env python3
"""
VLM + Milvus 向量資料庫整合版本
影像直接儲存在 Milvus 向量資料庫中，而不是儲存在電腦上
Collection = vlm_conversations_with_images
"""

import cv2
import numpy as np
from PIL import Image, ImageTk
import requests
import base64
import io
import json
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
from datetime import datetime
import threading
import time
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
from sentence_transformers import SentenceTransformer
import uuid
import os

class VLMWithMilvus:
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
        self.current_model = None
        
        self.setup_gui()
        self.init_vector_db()
    
    def init_vector_db(self):
        """初始化 Milvus 向量資料庫"""
        try:
            print("🔗 連接 Milvus 向量資料庫...")
            
            # 連接到 Milvus
            connections.connect(
                alias="default",
                host='localhost',
                port='19530'
            )
            
            # 初始化句子轉換器
            self.encoder = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            
            # 定義集合名稱
            self.collection_name = "vlm_conversations_with_images"
            
            # 檢查集合是否存在，不存在則建立
            if utility.has_collection(self.collection_name):
                print(f"ℹ️ 集合已存在: {self.collection_name}")
                self.collection = Collection(self.collection_name)
            else:
                # 定義字段結構
                fields = [
                    FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=100, is_primary=True),
                    FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=384),
                    FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=5000),
                    FieldSchema(name="question", dtype=DataType.VARCHAR, max_length=1000),
                    FieldSchema(name="answer", dtype=DataType.VARCHAR, max_length=3000),
                    FieldSchema(name="image_base64", dtype=DataType.VARCHAR, max_length=65000),  # 儲存 base64 影像
                    FieldSchema(name="image_size", dtype=DataType.VARCHAR, max_length=50),
                    FieldSchema(name="image_format", dtype=DataType.VARCHAR, max_length=20),
                    FieldSchema(name="timestamp", dtype=DataType.VARCHAR, max_length=50),
                    FieldSchema(name="date", dtype=DataType.VARCHAR, max_length=20),
                    FieldSchema(name="time", dtype=DataType.VARCHAR, max_length=20),
                    FieldSchema(name="model", dtype=DataType.VARCHAR, max_length=100),
                    FieldSchema(name="temperature", dtype=DataType.DOUBLE),
                    FieldSchema(name="conversation_type", dtype=DataType.VARCHAR, max_length=50)
                ]
                
                # 創建集合 schema
                schema = CollectionSchema(
                    fields=fields,
                    description="VLM conversations with embedded images"
                )
                
                # 創建集合
                self.collection = Collection(
                    name=self.collection_name,
                    schema=schema
                )
                
                print(f"✅ 建立集合: {self.collection_name}")
            
            # 創建索引（如果不存在）
            try:
                index_params = {
                    "metric_type": "COSINE",
                    "index_type": "IVF_FLAT",
                    "params": {"nlist": 128}
                }
                self.collection.create_index(field_name="vector", index_params=index_params)
                print("✅ 向量索引已建立")
            except Exception as e:
                if "already exist" in str(e):
                    print("ℹ️ 向量索引已存在")
                else:
                    print(f"⚠️ 建立索引時發生警告: {e}")
            
            # 載入集合到記憶體
            self.collection.load()
            
            self.vector_db_status = "✅ Milvus 向量資料庫已連接"
            print("✅ Milvus 向量資料庫初始化成功")
            
        except Exception as e:
            print(f"❌ Milvus 向量資料庫初始化失敗: {e}")
            self.vector_db_status = "❌ Milvus 向量資料庫連接失敗"
            self.collection = None
    
    def image_to_base64(self, frame, quality=85, max_size=800):
        """將影像轉換為 base64 字串用於儲存，並確保長度不超過限制"""
        try:
            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            if max(pil_image.size) > max_size:
                ratio = max_size / max(pil_image.size)
                new_size = tuple(int(dim * ratio) for dim in pil_image.size)
                pil_image = pil_image.resize(new_size, Image.Resampling.LANCZOS)

            # 反覆降低品質直到 base64 長度符合要求
            img_base64 = None
            current_quality = quality
            while current_quality >= 20: # 設定最低品質底線
                buffer = io.BytesIO()
                pil_image.save(buffer, format='JPEG', quality=current_quality)
                current_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

                if len(current_base64) <= 65000:
                    img_base64 = current_base64
                    print(f"ℹ️ 影像已壓縮至品質 {current_quality}，大小符合要求。")
                    break
                
                # 如果還是太大，降低品質再試一次
                current_quality -= 10

            if img_base64 is None:
                print("❌ 影像轉換失敗：即使降低品質，影像仍然太大")
                return None, None

            return img_base64, pil_image.size
            
        except Exception as e:
            print(f"❌ 影像轉換失敗: {e}")
            return None, None
    
    def base64_to_image(self, base64_str):
        """將 base64 字串轉換回影像"""
        try:
            # 解碼 base64
            image_data = base64.b64decode(base64_str)
            
            # 轉換為 PIL 影像
            pil_image = Image.open(io.BytesIO(image_data))
            
            return pil_image
            
        except Exception as e:
            print(f"❌ base64 轉影像失敗: {e}")
            return None
    
    def save_conversation_to_vector_db(self, question, answer, frame_base64, image_size, metadata=None):
        """將對話和影像儲存到 Milvus 向量資料庫"""
        if not self.collection:
            print("⚠️ Milvus 向量資料庫未連接，跳過儲存")
            return
        
        try:
            timestamp = datetime.now()
            
            # 建立包含完整對話的文本
            conversation_text = f"問題: {question}\n回答: {answer}"
            
            # 將對話文本向量化
            vector = self.encoder.encode([conversation_text])[0]
            
            # 準備要插入的數據
            data = [{
                "id": str(uuid.uuid4()),
                "vector": vector.tolist(),
                "content": conversation_text,
                "question": question,
                "answer": answer,
                "image_base64": frame_base64,  # 🔥 關鍵：影像直接儲存為 base64
                "image_size": f"{image_size[0]}x{image_size[1]}" if image_size else "unknown",
                "image_format": "JPEG",
                "timestamp": timestamp.isoformat(),
                "date": timestamp.strftime("%Y-%m-%d"),
                "time": timestamp.strftime("%H:%M:%S"),
                "model": self.current_model or "unknown",
                "temperature": self.temperature_var.get(),
                "conversation_type": "vlm_qa"
            }]
            
            # 插入數據到 Milvus
            insert_result = self.collection.insert(data)
            self.collection.flush()  # 確保數據寫入
            
            self.log_message(f"💾 對話和影像已儲存到 Milvus 向量資料庫")
            
            # 計算儲存大小
            image_size_kb = len(frame_base64) * 3 / 4 / 1024  # base64 大小估算
            print(f"✅ 對話儲存成功，影像大小: {image_size_kb:.1f} KB")
            
        except Exception as e:
            print(f"❌ 儲存對話失敗: {e}")
            self.log_message(f"⚠️ Milvus 向量資料庫儲存失敗: {e}")
    
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
            # 將搜尋查詢向量化
            query_vector = self.encoder.encode([query])[0]
            
            # 搜尋參數
            search_params = {
                "metric_type": "COSINE",
                "params": {"nprobe": 10}
            }
            
            # 在 Milvus 中搜尋
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

            # 為了效能與安全，最多獲取1000筆來排序
            limit = min(total_conversations, 1000)

            # 使用 query 直接獲取數據
            conversations = self.collection.query(
                expr="",  # 空表達式查詢所有數據
                output_fields=[
                    "id", "content", "question", "answer", "date", "time", 
                    "model", "image_size", "timestamp"
                ],
                limit=limit # Milvus 要求空 expr 必須有 limit
            )
            
            if conversations:
                history_text = f"\n📚 最近的對話歷史 (共 {total_conversations} 個對話)\n" + "="*60 + "\n"
                
                # 按時間排序（最新的在前）
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
        
        # 簡單的對話ID輸入視窗
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
                
                # 在背景線程獲取數據，然後在主線程顯示影像
                self._fetch_and_display_image(index)
            except ValueError:
                messagebox.showerror("錯誤", "請輸入有效的數字")
            except Exception as e:
                messagebox.showerror("錯誤", f"檢視影像失敗: {e}")
        
        ttk.Button(dialog, text="檢視影像", command=on_submit).pack(pady=10)
        ttk.Button(dialog, text="取消", command=dialog.destroy).pack()

    def _fetch_and_display_image(self, index):
        """在背景線程獲取數據，然後在主線程顯示影像"""
        try:
            num_entities = self.collection.num_entities
            if num_entities == 0:
                self.root.after(0, lambda: messagebox.showerror("錯誤", "資料庫中沒有對話記錄"))
                return

            # 為了效能與安全，最多獲取1000筆來排序
            limit = min(num_entities, 1000)

            # 使用 query 直接獲取所有數據
            conversations = self.collection.query(
                expr="",
                output_fields=[
                    "id", "question", "answer", "date", "time",
                    "image_base64", "timestamp"
                ],
                limit=limit  # Milvus 要求空 expr 必須有 limit
            )

            if conversations and index < len(conversations):
                # 按時間排序
                conversations.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
                selected_conversation = conversations[index]

                # 排程到主線程顯示視窗
                self.root.after(0, self._show_image_window, selected_conversation, index)
            else:
                self.root.after(0, lambda: messagebox.showerror("錯誤", "對話編號不存在"))

        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("錯誤", f"檢視影像失敗: {e}"))

    def _show_image_window(self, conversation_data, index):
        """(主線程執行) 創建並顯示影像視窗"""
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
                img_label.image = photo  # 保持引用

                info_text = f"""時間: {conversation_data.get('date')} {conversation_data.get('time')}
問題: {conversation_data.get('question', '')}
回答: {conversation_data.get('answer', '')[:100]}..."""

                info_label = ttk.Label(img_window, text=info_text, wraplength=580)
                info_label.pack(padx=10, pady=5)
            else:
                messagebox.showerror("錯誤", "無法載入影像")
        else:
            messagebox.showwarning("警告", "該對話沒有影像資料")
    
    def setup_gui(self):
        """設置圖形界面"""
        self.root = tk.Tk()
        self.root.title("VLM 系統 + Milvus 向量資料庫 (影像內嵌版)")
        self.root.geometry("1400x900")
        
        # 主框架
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 左側 - 影像顯示區域
        image_container = ttk.Frame(main_frame)
        image_container.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        # 即時影像區域
        live_frame = ttk.LabelFrame(image_container, text="📹 即時影像")
        live_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 5))
        
        self.live_video_label = ttk.Label(live_frame, text="即時相機畫面\n點擊「開始相機」啟動")
        self.live_video_label.pack(expand=True, padx=10, pady=10)
        
        # 分析照片區域
        captured_frame = ttk.LabelFrame(image_container, text="📸 正在分析的照片")
        captured_frame.pack(fill=tk.BOTH, expand=True, pady=(5, 0))
        
        self.captured_image_label = ttk.Label(captured_frame, text="尚未拍攝照片\n點擊「拍照分析」開始")
        self.captured_image_label.pack(expand=True, padx=10, pady=10)
        
        # 右側 - 控制和結果區域
        right_frame = ttk.Frame(main_frame, width=500)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(10, 0))
        right_frame.pack_propagate(False)
        
        # 系統狀態
        status_frame = ttk.LabelFrame(right_frame, text="系統狀態")
        status_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.ollama_status = ttk.Label(status_frame, text="Ollama 狀態: 檢查中...")
        self.ollama_status.pack(anchor=tk.W, padx=5, pady=3)
        
        # Milvus 狀態
        self.vector_db_label = ttk.Label(status_frame, text=getattr(self, 'vector_db_status', '檢查中...'))
        self.vector_db_label.pack(anchor=tk.W, padx=5, pady=3)
        
        # 相機控制
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
        
        # 問題輸入
        question_frame = ttk.LabelFrame(right_frame, text="問題輸入")
        question_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.question_entry = tk.Text(question_frame, height=3, wrap=tk.WORD)
        self.question_entry.pack(fill=tk.X, padx=5, pady=5)
        self.question_entry.insert(tk.END, "請描述這張圖片。")
        
        # 快速問題
        quick_frame = ttk.Frame(question_frame)
        quick_frame.pack(fill=tk.X, padx=5, pady=(5, 0))
        
        quick_questions = [
            ("有什麼？", "這張圖片裡有什麼？"),
            ("詳細描述", "請詳細描述這張圖片的內容。"),
            ("What's this?", "What do you see in this image?"),
            ("顏色？", "主要有什麼顏色？"),
            ("幾個人？", "圖片中有幾個人？"),
            ("Describe", "Describe this image in detail.")
        ]
        
        for i, (short_text, full_question) in enumerate(quick_questions):
            btn = ttk.Button(quick_frame, text=short_text, 
                           command=lambda q=full_question: self.set_question(q))
            btn.grid(row=i//3, column=i%3, padx=2, pady=2, sticky="ew")
        
        for i in range(3):
            quick_frame.columnconfigure(i, weight=1)
        
        # 🔥 新增：Milvus 向量資料庫控制區域
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
        
        # 搜尋框
        search_frame = ttk.Frame(vector_frame)
        search_frame.pack(fill=tk.X, padx=5, pady=(0, 5))
        
        ttk.Label(search_frame, text="搜尋:").pack(side=tk.LEFT)
        self.search_entry = ttk.Entry(search_frame)
        self.search_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 0))
        self.search_entry.bind('<Return>', lambda e: self.search_conversations())
        
        # 結果顯示
        result_frame = ttk.LabelFrame(right_frame, text="AI 回答")
        result_frame.pack(fill=tk.BOTH, expand=True)
        
        self.result_text = scrolledtext.ScrolledText(result_frame, height=12, wrap=tk.WORD)
        self.result_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 模型參數
        options_frame = ttk.LabelFrame(right_frame, text="模型參數")
        options_frame.pack(fill=tk.X, pady=(0, 10))
        
        temp_label = ttk.Label(options_frame, text="Temperature (0.0-1.0):")
        temp_label.pack(side=tk.LEFT, padx=(5,0))
        
        self.temperature_var = tk.DoubleVar(value=0.2)
        
        self.temp_value_label = ttk.Label(options_frame, text=f"{self.temperature_var.get():.2f}", width=4)
        self.temp_value_label.pack(side=tk.RIGHT, padx=(0,5))

        self.temperature_scale = ttk.Scale(options_frame, from_=0.0, to=1.0, 
                                          orient=tk.HORIZONTAL, variable=self.temperature_var,
                                          command=self.update_temp_label)
        self.temperature_scale.pack(fill=tk.X, expand=True, padx=5)

        # 狀態列
        self.status_label = ttk.Label(self.root, text="準備就緒")
        self.status_label.pack(fill=tk.X, side=tk.BOTTOM, padx=10, pady=5)
        
        # 更新 Milvus 狀態顯示
        if hasattr(self, 'vector_db_status'):
            self.vector_db_label.config(text=self.vector_db_status)
        
        # 延遲啟動 Ollama 連線檢查
        self.root.after(100, self.check_ollama_connection)
    
    def set_question(self, question):
        """設置問題"""
        self.question_entry.delete(1.0, tk.END)
        self.question_entry.insert(tk.END, question)
    
    def update_temp_label(self, value):
        """更新溫度顯示標籤"""
        self.temp_value_label.config(text=f"{float(value):.2f}")
    
    def check_ollama_connection(self):
        """檢查 Ollama 連接狀態"""
        def check_in_background():
            try:
                response = requests.get(f"{self.ollama_url}/api/tags", timeout=3)
                if response.status_code == 200:
                    models = response.json().get("models", [])
                    model_names = [model.get("name", "") for model in models]
                    
                    target_model = "llava-phi3:latest"
                    
                    if any(target_model in name for name in model_names):
                        self.current_model = target_model
                        self.root.after(0, lambda: self.ollama_status.config(
                            text="✅ Ollama + LLaVA-Phi3 可用", foreground="green"))
                        self.root.after(0, lambda: self.capture_btn.config(state=tk.NORMAL))
                        self.log_message(f"使用模型: {self.current_model}")
                    else:
                        phi3_models = [name for name in model_names if "llava-phi3" in name]
                        if phi3_models:
                            self.current_model = phi3_models[0]
                            self.root.after(0, lambda: self.ollama_status.config(
                                text=f"✅ Ollama + {self.current_model} 可用", foreground="green"))
                            self.root.after(0, lambda: self.capture_btn.config(state=tk.NORMAL))
                            self.log_message(f"使用模型: {self.current_model}")
                        else:
                            self.root.after(0, lambda: self.ollama_status.config(
                                text="⚠️ 需要安裝 LLaVA-Phi3", foreground="orange"))
                            self.suggest_install_phi3()
                else:
                    raise Exception("服務不可用")
                    
            except Exception:
                self.root.after(0, lambda: self.ollama_status.config(
                    text="❌ Ollama 未運行", foreground="red"))
                self.log_message("請執行: ollama serve")
        
        threading.Thread(target=check_in_background, daemon=True).start()
    
    def suggest_install_phi3(self):
        """建議安裝 LLaVA-Phi3"""
        message = """需要安裝 LLaVA-Phi3 模型！

請在終端執行：
ollama pull llava-phi3:latest

LLaVA-Phi3 特色：
✨ 基於 Microsoft Phi-3，效能優秀
💾 記憶體需求較低，適合 8GB MacBook
🚀 推理速度快
🌏 支援多語言回答

安裝完成後重新啟動程式。"""
        
        self.log_message(message)
    
    def start_camera(self):
        """開始相機"""
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
        """停止相機"""
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
        """更新即時影像"""
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
                    
                    time.sleep(0.033)  # 30 FPS
                    
            except Exception as e:
                print(f"影像更新錯誤: {e}")
                break
    
    def _update_live_video_label(self, photo):
        """更新即時影像標籤"""
        if self.live_video_label and self.update_video_flag:
            self.live_video_label.config(image=photo, text="")
            self.live_video_label.image = photo
    
    def capture_and_ask(self):
        """拍照並提問"""
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
        
        threading.Thread(target=self.process_with_ollama, 
                        args=(self.captured_frame.copy(), question), 
                        daemon=True).start()
    
    def display_captured_image(self):
        """顯示拍攝的照片"""
        if self.captured_frame is not None:
            display_frame = cv2.resize(self.captured_frame, (400, 300))
            display_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            
            pil_image = Image.fromarray(display_frame)
            photo = ImageTk.PhotoImage(pil_image)
            
            self.captured_image_label.config(image=photo, text="")
            self.captured_image_label.image = photo
    
    def process_with_ollama(self, frame, question):
        """處理圖像並儲存到 Milvus 向量資料庫（影像直接儲存）"""
        try:
            # 🔥 關鍵變化：將影像轉換為 base64 用於儲存
            frame_base64, image_size = self.image_to_base64(frame, quality=85, max_size=800)
            
            # 轉換圖像用於 API 呼叫（較小尺寸）
            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            # 適度縮放用於 API
            if max(pil_image.size) > 512:
                ratio = 512 / max(pil_image.size)
                new_size = tuple(int(dim * ratio) for dim in pil_image.size)
                pil_image = pil_image.resize(new_size, Image.Resampling.LANCZOS)
            
            # 轉換為 base64 用於 API
            buffer = io.BytesIO()
            pil_image.save(buffer, format='PNG')
            img_base64_api = base64.b64encode(buffer.getvalue()).decode()
            
            # 呼叫 Ollama API
            temperature = self.temperature_var.get()
            payload = {
                "model": self.current_model,
                "prompt": question,
                "images": [img_base64_api],
                "stream": False,
                "options": {
                    "temperature": temperature
                }
            }
            
            self.log_message(f"正在呼叫 Ollama API (Temperature: {temperature:.2f})...")
            self.log_message("⏰ LLaVA-Phi3 首次推理較慢，請耐心等待...")
            
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json=payload,
                timeout=120
            )
            
            if response.status_code == 200:
                result = response.json()
                answer = result.get("response", "無法獲得回答")
                
                if answer and answer.strip():
                    self.log_message(f"AI 回答: {answer}")
                    
                    # 🔥 重點：儲存到 Milvus 向量資料庫（影像直接儲存為 base64）
                    if frame_base64:
                        self.save_conversation_to_vector_db(
                            question=question,
                            answer=answer,
                            frame_base64=frame_base64,  # 完整質量的影像
                            image_size=image_size,
                            metadata={
                                "api_image_size": f"{pil_image.size[0]}x{pil_image.size[1]}",
                                "storage_image_size": f"{image_size[0]}x{image_size[1]}" if image_size else "unknown",
                                "response_length": len(answer),
                                "api_call_success": True,
                                "storage_method": "embedded_base64_milvus"
                            }
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
            self.log_message("💡 提示：第一次推理會較慢，後續會變快")
            self.log_message("🔄 請稍後重試，或嘗試較簡單的問題")
            self.is_analyzing = False
            self.root.after(0, lambda: self.status_label.config(text="首次推理超時，請重試"))
            
        except Exception as e:
            self.log_message(f"處理錯誤: {str(e)}")
            self.is_analyzing = False
            self.root.after(0, lambda: self.status_label.config(text="處理失敗"))
    
    def log_message(self, message):
        """記錄訊息"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_message = f"[{timestamp}] {message}\n"
        
        self.root.after(0, self._update_text, formatted_message)
    
    def _update_text(self, message):
        """更新文字區域"""
        self.result_text.insert(tk.END, message)
        self.result_text.see(tk.END)
    
    def run(self):
        """運行應用程式"""
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.mainloop()
    
    def on_closing(self):
        """關閉應用程式"""
        self.stop_camera()
        # 斷開 Milvus 連接
        try:
            connections.disconnect("default")
            print("✅ Milvus 連接已斷開")
        except:
            pass
        self.root.destroy()

def main():
    """主函數"""
    print("VLM 系統 + Milvus 向量資料庫 (影像內嵌版)")
    print("=" * 50)
    print("✨ 使用 LLaVA-Phi3:latest 模型")
    print("🖼️ 影像直接儲存在 Milvus 向量資料庫中")
    print("💾 無需管理檔案系統，統一儲存")
    print("🔍 支援對話搜尋和影像檢視")
    print("🌏 支援中英文問答")
    print("📊 可使用 Attu 管理介面查看資料")
    print("\n📦 需要的服務:")
    print("1. ollama pull llava-phi3:latest")
    print("2. docker run -p 19530:19530 -p 9091:9091 milvusdb/milvus:latest")
    print("3. docker run -p 8000:3000 -e MILVUS_URL=localhost:19530 zilliz/attu:latest")
    print("=" * 50)
    
    try:
        app = VLMWithMilvus()
        app.run()
    except KeyboardInterrupt:
        print("\n程式已停止")

if __name__ == "__main__":
    main()
            
