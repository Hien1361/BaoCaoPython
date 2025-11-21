import sys, os, time
import cv2
import numpy as np
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QToolBar, QToolButton,
    QFileDialog, QHBoxLayout, QVBoxLayout, QGroupBox, QSlider, QStatusBar, QStyle, 
    QMessageBox, QComboBox  # <-- ĐÃ THÊM QComboBox
)
from PySide6.QtGui import QAction, QImage, QPixmap, QIcon  # <-- ĐÃ THÊM QIcon
from PySide6.QtCore import Qt, QTimer, QSize, QThread, Signal
from ultralytics import YOLO

# --- Class VideoThread (Không thay đổi) ---
class VideoThread(QThread):
    frame_ready = Signal(np.ndarray) 
    stats_ready = Signal(str, str, str)

    def __init__(self, source, model, conf):
        super().__init__()
        self.source = source
        self.model = model
        self.conf = conf
        self.running = True

    def run(self):
        is_image = isinstance(self.source, str) and self.source.endswith(('.jpg', '.jpeg', '.png'))

        if is_image:
            frame = cv2.imread(self.source)
            if frame is not None:
                self.process_frame(frame, is_image=True)
            return

        try:
            cap = cv2.VideoCapture(self.source)
            if not cap.isOpened():
                print(f"Lỗi: Không thể mở nguồn {self.source}")
                return
        except Exception as e:
            print(f"Lỗi khi mở VideoCapture: {e}")
            return
            
        is_video_file = isinstance(self.source, str)
        
        while self.running and cap.isOpened():
            start_time = time.time()
            ret, frame = cap.read()

            if not ret:
                if is_video_file:
                    print("Video kết thúc.")
                    break
                else:
                    continue

            self.process_frame(frame)
            
            elapsed = (time.time() - start_time) * 1000 # ms
            fps = 1000 / max(elapsed, 1)
            
            self.stats_ready.emit(f"FPS: {fps:.1f}", f"Infer: {elapsed:.1f} ms", "")
            
        cap.release()
        print("VideoThread đã dừng.")

    def process_frame(self, frame, is_image=False):
        try:
            results = self.model(frame, conf=self.conf)
            annotated_frame = results[0].plot()
            self.frame_ready.emit(annotated_frame)

            if not is_image:
                count = len(results[0].boxes)
                self.stats_ready.emit("", "", f"Detections: {count}")

        except Exception as e:
            print(f"Lỗi trong quá trình inference: {e}")

    def stop(self):
        self.running = False
        self.wait()


# --- Class MainWindow (Đã được nâng cấp) ---
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Camera thông minh — Phát hiện đối tượng (YOLO)")
        self.resize(1000, 660)

        # === NÂNG CẤP 2: THÊM ICON CHO ỨNG DỤNG ===
        # (Tạo 1 file "app_icon.ico" và để ngang hàng file code)
        icon_path = "anh11/app_icon.ico.jpg"
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))
        # ========================================

        # Đọc file QSS (nếu có)
        qss_content = self.load_qss("style.qss")
        if qss_content:
            self.setStyleSheet(qss_content)

        self.source = None
        self.conf_val = 25
        self.thread = None
        self.model_path = "" # Sẽ được cập nhật khi nhấn Start

        # === NÂNG CẤP 3: TÙY CHỌN MODEL ===
        # Model sẽ được load khi nhấn "Start", không load cứng ở đây nữa
        self.model = None 

        # Định nghĩa các model bạn có
        self.available_models = {
            "Nhan Dien 4 Chat (Co, Ro, Chuon, Bich)": "runs/train/model_bai_tay1/weights/best.pt",
            "Nhan Dien 52 La Bai (Chuan)": "runs/train/model_52_la_bai_FINAL_V3/weights/best.pt"
            # (Lưu ý: Model 52 lá bài phải được train xong thì mới chạy được)
        }
        # ==================================

        self._build_toolbar()
        self._build_body()
        self._build_statusbar()
        
        self.preview.setText(self._preview_text_default())

    def load_qss(self, file_path):
        """Hàm helper để đọc file stylesheet .qss"""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError:
            print(f"INFO: Không tìm thấy file stylesheet '{file_path}'. Bỏ qua...")
            return ""
        except Exception as e:
            print(f"Lỗi khi đọc file QSS: {e}")
            return ""

    def _build_toolbar(self):
        tb = QToolBar()
        tb.setIconSize(QSize(18, 18))
        tb.setMovable(False)
        tb.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.addToolBar(tb)
        style = QApplication.style()
        act_open = QAction(style.standardIcon(QStyle.SP_DialogOpenButton), "Open", self)
        act_open.triggered.connect(self.on_open_file)
        btn_open = QToolButton(); btn_open.setDefaultAction(act_open)
        btn_open.setObjectName("BtnOpen")
        tb.addWidget(btn_open)
        act_cam = QAction(style.standardIcon(QStyle.SP_ComputerIcon), "Webcam (0)", self)
        act_cam.triggered.connect(lambda: self.on_use_cam(0)) 
        btn_cam = QToolButton(); btn_cam.setDefaultAction(act_cam)
        btn_cam.setObjectName("BtnCam")
        tb.addWidget(btn_cam)
        act_cam_1 = QAction(style.standardIcon(QStyle.SP_ComputerIcon), "DroidCam (IP)", self)
        droidcam_url = "http://192.168.137.111:4747/video"
        act_cam_1.triggered.connect(lambda: self.on_use_cam(droidcam_url))
        btn_cam_1 = QToolButton(); btn_cam_1.setDefaultAction(act_cam_1)
        tb.addWidget(btn_cam_1)
        tb.addSeparator()
        act_start = QAction(style.standardIcon(QStyle.SP_MediaPlay), "Start", self)
        act_start.triggered.connect(self.on_start)
        btn_start = QToolButton(); btn_start.setDefaultAction(act_start)
        btn_start.setObjectName("BtnStart")
        tb.addWidget(btn_start)
        act_stop = QAction(style.standardIcon(QStyle.SP_MediaStop), "Stop", self)
        act_stop.triggered.connect(self.on_stop)
        btn_stop = QToolButton(); btn_stop.setDefaultAction(act_stop)
        btn_stop.setObjectName("BtnStop")
        tb.addWidget(btn_stop)

    def _build_body(self):
        central = QWidget(); self.setCentralWidget(central)
        root = QHBoxLayout(central); root.setContentsMargins(16,16,16,16); root.setSpacing(16)

        self.preview = QLabel("(Chưa có gì)")
        self.preview.setObjectName("PreviewCard")
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setMinimumSize(580, 330)
        self.preview.setScaledContents(False) 
        root.addWidget(self.preview, 2)

        right = QWidget(); col = QVBoxLayout(right); col.setSpacing(12)

        box_info = QGroupBox("Nguồn & Mô hình")
        vinfo = QVBoxLayout(box_info)
        self.lbl_source = QLabel("Nguồn: (chưa chọn)")
        vinfo.addWidget(self.lbl_source)
        
        # === NÂNG CẤP 3: THÊM DROPDOWN CHỌN MODEL ===
        vinfo.addSpacing(10) # Thêm khoảng cách
        lbl_model_select = QLabel("Chọn mô hình AI:")
        self.combo_model = QComboBox()
        self.combo_model.addItems(self.available_models.keys())
        
        vinfo.addWidget(lbl_model_select)
        vinfo.addWidget(self.combo_model)
        
        # Xóa dòng QLabel model cũ
        # self.lbl_model = QLabel("Model: Nhan Dien 4 Chat (Co, Ro, Chuon, Bich)") 
        # vinfo.addWidget(self.lbl_model)
        # ==========================================

        col.addWidget(box_info)

        box_conf = QGroupBox("Confidence")
        vconf = QVBoxLayout(box_conf)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(5, 90)
        self.slider.setValue(self.conf_val)
        self.slider.valueChanged.connect(self.on_conf_changed)
        self.lbl_conf = QLabel(self._conf_str())
        vconf.addWidget(self.slider)
        vconf.addWidget(self.lbl_conf)
        col.addWidget(box_conf)

        col.addStretch(1)
        root.addWidget(right, 1)

    def _build_statusbar(self):
        # (Không thay đổi)
        sb = QStatusBar()
        self.lbl_state = QLabel("Sẵn sàng")
        self.lbl_fps = QLabel("FPS: --")
        self.lbl_ms = QLabel("Infer: -- ms")
        self.lbl_cnt = QLabel("Detections: --")
        sb.addPermanentWidget(self.lbl_state)
        sb.addPermanentWidget(self.lbl_fps)
        sb.addPermanentWidget(self.lbl_ms)
        sb.addPermanentWidget(self.lbl_cnt)
        self.setStatusBar(sb)

    def on_open_file(self):
        # (Không thay đổi)
        self.on_stop()
        path, _ = QFileDialog.getOpenFileName(self, "Chọn video/ảnh", "", "Media Files (*.mp4 *.avi *.mkv *.jpg *.png)")
        if path:
            self.source = path
            self.lbl_source.setText(f"Nguồn: {os.path.basename(self.source)}")
            self.preview.setText(self._preview_text_default())

    def on_use_cam(self, source_input):
        # (Không thay đổi)
        self.on_stop()
        self.source = source_input
        
        if isinstance(source_input, int):
            self.lbl_source.setText(f"Nguồn: Webcam {source_input}")
        else:
            self.lbl_source.setText(f"Nguồn: {source_input}")
            
        self.preview.setText(self._preview_text_default())

    def on_start(self):
        # === NÂNG CẤP 1: KIỂM TRA NGUỒN BẰNG POP-UP ===
        if self.source is None:
            QMessageBox.warning(self, "Chưa chọn nguồn", "Vui lòng chọn Webcam hoặc mở file media trước.")
            self.lbl_state.setText("Chưa chọn nguồn")
            return
        
        # === NÂNG CẤP 1 & 3: LOAD MODEL KHI NHẤN START VÀ BÁO LỖI ===
        try:
            # Lấy tên model từ dropdown
            selected_model_name = self.combo_model.currentText()
            # Lấy đường dẫn từ dictionary
            self.model_path = self.available_models[selected_model_name]
            
            self.lbl_state.setText(f"Đang tải: {selected_model_name}...")
            QApplication.processEvents() # Cập nhật giao diện ngay
            
            # Tải model
            self.model = YOLO(self.model_path)
            
            print(f"Đã tải model thành công: {self.model_path}")
            
        except Exception as e:
            # Dùng Pop-up báo lỗi nếu không tải được model
            QMessageBox.critical(
                self, 
                "Lỗi tải mô hình", 
                f"Không thể tải model!\n\nĐường dẫn: {self.model_path}\nLỗi: {e}\n\n(Bạn đã train model này chưa?)"
            )
            self.model = None
            self.lbl_state.setText("Model lỗi (Xem lại đường dẫn)")
            return
        # ====================================================

        if self.thread is not None and self.thread.isRunning():
             self.on_stop()
             time.sleep(0.1)

        # Khởi động luồng
        self.thread = VideoThread(self.source, self.model, self.conf_val/100)
        self.thread.frame_ready.connect(self.update_image)
        self.thread.stats_ready.connect(self.update_stats)
        self.thread.finished.connect(self.thread.deleteLater) 
        self.thread.start()
        self.lbl_state.setText("Đang chạy…")
        self.preview.setText("Đang khởi động luồng...")

    def on_stop(self):
        # (Không thay đổi)
        self.lbl_state.setText("Đang dừng...")
        if self.thread:
            self.thread.stop()
            self.thread = None
        
        self.lbl_state.setText("Đã dừng")
        self.lbl_fps.setText("FPS: --")
        self.lbl_ms.setText("Infer: -- ms")
        self.lbl_cnt.setText("Detections: --")
        self.preview.setText(self._preview_text_default())
        self.preview.setPixmap(QPixmap())

    def on_conf_changed(self, _):
        # (Không thay đổi)
        self.conf_val = self.slider.value()
        self.lbl_conf.setText(self._conf_str())
        if self.thread:
            self.thread.conf = self.conf_val / 100

    def _conf_str(self):
        # (Không thay đổi)
        return f"{self.conf_val/100:.2f}"

    def _preview_text_default(self):
        # (Không thay đổi)
        src_text = "(chưa chọn)"
        if isinstance(self.source, str):
            src_text = os.path.basename(self.source)
        elif isinstance(self.source, int):
            src_text = f"Webcam {self.source}"
        elif self.source is not None:
             src_text = str(self.source)
        
        return (
            "SMARTCAM — PyTorch + YOLO\n\n"
            f"Nguồn: {src_text}\n"
            "Trạng thái: ĐÃ DỪNG\n"
            f"Confidence: {self._conf_str()}\n\n"
            "(Chọn Nguồn và Mô hình, sau đó nhấn Start)"
        )

    def update_image(self, cv_img):
        # (Không thay đổi)
        try:
            rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            qt_pixmap = QPixmap.fromImage(qt_image)

            self.preview.setPixmap(qt_pixmap.scaled(
                self.preview.size(), 
                Qt.KeepAspectRatio, 
                Qt.SmoothTransformation
            ))
        except Exception as e:
            print(f"Lỗi update_image: {e}")

    def update_stats(self, fps_str, ms_str, cnt_str):
        # (Không thay đổi)
        if fps_str:
            self.lbl_fps.setText(fps_str)
        if ms_str:
            self.lbl_ms.setText(ms_str)
        if cnt_str:
            self.lbl_cnt.setText(cnt_str)
            
    def keyPressEvent(self, e):
        # (Không thay đổi)
        if e.key() == Qt.Key_Space:
            if self.thread and self.thread.isRunning():
                self.on_stop()
            else:
                self.on_start()
        elif e.key() == Qt.Key_O:
            self.on_open_file()
        elif e.key() == Qt.Key_W:
            self.on_use_cam(0)
        else:
            super().keyPressEvent(e)
            
    def closeEvent(self, event):
        # (Không thay đổi)
        self.on_stop()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # === NÂNG CẤP 2 (Tiếp): Đặt tên cho App ID (Giúp icon hiển thị đúng trên Windows) ===
    if os.name == 'nt':
        import ctypes
        myappid = 'my.smartcam.yolo.1' # Tên bất kỳ
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
    # ==============================================================================

    w = MainWindow()
    w.show()
    sys.exit(app.exec())