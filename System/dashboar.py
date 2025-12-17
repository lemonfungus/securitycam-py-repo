import cv2
import datetime
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

class SecurityDashboard:
    def __init__(self, window):
        self.window = window
        self.window.title("AI Security Monitor - Team 12")
        self.window.geometry("1100x700")
        self.window.configure(bg="#1e1e1e") # Dark mode background

        # --- 1. Load AI Models ---
        print("Loading AI Models...")
        self.model = YOLO("best.pt") 
        self.tracker = DeepSort(max_age=30)
        self.track_history = {} # Memory: {ID: Name}

        # --- 2. GUI Layout ---
        self.create_widgets()

        # --- 3. Camera Setup ---
        self.cap = cv2.VideoCapture(0)
        self.is_running = True
        
        # Start the video loop
        self.update_video()

    def create_widgets(self):
        # >> TITLE HEADER
        title_label = tk.Label(
            self.window, text="🔒 RESTRICTED AREA MONITORING", 
            font=("Arial", 20, "bold"), bg="#1e1e1e", fg="#00ff00"
        )
        title_label.pack(pady=10)

        # >> MAIN CONTENT FRAME (Video Left, Sidebar Right)
        content_frame = tk.Frame(self.window, bg="#1e1e1e")
        content_frame.pack(fill=tk.BOTH, expand=True, padx=20)

        # [LEFT] Video Display
        self.video_label = tk.Label(content_frame, bg="black", borderwidth=2, relief="solid")
        self.video_label.pack(side=tk.LEFT, padx=10, pady=10)

        # [RIGHT] Sidebar Controls
        sidebar = tk.Frame(content_frame, bg="#2d2d2d", width=300)
        sidebar.pack(side=tk.RIGHT, fill=tk.Y, padx=10, pady=10)

        # Status Label
        self.status_label = tk.Label(
            sidebar, text="SYSTEM: ONLINE", 
            font=("Consolas", 14, "bold"), bg="#2d2d2d", fg="#00ff00"
        )
        self.status_label.pack(pady=20)

        # Activity Log Title
        log_title = tk.Label(
            sidebar, text="Activity Log:", 
            font=("Arial", 12), bg="#2d2d2d", fg="white"
        )
        log_title.pack(anchor="w", padx=10)

        # Activity Log Listbox (The scrolling text)
        self.log_list = tk.Listbox(
            sidebar, height=20, width=35, 
            bg="black", fg="#00ff00", font=("Consolas", 10)
        )
        self.log_list.pack(padx=10, pady=5)

        # Snapshot Button
        btn_snap = tk.Button(
            sidebar, text="📸 TAKE SNAPSHOT", 
            font=("Arial", 12, "bold"), bg="#444", fg="white",
            command=self.take_snapshot
        )
        btn_snap.pack(pady=20, fill=tk.X, padx=10)

        # Quit Button
        btn_quit = tk.Button(
            sidebar, text="EXIT SYSTEM", 
            font=("Arial", 12, "bold"), bg="#8b0000", fg="white",
            command=self.close_app
        )
        btn_quit.pack(side=tk.BOTTOM, pady=20, fill=tk.X, padx=10)

    def log_event(self, message):
        """Adds a message to the sidebar log with a timestamp."""
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        entry = f"[{timestamp}] {message}"
        
        # Only log if it's not the exact same as the last message (reduce spam)
        if self.log_list.size() == 0 or self.log_list.get(tk.END) != entry:
            self.log_list.insert(tk.END, entry)
            self.log_list.see(tk.END) # Auto-scroll to bottom

    def update_video(self):
        """The main loop: Reads frame, runs AI, updates GUI."""
        if not self.is_running:
            return

        ret, frame = self.cap.read()
        if ret:
            # --- AI PROCESSING START ---
            results = self.model(frame, stream=True, verbose=False)
            detections = []

            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    detected_name = self.model.names[cls_id]

                    if conf > 0.5:
                        w = x2 - x1
                        h = y2 - y1
                        detections.append([[x1, y1, w, h], conf, detected_name])

            # Update Tracker
            tracks = self.tracker.update_tracks(detections, frame=frame)

            for track in tracks:
                if not track.is_confirmed():
                    continue

                track_id = track.track_id
                ltrb = track.to_ltrb()
                x1, y1, x2, y2 = map(int, ltrb)
                
                # Identity Logic
                det_class = track.get_det_class()
                if det_class is not None and det_class in ['Aum', 'Jiw', 'Prem']:
                    self.track_history[track_id] = det_class
                    name_label = det_class
                    # LOG THE EVENT
                    self.log_event(f"Detected: {name_label}")
                else:
                    name_label = self.track_history.get(track_id, "Unknown")

                # Draw UI on Frame
                color = (0, 255, 0) if name_label != "Unknown" else (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{name_label} #{track_id}", (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            # --- AI PROCESSING END ---

            # Convert frame for Tkinter
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(image=img)

            # Update the label
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)

        # Repeat this function after 10 milliseconds
        self.window.after(10, self.update_video)

    def take_snapshot(self):
        """Saves the current frame to a file."""
        ret, frame = self.cap.read()
        if ret:
            filename = f"snapshot_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(filename, frame)
            self.log_event(f"📸 Snapshot Saved: {filename}")
            print(f"Snapshot saved: {filename}")

    def close_app(self):
        """Clean shutdown."""
        self.is_running = False
        self.cap.release()
        self.window.destroy()

# --- ENTRY POINT ---
if __name__ == "__main__":
    root = tk.Tk()
    app = SecurityDashboard(root)
    root.mainloop()