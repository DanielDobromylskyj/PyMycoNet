import random
import threading
import tkinter as tk
from tkinter import ttk


class Monitor(tk.Tk):
    def __init__(self, network, learning_rate):
        self.network = network

        super().__init__()
        self.title("Training Monitor")
        self.configure(bg="#f0f0f0")
        self.geometry("1000x500")

        # 3 main columns
        self.columnconfigure((0, 1, 2), weight=1, uniform="col")
        self.rowconfigure(0, weight=1)

        # --- Left: OpenCL Context ---
        devices = self.network.cl.ctx.devices
        device = devices[0]
        left_frame = self.make_section("Open CL Context", col=0)
        self.make_label_pair(left_frame, "Devices:", f"{len(devices)}")
        self.make_label_pair(left_frame, "Device:", device.name)
        self.make_label_pair(left_frame, "Vendor:", device.vendor)
        self.make_label_pair(left_frame, "Version:", device.version)
        self.make_label_pair(left_frame, "Driver:", device.driver_version)

        self.make_label_pair(left_frame, "Max Compute Units:", device.max_compute_units)
        self.make_label_pair(left_frame, "Max Work Group Size:", device.max_work_group_size)
        self.make_label_pair(left_frame, "Max Work Dims:", device.max_work_item_dimensions)
        self.make_label_pair(left_frame, "Max Work Size:", f"({', '.join(map(str, device.max_work_item_sizes))})")
        self.make_label_pair(left_frame, "Global Mem Size:", f"{device.global_mem_size // (1024 * 1024 * 1024)} GB")
        self.make_label_pair(left_frame, "Local Memory Size:", f"{device.local_mem_size // 1024} KB")
        self.make_label_pair(left_frame, "Max Buffer Size:", f"{device.max_constant_buffer_size // (1024 * 1024 * 1024)} GB")
        self.make_label_pair(left_frame, "Max Alloc Mem:", f"{device.max_mem_alloc_size // (1024 * 1024 * 1024)} GB")


        # --- Middle: Model Info + Graph Placeholder ---
        middle_frame = self.make_section("", col=1)
        top_info = self.make_subframe(middle_frame)
        self.make_label_pair(top_info, "Version:", f"v{'.'.join(map(str, self.network.version))}")
        self.make_label_pair(top_info, "File Version:", f"v{'.'.join(map(str, self.network.pyn_version))}")
        self.make_label_pair(top_info, "Layer Count:", len(self.network.layout))
        self.make_label_pair(top_info, "Learning Rate:", learning_rate)

        graph_frame = self.make_subframe(middle_frame)
        tk.Label(graph_frame, text="Loss Curve").pack()
        self.canvas = tk.Canvas(graph_frame, width=300, height=150, bg="white")
        self.canvas.pack(padx=10, pady=10)
        self.graph_points = []

        # --- Right: Fast-updating info ---
        right_frame = self.make_section("Live Info", col=2)

        # Epoch progress bar
        self.epoch_var = tk.DoubleVar(value=0)
        self.epoch_text = tk.StringVar(value="Epoch: 0/100")

        self.epoch_label = ttk.Label(right_frame, textvariable=self.epoch_text)
        self.epoch_label.pack()


        self.progress = ttk.Progressbar(right_frame, orient="horizontal",
                                        mode="determinate", variable=self.epoch_var,
                                        maximum=100, length=200)
        self.progress.pack(pady=10)

        # GPU usage arc
        self.gpu_canvas = tk.Canvas(right_frame, width=200, height=150, bg="white")
        self.gpu_canvas.pack()
        self.gpu_usage = 0

        # Start update loop
        self.update_loop()

    def make_section(self, title, col):
        frame = tk.LabelFrame(self, text=title, padx=10, pady=10)
        frame.grid(row=0, column=col, sticky="nsew", padx=8, pady=8)
        frame.columnconfigure(0, weight=1)
        return frame

    def make_subframe(self, parent):
        f = tk.Frame(parent)
        f.pack(fill="x", pady=5)
        return f

    def make_label_pair(self, parent, key, value):
        row = tk.Frame(parent)
        row.pack(fill="x", anchor="w")
        tk.Label(row, text=key, width=16, anchor="w").pack(side="left")
        tk.Label(row, text=value, anchor="e").pack(side="right")

    def update_loop(self):
        if self.network.epoches > 0:
            self.epoch_var.set(round((self.network.epoche / self.network.epoches) * 100))

        self.epoch_text.set(f"Epoch: {self.network.epoche}/{self.network.epoches}")

        self.gpu_usage = 50 #(self.gpu_usage + random.uniform(-10, 10)) % 100 # todo

        self.draw_gpu_gauge()
        self.draw_loss_curve()
        self.after(500, self.update_loop)

    def draw_gpu_gauge(self):
        c = self.gpu_canvas
        c.delete("all")
        cx, cy, r = 100, 80, 60
        start_angle = 180 + 45  # bottom-left corner
        extent = int(self.gpu_usage * 2.7)

        c.create_arc(cx - r, cy - r, cx + r, cy + r, start=start_angle, extent=-270,
                     style="arc", width=20, outline="grey")

        c.create_arc(cx - r, cy - r, cx + r, cy + r, start=start_angle, extent=-extent,
                     style="arc", width=20, outline="green")

        c.create_text(cx, cy + 60, text=f"GPU: {self.gpu_usage:.1f}%", font=("Arial", 12))

    def draw_loss_curve(self):
        val = self.network.last_error
        self.graph_points.append(val)
        if len(self.graph_points) > 30:
            self.graph_points.pop(0)

        self.canvas.delete("all")
        w, h = 300, 170
        min_p, max_p = min(self.graph_points), max(self.graph_points)
        range_p = (max_p - min_p) + 0.00001

        scaled = [h - (h * ((p - min_p) / range_p)) + 5 for p in self.graph_points]
        for i in range(1, len(scaled)):
            self.canvas.create_line((i - 1) * 10, scaled[i - 1], i * 10, scaled[i], fill="gray")

    @staticmethod
    def start(*args):
        def worker(*args):
            monitor = Monitor(*args)
            monitor.mainloop()

        thread = threading.Thread(target=worker, daemon=True, args=args)
        thread.start()
