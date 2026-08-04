import cv2
import numpy as np
import threading
import time


class DepthViewer:
    def __init__(self, scale=6):
        self.scale = scale
        self.latest_frame = None
        self.lock = threading.Lock()
        self.running = True

        self.thread = threading.Thread(target=self._show_loop, daemon=True)
        self.thread.start()

    def update(self, depth_vis):
        """主线程调用，用于更新图像"""
        depth_vis_big = np.repeat(
            np.repeat(depth_vis, self.scale, axis=0),
            self.scale,
            axis=1
        )

        with self.lock:
            self.latest_frame = depth_vis_big.copy()

    def _show_loop(self):
        """显示线程"""
        cv2.namedWindow("depth", cv2.WINDOW_NORMAL)
        cv2.setWindowProperty("depth", cv2.WND_PROP_TOPMOST, 1)

        while self.running:
            frame = None
            with self.lock:
                if self.latest_frame is not None:
                    frame = self.latest_frame.copy()

            if frame is not None:
                cv2.imshow("depth", frame)
                cv2.setWindowProperty("depth", cv2.WND_PROP_TOPMOST, 1)

            cv2.waitKey(1)
            time.sleep(0.01)

        cv2.destroyWindow("depth")

    def stop(self):
        self.running = False
        self.thread.join()
