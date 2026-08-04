from pynput import keyboard
import atexit
import numpy as np
import threading


class KeyboardController:
    def __init__(self, max_vel=1.0, max_yaw_vel=None):
        self.max_vel = max_vel
        self.max_yaw_vel = max_vel if max_yaw_vel is None else max_yaw_vel
        self.x_vel = 0
        self.y_vel = 0
        self.ang_vel = 0
        self.scram = False
        self.listener = None
        self.listener_thread = None
        self.running = False
        atexit.register(self.stop)

    def on_key_press(self, key):
        if key == keyboard.Key.up:
            self.x_vel = self.max_vel
        elif key == keyboard.Key.down:
            self.x_vel = -self.max_vel
        elif key == keyboard.Key.left:
            self.ang_vel = self.max_yaw_vel
        elif key == keyboard.Key.right:
            self.ang_vel = -self.max_yaw_vel
        elif key == keyboard.KeyCode(char='q'):
            self.y_vel = self.max_vel
        elif key == keyboard.KeyCode(char='e'):
            self.y_vel = -self.max_vel
        elif key == keyboard.Key.space:
            self.scram = True

    def on_key_release(self, key):
        if key in [keyboard.Key.up, keyboard.Key.down]:
            self.x_vel = 0
        elif key in [keyboard.Key.left, keyboard.Key.right]:
            self.ang_vel = 0
        elif key == keyboard.KeyCode(char='q') or key == keyboard.KeyCode(char='e'):
            self.y_vel = 0

        if key == keyboard.Key.f1:
            self.stop()
            return False

    def start_listening(self):
        if self.running:
            return

        self.running = True
        self.listener = keyboard.Listener(on_press=self.on_key_press, on_release=self.on_key_release)
        self.listener_thread = threading.Thread(target=self.listener.start)
        self.listener_thread.daemon = True
        self.listener_thread.start()

    def stop(self):
        if not self.running:
            return

        self.running = False
        if self.listener:
            self.listener.stop()
        if self.listener_thread and self.listener_thread.is_alive():
            self.listener_thread.join(timeout=1.0)

    def get_velocities(self):
        return np.array([self.x_vel, self.y_vel, self.ang_vel])

    def get_scramble(self):
        return self.scram
