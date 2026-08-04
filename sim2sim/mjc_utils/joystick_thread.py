import pygame
from threading import Thread

def start_joystick_thread():
    # global joystick_opened, joystick_use

    # if not joystick_use:
    #     print("Joystick disabled.")
    #     return None

    pygame.init()
    try:
        joystick = pygame.joystick.Joystick(0)
        joystick.init()
        joystick_opened = True
        print("Joystick connected!")
    except Exception as e:
        print(f"无法打开手柄：{e}")
        joystick_opened = False
        return None

    exit_flag = False

    def handle_joystick_input():
        global x_vel_cmd, y_vel_cmd, yaw_vel_cmd, height_vel_cmd, reset_pos

        X_button_was_pressed = False
        A_button_was_pressed = False

        while not exit_flag:
            pygame.event.get()

            # 切换视角模式（可选）
            X_button_pressed = joystick.get_button(2)
            if X_button_pressed and not X_button_was_pressed:
                print("Camera mode switch pressed (X)")
            X_button_was_pressed = X_button_pressed

            # 复位机器人（可选）
            A_button_pressed = joystick.get_button(0)
            if A_button_pressed and not A_button_was_pressed:
                reset_pos = True
                print("press A")
            A_button_was_pressed = A_button_pressed

            # === 读取摇杆输入 ===
            x_vel_cmd = max(-1.5, min(-joystick.get_axis(1) * 2.0, 2.0))
            y_vel_cmd = -joystick.get_axis(0) * 1.5
            yaw_vel_cmd = -joystick.get_axis(3) * 1.5
            height_vel_cmd = -joystick.get_axis(4) * 0.4

            pygame.time.delay(50)

    # 创建线程
    joystick_thread = Thread(target=handle_joystick_input, daemon=True)
    joystick_thread.start()
    return joystick_thread