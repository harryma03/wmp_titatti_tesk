"""
从 TensorBoard event 文件读取训练数据并绘制趋势曲线。
解决 tensorboard 命令因 NumPy/SciPy 版本冲突无法启动的问题。

用法:
    cd /home/aiseon/other_robot_rl/tesk/WMP
    python scripts/plot_training_curves.py

输出: 在 logs/titatit_amp_example/WMP_titatit/ 下生成 PNG 图片
"""

import os
import sys
import struct
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def parse_tensorboard_event(filepath):
    """解析 TensorBoard event 文件，返回 {tag: [(step, value), ...]}"""
    data = {}
    with open(filepath, 'rb') as f:
        content = f.read()

    pos = 0
    while pos < len(content):
        if pos + 16 > len(content):
            break

        # 读取 payload length (第 8-15 字节)
        payload_len = struct.unpack('<Q', content[pos+8:pos+16])[0]

        if payload_len == 0 or payload_len > 1000000:  # 合理性检查
            pos += 1
            continue

        if pos + 16 + payload_len > len(content):
            break

        payload = content[pos+16:pos+16+payload_len]
        pos += 16 + payload_len

        # 解析 protobuf 格式的 Event
        try:
            text = payload.decode('utf-8', errors='replace')

            tag_match = re.search(r'tag:\s*"([^"]+)"', text)
            value_match = re.search(r'simple_value:\s*([-\d.e+]+)', text)
            step_match = re.search(r'step:\s*(\d+)', text)

            if tag_match and value_match and step_match:
                tag = tag_match.group(1)
                value = float(value_match.group(1))
                step = int(step_match.group(1))

                if tag not in data:
                    data[tag] = []
                data[tag].append((step, value))
        except:
            continue

    # 按 step 排序
    for tag in data:
        data[tag].sort(key=lambda x: x[0])

    return data


def plot_curves(data, output_dir):
    """绘制关键训练曲线"""

    groups = {
        '训练主指标': {
            'tags': ['Train/mean_episode_length', 'Train/mean_reward'],
            'ylabels': ['Episode Length', 'Mean Reward'],
            'filename': 'training_main.png',
        },
        'Episode 奖励分解': {
            'tags': [
                'Episode/tracking_lin_vel',
                'Episode/tracking_ang_vel',
                'Episode/terrain_progress',
                'Episode/lateral_deviation',
            ],
            'ylabels': ['tracking_lin_vel', 'tracking_ang_vel', 'terrain_progress', 'lateral_deviation'],
            'filename': 'episode_rewards.png',
        },
        'Episode 惩罚与地形': {
            'tags': [
                'Episode/action_rate',
                'Episode/dof_acc',
                'Episode/terrain_level',
            ],
            'ylabels': ['action_rate', 'dof_acc', 'terrain_level'],
            'filename': 'episode_penalties.png',
        },
        'AMP 相关 Loss': {
            'tags': [
                'Loss/AMP',
                'Loss/AMP_grad',
                'Loss/AMP_mean_policy_pred',
                'Loss/AMP_mean_expert_pred',
            ],
            'ylabels': ['AMP loss', 'AMP grad pen', 'AMP policy pred', 'AMP expert pred'],
            'filename': 'amp_losses.png',
        },
        '策略与性能': {
            'tags': [
                'Policy/mean_noise_std',
                'Perf/total_fps',
                'Loss/value_function',
                'Loss/surrogate',
            ],
            'ylabels': ['mean_noise_std', 'FPS', 'value_loss', 'surrogate_loss'],
            'filename': 'policy_perf.png',
        },
    }

    for group_name, group_info in groups.items():
        tags = group_info['tags']
        ylabels = group_info['ylabels']
        filename = group_info['filename']

        n_plots = len(tags)
        fig, axes = plt.subplots(n_plots, 1, figsize=(12, 3 * n_plots), squeeze=False)
        fig.suptitle(group_name, fontsize=14, fontweight='bold')

        for i, (tag, ylabel) in enumerate(zip(tags, ylabels)):
            ax = axes[i][0]
            if tag in data:
                steps = [x[0] for x in data[tag]]
                values = [x[1] for x in data[tag]]
                ax.plot(steps, values, 'b-', linewidth=1.5, alpha=0.8, label='raw')

                # 添加平滑曲线（移动平均）
                if len(values) > 20:
                    window = max(5, len(values) // 50)
                    kernel = np.ones(window) / window
                    smoothed = np.convolve(values, kernel, mode='valid')
                    smooth_steps = steps[len(steps) - len(smoothed):]
                    ax.plot(smooth_steps, smoothed, 'r-', linewidth=2, alpha=0.6, label='smoothed')

                ax.set_ylabel(ylabel, fontsize=10)
                ax.grid(True, alpha=0.3)
                ax.legend(loc='best', fontsize=8)

                # 显示最新值
                latest_val = values[-1]
                latest_step = steps[-1]
                ax.axhline(y=latest_val, color='gray', linestyle='--', alpha=0.5)
                ax.text(0.02, 0.95, f'latest: {latest_val:.4f} @ step {latest_step}',
                        transform=ax.transAxes, fontsize=9, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            else:
                ax.text(0.5, 0.5, f'Tag "{tag}" not found', ha='center', va='center',
                        transform=ax.transAxes, fontsize=10)

            if i == n_plots - 1:
                ax.set_xlabel('Iteration', fontsize=10)

        plt.tight_layout()
        output_path = os.path.join(output_dir, filename)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {output_path}")


def main():
    log_dir = 'logs/titatit_amp_example/WMP_titatit'
    output_dir = log_dir

    event_files = [f for f in os.listdir(log_dir) if f.startswith('events.out.tfevents')]
    if not event_files:
        print(f"Error: No TensorBoard event files found in {log_dir}")
        sys.exit(1)

    event_path = os.path.join(log_dir, event_files[0])
    print(f"Parsing event file: {event_path}")

    data = parse_tensorboard_event(event_path)
    print(f"Found {len(data)} unique tags")

    print("\nAvailable tags:")
    for tag in sorted(data.keys()):
        n_points = len(data[tag])
        if n_points > 0:
            latest = data[tag][-1][1]
            print(f"  {tag:50s}  {n_points:5d} points  latest={latest:.4f}")

    print("\nGenerating plots...")
    plot_curves(data, output_dir)

    print(f"\nDone! Plots saved to {os.path.abspath(output_dir)}/")
    print("You can view them by opening the PNG files in VS Code or your file browser.")


if __name__ == '__main__':
    main()
