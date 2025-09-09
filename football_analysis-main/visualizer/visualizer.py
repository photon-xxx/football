import os
import numpy as np
import matplotlib.pyplot as plt

def plot_team_ball_control(team_ball_control, save_dir="figures"):
    """
    绘制两队控球率随时间变化的折线图
    team_ball_control: numpy array，逐帧标记哪一方控球（1 或 2）
    """
    os.makedirs(save_dir, exist_ok=True)

    num_frames = len(team_ball_control)
    time = np.arange(num_frames)

    team1_possession = np.cumsum(team_ball_control == 1) / (np.arange(num_frames) + 1) * 100
    team2_possession = np.cumsum(team_ball_control == 2) / (np.arange(num_frames) + 1) * 100

    plt.figure(figsize=(12, 6))
    plt.plot(time, team1_possession, color="red", label="Team 1")
    plt.plot(time, team2_possession, color="blue", label="Team 2")
    plt.xlabel("Frame")
    plt.ylabel("Possession (%)")
    plt.title("Ball Possession Over Time")
    plt.legend()
    plt.savefig(os.path.join(save_dir, "possession_over_time.png"))
    plt.close()


def plot_players_speed_distance(tracks, save_dir="figures"):
    """
    绘制球员跑动距离和瞬时速度折线图
    依赖 SpeedAndDistance_Estimator 已经在 tracks 中添加 'speed' 和 'distance'
    """
    os.makedirs(save_dir, exist_ok=True)

    num_frames = len(tracks["players"])
    time = np.arange(num_frames)

    # 距离
    plt.figure(figsize=(12, 6))
    for player_id in set.union(*[set(f.keys()) for f in tracks["players"]]):
        distances = [
            tracks["players"][f].get(player_id, {}).get("distance", np.nan)
            for f in range(num_frames)
        ]
        plt.plot(time, distances, label=f"Player {player_id}")
    plt.xlabel("Frame")
    plt.ylabel("Distance (m)")
    plt.title("Players Running Distance Over Time")
    plt.legend()
    plt.savefig(os.path.join(save_dir, "players_distance.png"))
    plt.close()

    # 速度
    plt.figure(figsize=(12, 6))
    for player_id in set.union(*[set(f.keys()) for f in tracks["players"]]):
        speeds = [
            tracks["players"][f].get(player_id, {}).get("speed", np.nan)
            for f in range(num_frames)
        ]
        plt.plot(time, speeds, label=f"Player {player_id}")
    plt.xlabel("Frame")
    plt.ylabel("Speed (km/h)")
    plt.title("Players Instant Speed Over Time")
    plt.legend()
    plt.savefig(os.path.join(save_dir, "players_speed.png"))
    plt.close()
