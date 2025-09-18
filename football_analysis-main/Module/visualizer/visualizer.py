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
    原始版本：所有球员一起绘制
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


def plot_players_speed_distance_by_team(tracks, save_dir="figures"):
    """
    新版本：按队伍分别绘制
    依赖 team_assigner 给 tracks['players'][f][player_id]['team'] 打了标签
    """
    os.makedirs(save_dir, exist_ok=True)
    num_frames = len(tracks["players"])
    time = np.arange(num_frames)

    # 收集所有球员 id 和队伍
    player_team_map = {}
    for frame in tracks["players"]:
        for pid, pdata in frame.items():
            if "team" in pdata:
                player_team_map[pid] = pdata["team"]

    for team_id in [1, 2]:
        # 距离
        plt.figure(figsize=(12, 6))
        for pid, team in player_team_map.items():
            if team != team_id:
                continue
            distances = [
                tracks["players"][f].get(pid, {}).get("distance", np.nan)
                for f in range(num_frames)
            ]
            plt.plot(time, distances, label=f"Player {pid}")
        plt.xlabel("Frame")
        plt.ylabel("Distance (m)")
        plt.title(f"Team {team_id} Players Running Distance Over Time")
        plt.legend()
        plt.savefig(os.path.join(save_dir, f"team{team_id}_players_distance.png"))
        plt.close()

        # 速度
        plt.figure(figsize=(12, 6))
        for pid, team in player_team_map.items():
            if team != team_id:
                continue
            speeds = [
                tracks["players"][f].get(pid, {}).get("speed", np.nan)
                for f in range(num_frames)
            ]
            plt.plot(time, speeds, label=f"Player {pid}")
        plt.xlabel("Frame")
        plt.ylabel("Speed (km/h)")
        plt.title(f"Team {team_id} Players Instant Speed Over Time")
        plt.legend()
        plt.savefig(os.path.join(save_dir, f"team{team_id}_players_speed.png"))
        plt.close()
