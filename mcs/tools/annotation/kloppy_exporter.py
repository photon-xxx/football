"""
Kloppy / UnravelSports 数据导出器

将 MCS 标注数据 (annotations.json + camera_info.json) 转换为：
1. Kloppy TrackingDataset 对象
2. UnravelSports 兼容的 Polars DataFrame

复用项目已有模块：
- tools/transform.py  → video_pixel_to_pitch (含去畸变)
- soccer.py            → 各球场 Configuration 类 (length/width 用于归一化)
- tools/pitch_state_visualizer.py → interpolate_frame (可选插值)

用法：
    python tools/kloppy_exporter.py \\
        --annotations path/to/annotations.json \\
        --camera-info path/to/camera_info.json \\
        [--distortion path/to/distortion_params.json] \\
        [--output path/to/output.parquet] \\
        [--interpolate]
"""

import sys
import json
import argparse
import numpy as np
from pathlib import Path
from datetime import timedelta
from typing import Dict, List, Optional, Tuple, Any

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "tools"))

import soccer
from tools.transform import video_pixel_to_pitch

# ---------------------------------------------------------------------------
# 球场配置名称 → 类的映射
# ---------------------------------------------------------------------------
PITCH_CONFIG_MAP = {
    "SoccerPitchConfiguration": soccer.SoccerPitchConfiguration,
    "HalfPitchConfiguration": soccer.HalfPitchConfiguration,
    "QingYuanConfiguration": soccer.QingYuanConfiguration,
    "HalfQingYuanConfiguration": soccer.HalfQingYuanConfiguration,
    "WuChuanConfiguration": soccer.WuChuanConfiguration,
    "CUMTBPitchConfiguration": soccer.CUMTBPitchConfiguration,
    "HalfWuChuanConfiguration": soccer.HalfWuChuanConfiguration,
}


def resolve_pitch_config(name: str):
    cls = PITCH_CONFIG_MAP.get(name)
    if cls is None:
        raise ValueError(
            f"Unknown pitch config '{name}'. "
            f"Available: {list(PITCH_CONFIG_MAP.keys())}"
        )
    return cls()


# ---------------------------------------------------------------------------
# 坐标转换管道  pixel → pitch (cm) → normalized [0,1]
# ---------------------------------------------------------------------------

def load_distortion(camera_info: dict, distortion_path: Optional[str] = None):
    """从 camera_info 和可选的 distortion_params.json 加载畸变参数"""
    distortion_params = None
    distortion_info = None

    if not camera_info.get("distortion_correction_enabled", False):
        return distortion_params, distortion_info

    if distortion_path and Path(distortion_path).exists():
        with open(distortion_path, "r", encoding="utf-8") as f:
            dp = json.load(f)
        distortion_params = np.array(dp["params"], dtype=np.float64)
        distortion_info = {
            k: v for k, v in dp.items() if k != "params"
        }
    return distortion_params, distortion_info


def pixels_to_normalized(
    pixel_coords: np.ndarray,
    H: np.ndarray,
    pitch_length: float,
    pitch_width: float,
    distortion_params=None,
    distortion_info=None,
) -> np.ndarray:
    """
    pixel coords (N,2) → pitch coords (cm) → normalized [0,1] coords (N,2)

    Uses tools.transform.video_pixel_to_pitch for the pixel→pitch step
    (handles distortion correction internally).
    """
    if pixel_coords.size == 0:
        return np.empty((0, 2), dtype=np.float64)

    pitch = video_pixel_to_pitch(
        pixel_coords,
        H,
        distortion_params=distortion_params,
        distortion_info=distortion_info,
        enable_k3=True,
    )
    normalized = np.column_stack([
        pitch[:, 0] / pitch_length,
        pitch[:, 1] / pitch_width,
    ])
    return np.clip(normalized, 0.0, 1.0)


# ---------------------------------------------------------------------------
# 标注 JSON 解析  →  归一化关键帧
# ---------------------------------------------------------------------------

def _extract_pixel_points(frame_data: dict) -> Tuple[List[str], List[Optional[Tuple[float, float]]]]:
    """
    从单帧标注中提取所有目标的 key 和像素坐标。
    兼容旧格式 (o1: {x, y}) 和新格式 (p1: {foot: {x,y}, head: {x,y}})。
    foot 点优先用于位置追踪。
    """
    keys: List[str] = []
    coords: List[Optional[Tuple[float, float]]] = []

    for team in ("t1", "t2"):
        team_data = frame_data.get(team)
        if team_data is None:
            continue
        if isinstance(team_data, dict):
            for sub_group_key in ("players", "goalkeepers"):
                sub = team_data.get(sub_group_key)
                if isinstance(sub, dict):
                    for obj_id, val in sub.items():
                        key = f"{team}_{obj_id}"
                        keys.append(key)
                        if val is None:
                            coords.append(None)
                        elif "foot" in val:
                            foot = val["foot"]
                            coords.append((foot["x"], foot["y"]) if foot else None)
                        elif "x" in val:
                            coords.append((val["x"], val["y"]))
                        else:
                            coords.append(None)
                    continue
            for obj_id, val in team_data.items():
                if obj_id in ("players", "goalkeepers"):
                    continue
                key = f"{team}_{obj_id}"
                keys.append(key)
                if val is None:
                    coords.append(None)
                elif "foot" in val:
                    foot = val["foot"]
                    coords.append((foot["x"], foot["y"]) if foot else None)
                elif "x" in val:
                    coords.append((val["x"], val["y"]))
                else:
                    coords.append(None)

    ball = frame_data.get("ball")
    keys.append("ball")
    if ball is None:
        coords.append(None)
    elif "x" in ball:
        coords.append((ball["x"], ball["y"]))
    else:
        coords.append(None)

    test = frame_data.get("test")
    if isinstance(test, dict):
        for obj_id, val in test.items():
            key = f"test_{obj_id}"
            keys.append(key)
            if val is None:
                coords.append(None)
            elif "foot" in val:
                foot = val["foot"]
                coords.append((foot["x"], foot["y"]) if foot else None)
            elif "x" in val:
                coords.append((val["x"], val["y"]))
            else:
                coords.append(None)

    return keys, coords


def build_normalized_keyframes(
    annotations: dict,
    H: np.ndarray,
    pitch_length: float,
    pitch_width: float,
    distortion_params=None,
    distortion_info=None,
) -> Tuple[Dict[int, Dict[str, Optional[Tuple[float, float]]]], List[str]]:
    """
    Parse the annotation JSON and convert every frame's pixel coordinates
    into normalized [0,1] pitch coordinates using video_pixel_to_pitch.

    Returns:
        keyframes: {frame_idx: {player_key: (norm_x, norm_y) | None, ...}}
        all_keys: sorted union of all player/ball keys across frames
    """
    keyframes: Dict[int, Dict[str, Optional[Tuple[float, float]]]] = {}
    all_keys_set: set = set()

    for frame_str, frame_data in annotations["frames"].items():
        frame_idx = int(frame_str)
        keys, coords = _extract_pixel_points(frame_data)
        all_keys_set.update(keys)

        valid_indices = [i for i, c in enumerate(coords) if c is not None]
        if not valid_indices:
            keyframes[frame_idx] = {k: None for k in keys}
            continue

        px_array = np.array(
            [coords[i] for i in valid_indices], dtype=np.float32
        )
        norm_array = pixels_to_normalized(
            px_array, H, pitch_length, pitch_width,
            distortion_params, distortion_info,
        )

        kf: Dict[str, Optional[Tuple[float, float]]] = {}
        norm_idx = 0
        for i, k in enumerate(keys):
            if coords[i] is None:
                kf[k] = None
            else:
                kf[k] = (float(norm_array[norm_idx, 0]), float(norm_array[norm_idx, 1]))
                norm_idx += 1
        keyframes[frame_idx] = kf

    all_keys = sorted(all_keys_set)
    return keyframes, all_keys


# ---------------------------------------------------------------------------
# Kloppy TrackingDataset 构建
# ---------------------------------------------------------------------------

def _classify_key(key: str) -> Tuple[str, str, bool]:
    """
    Classify a player key into (team_id, player_id, is_goalkeeper).
    Examples:
        't1_o1'   -> ('t1', 't1_o1', False)
        't1_gk1'  -> ('t1', 't1_gk1', True)
        't2_p3'   -> ('t2', 't2_p3', False)
        'ball'    -> ('ball', 'ball', False)
        'test_o1' -> ('test', 'test_o1', False)
    """
    if key == "ball":
        return "ball", "ball", False
    parts = key.split("_", 1)
    team = parts[0]
    obj = parts[1] if len(parts) > 1 else key
    is_gk = obj.startswith("gk")
    return team, key, is_gk


def build_kloppy_dataset(
    keyframes: Dict[int, Dict[str, Optional[Tuple[float, float]]]],
    all_keys: List[str],
    meta: dict,
    fps: float,
):
    """
    Construct a kloppy TrackingDataset from normalized keyframes.
    """
    from kloppy.domain import (
        TrackingDataset,
        Frame,
        Point,
        Point3D,
        Player,
        Team,
        Ground,
        Period,
        Orientation,
        Provider,
        DatasetFlag,
        BallState,
        Metadata,
        PlayerData,
        PositionType,
    )
    from kloppy.domain.models.pitch import (
        PitchDimensions,
        Dimension,
        Unit,
    )
    from kloppy.domain import KloppyCoordinateSystem

    team1_name = meta.get("team1_name") or "Team1"
    team2_name = meta.get("team2_name") or "Team2"

    home_team = Team(
        team_id="t1",
        name=team1_name,
        ground=Ground.HOME,
    )
    away_team = Team(
        team_id="t2",
        name=team2_name,
        ground=Ground.AWAY,
    )
    teams = [home_team, away_team]

    player_objs: Dict[str, Player] = {}
    jersey_counter = {"t1": 1, "t2": 1, "test": 100}

    for key in all_keys:
        if key == "ball":
            continue
        team_id, player_id, is_gk = _classify_key(key)
        if team_id in ("t1", "t2"):
            team_obj = home_team if team_id == "t1" else away_team
        else:
            team_obj = home_team

        jno = jersey_counter.get(team_id, 100)
        jersey_counter[team_id] = jno + 1

        position = PositionType.Goalkeeper if is_gk else PositionType.Unknown

        player_objs[key] = Player(
            player_id=player_id,
            team=team_obj,
            jersey_no=jno,
            name=player_id,
            starting_position=position,
        )

    for t in teams:
        t.players = [p for p in player_objs.values() if p.team == t]

    period_id = meta.get("period_id") or 1
    period = Period(id=period_id, start_timestamp=timedelta(0), end_timestamp=None)

    sampling_interval = meta.get("sampling_interval", 1)
    effective_fps = fps / sampling_interval

    coord_system = KloppyCoordinateSystem(
        pitch_length=1.0,
        pitch_width=1.0,
    )

    pitch_dims = PitchDimensions(
        x_dim=Dimension(min=0.0, max=1.0),
        y_dim=Dimension(min=0.0, max=1.0),
        standardized=False,
        unit=Unit.NORMED,
        goal_width=0.0,
        goal_height=None,
        six_yard_width=0.0,
        six_yard_length=0.0,
        penalty_area_width=0.0,
        penalty_area_length=0.0,
        circle_radius=0.0,
        corner_radius=0.0,
        penalty_spot_distance=0.0,
        penalty_arc_radius=0.0,
    )

    frames: List[Frame] = []
    sorted_frame_ids = sorted(keyframes.keys())

    for seq_idx, frame_idx in enumerate(sorted_frame_ids):
        kf = keyframes[frame_idx]

        ball_pt = Point3D(x=float("nan"), y=float("nan"), z=None)
        ball_val = kf.get("ball")
        if ball_val is not None:
            ball_pt = Point3D(x=ball_val[0], y=ball_val[1], z=0.0)

        players_data: Dict[Player, PlayerData] = {}
        for key, player_obj in player_objs.items():
            val = kf.get(key)
            if val is not None:
                players_data[player_obj] = PlayerData(
                    coordinates=Point(x=val[0], y=val[1]),
                )

        frame = Frame(
            frame_id=frame_idx,
            timestamp=timedelta(seconds=frame_idx / fps),
            ball_coordinates=ball_pt,
            players_data=players_data,
            period=period,
            ball_state=BallState.ALIVE,
            ball_owning_team=None,
            other_data={},
            statistics=[],
        )
        frames.append(frame)

    metadata = Metadata(
        teams=teams,
        periods=[period],
        pitch_dimensions=pitch_dims,
        coordinate_system=coord_system,
        frame_rate=effective_fps,
        orientation=Orientation.HOME_AWAY,
        provider=None,
        flags=~(DatasetFlag.BALL_STATE | DatasetFlag.BALL_OWNING_TEAM),
        score=None,
    )

    dataset = TrackingDataset(
        metadata=metadata,
        records=frames,
    )
    return dataset


# ---------------------------------------------------------------------------
# 速度 / 加速度计算 (帧间差分)
# ---------------------------------------------------------------------------

def compute_velocities(
    keyframes: Dict[int, Dict[str, Optional[Tuple[float, float]]]],
    all_keys: List[str],
    fps: float,
    sampling_interval: int = 1,
) -> Dict[int, Dict[str, Dict[str, float]]]:
    """
    Compute per-frame velocity and acceleration for each tracked object.
    Returns: {frame_idx: {key: {'vx','vy','v','ax','ay','a'}}}

    Velocity units: normalized_units / second
    """
    sorted_frames = sorted(keyframes.keys())
    dt = sampling_interval / fps if fps > 0 else 1.0

    prev_pos: Dict[str, Optional[Tuple[float, float]]] = {}
    prev_vel: Dict[str, Optional[Tuple[float, float]]] = {}
    result: Dict[int, Dict[str, Dict[str, float]]] = {}

    for fi, frame_idx in enumerate(sorted_frames):
        kf = keyframes[frame_idx]
        frame_vel: Dict[str, Dict[str, float]] = {}

        for key in all_keys:
            pos = kf.get(key)
            zero = {"vx": 0.0, "vy": 0.0, "v": 0.0, "ax": 0.0, "ay": 0.0, "a": 0.0}

            if pos is None or prev_pos.get(key) is None:
                frame_vel[key] = zero
                prev_vel[key] = None
            else:
                pp = prev_pos[key]
                vx = (pos[0] - pp[0]) / dt
                vy = (pos[1] - pp[1]) / dt
                v = np.sqrt(vx ** 2 + vy ** 2)

                pv = prev_vel.get(key)
                if pv is not None:
                    ax = (vx - pv[0]) / dt
                    ay = (vy - pv[1]) / dt
                    a = np.sqrt(ax ** 2 + ay ** 2)
                else:
                    ax = ay = a = 0.0

                frame_vel[key] = {
                    "vx": vx, "vy": vy, "v": v,
                    "ax": ax, "ay": ay, "a": a,
                }
                prev_vel[key] = (vx, vy)

            prev_pos[key] = pos

        result[frame_idx] = frame_vel

    return result


# ---------------------------------------------------------------------------
# Polars DataFrame 导出 (兼容 UnravelSports)
# ---------------------------------------------------------------------------

def to_polars_dataframe(
    keyframes: Dict[int, Dict[str, Optional[Tuple[float, float]]]],
    all_keys: List[str],
    velocities: Dict[int, Dict[str, Dict[str, float]]],
    meta: dict,
    fps: float,
):
    """
    Build a Polars DataFrame matching the UnravelSports / Kloppy schema.
    Columns: period_id, timestamp, frame_id, ball_state, id, x, y, z,
             team_id, position_name, game_id, vx, vy, vz, v, ax, ay, az, a,
             ball_owning_team_id, is_ball_carrier
    """
    import polars as pl

    game_id = meta.get("game_id") or "match_001"
    period_id = meta.get("period_id") or 1

    rows: List[Dict[str, Any]] = []
    sorted_frames = sorted(keyframes.keys())

    for frame_idx in sorted_frames:
        kf = keyframes[frame_idx]
        vel_frame = velocities.get(frame_idx, {})
        timestamp = timedelta(seconds=frame_idx / fps)

        ball_pos = kf.get("ball")

        ball_carrier_key = None
        if ball_pos is not None:
            min_dist = float("inf")
            for key in all_keys:
                if key == "ball":
                    continue
                pos = kf.get(key)
                if pos is None:
                    continue
                d = np.sqrt((pos[0] - ball_pos[0]) ** 2 + (pos[1] - ball_pos[1]) ** 2)
                if d < min_dist:
                    min_dist = d
                    ball_carrier_key = key

        ball_owning_team = None
        if ball_carrier_key:
            team_id_carrier, _, _ = _classify_key(ball_carrier_key)
            ball_owning_team = team_id_carrier if team_id_carrier in ("t1", "t2") else None

        for key in all_keys:
            if key == "ball":
                continue
            pos = kf.get(key)
            if pos is None:
                continue

            team_id, player_id, is_gk = _classify_key(key)
            vel = vel_frame.get(key, {})

            rows.append({
                "period_id": period_id,
                "timestamp": timestamp,
                "frame_id": frame_idx,
                "ball_state": "alive",
                "id": player_id,
                "x": pos[0],
                "y": pos[1],
                "z": 0.0,
                "team_id": team_id,
                "position_name": "GK" if is_gk else "UNK",
                "game_id": game_id,
                "vx": vel.get("vx", 0.0),
                "vy": vel.get("vy", 0.0),
                "vz": 0.0,
                "v": vel.get("v", 0.0),
                "ax": vel.get("ax", 0.0),
                "ay": vel.get("ay", 0.0),
                "az": 0.0,
                "a": vel.get("a", 0.0),
                "ball_owning_team_id": ball_owning_team,
                "is_ball_carrier": (key == ball_carrier_key),
            })

    if not rows:
        return pl.DataFrame()

    df = pl.DataFrame(rows)
    return df


# ---------------------------------------------------------------------------
# CLI 主入口
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="MCS → Kloppy/UnravelSports exporter")
    parser.add_argument("--annotations", required=True, help="annotations.json path")
    parser.add_argument("--camera-info", required=True, help="camera_info.json path")
    parser.add_argument("--distortion", default=None, help="distortion_params.json path (optional)")
    parser.add_argument("--output", default=None, help="output .parquet path (optional)")
    parser.add_argument("--interpolate", action="store_true", help="interpolate between sampled frames")
    args = parser.parse_args()

    with open(args.annotations, "r", encoding="utf-8") as f:
        annotations = json.load(f)
    with open(args.camera_info, "r", encoding="utf-8") as f:
        camera_info = json.load(f)

    meta = annotations["meta"]
    fps = meta["fps"]
    sampling_interval = meta.get("sampling_interval", 1)

    H = np.array(camera_info["homography_matrix"], dtype=np.float64)

    distortion_params, distortion_info = load_distortion(camera_info, args.distortion)

    config_name = meta.get("pitch_config") or "SoccerPitchConfiguration"
    pitch_config = resolve_pitch_config(config_name)
    print(f"[Exporter] Pitch config: {config_name} "
          f"(length={pitch_config.length} cm, width={pitch_config.width} cm)")

    print("[Exporter] Converting pixel → pitch → normalized coordinates ...")
    keyframes, all_keys = build_normalized_keyframes(
        annotations, H,
        pitch_config.length, pitch_config.width,
        distortion_params, distortion_info,
    )
    print(f"  {len(keyframes)} keyframes, {len(all_keys)} tracked objects")

    if args.interpolate:
        from tools.annotation.pitch_state_visualizer import interpolate_frame, get_frame_objects
        print("[Exporter] Interpolating to full frame rate ...")
        sorted_indices = sorted(keyframes.keys())
        if len(sorted_indices) >= 2:
            start, end = sorted_indices[0], sorted_indices[-1]
            full_keyframes: Dict[int, Dict[str, Optional[Tuple[float, float]]]] = {}
            for fi in range(start, end + 1):
                full_keyframes[fi] = get_frame_objects(keyframes, sorted_indices, fi, all_keys)
            keyframes = full_keyframes
            sampling_interval = 1
            print(f"  Interpolated to {len(keyframes)} frames")

    print("[Exporter] Computing velocities ...")
    velocities = compute_velocities(keyframes, all_keys, fps, sampling_interval)

    print("[Exporter] Building Kloppy TrackingDataset ...")
    dataset = build_kloppy_dataset(keyframes, all_keys, meta, fps)
    print(f"  TrackingDataset: {len(dataset.records)} frames, "
          f"frame_rate={dataset.metadata.frame_rate:.2f}")

    print("[Exporter] Building Polars DataFrame ...")
    df = to_polars_dataframe(keyframes, all_keys, velocities, meta, fps)
    print(f"  DataFrame shape: {df.shape}")
    print(f"  Columns: {df.columns}")

    output_path = args.output
    if output_path is None:
        ann_path = Path(args.annotations)
        output_path = str(ann_path.parent / (ann_path.stem + "_kloppy.parquet"))

    df.write_parquet(output_path)
    print(f"\n[Exporter] Saved Polars DataFrame → {output_path}")

    print("\n[Exporter] Quick verification:")
    print(df.head(5))

    print("\n--- Usage example ---")
    print("from tools.annotation.kloppy_exporter import build_kloppy_dataset, to_polars_dataframe")
    print("# Or directly use the Kloppy dataset:")
    print("# df = dataset.to_df(engine='polars')")
    print("# from unravel.soccer import KloppyPolarsDataset, EFPI")
    print("# kloppy_polars = KloppyPolarsDataset(kloppy_dataset=dataset)")
    print("# model = EFPI(dataset=kloppy_polars)")
    print("# model.fit(formations=None, every='frame')")


if __name__ == "__main__":
    main()
