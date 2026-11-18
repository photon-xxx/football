import subprocess
import sys
import os
import argparse

# ================= 配置区域 =================

# 服务器配置列表
# 仓库里不保存内网地址与私钥路径，改由环境变量注入：
#   MCS_SERVER_HOST / MCS_SERVER_PORT / MCS_SERVER_USER / MCS_SERVER_KEY
#   MCS_SERVER_ROOT / MCS_SERVER_VIDEO_DIR
SERVERS = {
    "remote": {
        "host": os.getenv("MCS_SERVER_HOST", "127.0.0.1"),
        "port": os.getenv("MCS_SERVER_PORT", "22"),
        "user": os.getenv("MCS_SERVER_USER", "user"),
        "key": os.path.expanduser(os.getenv("MCS_SERVER_KEY", "~/.ssh/id_rsa")),
        "remote_root": os.getenv("MCS_SERVER_ROOT", "/data/football"),
        "remote_video_dir": os.getenv("MCS_SERVER_VIDEO_DIR", "/data/football/videos"),
    },
}

# 默认使用的服务器
DEFAULT_SERVER = "remote"

# 本地视频路径
LOCAL_VIDEO = os.getenv("MCS_LOCAL_VIDEO", "videos/sample.mp4")

# 任务配置
CAMERA_ID = 1
OUTPUT_JSON = "cam1_config.json"  # 仅用于兼容性，实际输出在 local_calibration/camera{id}/ 目录
PITCH_TYPE = "qingyuan_half"  # 可选: standard_half, qingyuan_full, qingyuan_half

# ===========================================

def main():
    parser = argparse.ArgumentParser(description="自动标定运行脚本")
    parser.add_argument("--server", type=str, default=DEFAULT_SERVER, choices=SERVERS.keys(), help="选择要使用的服务器")
    parser.add_argument("--video", type=str, default=LOCAL_VIDEO, help="本地视频路径")
    parser.add_argument("--id", type=int, default=CAMERA_ID, help="相机ID")
    parser.add_argument("--pitch", type=str, default=PITCH_TYPE, help="球场类型 (standard_half, qingyuan_full, qingyuan_half)")
    
    args = parser.parse_args()
    
    # 获取选中服务器的配置
    if args.server not in SERVERS:
        print(f"错误: 未知的服务器 '{args.server}'")
        return
        
    srv = SERVERS[args.server]
    video_name = os.path.basename(args.video)
    # 拼接服务器上的完整视频路径
    remote_video_path = f"{srv['remote_video_dir']}/{video_name}".replace("\\", "/")
    
    print(f"=== 启动全自动标定流程 (含畸变校正) ===")
    print(f"目标服务器: {args.server} ({srv['host']}:{srv['port']})")
    print(f"本地视频: {args.video}")
    print(f"远程视频路径: {remote_video_path}")
    print(f"球场配置: {args.pitch}")
    print(f"标定输出: local_calibration/camera{args.id}/")
    print("-" * 40)

    # 构造命令
    cmd = [
        sys.executable, "local_calibrator.py",
        "--video", args.video,
        "--id", str(args.id),
        "--output", OUTPUT_JSON,
        "--server-video-path", remote_video_path,
        "--host", srv["host"],
        "--port", srv["port"],
        "--user", srv["user"],
        "--key", srv["key"],
        "--remote-root", srv["remote_root"],
        "--pitch-type", args.pitch,
        "--no-trigger" # 暂时禁止自动触发计算，只传输文件
    ]
    
    print("\n📋 标定流程说明:")
    print("1️⃣  球场关键点标定")
    print("   - 点击球场关键点 (如 01, 06, B6, T6 等)")
    print("   - 按 's' 保存")
    print("\n2️⃣  横向线标定 (畸变校正)")
    print("   - 点击标注多条横向平行线")
    print("   - 每条线完成后按 'n'，全部完成按 's'")
    print("\n3️⃣  纵向线标定 (畸变校正)")
    print("   - 点击标注多条纵向平行线")
    print("   - 每条线完成后按 'n'，全部完成按 's'")
    print("\n4️⃣  训练区域标定 - 视频画面")
    print("   - 标注自定义点（可选，直接按 's' 跳过）")
    print("\n5️⃣  训练区域标定 - 球场俯视图")
    print("   - 控制台输入点序列 (如: 01, 06, B6, T6)")
    print("\n6️⃣  完成后提示传输命令")
    print("-" * 40)
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError:
        print("\n流程执行出错。")
    except KeyboardInterrupt:
        print("\n用户取消。")

if __name__ == "__main__":
    main()
