import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

try:
    import config
    # 尝试使用 config 中的默认值
    DEFAULT_OUTPUT = config.DEFAULT_CALIBRATION_OUTPUT
except ImportError:
    DEFAULT_OUTPUT = "calibration.json"

from camera import Camera
from soccer import HalfPitchConfiguration, QingYuanConfiguration, HalfQingYuanConfiguration, WuChuanConfiguration, HalfWuChuanConfiguration

PITCH_CONFIGS = {
    "standard_half": HalfPitchConfiguration,
    "qingyuan_full": QingYuanConfiguration,
    "qingyuan_half": HalfQingYuanConfiguration,
    "wuchuan_full": WuChuanConfiguration,
    "wuchuan_half": HalfWuChuanConfiguration
}

def run_command(cmd, description):
    """执行系统命令并打印输出"""
    print(f"\n[执行] {description}...")
    # print(f"CMD: {' '.join(cmd)}") # 调试用，生产环境可注释
    try:
        # Windows下 shutil.which 检查命令是否存在
        if sys.platform == 'win32' and cmd[0] == 'rsync':
            # 检查是否有 rsync，如果没有则警告
            if not shutil.which("rsync"):
                print("警告: 未找到 rsync 命令，尝试使用 scp...")
                return False

        result = subprocess.run(cmd, check=True, text=True, capture_output=True)
        print(f"成功: {result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"失败: {e.stderr}")
        return False
    except Exception as e:
        print(f"错误: {e}")
        return False

def transfer_file(local_path, remote_path, host, user, port, key_path, use_rsync=True):
    """传输文件到服务器"""
    # 构造目标字符串: user@host:path
    remote_dest = f"{user}@{host}:{remote_path}"
    
    if use_rsync:
        # rsync -avz -e "ssh -p 2222 -i key" local remote
        ssh_opt = f"ssh -p {port} -i \"{key_path}\""
        # 注意: Windows 路径传给 rsync 可能需要处理 (如 /cygdrive/d/...), 
        # 这里简单起见，如果 rsync 失败推荐 fallback 到 scp
        cmd = ["rsync", "-avz", "-e", ssh_opt, local_path, remote_dest]
        if run_command(cmd, "正在使用 rsync 传输标定文件"):
            return True
        print("rsync 传输失败，尝试切换到 scp...")
    
    # Fallback to SCP (Windows 原生支持更好)
    # scp -P 2222 -i key local user@host:path
    cmd = ["scp", "-P", str(port), "-i", key_path, local_path, remote_dest]
    return run_command(cmd, "正在使用 scp 传输标定文件")

def ensure_remote_dir(host, user, port, key_path, remote_dir):
    """确保远程目录存在 (mkdir -p)"""
    ssh_cmd = [
        "ssh", 
        "-p", str(port), 
        "-i", key_path, 
        f"{user}@{host}", 
        f"mkdir -p {remote_dir}"
    ]
    return run_command(ssh_cmd, f"正在检查/创建远程目录: {remote_dir}")

def trigger_remote_run(host, user, port, key_path, remote_project_root, config_file_name, server_video_path):
    """远程触发服务器上的计算脚本"""
    # 构造远程命令
    # 使用 bash -lc (Login Shell) 确保加载 ~/.bashrc 或 /etc/profile
    # 尝试激活 Conda 环境 'PGYfootball'
    
    env_name = "PGY"
    
    # 命令链：
    # 1. 尝试激活 conda 环境 (兼容 source activate 和 conda activate)
    # 2. cd 到项目目录
    # 3. 运行 python
    remote_cmd_str = (
        f"source ~/.bashrc; "
        f"if [ -f ~/miniconda3/bin/activate ]; then source ~/miniconda3/bin/activate {env_name}; "
        f"elif [ -f ~/anaconda3/bin/activate ]; then source ~/anaconda3/bin/activate {env_name}; "
        f"else conda activate {env_name} || source activate {env_name}; fi; "
        f"cd {remote_project_root} && "
        f"python server_runner.py --config configs/{config_file_name} --video-override \"{server_video_path}\""
    )
    
    ssh_cmd = [
        "ssh", 
        "-p", str(port), 
        "-i", key_path, 
        f"{user}@{host}", 
        f"bash -lc '{remote_cmd_str}'"
    ]
    
    return run_command(ssh_cmd, "正在触发服务器端计算任务")

def main():
    parser = argparse.ArgumentParser(description="本地标定 & 自动传输工具")
    
    # 基础标定参数
    parser.add_argument("--video", type=str, required=True, help="本地视频文件路径")
    parser.add_argument("--id", type=int, default=1, help="相机ID")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT, help="输出JSON文件名")
    
    # 服务器传输参数
    parser.add_argument("--server-video-path", type=str, required=True, help="视频在服务器上的绝对路径")
    parser.add_argument("--host", type=str, help="服务器IP")
    parser.add_argument("--port", type=str, default="22", help="SSH端口")
    parser.add_argument("--user", type=str, help="SSH用户名")
    parser.add_argument("--key", type=str, help="SSH私钥路径")
    parser.add_argument("--remote-root", type=str, help="服务器项目根目录 (例如 /data/shx/PGYfootball)")
    
    parser.add_argument("--no-trigger", action="store_true", help="仅传输不触发计算")
    parser.add_argument("--pitch-type", type=str, default="qingyuan_half", choices=PITCH_CONFIGS.keys(), help="球场配置类型")
    
    args = parser.parse_args()
    
    # 1. 本地标定
    if not os.path.exists(args.video):
        print(f"错误: 本地视频不存在 {args.video}")
        return

    pitch_class = PITCH_CONFIGS[args.pitch_type]
    config_pitch = pitch_class()
    print(f"=== 步骤 1/3: 本地标定 (相机 {args.id}) [{args.pitch_type}] ===")
    
    # 使用独立的标定目录
    calibration_dir = Path("local_calibration") / f"camera{args.id}"
    calibration_dir.mkdir(parents=True, exist_ok=True)
    print(f"标定文件将保存至: {calibration_dir}")
    
    try:
        # 初始化相机 (触发交互标定，包括畸变校正)
        camera = Camera(
            video_path=args.video, 
            pitch_config=config_pitch, 
            camera_id=args.id,
            calibration_dir=str(calibration_dir)
        )
        homography = camera.homography_matrix
        
        if homography is None:
            print("标定失败。")
            return
            
        # 训练区域标定已在 Camera 初始化时完成
        # Camera.__init__ 会自动调用 train_area.define_from_video() 和 define_from_pitch_map()
        region_points = camera.train_area.region_points
        
        if not region_points:
            print("警告: 未设置训练区域，将使用全场覆盖。")
        else:
            print(f"训练区域已设置: {len(region_points)} 个点")

        # 生成概览信息文件（可选，主要用于调试和查看）
        summary_data = {
            "camera_id": args.id,
            "video_path": args.server_video_path,
            "image_shape": list(camera.frame_shape),
            "calibration_dir": str(calibration_dir),
            "enable_distortion_correction": camera.enable_distortion_correction,
            "has_distortion_params": camera.distortion_params is not None,
            "train_area_points": len(region_points),
            "note": "完整标定数据在 calibration_dir 目录中"
        }
        
        # 保存概览文件
        summary_path = calibration_dir / "calibration_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=4, ensure_ascii=False)
        print(f"\n✅ 标定完成！")
        print(f"   - 标定目录: {calibration_dir}")
        print(f"   - 概览文件: {summary_path}")
        print(f"\n📁 标定文件列表:")
        for file in sorted(calibration_dir.glob("*")):
            print(f"   - {file.name}")
        
    except Exception as e:
        print(f"标定过程出错: {e}")
        import traceback
        traceback.print_exc()
        return

    # 2. 自动传输（需要实现）
    print(f"\n💡 传输到服务器:")
    print(f"   使用以下命令传输标定目录:")
    if args.host and args.user and args.remote_root:
        remote_store_dir = f"{args.remote_root}/store/camera{args.id}"
        print(f"\n   # 方式 1: 使用 rsync (推荐)")
        print(f"   rsync -avz -e \"ssh -p {args.port}\" {calibration_dir}/ {args.user}@{args.host}:{remote_store_dir}/")
        print(f"\n   # 方式 2: 使用 scp")
        print(f"   scp -P {args.port} -r {calibration_dir}/* {args.user}@{args.host}:{remote_store_dir}/")
        print(f"\n   服务器端使用:")
        print(f"   camera = Camera(")
        print(f"       video_path='{args.server_video_path}',")
        print(f"       pitch_config=config,")
        print(f"       camera_id={args.id},")
        print(f"       calibration_dir='store/camera{args.id}'")
        print(f"   )")
    else:
        print("   提示: 未提供服务器信息，无法生成传输命令。")
        print("   使用 --host, --user, --remote-root 参数指定服务器信息。")
    
    # TODO: 自动传输实现
    # if args.host and args.user and args.remote_root:
    #     print(f"\n=== 步骤 2/3: 传输文件到服务器 ({args.host}) ===")
    #     remote_store_dir = f"{args.remote_root}/store/camera{args.id}"
    #     key_abs = os.path.abspath(args.key) if args.key else None
    #
    #     # 确保远程目录存在
    #     ensure_remote_dir(args.host, args.user, args.port, key_abs, remote_store_dir)
    #
    #     # 传输整个标定目录
    #     success = transfer_directory(
    #         local_path=str(calibration_dir) + "/",
    #         remote_path=remote_store_dir,
    #         host=args.host,
    #         user=args.user,
    #         port=args.port,
    #         key_path=key_abs
    #     )

if __name__ == "__main__":
    import shutil # 延迟导入
    main()
