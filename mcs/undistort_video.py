"""
读取畸变参数，对视频去除广角（桶形）畸变。

去畸变映射构建原理（cv2.remap: dst(u,v) = src(map_x, map_y)）：
- 输出图像为理想（无畸变）坐标系。
- 对每个输出像素（理想坐标），用正向畸变模型 my_distort_point 计算其在
  原始畸变视频中的源像素位置，作为采样坐标。
- 这正是标准 OpenCV initUndistortRectifyMap 的做法。

标定得到的 params([k1,k2,p1,p2,k3]) 满足: undistort_point(畸变点)=理想点，
my_distort_point 是其逆运算，用来构建去畸变重采样表。

输出编码：通过管道传给 FFmpeg，使用 H.264 libx264 编码。
  --crf  画质参数，0=无损，18=高质量（默认），23=中等，越小越好
"""

import os
import sys
import json
import argparse
import subprocess

import numpy as np
import cv2

from distort import my_distort_point


def load_params(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    params = data["params"]
    k1, k2, p1, p2, k3 = params  # 注意顺序: k1, k2, p1, p2, k3
    info = data.get("info", {})
    center = info.get("center")
    if center is None:
        image_size = info.get("image_size", [0, 0])
        center = [image_size[0] / 2.0, image_size[1] / 2.0]
    src_cx, src_cy = float(center[0]), float(center[1])
    src_scale = src_cx
    return k1, k2, p1, p2, k3, src_cx, src_cy, src_scale


def parse_size(text):
    w, h = text.lower().split("x")
    return int(w), int(h)


def build_undistort_maps(
    out_w,
    out_h,
    k1,
    k2,
    p1,
    p2,
    k3,
    src_cx,
    src_cy,
    src_scale,
):
    """构建去畸变重采样映射表 (map_x, map_y)。

    输出像素在理想（无畸变）坐标系；map 指向源视频中的畸变采样位置。
    当 out 分辨率低于标定分辨率时，归一化用输出尺寸，采样坐标仍映射到源分辨率。
    """
    out_cx, out_cy = out_w / 2.0, out_h / 2.0
    out_scale = out_w / 2.0

    xs = np.arange(out_w, dtype=np.float64)
    ys = np.arange(out_h, dtype=np.float64)
    grid_x, grid_y = np.meshgrid(xs, ys)

    xn = (grid_x - out_cx) / out_scale
    yn = (grid_y - out_cy) / out_scale

    xd_n, yd_n = my_distort_point(xn, yn, k1, k2, k3, p1, p2)

    map_x = (xd_n * src_scale + src_cx).astype(np.float32)
    map_y = (yd_n * src_scale + src_cy).astype(np.float32)
    return map_x, map_y


def mux_audio(src_video, video_only_path, out_path):
    """用 FFmpeg 将源视频音轨复用到已编码的视频。"""
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        video_only_path,
        "-i",
        src_video,
        "-map",
        "0:v:0",
        "-map",
        "1:a?",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-shortest",
        out_path,
    ]
    print(f"FFmpeg 复用音轨: {' '.join(cmd)}")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print(proc.stderr)
        raise RuntimeError("音轨复用失败")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default=r"D:\视频\cumtb\视角1_1080_1m.mp4")
    parser.add_argument(
        "--params",
        default=r"d:\intern\teenager_football\MCS-main\store\CUMTB_mcs\camera1\distortion_params.json",
    )
    parser.add_argument("--suffix", default="_undistorted")
    parser.add_argument(
        "--out-size",
        default=None,
        help="输出分辨率 WxH，如 1920x1080；默认与输入相同",
    )
    parser.add_argument("--out", default=None, help="输出视频路径（优先于 --suffix）")
    parser.add_argument("--crf", type=int, default=18,
                        help="H.264 CRF 质量参数: 0=无损, 18=高质量(默认), 23=中等")
    parser.add_argument(
        "--preset",
        default="medium",
        help="x264 编码 preset，默认 medium（画质/速度均衡）",
    )
    parser.add_argument(
        "--audio",
        action="store_true",
        help="处理完成后用 FFmpeg 从源视频复用音轨",
    )
    parser.add_argument("--test", action="store_true", help="只处理首帧并保存对比图")
    args = parser.parse_args()

    k1, k2, p1, p2, k3, src_cx, src_cy, src_scale = load_params(args.params)
    print(f"畸变参数: k1={k1}, k2={k2}, k3={k3}, p1={p1}, p2={p2}")
    print(f"标定中心: ({src_cx}, {src_cy}), scale={src_scale}")

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"无法打开视频: {args.video}")
        sys.exit(1)

    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"源视频: {src_w}x{src_h} @ {fps:.3f}fps, {total} 帧")

    if args.out_size:
        out_w, out_h = parse_size(args.out_size)
    else:
        out_w, out_h = src_w, src_h
    print(f"输出: {out_w}x{out_h}")

    map_x, map_y = build_undistort_maps(
        out_w, out_h, k1, k2, p1, p2, k3, src_cx, src_cy, src_scale
    )

    base, ext = os.path.splitext(args.video)

    if args.test:
        ok, frame = cap.read()
        if not ok:
            print("读取首帧失败")
            sys.exit(1)
        undist = cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR)
        out_dir = os.path.dirname(os.path.abspath(args.video)) or "."
        for name, img in [("_test_original.jpg", frame), ("_test_undistorted.jpg", undist)]:
            path = os.path.join(out_dir, name)
            ok2, buf = cv2.imencode(".jpg", img)
            if ok2:
                with open(path, "wb") as f:
                    f.write(buf.tobytes())
        print(f"已保存测试帧到: {out_dir}")
        cap.release()
        return

    if args.out:
        out_path = args.out
    elif args.out_size:
        out_path = base + "_1080_undistorted.mp4"
    else:
        out_path = base + args.suffix + ".mp4"

    encode_path = out_path + ".video_only.mp4" if args.audio else out_path

    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", f"{out_w}x{out_h}",
        "-r", str(fps),
        "-i", "pipe:0",
        "-vcodec", "libx264",
        "-crf", str(args.crf),
        "-preset", args.preset,
        "-pix_fmt", "yuv420p",
        encode_path,
    ]
    print(f"FFmpeg 命令: {' '.join(ffmpeg_cmd)}")
    proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)

    idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            undist = cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR)
            proc.stdin.write(undist.tobytes())
            idx += 1
            if idx % 500 == 0 or idx == total:
                print(f"  处理进度: {idx}/{total}", flush=True)
    finally:
        proc.stdin.close()
        proc.wait()
        cap.release()

    if proc.returncode != 0:
        print(f"FFmpeg 编码失败 (code={proc.returncode})")
        sys.exit(1)

    if args.audio:
        mux_audio(args.video, encode_path, out_path)
        os.remove(encode_path)

    print(f"完成! 已保存: {out_path} (共 {idx} 帧, {out_w}x{out_h}, CRF={args.crf})")


if __name__ == "__main__":
    main()
