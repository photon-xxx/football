import os
import time
import json
import base64
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
from dashscope import MultiModalConversation

# 尝试导入 openai，如果不存在则需要在 requirements 中添加
try:
    from openai import OpenAI
    print("[OCR] Successfully imported openai")
except ImportError:
    OpenAI = None
    print("[OCR] Warning: 'openai' package not installed. Batch OCR will be disabled.")

def encode_image_to_base64(image_path: Path) -> str:
    """将图片转换为 Base64 字符串"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def check_has_number(image_path: str, api_key: str = os.getenv('DASHSCOPE_API_KEY', '')) -> bool:
    """
    检查图片中是否包含可见的球衣号码。
    (保持原有的实时调用逻辑，用于稀疏采样检查)
    """
    messages = [
        {
            'role': 'user',
            'content': [
                {'image': image_path},
                {'text': '这张图片中是否有足球球衣号码？请只回答"Yes"或"No"。'}
            ]
        }
    ]
    
    try:
        response = MultiModalConversation.call(
            api_key=api_key,
            model='qwen3-vl-plus', # 保持和原逻辑一致
            messages=messages
        )
        
        if response.status_code == 200:
            text = response.output.choices[0].message.content[0]["text"].strip().lower()
            return "yes" in text
        else:
            print(f"[OCR Check] Error: {response.code} - {response.message}")
            return False
            
    except Exception as e:
        print(f"[OCR Check] Exception: {e}")
        return False

def recognize_track(track_id: int, api_key: str = os.getenv('DASHSCOPE_API_KEY', '')) -> List[str]:
    """
    [旧接口 - 实时单轨识别] 
    保留此接口用于兼容性或测试，但建议使用 batch_recognize_tracks
    """
    input_dir = Path("ocr_input") / f"track{track_id}"
    if not input_dir.exists(): return []
    image_files = sorted(list(input_dir.glob("*.jpg")))
    if not image_files: return []

    results = []
    print(f"[OCR] Processing Track {track_id} (Real-time): {len(image_files)} images...")

    for img_file in image_files:
        abs_path = img_file.absolute()
        image_path = f"file://{abs_path}"
        messages = [
            {
                'role': 'user',
                'content': [
                    {'image': image_path},
                    {'text': '识别球员的球衣号码。只返回给我完全可以相信的结果，不确定的、球员侧身球衣显示不全的结果都返回"None"。'}
                ]
            }
        ]
        try:
            response = MultiModalConversation.call(
                api_key=api_key,
                model='qwen3-vl-plus',
                messages=messages
            )
            if response.status_code == 200:
                text = response.output.choices[0].message.content[0]["text"]
                results.append(text)
            else:
                results.append("Error")
        except Exception as e:
            print(f"Exception: {e}")
            results.append("Error")

    return results

def _run_batch_job(tasks: List[Dict], api_key: str, prompt: str, ocr_root: Path, file_suffix: str) -> Dict[str, str]:
    """
    [内部方法] 统一的 Batch Job 执行逻辑
    
    Returns:
        results: Dict[custom_id, content]
    """
    if OpenAI is None:
        print("[OCR Batch] Error: 'openai' package not installed.")
        return {}

    if not tasks:
        print("[OCR Batch] No tasks provided.")
        return {}

    print(f"[OCR Batch] Building request file for {len(tasks)} tasks...")

    # 1. 创建 JSONL 文件
    jsonl_path = ocr_root / f"batch_requests_{file_suffix}.jsonl"
    
    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for task in tasks:
            base64_img = encode_image_to_base64(task['path'])
            data_url = f"data:image/jpeg;base64,{base64_img}"
            
            # custom_id
            custom_id = task['custom_id']
            
            request_obj = {
                "custom_id": custom_id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": "qwen-vl-plus",
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text", 
                                    "text": prompt
                                },
                                {
                                    "type": "image_url", 
                                    "image_url": {"url": data_url}
                                }
                            ]
                        }
                    ]
                }
            }
            f.write(json.dumps(request_obj, ensure_ascii=False) + "\n")

    # 2. 初始化客户端
    client = OpenAI(
        api_key=api_key if api_key else os.getenv("DASHSCOPE_API_KEY", os.getenv('DASHSCOPE_API_KEY', '')),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )

    try:
        # 3. 上传文件
        print(f"[OCR Batch] Uploading file ({jsonl_path.stat().st_size / 1024 / 1024:.2f} MB)...")
        batch_input_file = client.files.create(
            file=open(jsonl_path, "rb"),
            purpose="batch"
        )
        file_id = batch_input_file.id
        print(f"[OCR Batch] File uploaded. ID: {file_id}")

        # 4. 创建任务
        print("[OCR Batch] Submitting job...")
        batch_job = client.batches.create(
            input_file_id=file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h"
        )
        batch_id = batch_job.id
        print(f"[OCR Batch] Job submitted. Batch ID: {batch_id}")
        print(f"[OCR Batch] Initial status: {batch_job.status}")
        
        # 保存 batch_id 到文件（用于恢复）
        batch_log = ocr_root / f"batch_{file_suffix}_latest.txt"
        with open(batch_log, 'w') as f:
            f.write(f"{batch_id}\n")
            f.write(f"Tasks: {len(tasks)}\n")
            f.write(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        print("[OCR Batch] Waiting for completion (this may take several minutes)...")
        print(f"[OCR Batch] Tip: Batch ID saved to {batch_log}")
        
        # 初始等待（批处理任务需要时间启动）
        time.sleep(5)

        # 5. 轮询（添加超时机制）
        max_wait_time = 3600  # 最大等待时间：1小时
        poll_interval = 10    # 轮询间隔：10秒
        elapsed_time = 0
        retry_count = 0
        max_retries = 3
        
        while elapsed_time < max_wait_time:
            try:
                batch_job = client.batches.retrieve(batch_id)
                status = batch_job.status
                completed = batch_job.request_counts.completed if batch_job.request_counts else 0
                total = batch_job.request_counts.total if batch_job.request_counts else 0
                failed = batch_job.request_counts.failed if batch_job.request_counts else 0
                
                # 显示详细进度
                progress_pct = (completed / total * 100) if total > 0 else 0
                print(f"[OCR Batch] Status: {status} | Progress: {completed}/{total} ({progress_pct:.1f}%) | Failed: {failed} | Time: {elapsed_time}s", end="\r")
                
                if status == "completed":
                    print(f"\n[OCR Batch] Job Completed! Total: {completed}, Failed: {failed}")
                    break
                elif status in ["failed", "expired", "cancelled"]:
                    print(f"\n[OCR Batch] Job Failed: {status}")
                    if batch_job.errors:
                        print(f"[OCR Batch] Errors: {batch_job.errors}")
                    return {}
                elif status in ["validating", "in_progress", "finalizing"]:
                    # 正常的中间状态，继续等待
                    pass
                else:
                    # 未知状态，记录但继续等待
                    print(f"\n[OCR Batch] Warning: Unknown status '{status}', continuing...")
                
                # 重置重试计数（成功获取状态）
                retry_count = 0
                
            except KeyboardInterrupt:
                print(f"\n[OCR Batch] User interrupted. Batch ID: {batch_id}")
                print(f"[OCR Batch] You can check status later with this Batch ID")
                return {}
            except Exception as e:
                retry_count += 1
                print(f"\n[OCR Batch] Error retrieving status (attempt {retry_count}/{max_retries}): {e}")
                if retry_count >= max_retries:
                    print(f"[OCR Batch] Max retries reached. Batch ID: {batch_id}")
                    return {}
            
            time.sleep(poll_interval)
            elapsed_time += poll_interval
        
        # 超时处理
        if elapsed_time >= max_wait_time:
            print(f"\n[OCR Batch] Timeout after {max_wait_time}s. Batch ID: {batch_id}")
            print(f"[OCR Batch] The job may still be processing. Check later with this Batch ID")
            return {}

        # 6. 下载结果
        output_file_id = batch_job.output_file_id
        if not output_file_id:
            print("[OCR Batch] Warning: No output file ID.")
            return {}

        print(f"[OCR Batch] Downloading results (File ID: {output_file_id})...")
        file_response = client.files.content(output_file_id)
        file_content = file_response.text
        
        # 7. 解析
        results = {}
        for line in file_content.strip().split('\n'):
            if not line: continue
            try:
                res = json.loads(line)
                cid = res.get("custom_id", "")
                
                content = "Error"
                if res.get("response") and res["response"].get("body"):
                    choices = res["response"]["body"].get("choices", [])
                    if choices:
                        content = choices[0]["message"]["content"]
                
                results[cid] = content
            except Exception as e:
                print(f"[OCR Batch] Error parsing result line: {e}")
                
        return results

    except Exception as e:
        print(f"\n[OCR Batch] Exception: {e}")
        import traceback
        traceback.print_exc()
        return {}

def check_batch_status(batch_id: str, api_key: str = None) -> dict:
    """
    检查批处理任务状态（用于调试和恢复）
    
    Args:
        batch_id: 批处理任务ID
        api_key: API密钥
    
    Returns:
        status_info: 状态信息字典
    """
    if OpenAI is None:
        print("[OCR Batch] Error: 'openai' package not installed.")
        return {}
    
    client = OpenAI(
        api_key=api_key if api_key else os.getenv("DASHSCOPE_API_KEY", os.getenv('DASHSCOPE_API_KEY', '')),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    
    try:
        batch_job = client.batches.retrieve(batch_id)
        status_info = {
            'id': batch_job.id,
            'status': batch_job.status,
            'created_at': batch_job.created_at,
            'completed': batch_job.request_counts.completed if batch_job.request_counts else 0,
            'failed': batch_job.request_counts.failed if batch_job.request_counts else 0,
            'total': batch_job.request_counts.total if batch_job.request_counts else 0,
            'output_file_id': batch_job.output_file_id if hasattr(batch_job, 'output_file_id') else None
        }
        
        print(f"\n[Batch Status Check]")
        print(f"  Batch ID: {status_info['id']}")
        print(f"  Status: {status_info['status']}")
        print(f"  Progress: {status_info['completed']}/{status_info['total']} (Failed: {status_info['failed']})")
        print(f"  Output File: {status_info['output_file_id']}")
        
        return status_info
    except Exception as e:
        print(f"[OCR Batch] Error checking status: {e}")
        return {}

def cancel_batch(batch_id: str, api_key: str = None) -> bool:
    """
    取消批处理任务
    
    Args:
        batch_id: 批处理任务ID
        api_key: API密钥
    
    Returns:
        success: 是否成功取消
    """
    if OpenAI is None:
        print("[OCR Batch] Error: 'openai' package not installed.")
        return False
    
    client = OpenAI(
        api_key=api_key if api_key else os.getenv("DASHSCOPE_API_KEY", os.getenv('DASHSCOPE_API_KEY', '')),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    
    try:
        print(f"\n[Batch Cancel] Attempting to cancel batch: {batch_id}")
        batch_job = client.batches.cancel(batch_id)
        print(f"[Batch Cancel] Success! New status: {batch_job.status}")
        return True
    except Exception as e:
        print(f"[Batch Cancel] Error: {e}")
        return False

def check_api_quota(api_key: str = None) -> dict:
    """
    查询API配额和使用情况
    
    Args:
        api_key: API密钥
    
    Returns:
        quota_info: 配额信息字典
    """
    if OpenAI is None:
        print("[API Quota] Error: 'openai' package not installed.")
        return {}
    
    client = OpenAI(
        api_key=api_key if api_key else os.getenv("DASHSCOPE_API_KEY", os.getenv('DASHSCOPE_API_KEY', '')),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    
    quota_info = {}
    
    print(f"\n{'='*70}")
    print(f"[API Quota Check]")
    print(f"{'='*70}\n")
    
    # 尝试1: 查询模型列表（测试API可用性）
    try:
        print("[1/3] 测试API连接...")
        models = client.models.list()
        available_models = [m.id for m in models.data[:5]]  # 只显示前5个
        print(f"  ✅ API连接正常")
        print(f"  可用模型: {', '.join(available_models)}")
        quota_info['api_available'] = True
        quota_info['models'] = available_models
    except Exception as e:
        print(f"  ❌ API连接失败: {e}")
        quota_info['api_available'] = False
        return quota_info
    
    # 尝试2: 查询订阅信息
    try:
        print("\n[2/3] 查询订阅信息...")
        # 注意：不同的API提供商端点可能不同
        # 这里尝试标准的OpenAI端点
        import requests
        
        base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        headers = {
            "Authorization": f"Bearer {api_key if api_key else os.getenv('DASHSCOPE_API_KEY', os.getenv('DASHSCOPE_API_KEY', ''))}"
        }
        
        # 尝试查询订阅
        response = requests.get(f"{base_url}/dashboard/billing/subscription", headers=headers, timeout=10)
        if response.status_code == 200:
            subscription = response.json()
            print(f"  ✅ 订阅信息:")
            if 'plan' in subscription:
                print(f"    - 套餐: {subscription.get('plan', {}).get('title', 'N/A')}")
            if 'hard_limit_usd' in subscription:
                print(f"    - 限额: ${subscription.get('hard_limit_usd', 'N/A')}")
            quota_info['subscription'] = subscription
        else:
            print(f"  ⚠️ 无法查询订阅信息 (状态码: {response.status_code})")
            print(f"     可能是API提供商不支持此端点")
    except Exception as e:
        print(f"  ⚠️ 查询订阅信息失败: {e}")
    
    # 尝试3: 查询使用情况
    try:
        print("\n[3/3] 查询使用情况...")
        import requests
        from datetime import datetime, timedelta
        
        # 查询最近30天的使用情况
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        params = {
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date": end_date.strftime("%Y-%m-%d")
        }
        
        response = requests.get(
            f"{base_url}/dashboard/billing/usage",
            headers=headers,
            params=params,
            timeout=10
        )
        
        if response.status_code == 200:
            usage = response.json()
            print(f"  ✅ 使用情况 (最近30天):")
            if 'total_usage' in usage:
                total = usage.get('total_usage', 0) / 100  # 转换为美元
                print(f"    - 总使用: ${total:.2f}")
            if 'daily_costs' in usage and usage['daily_costs']:
                recent_cost = usage['daily_costs'][-1].get('amount', 0) / 100
                recent_date = usage['daily_costs'][-1].get('date', 'N/A')
                print(f"    - 最近一天 ({recent_date}): ${recent_cost:.2f}")
            quota_info['usage'] = usage
        else:
            print(f"  ⚠️ 无法查询使用情况 (状态码: {response.status_code})")
            print(f"     可能是API提供商不支持此端点")
    except Exception as e:
        print(f"  ⚠️ 查询使用情况失败: {e}")
    
    # 尝试4: 查询当前批处理配额
    try:
        print("\n[4/4] 查询批处理配额...")
        batches = client.batches.list(limit=100)
        all_batches = batches.data
        
        # 统计不同状态的批处理
        active = sum(1 for b in all_batches if b.status in ['validating', 'in_progress', 'finalizing'])
        completed = sum(1 for b in all_batches if b.status == 'completed')
        failed = sum(1 for b in all_batches if b.status in ['failed', 'expired', 'cancelled'])
        
        print(f"  ✅ 批处理统计 (最近100个):")
        print(f"    - 活跃中: {active} 个")
        print(f"    - 已完成: {completed} 个")
        print(f"    - 失败/取消: {failed} 个")
        
        if active >= 3:
            print(f"    ⚠️ 警告: {active} 个活跃批处理可能接近并发限制")
        
        quota_info['batches'] = {
            'active': active,
            'completed': completed,
            'failed': failed,
            'total': len(all_batches)
        }
    except Exception as e:
        print(f"  ⚠️ 查询批处理配额失败: {e}")
    
    print(f"\n{'='*70}")
    print(f"[Quota Check Complete]")
    print(f"{'='*70}\n")
    
    return quota_info

def list_batches(api_key: str = None, limit: int = 10) -> list:
    """
    列出所有批处理任务（最近的N个）
    
    Args:
        api_key: API密钥
        limit: 返回的任务数量
    
    Returns:
        batches: 批处理任务列表
    """
    if OpenAI is None:
        print("[OCR Batch] Error: 'openai' package not installed.")
        return []
    
    client = OpenAI(
        api_key=api_key if api_key else os.getenv("DASHSCOPE_API_KEY", os.getenv('DASHSCOPE_API_KEY', '')),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    
    try:
        print(f"\n[Batch List] Fetching last {limit} batches...")
        batches_response = client.batches.list(limit=limit)
        batches = batches_response.data
        
        print(f"\n{'='*80}")
        print(f"{'ID':<40} {'Status':<15} {'Progress':<20} {'Created'}")
        print(f"{'='*80}")
        
        for batch in batches:
            batch_id = batch.id
            status = batch.status
            completed = batch.request_counts.completed if batch.request_counts else 0
            total = batch.request_counts.total if batch.request_counts else 0
            created = batch.created_at if hasattr(batch, 'created_at') else 'N/A'
            
            progress = f"{completed}/{total}"
            print(f"{batch_id:<40} {status:<15} {progress:<20} {created}")
        
        print(f"{'='*80}\n")
        
        # 提示活跃任务
        active_batches = [b for b in batches if b.status in ['validating', 'in_progress', 'finalizing']]
        if active_batches:
            print(f"⚠️  Active batches: {len(active_batches)}")
            print(f"   These may block new batch requests if there's a concurrency limit\n")
        
        return batches
    except Exception as e:
        print(f"[Batch List] Error: {e}")
        return []

def batch_recognize_tracks(track_ids: List[int], api_key: str = None) -> Dict[int, List[str]]:
    """
    [批量识别接口] (Pass 4)
    """
    ocr_root = Path("ocr_input")
    tasks = []
    
    print(f"[OCR Batch] Collecting images for {len(track_ids)} tracks (Recognition)...")
    
    for tid in track_ids:
        track_dir = ocr_root / f"track{tid}"
        if not track_dir.exists(): continue
        
        images = sorted(list(track_dir.glob("*.jpg")))
        for img_path in images:
            tasks.append({
                "custom_id": f"{tid}|{img_path.name}",
                "path": img_path
            })
            
    if not tasks: return {}
    
    prompt = '识别球员的球衣号码。只返回给我完全可以相信的结果，不确定的、球员侧身球衣显示不全的结果都返回"None"。'
    raw_results = _run_batch_job(tasks, api_key, prompt, ocr_root, "recog")
    
    # 解析结果回 Map
    results_map = {tid: [] for tid in track_ids}
    for cid, content in raw_results.items():
        if "|" in cid:
            tid_str, _ = cid.split("|", 1)
            try:
                tid = int(tid_str)
                if tid in results_map:
                    results_map[tid].append(content)
            except: pass
            
    return results_map

def batch_check_has_number(requests: List[Tuple[int, Path]], api_key: str = None) -> Dict[int, Dict[str, bool]]:
    """
    [批量检测接口] (Pass 2 - First Glance)
    
    Args:
        requests: List of (track_id, image_path)
    
    Returns:
        result_map: {track_id: {filename: has_number(bool)}}
    """
    ocr_root = Path("ocr_input")
    tasks = []
    
    print(f"[OCR Batch] Preparing {len(requests)} images for First Glance Check...")
    
    for tid, img_path in requests:
        tasks.append({
            "custom_id": f"{tid}|{img_path.name}",
            "path": img_path
        })
        
    if not tasks: return {}
    
    prompt = '这张图片中是否有足球球衣号码？请只回答"Yes"或"No"。'
    raw_results = _run_batch_job(tasks, api_key, prompt, ocr_root, "check")
    
    # 解析结果
    result_map = {}
    
    for cid, content in raw_results.items():
        if "|" in cid:
            tid_str, fname = cid.split("|", 1)
            try:
                tid = int(tid_str)
                if tid not in result_map: result_map[tid] = {}
                
                # Check yes/no
                text = content.strip().lower()
                has_num = "yes" in text
                
                result_map[tid][fname] = has_num
            except: pass
            
    return result_map

def detect_jersey_color(image_path: Path, api_key: str = os.getenv('DASHSCOPE_API_KEY', '')) -> str:
    """
    单张图片检测球衣颜色（实时调用）
    
    Args:
        image_path: 图片路径（裁剪后的检测框）
        api_key: API密钥
    
    Returns:
        color_label: 'white'/'red'/'green'/'blue'/'black'/'unknown'
    """
    abs_path = image_path.absolute()
    file_url = f"file://{abs_path}"
    
    messages = [
        {
            'role': 'user',
            'content': [
                {'image': file_url},
                {'text': '''请识别这张足球运动员图片中的球衣颜色。
                
    要求：
    1. 只关注球衣主体颜色（忽略号码、队徽等细节）
    2. 从以下5个类别中选择最接近的一个：white（白色）、red（红色）、green（绿色）、blue（蓝色）、black（黑色）
    3. 如果球衣颜色不清晰或不属于上述5类，返回 "unknown"
    4. 只返回颜色标签（如 "red"），不要返回其他解释文字

    颜色标签：'''}
            ]
        }
    ]
    
    try:
        response = MultiModalConversation.call(
            api_key=api_key,
            model='qwen3-vl-plus',
            messages=messages
        )
        
        if response.status_code == 200:
            text = response.output.choices[0].message.content[0]["text"].strip().lower()
            
            # 解析颜色标签
            valid_colors = ['white', 'red', 'green', 'blue', 'black']
            for color in valid_colors:
                if color in text:
                    return color
            
            return 'unknown'
        else:
            print(f"[Color API] Error: {response.code} - {response.message}")
            return 'unknown'
            
    except Exception as e:
        print(f"[Color API] Exception: {e}")
        return 'unknown'


def batch_detect_jersey_colors(crop_paths: List[Path], api_key: str = None) -> Dict[str, str]:
    """
    批量检测球衣颜色（使用 Batch API）
    
    Args:
        crop_paths: 裁剪后的检测框图片路径列表
        api_key: API密钥
    
    Returns:
        results: {filename: color_label}
    """
    if not crop_paths:
        print("[Color Batch] No images provided.")
        return {}
    
    ocr_root = Path("ocr_input")
    tasks = []
    
    print(f"[Color Batch] Preparing {len(crop_paths)} images for color detection...")
    
    for img_path in crop_paths:
        tasks.append({
            "custom_id": img_path.name,
            "path": img_path
        })
    
    prompt = '''请识别这张足球运动员图片中的球衣颜色。

    要求：
    1. 只关注球衣主体颜色（忽略号码、队徽等细节）
    2. 从以下7个类别中选择最接近的一个：white（白色）、red（红色）、green（绿色）、blue（蓝色）、black（黑色）、yellow（黄色）、purple（紫色）
    3. 如果球衣颜色不清晰或不属于上述7类，返回 "unknown"
    4. 只返回颜色标签（如 "red"），不要返回其他解释文字

    颜色标签：'''
    
    raw_results = _run_batch_job(tasks, api_key, prompt, ocr_root, "color")
    
    # 解析颜色标签
    valid_colors = ['white', 'red', 'green', 'blue', 'black', 'yellow', 'purple']
    results = {}
    
    for filename, content in raw_results.items():
        text = content.strip().lower()
        
        # 查找有效颜色
        detected_color = 'unknown'
        for color in valid_colors:
            if color in text:
                detected_color = color
                break
        
        results[filename] = detected_color
    
    print(f"[Color Batch] Detected colors for {len(results)} images")
    return results


if __name__ == "__main__":
    # 测试代码
    cancel_batch('batch_78410ef5-687d-478b-9a7e-3a3a908d2235', api_key=os.getenv('DASHSCOPE_API_KEY', ''))
