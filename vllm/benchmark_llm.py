import asyncio
import aiohttp
import time
import os

# ================= 压测与监控配置 =================
URL_CHAT = "http://localhost:8000/v1/chat/completions"
URL_METRICS = "http://localhost:8000/metrics" # vLLM 内部监控接口
MODEL = "Qwen3.6-27B"

CONCURRENCY = 160        # 并发数 (现在可以真正突破 100 了)
TEST_DURATION = 120      # 压测持续时间(秒)
MAX_TOKENS = 1024         # 单次请求生成的最大长度

DATA_FILE = "war_and_peace.txt" 

# ================= 数据加载 =================
def load_real_context():
    if not os.path.exists(DATA_FILE):
        print(f"❌ 找不到 {DATA_FILE}，请确保文本文件和脚本在同一目录下。")
        exit(1)
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        text = f.read()
        # 【安全阀门】：防止整本小说百万字直接导致 OOM
        if len(text) > 20000:
            text = text[:20000]
        return text

REAL_CONTEXT = load_real_context()

# ================= 监控后台线程 =================
async def monitor_vllm_metrics(session, stop_event, kv_history):
    """后台监控协程：独立抓取，不影响性能统计"""
    print("👁️  内部监控线程已启动...")
    # 监控接口设置极短的 2 秒超时，绝不死等
    monitor_timeout = aiohttp.ClientTimeout(total=2)
    
    while not stop_event.is_set():
        try:
            async with session.get(URL_METRICS, timeout=monitor_timeout) as response:
                if response.status == 200:
                    text = await response.text()
                    
                    kv_usage = 0.0
                    running_reqs = 0
                    waiting_reqs = 0
                    
                    for line in text.split('\n'):
                        if line.startswith('#'): continue 
                            
                        # 精确狙击最新的 KV Cache 命名
                        if 'kv_cache_usage_perc' in line:
                            kv_usage = float(line.split()[-1]) * 100
                        elif 'num_requests_running' in line:
                            running_reqs = int(float(line.split()[-1]))
                        elif 'num_requests_waiting' in line:
                            waiting_reqs = int(float(line.split()[-1]))

                    kv_history.append((time.time(), kv_usage))
                    print(f"📊 [监控] KV Cache: {kv_usage:05.2f}% | 运行中: {running_reqs} | 队列等待: {waiting_reqs}")
                    
        except asyncio.TimeoutError:
            print("📊 [监控] 抓取超时 (vLLM API 正在全力计算长文本，暂时无暇汇报...)")
        except Exception as e:
            pass # 忽略其他偶发网络中断
            
        await asyncio.sleep(2)

# ================= 压测工作线程 =================
async def worker(session, worker_id, end_time, stats):
    """模拟真实用户的发包机器 (自带高精度独立秒表)"""
    messages = [
        {"role": "system", "content": "你是一个文学分析专家。"},
        {"role": "user", "content": f"请详细阅读以下名著节选并做深度剧情分析：\n\n{REAL_CONTEXT}"}
    ]
    payload = {
        "model": MODEL,
        "messages": messages,
        "max_tokens": MAX_TOKENS,
        "temperature": 0.1,
        "stream": True 
    }

    req_count = 0
    while time.time() < end_time:
        start_time = time.perf_counter()
        first_token_time = None
        token_count = 0
        
        try:
            async with session.post(URL_CHAT, json=payload) as response:
                if response.status == 200:
                    async for line in response.content:
                        if line:
                            line = line.decode('utf-8').strip()
                            if line.startswith("data: ") and line != "data: [DONE]":
                                if first_token_time is None:
                                    # 精准记录首字到达的物理时间
                                    first_token_time = time.perf_counter()
                                token_count += 1
                                
                    req_count += 1
                    ttft = first_token_time - start_time
                    tpot = (time.perf_counter() - first_token_time) / max(1, token_count - 1)
                    
                    stats['ttfts'].append(ttft)
                    stats['tpots'].append(tpot)
                    stats['total_tokens'] += token_count
                else:
                    await asyncio.sleep(1)
                    
        except Exception as e:
            await asyncio.sleep(1)

# ================= 主函数 =================
async def main():
    print(f"🚀 开始持续 {TEST_DURATION} 秒的压测与监控任务")
    print(f"模型: {MODEL} | 并发数: {CONCURRENCY} | 文本长度: {len(REAL_CONTEXT)} 字符")
    print("-" * 60)
    
    stats = {'ttfts': [], 'tpots': [], 'total_tokens': 0}
    kv_history = [] 
    stop_event = asyncio.Event() 
    end_time = time.time() + TEST_DURATION
    
    # 💥【致命修复】：解除 aiohttp 默认只允许 100 个并发连接的封印！
    connector = aiohttp.TCPConnector(limit=0)
    
    # 工作线程的超时时间要设得非常长，允许在大并发时排队
    timeout = aiohttp.ClientTimeout(total=600) 
    
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        monitor_task = asyncio.create_task(monitor_vllm_metrics(session, stop_event, kv_history))
        
        workers = [worker(session, i, end_time, stats) for i in range(CONCURRENCY)]
        global_start = time.perf_counter()
        await asyncio.gather(*workers)
        global_end = time.perf_counter()
        
        stop_event.set()
        await monitor_task

    # ================= 结果统计 =================
    total_time = global_end - global_start
    throughput = stats['total_tokens'] / total_time
    
    avg_ttft = sum(stats['ttfts']) / len(stats['ttfts']) if stats['ttfts'] else 0
    avg_tpot = sum(stats['tpots']) / len(stats['tpots']) if stats['tpots'] else 0
    max_kv = max([h[1] for h in kv_history]) if kv_history else 0

    print("\n" + "=" * 60)
    print("🎯 压测与监控报告汇总")
    print("=" * 60)
    print(f"实际压测耗时:   {total_time:.1f} 秒")
    print(f"完成总请求数:   {len(stats['ttfts'])} 次")
    print(f"Decode 吞吐量:  {throughput:.1f} tokens/s")
    print("-" * 60)
    print(f"平均 TTFT (首字延迟): {avg_ttft:.3f} 秒")
    print(f"平均 TPOT (单字延迟): {avg_tpot*1000:.1f} 毫秒")
    print("-" * 60)
    print(f"🚨 KV Cache 峰值: {max_kv:.2f}%")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(main())