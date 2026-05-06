import argparse
import subprocess
import os

def main():
    parser = argparse.ArgumentParser(description="启动 vLLM 推理服务")
    parser.add_argument("--model_name", type=str, default="Qwen-7B-Chat", help="模型在 Models 文件夹下的名称")
    parser.add_argument("--port", type=int, default=8000, help="服务端口")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9, help="GPU 显存利用率")
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="张量并行大小(GPU数量)")
    
    args = parser.parse_args()
    
    # 构造模型路径
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(base_dir, "Models", args.model_name)
    
    if not os.path.exists(model_path):
        print(f"[警告] 模型路径 {model_path} 当前不存在，请确保模型已经下载并放在正确位置。")
    
    command = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_path,
        "--served-model-name", args.model_name,
        "--host", "0.0.0.0",
        "--port", str(args.port),
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
        "--tensor-parallel-size", str(args.tensor_parallel_size),
        # 兼容性设置，根据具体模型可增加参数如 --trust-remote-code 等
        "--trust-remote-code"
    ]
    
    print("启动命令:", " ".join(command))
    
    # 启动进程
    try:
        subprocess.run(command)
    except KeyboardInterrupt:
        print("服务已关闭。")

if __name__ == "__main__":
    main()
