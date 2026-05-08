import copy
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sampling_params import SamplingParams
from vllm.utils import random_uuid

# 配置模型路径
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_NAME = "Qwen-7B-Chat" # 默认模型名称
MODEL_PATH = os.path.join(BASE_DIR, "Models", MODEL_NAME)

engine = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    # 初始化 vLLM engine
    engine_args = AsyncEngineArgs(
        model=MODEL_PATH,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
        trust_remote_code=True
    )
    if os.path.exists(MODEL_PATH):
        engine = AsyncLLMEngine.from_engine_args(engine_args)
        print("vLLM Engine 初始化成功!")
    else:
        print(f"[警告] 模型路径 {MODEL_PATH} 不存在。请添加模型并重启服务。")
    yield

app = FastAPI(lifespan=lifespan)

@app.post("/generate")
async def generate(request: Request):
    """自定义生成接口示例"""
    if engine is None:
        return JSONResponse({"error": "Engine 尚未初始化 (可能模型不存在)"}, status_code=500)
        
    request_dict = await request.json()
    prompt = request_dict.pop("prompt")
    stream = request_dict.pop("stream", False)
    
    # 采样参数
    sampling_params = SamplingParams(**request_dict)
    request_id = random_uuid()
    
    results_generator = engine.generate(prompt, sampling_params, request_id)
    
    if stream:
        async def stream_results():
            async for request_output in results_generator:
                text = request_output.outputs[0].text
                yield text
        return StreamingResponse(stream_results(), media_type="text/plain")
    
    # 阻塞式等待结果
    final_output = None
    async for request_output in results_generator:
        # TODO: Handle cancellation correctly but simple implementation here
        final_output = request_output
    
    assert final_output is not None
    text = final_output.outputs[0].text
    return JSONResponse({"text": text})

if __name__ == "__main__":
    import uvicorn
    # 为了防止子进程出错，推荐直接在命令行用 uvicorn 启动: 
    # uvicorn custom_server:app --host 0.0.0.0 --port 8000
    uvicorn.run("custom_server:app", host="0.0.0.0", port=8000)
