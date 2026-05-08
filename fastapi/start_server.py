import argparse
import os
import torch
from threading import Thread
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
import uvicorn
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

app = FastAPI()
model = None
tokenizer = None

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    max_new_tokens: int = 1024
    temperature: float = 0.7
    stream: bool = False

@app.post("/chat")
async def chat(request: ChatRequest):
    # 使用 Qwen 等模型自带的 chat template 进行格式化
    messages = [{"role": m.role, "content": m.content} for m in request.messages]
    
    try:
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        text = ""
        for m in messages:
            text += f"{m['role']}: {m['content']}\n"
        text += "assistant: "
        
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    if request.stream:
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        generation_kwargs = dict(
            inputs,
            streamer=streamer,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            do_sample=True if request.temperature > 0 else False
        )
        
        thread = Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()
        
        async def stream_generator():
            for new_text in streamer:
                yield new_text
                
        return StreamingResponse(stream_generator(), media_type="text/plain")
    else:
        outputs = model.generate(
            **inputs,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            do_sample=True if request.temperature > 0 else False
        )
        response_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)
        return JSONResponse({"response": response_text})

def load_model(model_name):
    global model, tokenizer
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(base_dir, "Models", model_name)
    
    print(f"[*] 正在从 {model_path} 加载模型...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True
    ).eval()
    print("[*] 模型加载并准备完毕！")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen3.5-0.8B")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    
    load_model(args.model_name)
    print(f"[*] 启动 FastAPI API 服务，监听端口: {args.port}")
    uvicorn.run(app, host="0.0.0.0", port=args.port)

if __name__ == "__main__":
    main()
