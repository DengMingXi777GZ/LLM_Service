import json
from openai import OpenAI

# 配置连接信息
client = OpenAI(
    api_key="EMPTY", # vLLM 默认不需要鉴权，或者随便填
    base_url="http://localhost:8000/v1"
)

# 这里替换为你启动服务时使用的模型名
MODEL_NAME = "Qwen-7B-Chat" 

def test_chat():
    try:
        models = client.models.list()
        print("当前运行的模型:", models.data[0].id)
        model_id = models.data[0].id
        
        print(f"开始测试与模型 {model_id} 的对话...\n")
        response = client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": "你是一个人工智能助手。"},
                {"role": "user", "content": "你好，请给我讲一个关于程序员的笑话。"}
            ],
            stream=True
        )
        
        print("Assistant: ", end="", flush=True)
        for chunk in response:
            if chunk.choices[0].delta.content is not None:
                print(chunk.choices[0].delta.content, end="", flush=True)
        print("\n")
        
    except Exception as e:
        print(f"请求失败，请检查服务是否启动: {e}")

if __name__ == "__main__":
    test_chat()
