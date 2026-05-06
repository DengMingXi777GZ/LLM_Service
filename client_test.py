import requests
import json

url = "http://localhost:8000/chat"
payload = {
    "messages": [
        {"role": "system", "content": "你是一个人工智能助手。"},
        {"role": "user", "content": "你好，请给我讲一个关于程序员的笑话。"}
    ],
    "stream": True,
    "temperature": 0.7
}

def test_chat():
    print(f"正在请求 {url} ...\n")
    try:
        print("Assistant: ", end="", flush=True)
        with requests.post(url, json=payload, stream=True) as r:
            if r.status_code != 200:
                print(f"请求报错 (Status {r.status_code}): {r.text}")
                return
            for chunk in r.iter_content(chunk_size=None, decode_unicode=True):
                if chunk:
                    print(chunk, end="", flush=True)
        print("\n")
    except requests.exceptions.ConnectionError:
        print("\n\n[错误] 连接失败。请确认服务已启动并且端口8000一致。")
    except Exception as e:
        print(f"\n\n[错误] 请求发生异常: {e}")

if __name__ == "__main__":
    test_chat()
