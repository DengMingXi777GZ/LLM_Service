import requests

url = "http://localhost:8000/v1/chat/completions"

data = {
    "model": "Qwen3.6-27B",
    "messages": [
        {"role": "system", "content": "你是一个幽默的人工智能助手。"},
        {"role": "user", "content": "讲个很好笑的笑话，但是不要有谐音"}
    ],
    "max_tokens": 4096, 
    "temperature": 0.7
}

try:
    print("正在呼叫 Qwen3.6-27B，请稍候...")
    response = requests.post(url, json=data)
    
    if response.status_code == 200:
        message = response.json()["choices"][0]["message"]
        
        # 1. 提取回答的主体内容
        final_answer = message.get("content", "")
        
        # 2. 终极过滤大法：不管前面输出了什么乱七八糟的思考过程，
        # 只要存在 </think>，我们就把它劈成两半，只取最后面的那一部分！
        if "</think>" in final_answer:
            final_answer = final_answer.split("</think>")[-1]
            
        # 3. 清除首尾的空格和换行符
        final_answer = final_answer.strip()
        
        print("✅ 成功获取干净回答：\n")
        print(final_answer)
        
    else:
        print(f"❌ 服务异常，状态码：{response.status_code}")
        print(response.text)

except Exception as e:
    print("❌ 无法连接 vLLM 服务！")
    print("错误原因：", e)