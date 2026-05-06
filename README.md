# LLM 推理服务框架

本目录包含使用 `vllm` 构建的大模型推理服务框架。

## 前置依赖

首先确保安装了所需的依赖：
```bash
pip install vllm openai fastapi uvicorn
```

## 模型存放路径

模型统一存放在工作区根目录下的 `Models` 文件夹中。例如：
`../Models/Qwen1.5-7B-Chat`

## 启动服务

提供两种启动服务的方式：

### 1. 使用命令行启动 OpenAI 兼容格式服务 (推荐)

最直接的方法是使用 vLLM 内置的 API Server：

```bash
python -m vllm.entrypoints.openai.api_server \
    --model "../Models/Your-Model-Name" \
    --served-model-name "your-model" \
    --host 0.0.0.0 \
    --port 8000
```
您可以修改 `run_vllm_server.bat` 或 `run_vllm_server.sh` 脚本来快速启动。

### 2. 使用自定义 FastAPI 服务启动

如果您需要在服务中添加自定义逻辑（例如身份验证、日志记录、特殊的前后处理逻辑），请运行：

```bash
python custom_server.py
```

## 测试服务

服务启动后，可以运行测试脚本调用：

```bash
python client_test.py
```
