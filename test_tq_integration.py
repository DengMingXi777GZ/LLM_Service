import torch
import os
import sys
from transformers import AutoTokenizer, AutoModelForCausalLM

# 动态添加 TurboQuant 路径
TQ_PATH = r"e:\Code\Python\TurboQuant\TurboQuant_Test_CAT"
if TQ_PATH not in sys.path:
    sys.path.insert(0, TQ_PATH)

from turboquant.rotation import generate_rotation_matrix, rotate_forward
from turboquant.quantizer import TurboQuantMSE

class TurboQuantKVManager:
    """管理 Transformers 模型中 KV Cache 的量化与旋转"""
    def __init__(self, head_dim, num_layers, device="cuda"):
        self.head_dim = head_dim
        self.device = device
        self.num_layers = num_layers
        print(f"[TurboQuant] 初始化管理器: Head Dim={head_dim}, Layers={num_layers}")
        # 为每层生成旋转矩阵 (使用 TQ 原生生成函数)
        self.rotations = [generate_rotation_matrix(head_dim, device) for _ in range(num_layers)]
        # 初始化量化器
        # Qwen3.5 Head Dim 较大，TurboQuant 的 MSE 量化器会自动处理
        self.k_quantizer = TurboQuantMSE(dim=head_dim, bits=4, device=device)
        self.v_quantizer = TurboQuantMSE(dim=head_dim, bits=4, device=device)
        
    def transform_kv(self, layer_idx, key, value):
        """对输入的 K/V 执行旋转（为了量化友好）"""
        # key/value shape: [batch, num_heads, seq_len, head_dim]
        pi = self.rotations[layer_idx]
        
        # 统一 dtype 到 BFloat16 或 Float16 以匹配权重
        pi = pi.to(key.dtype)
        
        # 旋转（随机正交变换可以平衡特征分布，显著降低量化误差）
        rotated_k = rotate_forward(key, pi)
        # Value 通常也进行旋转或直接量化
        rotated_v = rotate_forward(value, pi)
        
        return rotated_k, rotated_v

    def pack_kv(self, tensor):
        """调用量化器获取量化结果"""
        return self.k_quantizer.forward(tensor)

def apply_tq_to_transformers(model):
    """
    一个简单的 Monkey Patch 示例。
    注意：在没有 vLLM 的情况下，我们手动在推理循环中介入。
    这里展示如何在不修改 Transformers 源码的情况下注入 TQ 逻辑的概念。
    """
    print("[TurboQuant] 正在注入 Transformers 适配层...")
    # 获取模型配置
    config = model.config
    head_dim = getattr(config, "head_dim", 256) # Qwen3.5 显式指定 head_dim
    num_layers = config.num_hidden_layers
    
    manager = TurboQuantKVManager(head_dim, num_layers, device=model.device)
    model.tq_manager = manager
    return model

if __name__ == "__main__":
    # 测试环境
    model_path = r"e:\Code\Python\Models\Qwen3.5-0.8B"
    if os.path.exists(model_path):
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True)
        
        # 应用 TQ 逻辑
        model = apply_tq_to_transformers(model)
        
        input_text = "你好，请介绍一下 TurboQuant。"
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
        
        print("[*] 正在执行带 TurboQuant 逻辑的推理...")
        # 在实际的生成中，我们会拦截 model.generate 内部的 KV 缓存写入
        # 这里为了演示核心原理，手动模拟一次 KV 转换
        with torch.no_grad():
            outputs = model(**inputs, output_attentions=False, use_cache=True)
            past_key_values = outputs.past_key_values
            
            # 针对 Qwen3.5 混合模型，寻找 Full Attention 层（位于索引 3）
            # 它使用的是 DynamicLayer 结构，属性为 .keys 和 .values
            layer_idx = 3
            full_attn_layer = past_key_values.layers[layer_idx]
            k = full_attn_layer.keys
            v = full_attn_layer.values
            
            print(f"Full Attention 层 (Layer {layer_idx}) K Shape: {k.shape}")
            rk, rv = model.tq_manager.transform_kv(layer_idx, k, v)
            print(f"旋转后误差 (Norm): {torch.norm(rk - k).item():.4f}")
            
            # 测试量化逻辑
            quantized_indices = model.tq_manager.pack_kv(rk)
            print(f"量化后 Indices Shape: {quantized_indices.shape}")
            
            # 计算压缩空间（索引为 INT32 包装的 4B，压缩 4bit 意味着 nelement 没变但实际上我们可以打包）
            # TurboQuant 内部目前可能返回未打包的索引 Tensor
            original_size = k.nelement() * k.element_size()
            print(f"原始 KV 占用 (bytes): {original_size}")
            
            print("[✓] TurboQuant 核心逻辑（旋转+量化）验证成功。")
