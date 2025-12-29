import os
import sys
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
from transformers import AutoTokenizer, CLIPImageProcessor

# 将 src 加入路径，确保能导入项目模块
sys.path.append(os.path.join(os.getcwd(), "src"))

from src.bert_layers.configuration_bert import FlexBertConfig
# 假设你在 src/bert_layers/model.py 中实现了 FlexBertWithCLIP
from src.bert_layers.model import FlexBertWithCLIP 

def create_dummy_data(batch_size=2):
    """
    创建模拟数据：
    1. SVG 文本
    2. 对应的渲染图片 (使用随机噪声模拟)
    """
    print(f"Creating dummy data for batch_size={batch_size}...")
    
    # 1. 模拟 SVG 文本 (长短不一，测试 Unpadding)
    svg_texts = [
        '<svg width="100" height="100"><circle cx="50" cy="50" r="40" stroke="green" stroke-width="4" fill="yellow" /></svg>',
        '<svg height="140" width="500"><ellipse cx="200" cy="80" rx="100" ry="50" style="fill:yellow;stroke:purple;stroke-width:2" /></svg>'
    ]
    assert len(svg_texts) == batch_size

    # 2. 模拟对应的图片 (RGB)
    # 真实场景中这是 SVG 渲染后的图片，这里用 PIL 生成随机图
    images = []
    for _ in range(batch_size):
        # CLIP 默认输入大概是 224x224，这里生成随机像素
        data = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        img = Image.fromarray(data)
        images.append(img)

    return svg_texts, images

def main():
    # 检查环境
    if not torch.cuda.is_available():
        raise RuntimeError("This script requires a GPU (e.g., RTX 4090).")
    
    device = torch.device("cuda")
    dtype = torch.bfloat16 # 4090 推荐使用 bf16
    
    print(f"Running on {torch.cuda.get_device_name(0)} with {dtype}")

    # ==========================================
    # 1. 配置模型
    # ==========================================
    print("\n[1/5] Configuring FlexBertWithCLIP...")
    
    # 这是一个 ModernBERT Base 的配置，开启了 Cross Attention
    config = FlexBertConfig(
        vocab_size=30528, # 对齐到 8 的倍数
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        
        # 核心优化开关
        padding="unpadded",       # 开启 Unpadding
        unpad_embeddings=True,    # Embedding 层也 Unpadding
        use_fa2=True,             # 开启 Flash Attention 2
        
        # 多模态开关 (我们在之前步骤中添加的)
        add_cross_attention=True,
        cross_attention_frequency=2, # 每2层加一个 Cross Attention
        vision_hidden_size=768,      # CLIP ViT-Base 的输出维度
        cross_attention_layer="cross_unpad", # 指定使用我们写的 Cross Attention 类
        
        # CLIP 设置
        vision_model_name="openai/clip-vit-base-patch32"
    )

    # ==========================================
    # 2. 初始化模型
    # ==========================================
    print("[2/5] Initializing Model...")
    try:
        model = FlexBertWithCLIP(config)
        model.to(device=device, dtype=dtype)
        print("Model initialized successfully.")
        
        # 打印一下模型结构，确认 Cross Attention 存在
        print("\n--- Layer Inspection ---")
        layer_0_type = type(model.bert.encoder.layers[0])
        print(f"Layer 0 type: {layer_0_type.__name__} (Should be standard/pre-norm)")
        
        # 假设 frequency=2，Layer 2 应该是 Cross Attention Layer
        # 注意 layer_id 是 0-indexed，如果 freq=2，且逻辑是 layer_id % 2 == 0，则 Layer 0, 2, 4... 是 Cross
        # 如果逻辑不一样，请根据你的实现调整这里的检查
        target_layer_idx = 0 if config.cross_attention_frequency > 0 else -1
        layer_target = model.bert.encoder.layers[target_layer_idx]
        print(f"Layer {target_layer_idx} type: {type(layer_target).__name__}")
        
        if hasattr(layer_target, 'cross_attn'):
            print(f"✅ Confirmed: Layer {target_layer_idx} has 'cross_attn' module.")
        else:
            print(f"⚠️ Warning: Layer {target_layer_idx} does not seem to have 'cross_attn'. Check your config frequency.")

    except Exception as e:
        print(f"❌ Model initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # ==========================================
    # 3. 数据预处理
    # ==========================================
    print("\n[3/5] Processing Data...")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")

    svg_texts, raw_images = create_dummy_data(batch_size=2)

    # 处理文本 (Padding 到 batch 内最长，ModernBERT 内部会自动 Unpad)
    text_inputs = tokenizer(svg_texts, padding=True, return_tensors="pt")
    input_ids = text_inputs["input_ids"].to(device)
    attention_mask = text_inputs["attention_mask"].to(device)

    # 处理图片
    image_inputs = image_processor(images=raw_images, return_tensors="pt")
    pixel_values = image_inputs["pixel_values"].to(device=device, dtype=dtype)

    print(f"Input IDs shape: {input_ids.shape} (Batch, Seq)")
    print(f"Pixel Values shape: {pixel_values.shape} (Batch, C, H, W)")

    # ==========================================
    # 4. 前向传播 (Forward Pass)
    # ==========================================
    print("\n[4/5] Running Forward Pass...")
    
    # 统计有效 Token 数量 (Total NNZ)
    expected_nnz = attention_mask.sum().item()
    print(f"Expected unpadded tokens (NNZ): {expected_nnz}")

    try:
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values
        )
        
        # 检查输出
        # FlexBertModel 通常返回 Unpadded 的 hidden states [Total_NNZ, Hidden]
        # 或者 Padded 的 [Batch, Seq, Hidden]
        # 这取决于 FlexBertModel 的实现细节（是否最后调用了 pad_input）
        # 基于我们之前的修改，FlexBertModel 默认如果接收未压缩输入，会先压缩计算，最后还原(pad_input)
        # 除非我们在 model_config 显式设置了某些 flag。
        # 让我们检查一下 output shape。
        
        output_tensor = outputs if isinstance(outputs, torch.Tensor) else outputs[0]
        print(f"Output shape: {output_tensor.shape}")
        
        if output_tensor.dim() == 2:
            print("Output format: Unpadded [Total_Tokens, Hidden]")
            assert output_tensor.shape[0] == expected_nnz, f"Output tokens {output_tensor.shape[0]} != Input effective tokens {expected_nnz}"
        elif output_tensor.dim() == 3:
            print("Output format: Padded [Batch, Seq, Hidden]")
            assert output_tensor.shape[0] == 2
            assert output_tensor.shape[1] == input_ids.shape[1]
        
        print("✅ Forward pass successful!")

    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # ==========================================
    # 5. 反向传播 (Backward Pass Check)
    # ==========================================
    print("\n[5/5] Checking Gradients (Backward Pass)...")
    try:
        # 造一个简单的 Loss
        loss = output_tensor.mean()
        loss.backward()
        
        # 检查 Cross Attention 层的参数是否有梯度
        # 这能证明图片信息真的参与了计算
        target_layer = model.bert.encoder.layers[target_layer_idx] # 也就是之前确认有 Cross Attn 的层
        
        if hasattr(target_layer, 'cross_attn'):
            # 检查 Wkv (Key/Value projection from Image)
            wkv_grad = target_layer.cross_attn.Wkv.weight.grad
            if wkv_grad is not None:
                grad_norm = wkv_grad.norm().item()
                print(f"✅ Gradient check passed! Cross-Attention Wkv grad norm: {grad_norm:.6f}")
                if grad_norm == 0.0:
                    print("⚠️ Warning: Gradient is zero. Check connection.")
            else:
                print("❌ Error: Cross-Attention Wkv has no gradient!")
        else:
            print("Skipping specific layer check (layer structure differs from expectation).")
            
    except Exception as e:
        print(f"❌ Backward pass failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()