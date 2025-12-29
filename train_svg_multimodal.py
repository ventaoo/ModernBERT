import os
import sys
import re
import io
import torch
import torch.nn as nn
import numpy as np
import cairosvg
from PIL import Image
from typing import Optional, Tuple, Union, Dict, Any
from torch.utils.data import DataLoader, Dataset

# 引入 HuggingFace 和 Composer
import transformers
from transformers import AutoTokenizer, CLIPImageProcessor, CLIPVisionModel
from datasets import load_dataset
from composer import Trainer
from composer.models import HuggingFaceModel
from composer.optim import DecoupledAdamW
from composer.optim.scheduler import LinearWithWarmupScheduler
from composer.utils import dist, reproducibility

# 引入项目源码
sys.path.append(os.path.join(os.getcwd(), "src"))
from src.bert_layers.configuration_bert import FlexBertConfig
from src.bert_layers.model import FlexBertPreTrainedModel, FlexBertModel, FlexBertPredictionHead
from src.bert_layers.loss import get_loss_fn
from src.bert_layers.initialization import init_weights, ModuleType
from src.bert_layers.padding import unpad_input, pad_input
from src.sequence_packer import SequencePacker # 复用Masking逻辑

# =========================================================================
# 1. 定义支持多模态 MLM 的模型类
#    (基于 FlexBertForMaskedLM 修改，加入 CLIP 支持)
# =========================================================================

class FlexBertWithCLIPForMaskedLM(FlexBertPreTrainedModel):
    def __init__(self, config: FlexBertConfig):
        super().__init__(config)
        
        # 1. 视觉编码器 (CLIP)
        vision_model_name = getattr(config, "vision_model_name", "openai/clip-vit-base-patch32")
        self.vision_encoder = CLIPVisionModel.from_pretrained(vision_model_name)
        # 冻结 CLIP
        for param in self.vision_encoder.parameters():
            param.requires_grad = False
            
        # 2. 文本编码器 (FlexBert)
        # 注意：Config 必须开启 add_cross_attention=True
        self.bert = FlexBertModel(config)
        
        # 3. MLM 预测头
        self.head = FlexBertPredictionHead(config)
        if config.tie_word_embeddings:
            decoder_weights = self.bert.embeddings.tok_embeddings.weight
        else:
            decoder_weights = nn.Linear(config.hidden_size, config.vocab_size, bias=False).weight
        self.decoder = nn.Linear(decoder_weights.size(1), decoder_weights.size(0), bias=config.decoder_bias)
        self.decoder.weight = decoder_weights

        # 4. Loss & Utils
        self.loss_fn = nn.CrossEntropyLoss() if not hasattr(config, "loss_function") else get_loss_fn(config)
        self.return_z_loss = config.loss_kwargs.get("return_z_loss", False)
        self.unpad_embeddings = config.unpad_embeddings
        self.masked_prediction = config.masked_prediction # ModernBERT 默认只计算 Mask 部分的 Loss

        # 初始化 FlexBert 权重 (CLIP 已预训练，无需初始化)
        self.post_init()

    def post_init(self):
        self._init_weights(reset_params=False)

    def _init_weights(self, module: Optional[nn.Module] = None, reset_params: Optional[bool] = None):
        if module:
            self._init_module_weights(module)
        else:
            self.bert._init_weights(reset_params=reset_params)
            self.head._init_weights(reset_params=reset_params)
            if not self.config.tie_word_embeddings:
                init_weights(self.config, self.decoder, self.config.hidden_size, type_of_module=ModuleType.final_out)

    @torch.no_grad()
    def unpad_inputs(self, input_ids, attention_mask, position_ids, labels):
        # 调用源码中的 helper
        return unpad_input(input_ids, attention_mask, position_ids, labels)

    def forward(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor, # [Batch, 3, H, W]
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        **kwargs
    ):
        # 1. 获取图片特征 (Freeze)
        with torch.no_grad():
            vision_outputs = self.vision_encoder(pixel_values=pixel_values)
            # [Batch, Seq_Img, Dim]
            image_embeds = vision_outputs.last_hidden_state

        # 2. 处理文本 Unpadding
        indices = None
        cu_seqlens = None
        max_seqlen = None
        
        if self.unpad_embeddings:
            # 现场进行 Unpad
            batch_size, seq_len = input_ids.shape[:2]
            input_ids, indices, cu_seqlens, max_seqlen, position_ids, labels = self.unpad_inputs(
                input_ids, attention_mask, position_ids, labels
            )

        # 3. BERT Forward (Cross Attention 发生在这里)
        # encoder_hidden_states 传入 image_embeds
        encoder_outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            indices=indices,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            encoder_hidden_states=image_embeds, # <--- 关键点
            encoder_attention_mask=None,
        )

        # 4. MLM Head
        # 如果只预测 Masked Token (ModernBERT 优化)
        if self.masked_prediction and labels is not None:
            labels = labels.view(-1)
            encoder_outputs = encoder_outputs.view(labels.shape[0], -1)
            mask_tokens = labels != self.loss_fn.ignore_index
            encoder_outputs = encoder_outputs[mask_tokens]
            labels = labels[mask_tokens]

        logits = self.decoder(self.head(encoder_outputs))

        # 5. Loss 计算
        loss = None
        if labels is not None:
            if not self.masked_prediction:
                labels = labels.view(-1)
                logits = logits.view(labels.shape[0], -1)
            loss = self.loss_fn(logits, labels)

        return {"loss": loss, "logits": logits}

# =========================================================================
# 2. 数据集与 SVG 处理
# =========================================================================

class SVGRenderer:
    @staticmethod
    def clean_and_render(svg_content: str, target_size=224) -> Image.Image:
        try:
            # 强制设置宽高为 100% 以适应画布，防止原始内容过小
            # 1. 移除现有的 width/height 属性
            svg_content = re.sub(r'(<svg[^>]*?)\s+width="[^"]*"', r'\1', svg_content, flags=re.IGNORECASE)
            svg_content = re.sub(r'(<svg[^>]*?)\s+height="[^"]*"', r'\1', svg_content, flags=re.IGNORECASE)
            
            # 2. 添加 width="100%" height="100%"
            if "<svg" in svg_content:
                svg_content = svg_content.replace("<svg", '<svg width="100%" height="100%"', 1)
            
            # 3. 渲染为 PNG
            png_bytes = cairosvg.svg2png(
                bytestring=svg_content.encode('utf-8'),
                output_width=target_size,
                output_height=target_size,
                unsafe=True # 允许加载外部资源(如果有的话)，注意安全风险，但在训练受控数据集时通常没问题
            )
            
            image = Image.open(io.BytesIO(png_bytes)).convert("RGB")
            return image
        except Exception as e:
            print(f"Error rendering SVG: {e}")
            # 返回全黑图片作为 fallback，避免训练中断
            return Image.new("RGB", (target_size, target_size), (0, 0, 0))

class MultimodalSVGDataset(Dataset):
    def __init__(self, hf_dataset, tokenizer, image_processor, max_length=512):
        self.dataset = hf_dataset
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.max_length = max_length
        self.renderer = SVGRenderer()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        raw_svg = item['raw_svg']
        
        # 1. 处理图片
        image = self.renderer.clean_and_render(raw_svg)
        # CLIP Processor 会处理归一化和 Resize
        pixel_values = self.image_processor(images=image, return_tensors="pt")["pixel_values"][0]

        # 2. 处理文本 (Raw SVG string)
        # 这里暂时不进行 Padding，由 Collate_fn 处理
        # 或者为了配合 ModernBERT 的 Unpadding 逻辑，我们先 Pad 到 max_length
        # ModernBERT 内部会自动 unpad
        text_inputs = self.tokenizer(
            raw_svg,
            max_length=self.max_length,
            truncation=True,
            padding="max_length", # Pad 到最大长度，方便 batch
            return_tensors="pt"
        )
        
        return {
            "input_ids": text_inputs["input_ids"][0],
            "attention_mask": text_inputs["attention_mask"][0],
            "pixel_values": pixel_values
        }

class MLMCollator:
    def __init__(self, tokenizer, mask_prob=0.30): # ModernBERT 推荐 30% masking
        self.tokenizer = tokenizer
        self.mask_prob = mask_prob
        self.mask_token_id = tokenizer.mask_token_id
        self.pad_token_id = tokenizer.pad_token_id
        
    def __call__(self, batch):
        # 1. 堆叠 Tensor
        pixel_values = torch.stack([item["pixel_values"] for item in batch])
        input_ids = torch.stack([item["input_ids"] for item in batch])
        attention_mask = torch.stack([item["attention_mask"] for item in batch])
        
        # 2. 进行 MLM Masking
        # 使用 numpy 进行随机掩码 (参考 src/sequence_packer.py)
        # 将 input_ids 转为 numpy
        input_ids_np = input_ids.numpy()
        
        # 调用 SequencePacker 中的静态方法进行 Masking
        masked_ids, labels = SequencePacker.mlm_masking(
            input_ids_np,
            mask_prob=self.mask_prob,
            mask_token=self.mask_token_id,
            pad_token=self.pad_token_id,
            ignore_index=-100
        )
        
        return {
            "input_ids": torch.from_numpy(masked_ids),
            "attention_mask": attention_mask,
            "labels": torch.from_numpy(labels),
            "pixel_values": pixel_values
        }

# =========================================================================
# 3. 主训练流程
# =========================================================================

def main():
    # 0. 设置随机种子
    reproducibility.seed_all(42)
    
    # 1. 配置参数
    MAX_SEQ_LEN = 1024 # SVG 文本通常较长
    BATCH_SIZE = 16    # 4090 上根据显存调整，启用 FA2 可以稍微大点
    
    print("Loading Dataset...")
    # 加载数据集 (使用 streaming=False 以便简单处理，数据量不大)
    # 替换为你实际的 Token
    # os.environ["HF_TOKEN"] = "hf_QLisQanMCFlhJokmHFOJqwzbZMqgxh1231" 
    ds = load_dataset('VectorGraphics/svg-corpus-private', 'svg_viewer_dataset', split='train')
    
    print("Loading Tokenizer & Processor...")
    # 使用 ModernBERT 推荐的 tokenizer 或者普通的 bert tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    # 封装数据集
    train_dataset = MultimodalSVGDataset(ds, tokenizer, image_processor, max_length=MAX_SEQ_LEN)
    
    # DataLoader
    collator = MLMCollator(tokenizer)
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=8,
        collate_fn=collator,
        pin_memory=True
    )

    print("Configuring Model...")
    # ModernBERT Config
    config = FlexBertConfig(
        vocab_size=30528, # 对齐到 8 的倍数
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        max_position_embeddings=MAX_SEQ_LEN,
        
        # 效率开关
        padding="unpadded",
        unpad_embeddings=True,
        use_fa2=True, # 4090 必须开启
        
        # 多模态开关
        add_cross_attention=True,
        cross_attention_frequency=2, # 每2层做一次 Cross Attn
        vision_hidden_size=768,      # CLIP Base 维度
        cross_attention_layer="cross_unpad",
        
        # 训练任务
        masked_prediction=True,      # 只预测 mask 部分，加速
        loss_function="fa_cross_entropy",

        embedding_layer='sans_pos'
    )
    
    # 实例化模型
    model = FlexBertWithCLIPForMaskedLM(config)
    
    # 打印参数量
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable Parameters: {n_params / 1e6:.2f} M (CLIP is frozen)")

    # 包装为 Composer Model
    # HuggingFaceModel 会处理 metrics 计算和 batch 拆包
    composer_model = HuggingFaceModel(
        model=model,
        tokenizer=tokenizer,
        use_logits=True,
        metrics=[], # 可以添加 MaskedAccuracy
        allow_embedding_resizing=True
    )

    # 优化器
    optimizer = DecoupledAdamW(
        composer_model.parameters(),
        lr=5e-4,
        betas=(0.9, 0.98),
        eps=1e-6,
        weight_decay=1e-5
    )
    
    # 学习率调度
    scheduler = LinearWithWarmupScheduler(t_warmup="0.06dur")

    print("Starting Trainer...")
    trainer = Trainer(
        model=composer_model,
        train_dataloader=train_dataloader,
        optimizers=optimizer,
        schedulers=scheduler,
        max_duration="10ep", # 训练 10 个 epoch
        device="gpu",
        precision="amp_bf16", # 4090 推荐 bf16
        seed=42,
        # logging
        progress_bar=True,
        log_to_console=True,
        console_log_interval="10ba",
        # checkpoint
        save_folder="./checkpoints_svg_multimodal",
        save_interval="1ep",
        save_overwrite=True
    )

    trainer.fit()

if __name__ == "__main__":
    main()