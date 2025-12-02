## 系统架构图

```
bench_multimodal.py
    │
    └── multimodal_llm_engine.py (MultimodalLLMEngine)
            │
            ├── multimodal_sequence.py (MultimodalSequence)
            │
            ├── multimodal_block_manager.py (MultimodalBlockManager)
            │
            └── multimodal_scheduler.py (MultimodalScheduler)
                    │
                    ├── VisionEncoder [cuda:0]
                    │   └── nanovllm/models/clip.py (CLIPVisionModel)
                    │   └── nanovllm/models/clip_image_processing.py
                    │
                    └── LLMDecoder [cuda:1]
                        ├── nanovllm/models/llama.py (LlamaForCausalLM)
                        └── nanovllm/models/llava.py (LlavaMultiModalProjector)
```

## 新添加的文件功能说明

### 1. bench_multimodal.py

主要测两个东西：延迟和吞吐

里面有几个重要的函数：
- `load_coco_images()` - 加载COCO数据集的图片来测试
- `benchmark_throughput_with_output()` - 测试同时处理多个请求
- `benchmark_latency_with_output()` - 测试单个请求的速度


---

### 2. multimodal_scheduler.py
#### 2.1 VisionEncoder
**功能：** 在独立 GPU 上运行的视觉编码器
- `_init_vision_model()`: 初始化 CLIP 视觉模型
- `_load_vision_weights()`: 加载预训练权重
- `encode_images()`: 编码图像为特征向量
- 支持批量图像处理

#### 2.2 LanguageDecoder
**功能：** 在独立 GPU 上运行的语言解码器
- `_init_language_model()`: 初始化语言模型
- `_load_language_weights()`: 加载模型权重
- `generate_text()`: 生成文本

#### 2.3 MultimodalScheduler
**关键方法：**
- `submit_task()`: 提交新任务
- `_vision_worker()`: 视觉处理工作线程
  - 批量处理图像
  - 提取视觉特征

- `_language_worker()`: 语言处理工作线程
  - 批量生成文本
  - 管理 KV cache

- `_schedule_tasks()`: 任务调度逻辑
  - 优先级调度

---