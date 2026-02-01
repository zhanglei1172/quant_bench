# Qwen3-8B 量化工作流

本项目提供了 Qwen3-8B 模型从预平滑 (Pre-smoothing) 到量化 (Quantization) 再到最终评测 (Evaluation) 的完整技术链路。

## 核心流程

1.  **预平滑 (Pre-smoothing)**：在保持模型结构不变的前提下，通过旋转平滑（Rotation Smoothing）和权重等效变换（Equivalent Transformation）处理激活值，保存为 `float32` 权重。此步骤理论上无精度损失。
2.  **模型量化 (Quantization)**：采用 GPTQ 等算法对权重进行 8-bit/4-bit 量化，并产生相应的激活量化参数。此环节是精度损失的主要来源。
3.  **指标评测 (Evaluation)**：基于 `vLLM` 部署推理服务，并结合 `evalscope` 进行自动化测评，通过比对各阶段指标分析量化损失原因。

## 评测数据

### 主流程测评
| 模型版本 | MMLU | GSM8K | 备注 |
| :--- | :---: | :---: | :--- |
| Qwen3-8B (Base) | 0.8321 | 0.8809 | 原始 FP16 模型 |
| Qwen3-8B-R1R2 (Smooth) | 0.8428 | 0.8887 | 预平滑处理后 |
| Qwen3-8B-R1R2-R4-gptq-W8A8-static | 0.7598 | 0.7891 | 静态 W8A8 量化 |

### 消融实验与分析
通过以下额外测评，可以发现影响量化精度的关键因素：

*   **激活量化是瓶颈**：在 Weight-only (W8) 模式下，无论 RTN 还是 GPTQ 损失均极小，激活量化误差主因是down_proj层激活的8bit静态量化造成的。
*   **动态量化显著提升**：将激活量化由 `static per-tensor` 转为 `dynamic per-token` 后，精度接近 Weight-only 效果。
*   **关键敏感层**：`down_proj` 层对量化误差最为敏感。排除该层的权重与激活量化后，模型指标几乎恢复至无损水平。

| 模型版本 | MMLU | GSM8K |
| :--- | :---: | :---: |
| Qwen3-8B-R1R2-gptq-W8-weightonly | 0.8301 | 0.8672 |
| Qwen3-8B-R1R2-rtn-W8-weightonly | 0.8301 | 0.8691 |
| Qwen3-8B-R1R2-R4-gptq-W8A8-dynamic | 0.8359 | 0.8652 |
| Qwen3-8B-R1R2-gptq-ignore-down-W8A8-static | 0.8438 | 0.8770 |




