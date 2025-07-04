简体中文 | [English](README_en.md)

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="./docs/source/assets/logos/angelslim_logo_light.png">
    <img alt="AngelSlim" src="./docs/source/assets/logos/angelslim_logo.png" width=55%>
  </picture>
</p>

<h3 align="center">
致力于打造更易用、更全面和更高效的大模型压缩工具包
</h3>

<p align="center">
          📖 <a href="https://angelslim.readthedocs.io/">Documentation</a>&nbsp&nbsp | &nbsp&nbsp🤗 <a href="https://huggingface.co/AngelSlim">Hugging Face</a>&nbsp&nbsp | &nbsp&nbsp🤖 <a href="https://modelscope.cn/organization/AngelSlim">ModelScope</a>&nbsp&nbsp | &nbsp&nbsp💬 <a href="./docs/source/assets/angel_slim_wechat.png">WeChat (微信)</a>
<br>
</p>


## 目录
- [最新进展](#最新进展)
- [主要特性](#主要特性)
- [支持模型](#支持模型)
- [如何使用](#如何使用)
  - [安装 AngelSlim](#安装-AngelSlim)
  - [快速开始](#快速开始)
  - [部署与测试](#部署与测试)
- [Benchmark](#benchmark)
- [许可协议](#许可协议)
- [引用](#引用)
- [技术交流](#技术交流)

## 📣最新进展

- [25/07/04] 我们支持了`Hunyuan/Qwen2.5/Qwen3/DeepSeek-R1-Distill-Qwen`等模型的量化，包含INT8、FP8、INT4等算法。
我们还开源了`Qwen3-8B`模型的Eagle3权重。

Coming soon：
- [ ] DeepSeek-R1的W4A8量化支持
- [ ] 多模态Qwen-VL模型的量化支持
- [ ] 投机采样新算法发布

## 🌟主要特性

- **高度集成化**：本工具将主流的压缩算法集成到工具，开发者可一键式调用，具有很好的易用性。
- **持续算法创新**：本工具除了集成工业界使用最广的算法，还持续自研更好的压缩算法，并且会陆续开源。
- **追求极致性能**：在模型压缩流程、压缩算法部署方面，本工具持续端到端优化，例如单卡GPU可量化Qwen3-235B和Deepseek-R1。

## 💼支持模型

### 量化
目前已支持文生文任务Hunyuan-Dense、Hunyuan-MoE、Qwen3-Dense、Qwen3-MoE、Qwen2.5、DeepSeek-R1蒸馏Qwen模型、QwQ等系列的主要模型：

| 模型名      | FP8-Dynamic       | FP8-Static        | INT8-Dynamic | INT4-GPTQ         | INT4-AWQ          |
| ---------- | ----------------- | ----------------- | ----------------- | ----------------- | ----------------- |
| [Hunyuan-Dense](https://huggingface.co/tencent/Hunyuan-7B-Instruct)   |      ✅           |         ✅           | ✅           |    ✅               |         [ ]           |
| [Hunyuan-MoE](https://huggingface.co/collections/tencent/hunyuan-a13b-685ec38e5b46321e3ea7c4be)   |      ✅           |         ✅           | ✅           |    ✅               |         [ ]           |
| [Qwen3-Dense](https://huggingface.co/collections/AngelSlim/qwen3-quant-68652e26da31740739d154f8)               |      ✅           |         ✅           | ✅           |    ✅               |         ✅           |
| [Qwen3-MoE](https://huggingface.co/collections/AngelSlim/qwen3-quant-68652e26da31740739d154f8)            |      ✅           |         ✅           | ✅           |     ✅             |        ✅            |
| [Qwen2.5](https://huggingface.co/collections/AngelSlim/qwen2-25-quant-68652d6cbdf5c0d4b1c4499a)            |      ✅           |         ✅           | ✅           |     ✅             |        ✅            |
| [DeepSeek-R1-Distill-Qwen](https://huggingface.co/collections/AngelSlim/deepseek-r1-distill-quant-68652f16a9c206b030b05f7f) |      ✅           |         ✅           | ✅           |      ✅             |        ✅            |
| [QwQ](https://huggingface.co/collections/AngelSlim/qwen3-quant-68652e26da31740739d154f8) |      ✅           |         ✅           |       ✅             | ✅           |       ✅            |

### 投机采样
目前已开源Qwen3-8B模型的Eagle3权重，Qwen3系列其他模型的Eagle3权重也即将开放。

| 模型名     |      Eagle3       | 
| ----------| ----------------- | 
| [Qwen3-8B](https://huggingface.co/AngelSlim/Qwen3-8B_eagle3/tree/main) |      ✅           | 
| Qwen3-14B | coming soon |
| Qwen3-32B | coming soon |

## 🛎️如何使用

### 安装 AngelSlim

推荐使用`pip`直接安装最新稳定版`AngelSlim`：

```shell
pip install angelslim
```

也可以选择克隆代码仓库后，以可编辑的方式从源代码安装：

```shell
cd AngelSlim && python setup.py install
```

更详细的安装说明可参考[安装文档](./docs/source/getting_started/installation.md)。

### 快速开始

完成安装`AngelSlim`后，您可以通过以下脚本快速开始，完成`Qwen3-1.7B`模型的静态`FP8`量化：

- 一键式启动

  ```shell
  python3 tools/run.py -c configs/qwen3/fp8_static/qwen3-1_7b_fp8_static.yaml
  ```

  该示例将会加载`HugggingFace`模型， 使用`config`配置的`dataset`数据进行激活值校准，量化产出模型权重.

- 源码启动

  对`Qwen3-1.7B`完成动态`FP8`量化：

  ```python
  from angelslim.engine import Engine

  slim_engine = Engine()
  # Prepare model
  slim_engine.prepare_model(model_name="Qwen", model_path="Qwen/Qwen3-1.7B",)
  # Initialize compressor
  slim_engine.prepare_compressor("PTQ", default_method="fp8_dynamic")
  # Compress model
  slim_engine.run()
  # Save compressed model
  slim_engine.save("./output")
  ```

详情请参考[快速开始文档](./docs/source/getting_started/quickstrat.md)。

### 部署与测试

#### 1. 服务部署

指定量化模型路径 `MODEL_PATH` 后，支持通过以下推理框架部署 OpenAI 兼容的 API 服务：

**vLLM**

[vLLM](https://github.com/vllm-project/vllm) 服务启动脚本，建议版本`vllm>=0.8.5.post1`，部署MOE INT8量化模型需要`vllm>=0.9.0`。

```shell
bash deploy/run_vllm.sh $MODEL_PATH
```

**SGLang**

[SGLang](https://github.com/sgl-project/sglang) 服务启动脚本，建议版本 `sglang>=0.4.6.post1`：

```shell
bash deploy/run_sglang.sh $MODEL_PATH
```

#### 2. 服务调用

通过 [OpenAI 格式](https://platform.openai.com/docs/api-reference/introduction) 接口发起请求：

```shell
bash deploy/openai.sh $MODEL_PATH
```

#### 3. 效果验证

使用 [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) 评估量化模型精度，建议版本`lm-eval>=0.4.8`：

```shell
bash deploy/lm_eval.sh $MODEL_PATH
```

详细操作指南请参阅[部署文档](./docs/source/deployment/deploy.md)。

## 📈Benchmark

### 量化

下面只展示了部分模型的效果测试情况，完整Benchmark可以参考[Benchmark文档](./docs/source/performance/quantization/benchmarks.md)

#### Hunyuan系列模型

Hunyuan-A13B-Instruct的`BF16`、`FP8`、`INT4-GPTQ`在`AIME 2024`、`GSM8K`、`BBH`、`DROP`上的评测结果如下：

|   Bench   | Hunyuan-A13B-Instruct | Hunyuan-A13B-Instruct-FP8 | Hunyuan-A13B-Instruct-Int4-GPTQ | 
|:---------:|:---------------------:|:-------------------------:|:-------------------------------:|
| AIME 2024 |         87.30         |           86.70           |              86.70              |
|   GSM8K   |         94.39         |           94.01           |              94.24              |
|    BBH    |         89.10         |           88.34           |              87.91              |
|   DROP    |         91.10         |           91.10           |              91.05              |

#### Qwen3系列模型

Qwen3系列模型的`BF16`、`FP8-Static`、`FP8-Dynamic`、`INT8-Dynamic`、`INT4-GPTQ`、`INT4-AWQ`在`CEVAL`、`MMLU`、`GSM8K`、`HUMANEVAL`上的评测结果如下：

| Model                         | Quantization | CEVAL | MMLU  | GSM8K | HUMANEVAL |
|-------------------------------|--------------|-------|-------|-------|-----------|
| Qwen3-0.6B                    | BF16         | 45.84 | 47.21 | 42.99 | 19.51     |
|                               | FP8-Static   | 45.99 | 46.87 | 38.06 | 18.90     |
|                               | FP8-Dynamic  | 45.99 | 46.93 | 38.29 | 20.73     |
|                               | INT8-Dynamic | 45.17 | 46.95 | 41.17 | 21.34     |
| Qwen3-8B                      | BF16         | 79.27 | 74.78 | 87.79 | 63.41     |
|                               | FP8-Static   | 78.23 | 74.79 | 86.96 | 62.20     |
|                               | FP8-Dynamic  | 78.45 | 74.75 | 87.64 | 62.80     |
|                               | INT8-Dynamic | 78.01 | 74.84 | 86.96 | 67.07     |
|                               | INT4-GPTQ    | 77.19 | 73.26 | 86.43 | 62.20     |
|                               | INT4-AWQ     | 76.15 | 73.59 | 86.96 | 63.41     |
| Qwen3-14B                     | BF16         | 83.06 | 78.90 | 88.40 | 55.49     |
|                               | FP8-Static   | 82.62 | 78.57 | 89.46 | 57.32     |
|                               | FP8-Dynamic  | 82.24 | 78.92 | 88.32 | 52.44     |
|                               | INT8-Dynamic | 81.87 | 78.13 | 86.28 | 56.10     |
|                               | INT4-GPTQ    | 81.05 | 78.02 | 87.34 | 57.93     |
|                               | INT4-AWQ     | 82.02 | 77.68 | 84.23 | 61.59     |
| Qwen3-32B                     | BF16         | 86.55 | 82.00 | 74.53 | 37.80     |
|                               | FP8-Static   | 86.92 | 81.78 | 70.20 | 39.63     |
|                               | FP8-Dynamic  | 86.55 | 81.89 | 70.43 | 38.41     |
|                               | INT4-GPTQ    | 86.18 | 81.01 | -     | 43.29     |
|                               | INT4-AWQ     | 86.18 | 81.54 | -     | 36.59     |
| Qwen3-30B-A3B                 | BF16         | 83.66 | 79.36 | 89.99 | 31.71     |
|                               | FP8-Static   | 83.95 | 79.47 | 89.01 | 31.10     |
|                               | FP8-Dynamic  | 84.10 | 79.40 | 89.16 | 32.93     |
|                               | INT8-Dynamic | 83.36 | 79.48 | 89.16 | 34.15     |
| Qwen3-235B-A22B               | BF16         | 89.60 | 86.28 | 85.29 | 27.44     |
|                               | FP8-Static   | 89.67 | 86.19 | 86.96 | 27.44     |
|                               | FP8-Dynamic  | 89.67 | 86.18 | 85.22 | 28.05     |
|                               | INT8-Dynamic | 88.93 | 86.20 | 86.20 | 23.78     |
| QwQ-32B                       | BF16         | 85.74 | 82.03 | 73.31 | 42.68     |
|                               | FP8-Static   | 85.44 | 81.91 | 75.36 | 42.68     |
|                               | FP8-Dynamic  | 85.07 | 81.93 | 75.66 | 42.07     |
|                               | INT4-GPTQ    | 84.03 | 81.26 | 68.23 | 45.73     |
|                               | INT4-AWQ     | 83.58 | 81.01 | 68.69 | 43.29     |

#### 其他模型

其他模型的`BF16`、`FP8-Static`、`FP8-Dynamic`、`INT4-GPTQ`、`INT4-AWQ`在`CEVAL`、`MMLU`、`GSM8K`上的评测结果如下：

| Model                         | Quantization | CEVAL | MMLU  | GSM8K |
|-------------------------------|--------------|-------|-------|-------|
| Qwen2.5-1.5B-Instruct         | BF16         | 67.01 | 60.05 | 54.28 |
|                               | FP8-Static   | 66.27 | 60.23 | -     |
|                               | FP8-Dynamic  | 66.79 | 60.08 | 51.71 |
| Qwen2.5-7B-Instruct           | BF16         | 81.20 | 74.55 | 79.98 |
|                               | FP8-Static   | 81.13 | 74.03 | 79.30 |
|                               | FP8-Dynamic  | 80.31 | 74.07 | 79.00 |
|                               | INT4-GPTQ    | 79.05 | 73.05 | 74.75 |
|                               | INT4-AWQ     | 79.35 | 73.22 | 79.38 |
| Qwen2.5-32B-Instruct          | BF16         | 87.30 | 83.21 | 81.73 |
|                               | FP8-Static   | 87.59 | 83.08 | 81.58 |
|                               | FP8-Dynamic  | 87.30 | 83.04 | 81.58 |
|                               | INT4-GPTQ    | 86.70 | 82.45 | 82.03 |
|                               | INT4-AWQ     | 87.00 | 82.64 | -     |
| DeepSeek-R1-Distill-Qwen-7B   | BF16         | 53.49 | 53.80 | 75.74 |
|                               | FP8-Static   | 53.57 | 54.17 | 76.19 |
|                               | FP8-Dynamic  | 52.97 | 54.13 | 74.15 |
|                               | INT4-GPTQ    | 51.86 | 52.44 | 75.89 |
|                               | INT4-AWQ     | 53.49 | 53.70 | -     |
| DeepSeek-R1-Distill-Qwen-14B  | BF16         | 77.71 | 74.28 | 85.67 |
|                               | FP8-Static   | 77.56 | 74.66 | 86.73 |
|                               | FP8-Dynamic  | 76.82 | 74.63 | 87.11 |
|                               | INT4-GPTQ    | 74.29 | 72.37 | 84.61 |
|                               | INT4-AWQ     | 74.81 | 73.00 | 86.05 |
| DeepSeek-R1-Distill-Qwen-32B  | BF16         | 84.18 | 80.89 | 87.41 |
|                               | FP8-Static   | 83.43 | 80.90 | 87.57 |
|                               | FP8-Dynamic  | 83.73 | 81.10 | 86.43 |
|                               | INT4-GPTQ    | 84.10 | 79.80 | 86.73 |
|                               | INT4-AWQ     | 82.84 | 80.15 | 87.19 |

### 投机采样
Qwen3系列的Eagle3模型在MT-bench/HunmanEval/GSM8K/Alpaca上的加速结果如下：
#### Qwen3-8B

|             |        | Datasets |              |           |                |         |               |         |        | 
| ----------- | ------ | -------- | ------       | --------- | ------         | ------- | ------        | ------- | ------ |
|             |        | MT-bench |              | HumanEval |                | GSM8K   |               | Alpaca  |        |
| Temperature | Method | Speedup  | Accept length| Speedup   | Accept length  | Speedup | Accept length | Speedup | Accept length |
| T=0         | Eagle3 | 2.63x    | 3.65         | 2.76x     | 3.85            | 2.82x   | 3.90          | 2.62x   | 3.48   |
| T=1         | Eagle3 | 1.98x    | 2.75         | 2.25x     | 3.11            | 2.31x   | 3.15          | 2.10x   | 2.76   |


## 📝许可协议
本项目的代码依照 [License for AngelSlim](LICENSE) 协议开源。

## 🔗引用
```
@software{AngelSlim2025,
    title={{AngelSlim}},
    author={Tencent AngelSlim Project Contributors},
    year={2025},
    month={7},
    url={https://github.com/Tencent/AngelSlim},
}
```

## 💬技术交流

- AngelSlim正在快速迭代更新中，后续会推出更多的功能，有问题或建议欢迎通过GitHub Issues给我们提issue，或者加入[微信技术交流群](./docs/source/assets/angel_slim_wechat.png)。
