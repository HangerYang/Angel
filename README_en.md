English | [简体中文](README.md)

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="./docs/source/assets/logos/angelslim_logo_light.png">
    <img alt="AngelSlim" src="./docs/source/assets/logos/angelslim_logo.png" width=45%>
  </picture>
</p>

<h3 align="center">
Dedicated to building a more intuitive, comprehensive, and efficient LLMs compression toolkit.
</h3>

<p align="center">
          📖 <a href="https://angelslim.readthedocs.io/">Documentation</a>&nbsp&nbsp | &nbsp&nbsp🤗 <a href="https://huggingface.co/AngelSlim">Hugging Face</a>&nbsp&nbsp | &nbsp&nbsp🤖 <a href="https://modelscope.cn/organization/AngelSlim">ModelScope</a>&nbsp&nbsp | &nbsp&nbsp💬 <a href="./docs/source/assets/angel_slim_wechat.png">WeChat</a>
<br>
</p>

## Table of Contents

* [Latest Updates](#latest-updates)
* [Key Features](#key-features)
* [Supported Models](#supported-models)
* [How to Use](#how-to-use)

  * [Install AngelSlim](#install-angelslim)
  * [Quick Start](#quick-start)
  * [deployment & Evaluation](#deployment)
* [Benchmark](#benchmark)
* [License](#license)
* [Citation](#citation)
* [Technical Discussion](#technical-discussion)

## 📣 Latest Updates

* \[25/07/04] We now support quantization for Hunyuan/Qwen2.5/Qwen3/DeepSeek-R1-Distill-Qwen and other models, including INT8/FP8/INT4 algorithms.
              We also opensource Qwen3-8B`s Eagle3 model weight.

Coming soon:

* [ ] Support W4A8 quantization for DeepSeek-R1.
* [ ] Support quantization for multimodal models like Qwen-VL.
* [ ] Release of new algorithm for speculative sampling.

## 🌟 Key Features

* **Highly Integrated**: This toolkit integrates mainstream compression algorithms into a unified framework, offering developers one-click access with exceptional ease of use.
* **Continuous Innovation**: Beyond integrating widely-used industry algorithms, we are continuously researching better compression algorithms, which will be gradually open-sourced in the future.
* **Performance-Driven**: We continuously optimize end-to-end performance in model compression workflows and algorithm deployment, such as enabling quantization of models like Qwen3-235B and DeepSeek-R1 on a single GPU.

## 💼 Supported Models

### Quantization
Currently supports the following LLMs, including Hunyuan-Dense, Hunyuan-MoE, Qwen3-Dense, Qwen3-MoE, Qwen2.5, DeepSeek-R1 distilled Qwen models, and QwQ::

| Model | FP8-Dynamic | FP8-Static | INT8-Dynamic | INT4-GPTQ | INT4-AWQ |
| --------------------------------------------------------------------------------------------------------------------------- | ----------- | ---------- | ------------ | --------- | -------- |
| [Hunyuan-Dense](https://huggingface.co/tencent/Hunyuan-7B-Instruct)                                                         | ✅           | ✅          | ✅            | ✅         | ✅        |
| [Hunyuan-MoE](https://huggingface.co/collections/tencent/hunyuan-a13b-685ec38e5b46321e3ea7c4be)                             | ✅           | ✅          | ✅            | ✅         | ✅        |
| [Qwen3-Dense](https://huggingface.co/collections/AngelSlim/qwen3-quant-68652e26da31740739d154f8)                            | ✅           | ✅          | ✅            | ✅         | ✅        |
| [Qwen3-MoE](https://huggingface.co/collections/AngelSlim/qwen3-quant-68652e26da31740739d154f8)                              | ✅           | ✅          | ✅            | ✅         | ✅        |
| [Qwen2.5](https://huggingface.co/collections/AngelSlim/qwen2-25-quant-68652d6cbdf5c0d4b1c4499a)                             | ✅           | ✅          | ✅            | ✅         | ✅        |
| [DeepSeek-R1-Distill-Qwen](https://huggingface.co/collections/AngelSlim/deepseek-r1-distill-quant-68652f16a9c206b030b05f7f) | ✅           | ✅          | ✅            | ✅         | ✅        |
| [QwQ](https://huggingface.co/collections/AngelSlim/qwen3-quant-68652e26da31740739d154f8)                                    | ✅           | ✅          | ✅            | ✅         | ✅        |

### Speculative Decoding
The Eagle3 weights for the Qwen3-8B model are now available, with Eagle3 weights for other models in the Qwen3 series to be released soon.

| Model     |      Eagle3       | 
| ----------| ----------------- | 
| [Qwen3-8B](https://huggingface.co/AngelSlim/Qwen3-8B_eagle3/tree/main) |      ✅           | 
| Qwen3-14B | coming soon |
| Qwen3-32B | coming soon |

## 🛎️ How to Use

### Install AngelSlim

We recommend using `pip` to install the latest stable version of `AngelSlim`:

```shell
pip install angelslim
```

Alternatively, you can clone the repository and install from source in editable mode:

```shell
cd AngelSlim && python setup.py install
```

For more detailed installation instructions, please refer to the [Installation Documentation](./docs/source/getting_started/installation.md).

### Quick Start

After installing `AngelSlim`, you can quickly start by running the following script to perform static `FP8` quantization on the `Qwen3-1.7B` model:

* One-click Start

  ```shell
  python3 tools/run.py -c configs/qwen3/fp8_static/qwen3-1_7b_fp8_static.yaml
  ```

  This example will load the HuggingFace model and perform activation value calibration using the `dataset` specified in the config file, saving the quantized model weights.

* Code-based Start

  To perform dynamic `FP8` quantization on `Qwen3-1.7B`:

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

For more details, please refer to the [Quick Start Documentation](./docs/source/getting_started/quickstart.md).

### 🖥️ Deployment and Testing

#### 1. API Service Deployment

After specifying the quantized model path `MODEL_PATH`, you can deploy an OpenAI-compatible API service using the following LLMs inference frameworks:

**vLLM**

Use the following script to launch a [vLLM](https://github.com/vllm-project/vllm) server, recommended version `vllm>=0.8.5.post1`. For MOE INT8 quantized models, vllm>=0.9.0 is required.


```shell
bash deploy/run_vllm.sh $MODEL_PATH
```

**SGLang**


Use the following script to launch a [SGLang](https://github.com/sgl-project/sglang) server, recommended version `sglang>=0.4.6.post1`.

```shell
bash deploy/run_sglang.sh $MODEL_PATH
```

#### 2. Service Invocation

Invoke requests via [OpenAI's API format](https://platform.openai.com/docs/api-reference/introduction):

```shell
bash deploy/openai.sh $MODEL_PATH
```

#### 3. Performance Evaluation

Evaluate the performance of quantized model using [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness), recommended version`lm-eval>=0.4.8`:

```shell
bash deploy/lm_eval.sh $MODEL_PATH
```

For more detaileds, please refer to the [Deployment Documentation](./docs/source/deployment/deploy.md).


## 📈 Benchmark

### Quantization

The performance test results for selected models are shown below. For the complete benchmark, refer to the [Benchmark documentation](./docs/source/performance/quantization/benchmarks.md)

#### Hunyuan Series Models

Benchmark results for the `Hunyuan-A13B-Instruct` model with `FP8` and `INT4-GPTQ` quantization algorithms on datasets including `AIME 2024`, `GSM8K`, `BBH`, and `DROP`:

|   Bench   | Hunyuan-A13B-Instruct | Hunyuan-A13B-Instruct-FP8 | Hunyuan-A13B-Instruct-Int4-GPTQ | 
|:---------:|:---------------------:|:-------------------------:|:-------------------------------:|
| AIME 2024 |         87.3          |           86.7            |              86.7               |
|   GSM8K   |         94.39         |           94.01           |              94.24              |
|    BBH    |         89.1          |           88.34           |              87.91              |
|   DROP    |         91.1          |           91.1            |              91.05              |

#### Qwen3 Series Models

Benchmark results for Qwen3 series models with `FP8-Static`, `FP8-Dynamic`, `INT4-GPTQ`, and `INT4-AWQ` quantization algorithms on datasets including `CEVAL`, `MMLU`, `GSM8K`, and `HUMANEVAL`:

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

#### Other Models

Benchmark results for other models with `FP8-Static`, `FP8-Dynamic`, `INT4-GPTQ`, and `INT4-AWQ` quantization algorithms on datasets including `CEVAL`, `MMLU` and `GSM8K`:

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

### Speculative Decoding
Benchmark results for Qwen3 series models with `Eagle3` speculative decoding algorithm on datasets including `MT-bench`, `HunmanEval`, `GSM8K`, and `Alpaca`:

#### Qwen3-8B
|             |        | Datasets |              |           |                |         |               |         |        | 
| ----------- | ------ | -------- | ------       | --------- | ------         | ------- | ------        | ------- | ------ |
|             |        | MT-bench |              | HumanEval |                | GSM8K   |               | Alpaca  |        |
| Temperature | Method | Speedup  | Accept length| Speedup   | Accept length  | Speedup | Accept length | Speedup | Accept length |
| T=0         | Eagle3 | 2.63x    | 3.65         | 2.76x     | 3.85            | 2.82x   | 3.90          | 2.62x   | 3.48   |
| T=1         | Eagle3 | 1.98x    | 2.75         | 2.25x     | 3.11            | 2.31x   | 3.15          | 2.10x   | 2.76   |


## 📝 License

The code for this project is open-sourced under the [License for AngelSlim](LICENSE).

## 🔗 Citation

```
@software{AngelSlim2025,
    title={{AngelSlim}},
    author={Tencent AngelSlim Project Contributors},
    year={2025},
    month={6},
    url={https://github.com/Tencent/AngelSlim},
}
```

## 💬 Technical Discussion

* AngelSlim is continuously iterating and new features will be released soon. If you have any questions or suggestions, please open an issue on GitHub or join our [WeChat technical discussion group](./docs/source/assets/angel_slim_wechat.png).
