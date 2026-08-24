# Generation prompts: what actually makes SmolVLM-256M answer at length

Generated 2026-08-20T19:46:18-07:00.

| | |
|---|---|
| population | N=10 prompts per benchmark — the first 10 of the same ordered subset the N=80 sweeps use |
| arm | target-only (`USE_EAGLE=0`, no draft): this measures prompt compliance, not acceptance |
| prompt styles | `raw` (default) plus the styles below, via `--prompt_style` |
| temperature | 0 |
| decode | `output_len=1024`, `max_num_seqs=1`, `enforce_eager`, gpu-mem 0.8, tp 1 |
| target | HuggingFaceTB/SmolVLM-256M-Instruct |

N=10 means single samples move a mean: chartqa is 138.3 here but 86.0 at N=80,
purely from one 522-word outlier. The first 10 generations are character-identical
between the two populations, so this is subsampling, not a different configuration.

| style | prompt |
|---|---|
| `raw` | the dataset question, verbatim |
| `verbose` | question + 'Please answer with an explanation.' |
| `detail_prefix` | 'Answer the following question in detail, explaining your reasoning: ' + question |
| `cot` | question + newline + "Let's think step by step and explain the reasoning before giving the answer." |
| `describe_first` | 'Describe what you see in the image in detail, then answer this question: ' + question |
| `min_words` | question + 'Please answer with at least 100 words.' |
| `answer_then_describe` | 'Answer this question: ' + question + ' Then describe the image in detail to justify your answer.' |

## VQA benchmarks — avg output tokens

| benchmark | `raw` | `verbose` | `detail_prefix` | `cot` | `describe_first` | `min_words` | `answer_then_describe` |
|---|---|---|---|---|---|---|---|
| MMStar | 8.7 | 9.1 (1.05x) | 23.0 (2.64x) | 26.2 (3.01x) | 20.9 (2.40x) | 9.0 (1.03x) | 89.1 (10.24x) |
| MMMU | 114.3 | 20.6 (0.18x) | 24.6 (0.22x) | 12.1 (0.11x) | 41.1 (0.36x) | 111.1 (0.97x) | 215.8 (1.89x) |
| textvqa | 10.3 | 5.2 (0.50x) | 8.1 (0.79x) | 4.6 (0.45x) | 425.8 (41.34x) | 5.7 (0.55x) | 140.6 (13.65x) |
| chartqa | 138.3 | 141.0 (1.02x) | 140.3 (1.01x) | 142.4 (1.03x) | 45.9 (0.33x) | 143.4 (1.04x) | 142.5 (1.03x) |
| mathvista | 20.6 | 18.7 (0.91x) | 30.2 (1.47x) | 134.9 (6.55x) | 20.8 (1.01x) | 11.2 (0.54x) | 162.3 (7.88x) |
| **mean** | **58.4** | **38.9** | **45.2** | **64.0** | **110.9** | **56.1** | **150.1** |

## Answer retention (raw answer still present in the styled answer)

| benchmark | `verbose` | `detail_prefix` | `cot` | `describe_first` | `min_words` | `answer_then_describe` |
|---|---|---|---|---|---|---|
| MMStar | 6/10 | 7/10 | 6/10 | 7/10 | 5/10 | 7/10 |
| MMMU | 6/10 | 6/10 | 2/10 | 8/10 | 3/10 | 6/10 |
| textvqa | 2/10 | 6/10 | 3/10 | 4/10 | 2/10 | 4/10 |
| chartqa | 8/10 | 9/10 | 4/10 | 6/10 | 9/10 | 8/10 |
| mathvista | 8/10 | 8/10 | 5/10 | 8/10 | 8/10 | 6/10 |

## Long-output benchmarks (raw vs the papers' prompts)

| benchmark | change | raw | verbose | ratio |
|---|---|---:|---:|---:|
| OmniDocBench | whole prompt replaced with the OCR instruction | 451.5 | 756.2 | **1.67x** |
| COCO-Caption | whole prompt replaced with the caption instruction | 558.1 | 754.8 | **1.35x** |
| MATH-500 | never modified — control | 476.4 | 476.4 | **1.00x** |

## Sample answers (first prompt of each VQA benchmark)

> `results.json` records the raw question, not the styled prompt, so only
> the answers are shown — the applied prompt is echoed in each run's
> `_logs/<bench>.log` as `prompt_style=... -> ...`.

### MMStar

Question: `Which option describe the object relationship in the image correctly?
Options: A: The suitcase is on the book., B: The suitcase is beneath the cat., C: The suit`

- `raw`: ' D: The suitcase is beneath the book.'
- `verbose`: ' D: The suitcase is beneath the book.'
- `detail_prefix`: ' D: The suitcase is beneath the book.'
- `cot`: ' D: The suitcase is beneath the book.'
- `describe_first`: ' D: The suitcase is beneath the book.'
- `min_words`: ' D: The suitcase is beneath the book.'
- `answer_then_describe`: ' Answer: D: The suitcase is beneath the book.\n\nStep-by-step justification:\n1. The image shows a brown suitcase with various stickers and patches on it.\n2. The stickers and patches are in the British flag colors, which ar'

### MMMU

Question: `Which of the following best explains the overall trend shown in the <image 1>?`

- `raw`: ' Population (millions).'
- `verbose`: ' Population (millions).'
- `detail_prefix`: ' Roman.'
- `cot`: ' Population (millions).'
- `describe_first`: ' Population (millions).'
- `min_words`: ' Population (millions).'
- `answer_then_describe`: ' Population (millions).'

### textvqa

Question: `what brand of phone?`

- `raw`: ' Nokia.'
- `verbose`: ' Nokia.'
- `detail_prefix`: ' The phone is a Nokia.'
- `cot`: ' Nokia.'
- `describe_first`: ' The image shows a Nokia phone. The phone is a model with a silver color and a rectangular shape. The brand name "Nokia" is visible on the top of the phone, and the model name "We Wait" is printed on the bottom. The phon'
- `min_words`: ' Nokia.'
- `answer_then_describe`: ' The image belongs to a Nokia phone. The phone is shown in the center of the image, and it is the only object in the image. The phone has a silver color, and it has a rectangular shape. The phone has a camera on the righ'

### chartqa

Question: `How many food item is shown in the bar graph?`

- `raw`: ' 10'
- `verbose`: ' 10.'
- `detail_prefix`: ' 10.'
- `cot`: ' 10.'
- `describe_first`: ' 10.'
- `min_words`: ' 10.'
- `answer_then_describe`: ' 10.'

### mathvista

Question: `When a spring does work on an object, we cannot find the work by simply multiplying the spring force by the object's displacement. The reason is that there is n`

- `raw`: ' The canister slides across the counter with a force of $v=0.50 \\mathrm{~m} / \\mathrm{s}$. The spring is stretched $k=750 \\mathrm{~N} / \\mathrm{m}$. The force of the spring is $v=0.50 \\mathrm{~m} / \\mathrm{s}$. When the '
- `verbose`: ' From the given information, we can see that the canister is sliding across the counter with a force of $v=0.50 \\mathrm{~m} / \\mathrm{s}$. Since the canister is sliding across the counter, the force of the canister is eq'
- `detail_prefix`: ' The canister slides across the counter with a force of $v=0.50 \\mathrm{~m} / \\mathrm{s}$. The spring is stretched $m=0.40 \\mathrm{~kg} = 0.40 \\mathrm{~kg} \\cdot \\mathrm{s} = 0.40 \\mathrm{~kg} \\cdot \\mathrm{s} = 0.40 \\ma'
- `cot`: ' To find the distance $d$ the spring is compressed, we can use the formula:\n\\[d = \\frac{k}{m} \\]\nSubstituting $k=750 \\mathrm{~N} / \\mathrm{m}$, we get:\n\\[d = \\frac{750}{750} \\]\n\\[d = 1 \\]\nSo, the spring is compressed $1$'
- `describe_first`: ' The canister slides across the counter with a force of $v=0.50 \\mathrm{~m} / \\mathrm{s}$. Since the canister is stopped by the spring, the force of the spring is equal to the force of the canister. So, the force of the '
- `min_words`: ' The canister of mass $m=0.40 \\mathrm{~kg}$ slides across a horizontal frictionless counter with speed $v=0.50 \\mathrm{~m} / \\mathrm{s}$. When the canister is momentarily stopped by the spring, by what distance $d$ is th'
- `answer_then_describe`: " The spring is compressed by the force of the canister's motion. Since the canister is sliding across the counter, the force of the canister's motion is equal to the force of the spring. Therefore, the force of the sprin"

