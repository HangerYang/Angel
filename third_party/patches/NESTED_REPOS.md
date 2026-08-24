# Nested Repository Preservation

This file records local nested repositories seen while creating the mess branch. The nested working directories remain local; this branch avoids committing fragile embedded gitlinks without gitmodules.

## ViSpec

- Path: ViSpec/
- Remote: https://github.com/KangJialiang/ViSpec.git
- Commit: 49a33ff0ca31e281b9dc8803fd6f3041c6c0e135
- Local status at capture: clean

## HiViS

- Path: third_party/HiViS/
- Remote: https://github.com/lnn-ops/HiViS.git
- Commit: 40c85840c37b5dda4300ba892e2c510a26e80774
- Local SmolVLM changes preserved in: third_party/patches/hivis-local-smolvlm.patch
- Untracked local TextVQA images under third_party/HiViS/eval_data/textvqa_images/ are treated as local data artifacts.
