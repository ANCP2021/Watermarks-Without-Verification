# Third-party data

The code in this repository is licensed under Apache 2.0 (see `LICENSE`). The two files below contain third-party datasets and are distributed under those datasets' own licenses, not under Apache 2.0.

## `prompts.json`

- **Source:** `databricks-dolly-15k`, Copyright (2023) Databricks, Inc. — <https://huggingface.co/datasets/databricks/databricks-dolly-15k> (revision `bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a`)
- **License:** [Creative Commons Attribution-ShareAlike 3.0 Unported (CC BY-SA 3.0)](https://creativecommons.org/licenses/by-sa/3.0/)
- **Changes:** 500 of the 3,742 `open_qa` rows, sampled with a fixed seed; only the instruction text is kept, with an id and the source row index added.

`prompts.json` is an adaptation of `databricks-dolly-15k` and is itself licensed under CC BY-SA 3.0.

## `problems.json`

### HumanEval (`humaneval-*`, 164 problems)

- **Source:** HumanEval, Chen et al., *Evaluating Large Language Models Trained on Code* (2021) — <https://github.com/openai/human-eval>, via <https://huggingface.co/datasets/openai/openai_humaneval> (revision `7dce6050a7d6d172f3cc5c32aa97f52fa1a2e544`)
- **License:** MIT, reproduced below.
- **Changes:** all 164 test problems, fields renamed into a common schema.

```
The MIT License

Copyright (c) OpenAI (https://openai.com)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
```

### MBPP (`mbpp-*`, 200 problems)

- **Source:** MBPP (sanitized), Austin et al., *Program Synthesis with Large Language Models* (2021) — <https://huggingface.co/datasets/google-research-datasets/mbpp> (config `sanitized`, split `test`, revision `4bb6404fdc6cacfda99d4ac4205087b89d32030c`)
- **License:** [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/)
- **Changes:** the first 200 of the 257 sanitized test problems by `task_id`; `code` renamed to `canonical_solution`, and an `entry_point` field added, inferred from the asserts.

## Model outputs in `results/`

The generations in `results/` were produced by `google/gemma-2-9b-it` ([Gemma Terms of Use](https://ai.google.dev/gemma/terms)) and `meta-llama/Llama-3.1-8B-Instruct` ([Llama 3.1 Community License](https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/LICENSE)). The files in `results/` identify prompts and problems by id only and do not include the dataset text. Under the Llama 3.1 license (§1.b.i), anyone who uses the Llama outputs to create, train, fine-tune or otherwise improve an AI model that is distributed or made available must include "Llama" at the beginning of that model's name.
