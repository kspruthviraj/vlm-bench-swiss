# VLM Benchmark: Swiss Document Understanding

Evaluation benchmark for Vision-Language Models on Swiss-specific document understanding tasks.

## Tasks

| Task | Description | Language(s) |
|------|-------------|-------------|
| `timetable_qa` | SBB/ZVV transport timetable question answering | DE, FR, EN, IT |
| `receipt_ocr` | Swiss retailer receipt OCR + total extraction | DE, FR, EN |
| `multilingual_captioning` | Image captioning in DE/FR/EN/IT | DE, FR, EN, IT |
| `form_extraction` | Swiss tax/form field extraction | DE, EN |
| `news_description` | Swiss news image description | DE, FR, EN |

## Metrics

- **Exact Match** — exact string match against accepted answers
- **Contains Match** — prediction contains an accepted answer
- **ANLS** — Average Normalized Levenshtein Similarity (threshold 0.5)
- **Edit Distance Similarity** — 1 - (levenshtein / max_len)
- **BLEU** — n-gram precision (up to 4-gram)
- **ROUGE-L** — longest common subsequence F1

## Quick Start

```bash
pip install matplotlib plotly

# Run with mock model (for testing)
python benchmark.py --model mock

# Run with Xiaomi MiMo (pay-as-you-go)
export MIMO_API_KEY=sk-...
python benchmark.py --model mimo --model-id mimo-v2.5-pro

# Run with Xiaomi MiMo (Token Plan subscription)
export MIMO_API_KEY=tp-...
python benchmark.py --model mimo --model-id mimo-v2.5-pro --plan token_plan

# Run with OpenAI
export OPENAI_API_KEY=sk-...
python benchmark.py --model openai --model-id gpt-4o

# Run with Anthropic
export ANTHROPIC_API_KEY=sk-ant-...
python benchmark.py --model anthropic --model-id claude-sonnet-4-20250514

# Run with HuggingFace model
python benchmark.py --model hf --model-id llava-hf/llava-1.5-7b-hf
```

## Compare Models

```bash
# Run multiple models
python benchmark.py --model mock --output-dir results
python benchmark.py --model openai --output-dir results

# View leaderboard
python leaderboard.py --results-dir results
python leaderboard.py --by-task --by-language

# Export
python leaderboard.py --export-csv leaderboard.csv --export-md leaderboard.md
```

## Visualize Results

```bash
# Generate all plots
python visualize.py

# Specific plot
python visualize.py --plot radar --format png
python visualize.py --plot interactive  # generates HTML with plotly
```

## Adding a New Model

Create a class that extends `VLMModel`:

```python
from benchmark import VLMModel

class MyVLM(VLMModel):
    def __init__(self):
        super().__init__("my_vlm")

    def predict(self, image_path: str, prompt: str) -> str:
        # Load image, run inference, return text
        return "model prediction"
```

Then run it:

```python
from benchmark import run_benchmark
model = MyVLM()
run_benchmark([model], output_dir="results")
```

## Adding New Test Cases

Create a JSON file in `sample_data/` matching the naming pattern `*_cases.json`:

```json
[
  {
    "id": "unique_id",
    "task": "timetable_qa",
    "image_path": "sample_data/images/my_image.png",
    "question": "What is the departure time?",
    "expected_answer": "14:30",
    "accepted_answers": ["14:30", "2:30 PM"],
    "language": "en"
  }
]
```

### Task-Specific Fields

**timetable_qa / receipt_ocr / news_description:**
- `expected_answer`: string
- `accepted_answers`: list of acceptable string answers
- `reference_descriptions`: list of reference descriptions (for BLEU/ROUGE)

**multilingual_captioning:**
- `reference_answers`: dict mapping language code to reference caption
- `reference_descriptions`: list of reference captions

**form_extraction:**
- `expected_answer`: dict of field_name → expected_value
- `accepted_fields`: list of fields to evaluate

## Adding a New Task

1. Add test cases with a new task name in `sample_data/`
2. Register the task type in `benchmark.py`:
   ```python
   TASK_TYPES["my_new_task"] = "qa"  # or "captioning" or "form_extraction"
   ```
3. If needed, add a custom metric function in `metrics.py`

## Project Structure

```
vlm-bench-swiss/
├── benchmark.py          # Core evaluation framework + VLM model wrappers
├── metrics.py            # All metric implementations
├── leaderboard.py        # Leaderboard generation and export
├── visualize.py          # Matplotlib + Plotly visualization
├── README.md
├── sample_data/
│   ├── timetable_cases.json
│   ├── receipt_cases.json
│   ├── captioning_cases.json
│   ├── form_cases.json
│   ├── news_cases.json
│   └── images/           # Place test images here
├── results/              # Auto-generated evaluation results
└── plots/                # Auto-generated visualizations
```
