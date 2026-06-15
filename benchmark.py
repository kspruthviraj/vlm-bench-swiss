import json
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

from metrics import compute_all_metrics, aggregate_metrics


@dataclass
class TestCase:
    id: str
    task: str
    image_path: str
    question: str
    expected_answer: Any
    accepted_answers: list[str] | None = None
    reference_answers: dict | None = None
    reference_descriptions: list[str] | None = None
    accepted_fields: list[str] | None = None
    metadata: dict | None = None
    language: str = "en"


@dataclass
class EvalResult:
    test_id: str
    task: str
    language: str
    prediction: str
    expected: Any
    metrics: dict
    latency_ms: float = 0.0
    error: str | None = None


TASK_TYPES = {
    "timetable_qa": "qa",
    "receipt_ocr": "qa",
    "multilingual_captioning": "captioning",
    "form_extraction": "form_extraction",
    "news_description": "captioning",
}


class VLMModel(ABC):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def predict(self, image_path: str, prompt: str) -> str:
        ...


class MockVLM(VLMModel):
    def __init__(self):
        super().__init__("mock_vlm")

    def predict(self, image_path: str, prompt: str) -> str:
        return "mock prediction"


class OpenAIVLM(VLMModel):
    def __init__(self, model: str = "gpt-4o", api_key: str | None = None, base_url: str | None = None):
        super().__init__(f"openai_{model}")
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.base_url = base_url
        self._client = None

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI
            kwargs = {"api_key": self.api_key}
            if self.base_url:
                kwargs["base_url"] = self.base_url
            self._client = OpenAI(**kwargs)
        return self._client


class XiaomiMiMoVLM(VLMModel):
    """Xiaomi MiMo VLM via OpenAI-compatible API.

    Supports pay-as-you-go (sk-*) and Token Plan (tp-*) keys.
    Models: mimo-v2.5-pro, mimo-v2.5, mimo-v2-flash
    """

    BASE_URLS = {
        "payg": "https://api.xiaomimimo.com/v1",
        "token_plan": "https://token-plan-sgp.xiaomimimo.com/v1",
        "token_plan_cn": "https://token-plan-cn.xiaomimimo.com/v1",
        "token_plan_sgp": "https://token-plan-sgp.xiaomimimo.com/v1",
        "token_plan_ams": "https://token-plan-ams.xiaomimimo.com/v1",
    }

    def __init__(
        self,
        model: str = "mimo-v2.5-pro",
        api_key: str | None = None,
        plan: str = "payg",
        base_url: str | None = None,
    ):
        super().__init__(f"mimo_{model}")
        self.model = model
        self.api_key = api_key or os.environ.get("MIMO_API_KEY")
        if not self.api_key:
            raise ValueError(
                "MiMo API key required. Set MIMO_API_KEY env var or pass api_key. "
                "Get one at https://platform.xiaomimimo.com/#/console/api-keys"
            )
        self.base_url = base_url or self.BASE_URLS.get(plan, self.BASE_URLS["payg"])
        self._client = None

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        return self._client

    def predict(self, image_path: str, prompt: str) -> str:
        import base64
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        ext = Path(image_path).suffix.lstrip(".")
        mime = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg", "pdf": "application/pdf"}.get(ext, "image/png")
        client = self._get_client()
        resp = client.chat.completions.create(
            model=self.model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                ],
            }],
            max_tokens=1024,
        )
        return resp.choices[0].message.content or ""


class AnthropicVLM(VLMModel):
    def __init__(self, model: str = "claude-sonnet-4-20250514", api_key: str | None = None):
        super().__init__(f"anthropic_{model}")
        self.model = model
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self._client = None

    def _get_client(self):
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic(api_key=self.api_key)
        return self._client

    def predict(self, image_path: str, prompt: str) -> str:
        import base64
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        ext = Path(image_path).suffix.lstrip(".")
        media = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg"}.get(ext, "image/png")
        client = self._get_client()
        resp = client.messages.create(
            model=self.model,
            max_tokens=1024,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": media, "data": b64}},
                    {"type": "text", "text": prompt},
                ],
            }],
        )
        return resp.content[0].text


class HFVLM(VLMModel):
    def __init__(self, model_id: str = "llava-hf/llava-1.5-7b-hf", device: str = "auto"):
        super().__init__(f"hf_{model_id.split('/')[-1]}")
        self.model_id = model_id
        self.device = device
        self._pipe = None

    def _load(self):
        if self._pipe is None:
            from transformers import pipeline
            self._pipe = pipeline("image-to-text", model=self.model_id, device_map=self.device)

    def predict(self, image_path: str, prompt: str) -> str:
        self._load()
        from PIL import Image
        img = Image.open(image_path).convert("RGB")
        result = self._pipe(images=img, prompt=prompt)
        if isinstance(result, list) and result:
            return result[0].get("generated_text", "")
        return str(result)


def load_test_cases(data_dir: str = "sample_data") -> list[TestCase]:
    cases = []
    data_path = Path(data_dir)
    for json_file in sorted(data_path.glob("*_cases.json")):
        with open(json_file) as f:
            raw = json.load(f)
        for item in raw:
            cases.append(TestCase(**{k: v for k, v in item.items() if k in TestCase.__dataclass_fields__}))
    return cases


def evaluate_model(
    model: VLMModel,
    test_cases: list[TestCase] | None = None,
    data_dir: str = "sample_data",
    tasks: list[str] | None = None,
    languages: list[str] | None = None,
    max_cases: int | None = None,
    verbose: bool = True,
) -> list[EvalResult]:
    if test_cases is None:
        test_cases = load_test_cases(data_dir)
    if tasks:
        test_cases = [c for c in test_cases if c.task in tasks]
    if languages:
        test_cases = [c for c in test_cases if c.language in languages]
    if max_cases:
        test_cases = test_cases[:max_cases]

    results = []
    for i, tc in enumerate(test_cases):
        if verbose:
            print(f"  [{i+1}/{len(test_cases)}] {tc.id} ({tc.task}/{tc.language})")
        try:
            start = time.perf_counter()
            prediction = model.predict(tc.image_path, tc.question)
            latency = (time.perf_counter() - start) * 1000
        except Exception as e:
            results.append(EvalResult(
                test_id=tc.id, task=tc.task, language=tc.language,
                prediction="", expected=tc.expected_answer,
                metrics={}, error=str(e),
            ))
            continue

        references = None
        if tc.reference_answers and isinstance(tc.reference_answers, dict):
            references = list(tc.reference_answers.values())
        elif tc.reference_descriptions:
            references = tc.reference_descriptions

        task_type = TASK_TYPES.get(tc.task, "qa")
        metrics = compute_all_metrics(
            prediction=prediction,
            expected_answer=tc.expected_answer,
            accepted_answers=tc.accepted_answers,
            references=references,
            task_type=task_type,
        )

        results.append(EvalResult(
            test_id=tc.id, task=tc.task, language=tc.language,
            prediction=prediction, expected=tc.expected_answer,
            metrics=metrics, latency_ms=latency,
        ))

    return results


def run_benchmark(
    models: list[VLMModel],
    data_dir: str = "sample_data",
    tasks: list[str] | None = None,
    languages: list[str] | None = None,
    output_dir: str = "results",
) -> dict[str, list[EvalResult]]:
    os.makedirs(output_dir, exist_ok=True)
    all_results = {}
    for model in models:
        print(f"\nEvaluating: {model.name}")
        results = evaluate_model(model, data_dir=data_dir, tasks=tasks, languages=languages)
        all_results[model.name] = results
        out_path = os.path.join(output_dir, f"{model.name}_results.json")
        with open(out_path, "w") as f:
            json.dump([asdict(r) for r in results], f, indent=2, default=str)
        agg = aggregate_metrics([r.metrics for r in results])
        print(f"  Summary: {json.dumps({k: round(v['mean'], 4) for k, v in agg.items()}, indent=2)}")
    return all_results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Swiss VLM Benchmark")
    parser.add_argument("--model", choices=["mock", "openai", "anthropic", "hf", "mimo"], default="mock")
    parser.add_argument("--tasks", nargs="+", default=None)
    parser.add_argument("--languages", nargs="+", default=None)
    parser.add_argument("--data-dir", default="sample_data")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--model-id", default=None, help="HF model ID or API model name")
    parser.add_argument("--plan", choices=["payg", "token_plan"], default="payg", help="MiMo plan type")
    args = parser.parse_args()

    model_map = {
        "mock": lambda: MockVLM(),
        "openai": lambda: OpenAIVLM(model=args.model_id or "gpt-4o"),
        "anthropic": lambda: AnthropicVLM(model=args.model_id or "claude-sonnet-4-20250514"),
        "hf": lambda: HFVLM(model_id=args.model_id or "llava-hf/llava-1.5-7b-hf"),
        "mimo": lambda: XiaomiMiMoVLM(model=args.model_id or "mimo-v2.5", plan=args.plan),
    }
    model = model_map[args.model]()
    results = run_benchmark([model], data_dir=args.data_dir, tasks=args.tasks, languages=args.languages, output_dir=args.output_dir)
    print("\nDone. Results saved to", args.output_dir)
