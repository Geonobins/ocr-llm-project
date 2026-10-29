import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, BitsAndBytesConfig

from pipeline.config import FIELDS, LLM_NAME, MAX_INPUT_TOKENS, QUESTIONS
from pipeline.normalize import clean_prediction

PRECISIONS = ("fp32", "fp16", "bf16", "8bit", "4bit")


def build_prompt(context: str, field: str) -> str:
    return f"{context}\n\nQuestion: {QUESTIONS[field]}\nAnswer:"


def load_tokenizer():
    return AutoTokenizer.from_pretrained(LLM_NAME)


def load_model(precision: str, device: str = "cuda"):
    if precision not in PRECISIONS:
        raise ValueError(f"Unknown precision {precision}; expected one of {PRECISIONS}")
    if precision == "8bit":
        config = BitsAndBytesConfig(load_in_8bit=True)
        return AutoModelForSeq2SeqLM.from_pretrained(LLM_NAME, quantization_config=config, device_map=device).eval()
    if precision == "4bit":
        config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        return AutoModelForSeq2SeqLM.from_pretrained(LLM_NAME, quantization_config=config, device_map=device).eval()
    dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[precision]
    model = AutoModelForSeq2SeqLM.from_pretrained(LLM_NAME, torch_dtype=dtype)
    return model.to(device).eval()


class FieldExtractor:
    def __init__(self, precision: str, device: str = "cuda", tokenizer=None):
        self.precision = precision
        self.device = device
        self.tokenizer = tokenizer or load_tokenizer()
        self.model = load_model(precision, device)

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer(text, truncation=False)["input_ids"])

    @torch.no_grad()
    def _generate(self, prompts: list[str], max_new_tokens: int) -> list[str]:
        encoded = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_INPUT_TOKENS,
        ).to(self.device)
        output = self.model.generate(**encoded, max_new_tokens=max_new_tokens, do_sample=False)
        return self.tokenizer.batch_decode(output, skip_special_tokens=True)

    def extract_sequential(self, context: str, max_new_tokens: int) -> dict:
        result = {}
        for field in FIELDS:
            answer = self._generate([build_prompt(context, field)], max_new_tokens)[0]
            result[field] = clean_prediction(field, answer)
        return result

    def extract_batch(self, contexts: list[str], max_new_tokens: int) -> list[dict]:
        if not contexts:
            return []
        prompts = [build_prompt(context, field) for context in contexts for field in FIELDS]
        answers = self._generate(prompts, max_new_tokens)
        results = []
        for doc_index in range(len(contexts)):
            result = {}
            for field_index, field in enumerate(FIELDS):
                answer = answers[doc_index * len(FIELDS) + field_index]
                result[field] = clean_prediction(field, answer)
            results.append(result)
        return results
