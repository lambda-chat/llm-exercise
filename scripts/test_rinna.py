import json
import logging

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast
from transformers.models.gpt_neo.modeling_gpt_neo import GPTNeoModel

# default config
logging.basicConfig(level=logging.INFO)

# config for this file
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# type aliases
Model = GPTNeoModel
Tokenizer = PreTrainedTokenizer | PreTrainedTokenizerFast
IntTensorRank1 = torch.Tensor

# constants
MODEL_NAME = "rinna/bilingual-gpt-neox-4b"


def load_model() -> tuple[Model, Tokenizer]:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

    cuda_is_available = torch.cuda.is_available()
    logger.debug(f"cuda_is_available: {cuda_is_available}")
    if cuda_is_available:
        model.to("cuda")

    return model, tokenizer


def answer(model: Model, tokenizer: Tokenizer, text: str) -> str:
    token_ids: IntTensorRank1 = tokenizer.encode(text, add_special_tokens=False, return_tensors="pt")  # type: ignore
    with torch.no_grad():
        output_ids: torch.LongTensor = model.generate(  # type: ignore
            token_ids.to(model.device),
            max_new_tokens=300,
            min_new_tokens=100,
            do_sample=True,
            temperature=0.1,
            top_p=0.95,
            pad_token_id=tokenizer.pad_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(output_ids.tolist()[0])


def main() -> None:
    model, tokenizer = load_model()
    input_text = "フェネックギツネの特徴について簡単に教えて下さい。"
    output_text = answer(model, tokenizer, input_text)
    print(
        json.dumps(
            {"input_text": input_text, "output_text": output_text},
            indent=2,
            ensure_ascii=False,
        ),
    )


if __name__ == "__main__":
    main()
