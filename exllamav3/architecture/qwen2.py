from __future__ import annotations

from typing import TYPE_CHECKING

from typing_extensions import override

from .llama import LlamaConfig, LlamaModel

if TYPE_CHECKING:
    from .qwen2_5_vl import Qwen2_5VLConfig

# Qwen2 is identical to Llama except for bias on Q, K and V projections, but Linear module automatically
# detects *.bias tensor

class Qwen2Config(LlamaConfig):
    arch_string = "Qwen2ForCausalLM"

    def __init__(
        self,
        directory: str,
        **kwargs,
    ):
        super().__init__(
            directory,
            derived_model = {"text": Qwen2Model},
            **kwargs
        )


class Qwen2Model(LlamaModel):
    config_class = Qwen2Config

    def __init__(
        self,
        config: Qwen2Config | Qwen2_5VLConfig,
        **kwargs
    ):
        super().__init__(config, **kwargs)


    @override
    def default_chat_prompt(self, prompt: str, system_prompt: str = None) -> str:
        p = ""
        if system_prompt:
            p += "<|im_start|>system\n"
            p += f"{system_prompt}<|im_end|>\n"
        p += "<|im_start|>user\n"
        p += f"{prompt}<|im_end|>\n"
        p += "<|im_start|>assistant\n"
        return p
