import gc
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# -- end do not touch

import unittest  # noqa: E402

import torch  # noqa: E402
from gptqmodel import BACKEND, GPTQModel  # noqa: E402


class TestLoadVLLM(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        from vllm import SamplingParams  # noqa: E402
        self.MODEL_ID = "LnL-AI/TinyLlama-1.1B-Chat-v1.0-GPTQ-4bit"
        self.SHARDED_MODEL_ID = "ModelCloud/TinyLlama-1.1B-Chat-v1.0-GPTQ-4bit-sharded"
        self.prompts = [
            "The capital of France is",
        ]
        self.sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

    def release_vllm_model(self):
        from vllm.distributed.parallel_state import destroy_model_parallel  # noqa: E402

        destroy_model_parallel()
        gc.collect()
        torch.cuda.empty_cache()

    def test_load_vllm(self):
        model = GPTQModel.from_quantized(
            self.MODEL_ID,
            device="cuda:0",
            backend=BACKEND.VLLM,
            gpu_memory_utilization=0.2,
        )
        outputs = model.generate(
            prompts=self.prompts,
            sampling_params=self.sampling_params,
        )
        for output in outputs:
            prompt = output.prompt
            generated_text = output.outputs[0].text
            print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
            self.assertEquals(generated_text, " Paris, which is also the capital of France.")
        outputs_param = model.generate(
            prompts=self.prompts,
            temperature=0.8,
            top_p=0.95,
        )
        for output in outputs_param:
            prompt = output.prompt
            generated_text = output.outputs[0].text
            print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
            self.assertEquals(generated_text, " ___________.\n6. City Name: Paris, France\n7. C")

        del model
        self.release_vllm_model()

    def test_load_shared_vllm(self):
        model = GPTQModel.from_quantized(
            self.SHARDED_MODEL_ID,
            device="cuda:0",
            backend=BACKEND.VLLM,
            gpu_memory_utilization=0.2,
        )
        outputs = model.generate(
            prompts=self.prompts,
            temperature=0.8,
            top_p=0.95,
        )
        for output in outputs:
            prompt = output.prompt
            generated_text = output.outputs[0].text
            print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
            self.assertEquals(generated_text,
                              " Paris, which is also known as the city of love.")

        del model
        self.release_vllm_model()
