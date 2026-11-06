
"""
Bark RTX GPU
------------
Reusable Bark text-to-speech implementation.

Tested baseline:
    Python 3.11
    PyTorch 2.7.1
    Transformers 4.48.0
    CUDA 12.8
    NVIDIA RTX 5090 Laptop GPU

Model:
    suno/bark-small
"""

from pathlib import Path
import time

import numpy as np
import scipy.io.wavfile
import torch

from transformers import AutoProcessor, BarkModel


class BarkSpeech:
    """Reusable Bark speech generation engine."""

    MODEL_NAME = "suno/bark-small"

    DEFAULT_VOICE = "v2/en_speaker_9"

    def __init__(
        self,
        model_name=None,
        device=None,
        voice=None,
    ):
        self.model_name = model_name or self.MODEL_NAME
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.default_voice = voice or self.DEFAULT_VOICE

        print(f"Using device: {self.device}")
        print(f"Loading Bark model: {self.model_name}")

        self.processor = AutoProcessor.from_pretrained(
            self.model_name
        )

        self.model = BarkModel.from_pretrained(
            self.model_name
        )

        self.model = self.model.to(self.device)
        self.model.eval()

        self._install_embedding_device_hooks()
        self._install_generation_device_fix()

        print("Bark model loaded successfully.")

        if self.device.type == "cuda":
            print(
                "GPU:",
                torch.cuda.get_device_name(self.device)
            )

    # ---------------------------------------------------------
    # Embedding device compatibility
    # ---------------------------------------------------------

    def _install_embedding_device_hooks(self):

        def move_embedding_input_to_device(module, inputs):

            if not inputs:
                return inputs

            input_ids = inputs[0]

            if isinstance(input_ids, torch.Tensor):

                expected_device = module.weight.device

                if input_ids.device != expected_device:

                    inputs = (
                        input_ids.to(expected_device),
                    ) + inputs[1:]

            return inputs

        self.embedding_hooks = []

        for module in self.model.modules():

            if isinstance(module, torch.nn.Embedding):

                handle = module.register_forward_pre_hook(
                    move_embedding_input_to_device
                )

                self.embedding_hooks.append(handle)

        print(
            f"Configured {len(self.embedding_hooks)} "
            "embedding device hooks."
        )

    # ---------------------------------------------------------
    # Generation token suppression compatibility
    # ---------------------------------------------------------

    @staticmethod
    def _install_generation_device_fix():

        import transformers.generation.logits_process as logits_process

        if getattr(
            logits_process,
            "_bark_device_fix_installed",
            False
        ):
            return

        original_isin = logits_process.isin_mps_friendly

        def device_safe_isin(elements, test_elements):

            if isinstance(test_elements, torch.Tensor):

                test_elements = test_elements.to(
                    elements.device
                )

            return original_isin(
                elements,
                test_elements
            )

        logits_process.isin_mps_friendly = device_safe_isin

        logits_process._bark_device_fix_installed = True

        print("Configured device-safe token suppression.")

    # ---------------------------------------------------------
    # Recursive device conversion
    # ---------------------------------------------------------

    def _move_to_device(self, obj):

        if isinstance(obj, torch.Tensor):

            return obj.to(self.device)

        if isinstance(obj, dict):

            return {
                key: self._move_to_device(value)
                for key, value in obj.items()
            }

        if isinstance(obj, list):

            return [
                self._move_to_device(value)
                for value in obj
            ]

        if isinstance(obj, tuple):

            return tuple(
                self._move_to_device(value)
                for value in obj
            )

        return obj

    # ---------------------------------------------------------
    # Audio generation
    # ---------------------------------------------------------

    def generate(
        self,
        text,
        voice=None,
        output_file="output.wav",
        output_format="wav",
    ):

        if not isinstance(text, str) or not text.strip():

            raise ValueError("Text must be a non-empty string.")

        if output_format.lower() != "wav":

            raise ValueError(
                "Version 0.1.0 currently supports WAV output only."
            )

        voice = voice or self.default_voice

        if not voice.startswith("v2/"):

            raise ValueError(
                "This initial version expects a v2 English speaker preset."
            )

        print(f"\nVoice: {voice}")
        print(f"Text: {text}")

        inputs = self.processor(
            text,
            voice_preset=voice,
            return_tensors="pt"
        )

        inputs = self._move_to_device(inputs)

        if self.device.type == "cuda":

            torch.cuda.synchronize(self.device)

            torch.cuda.reset_peak_memory_stats(
                self.device
            )

        start_time = time.perf_counter()

        with torch.inference_mode():

            audio = self.model.generate(**inputs)

        if self.device.type == "cuda":

            torch.cuda.synchronize(self.device)

        elapsed = time.perf_counter() - start_time

        audio_array = (
            audio.detach()
            .float()
            .cpu()
            .numpy()
            .squeeze()
        )

        audio_array = np.asarray(
            audio_array,
            dtype=np.float32
        )

        sample_rate = self.model.generation_config.sample_rate

        output_path = Path(output_file)

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        scipy.io.wavfile.write(
            str(output_path),
            sample_rate,
            audio_array
        )

        print("\nGeneration completed.")
        print(f"Output: {output_path.resolve()}")
        print(f"Sample rate: {sample_rate}")
        print(f"Generation time: {elapsed:.2f} seconds")

        if self.device.type == "cuda":

            peak_memory = (
                torch.cuda.max_memory_allocated(self.device)
                / (1024 ** 3)
            )

            print(
                f"Peak GPU memory: {peak_memory:.2f} GB"
            )

        return str(output_path)


# -------------------------------------------------------------
# Standalone execution
# -------------------------------------------------------------

if __name__ == "__main__":

    bark = BarkSpeech()

    bark.generate(
        text="Hello! Welcome to artificial intelligence.",
        voice="v2/en_speaker_9",
        output_file="output/output.wav"
    )
