# Bark RTX GPU

**GPU-accelerated text-to-speech using Suno Bark and NVIDIA GPUs.**

Bark RTX GPU is a project exploring reliable execution and optimization of the Suno Bark text-to-speech model on NVIDIA GPUs, including the RTX 5090 Laptop GPU.

The project currently includes a working GPU implementation with device compatibility workarounds, configurable speaker selection, and WAV audio generation.

Additional objectives include multilingual speech generation, voice comparisons, and a FastAPI service.

## Project Status

**Current release target: v0.1.0 — Working GPU Implementation**

|Feature|Status|
|---|---|
|Bark model loading|Working|
|NVIDIA CUDA execution|Tested|
|RTX 5090 Laptop GPU|Tested|
|Embedding device compatibility|Working|
|Generation token-suppression workaround|Working|
|Speaker preset selection|Initial support|
|WAV audio generation|Working|
|British English experiments|Planned|
|Indian English experiments|Planned|
|Chinese speech experiments|Planned|
|Female voice comparisons|Planned|
|FastAPI service|Planned|
|GPU performance optimization|Planned|

Features will be marked as tested only after successful validation.

## Hardware and Software

### Tested hardware

|Component|Configuration|
|---|---|
|GPU|NVIDIA GeForce RTX 5090 Laptop GPU|
|GPU execution|CUDA|

### Tested software

|Component|Version|
|---|---|
|Python|3.11|
|PyTorch|2.7.1|
|CUDA|12.8|
|Transformers|4.48.0|
|Model|suno/bark-small|

These are the versions used for the initial successful generation test, not a guarantee that every hardware or software combination will work.

## Project Structure

text

```
bark-rtx-gpu/
├── README.md
├── LICENSE
├── THIRD_PARTY_NOTICES.md
├── .gitignore
├── requirements.txt
├── barkmodel.py
├── examples/
│   └── basic_generation.py
├── patches/
│   ├── README.md
│   └── transformers-4.48.0-bark-cuda.patch
└── output/
    └── .gitkeep
```

## Installation

This project assumes an existing Python environment with a compatible NVIDIA driver and CUDA-enabled PyTorch installation.

### 1. Clone the repository

bash

```
git clone git@github.com:turiya9710/bark-rtx-gpu.git

cd bark-rtx-gpu
```

Alternatively, use HTTPS:

bash

```
git clone https://github.com/turiya9710/bark-rtx-gpu.git

cd bark-rtx-gpu
```

### 2. Activate your Python environment

Example using Conda:

bash

```
conda activate genai_fastapi_p311
```

### 3. Install dependencies

bash

```
pip install -r requirements.txt
```

The tested environment uses:

text

```
Python 3.11
PyTorch 2.7.1
CUDA 12.8
Transformers 4.48.0
```

For a CUDA-enabled PyTorch installation, follow the official [PyTorch installation instructions](https://pytorch.org/get-started/locally/).

## Basic Speech Generation

The current implementation loads the Bark model and generates speech using a selected speaker preset.

Run:

bash

```
python barkmodel.py
```

The default example generates:

text

```
Hello! Welcome to artificial intelligence.
```

Using:

text

```
v2/en_speaker_9
```

The generated audio is saved to:

text

```
output/output.wav
```

The output sample rate is 24,000 Hz.

## Using BarkSpeech from Python

The reusable interface is:

python

```
from barkmodel import BarkSpeech

speech = BarkSpeech()

speech.generate(
    text="Hello! Welcome to our artificial intelligence platform.",
    voice="v2/en_speaker_9",
    output_file="output/hello.wav"
)
```

### Available parameters

|Parameter|Description|
|---|---|
|`text`|Text to convert into speech|
|`voice`|Bark speaker preset|
|`output_file`|Destination WAV filename|
|`output_format`|Currently WAV|

The model is loaded once when the `BarkSpeech` instance is created and reused for subsequent generation calls.

## GPU Compatibility Workarounds

During development on the NVIDIA RTX 5090 Laptop GPU, several tensor-device mismatches were encountered in the Bark generation pipeline.

The implementation includes the following compatibility workarounds.

### 1. Embedding input device handling

Forward pre-hooks are registered on PyTorch embedding layers.

These hooks move embedding input tensors to the same device as the embedding weights.

This addresses embedding input and weight device mismatches encountered during generation.

### 2. Recursive processor-output device conversion

Processor outputs are recursively moved to the selected execution device.

This handles tensors nested inside dictionaries, lists, and tuples.

### 3. Generation token-suppression compatibility

A device-safe wrapper is installed around:

python

```
transformers.generation.logits_process.isin_mps_friendly
```

The wrapper moves the token-suppression tensor to the same device as the input elements before calling the original function.

This is a process-wide Transformers compatibility workaround.

### 4. Bark history tensor modifications

The tested Transformers installation also contains two targeted modifications in:

text

```
transformers/models/bark/modeling_bark.py
```

**Semantic and coarse history tensors**

The following tensors are moved to the semantic generation device before concatenation:

python

```
generation_device = semantic_output.device

x_semantic_history = x_semantic_history.to(generation_device)
x_coarse = x_coarse.to(generation_device)
```

**Fine history tensor**

The fine history tensor is moved to the fine input device before concatenation:

python

```
x_fine_history = x_fine_history.to(fine_input.device)
```

These modifications address device mismatches involving Bark history tensors.

The source modifications are environment-specific compatibility changes, not official upstream Transformers fixes.

### Applying the patch

The repository includes a reproducible patch for Transformers 4.48.0:

`patches/transformers-4.48.0-bark-cuda.patch`

See [patches/README.md](patches/README.md) for backup,
application, verification, and rollback instructions.

The patch was tested against the original Transformers 4.48.0
source extracted from the official Python wheel.

## Initial Performance Baseline

The first successful generation using the consolidated implementation produced the following results:

|Metric|Result|
|---|---|
|GPU|NVIDIA GeForce RTX 5090 Laptop GPU|
|Model|suno/bark-small|
|Voice preset|v2/en_speaker_9|
|Sample rate|24,000 Hz|
|Generation time|9.01 seconds|
|Peak GPU memory|1.75 GB|
|Output|WAV|
|Audio verification|Audible and clear|

These measurements represent one test run, not a comprehensive benchmark.

Performance will vary depending on text length, GPU power limits, model configuration, and system load.

## Current Warnings

The initial successful generation produced the following Transformers warning:

text

```
The attention mask and the pad token id were not set.
```

Transformers also reported:

text

```
Setting pad_token_id to eos_token_id
```

Despite these warnings, audio generation completed successfully.

The warnings are being retained as known issues for investigation. Future versions will evaluate explicit attention-mask handling without changing the currently working generation behavior prematurely.

## Speaker Presets

The current implementation supports selecting a Bark speaker preset through the `voice` parameter.

Example:

python

```
speech.generate(
    text="Hello, how are you today?",
    voice="v2/en_speaker_9",
    output_file="output/speaker_test.wav"
)
```

The current implementation accepts a speaker preset through the voice parameter. A structured voice registry and systematic speaker comparisons are planned for future releases.

Voice labels and regional accent characteristics will be verified through listening tests rather than assumed from preset numbers.

## Multilingual Speech and Accent Experiments

Planned language experiments include:

- British English
- Indian English
- Chinese

The project will evaluate:

- Pronunciation accuracy
- Naturalness
- Voice consistency
- Accent characteristics
- Speech intelligibility

Bark's multilingual generation capabilities do not guarantee a specific regional accent. Accent reproduction will therefore be treated as an experimental evaluation.

## Roadmap

|Release|Planned functionality|
|---|---|
|v0.1.0|Working NVIDIA GPU implementation|
|v0.2.0|Clean implementation and configurable speaker presets|
|v0.3.0|Female voice and multilingual experiments|
|v0.4.0|FastAPI speech generation service|
|v0.5.0|GPU performance optimization|
|v1.0.0|Application-ready release|

## Future FastAPI Integration

The planned REST interface will accept:

json

```
{
  "text": "Hello, welcome to our application.",
  "voice": "v2/en_speaker_9",
  "format": "wav"
}
```

The service will eventually support:

- HTTP speech generation requests
- Configurable speaker selection
- WAV and additional output formats
- GPU-aware inference
- Request validation
- Error handling
- Model lifecycle management

## Known Limitations

- The current implementation has been validated with `suno/bark-small`.
- GPU compatibility workarounds are specific to the tested Transformers implementation.
- The generation token-suppression workaround modifies a Transformers module at runtime.
- Only WAV output is currently implemented.
- Regional accent capabilities have not yet been systematically evaluated.
- Concurrent API generation has not yet been implemented or tested.

## Licensing

This repository is distributed under the MIT License for its original project code.

Suno Bark, Transformers, PyTorch, and associated model weights are third-party components subject to their respective licenses and terms.

Refer to the original [Suno Bark repository](https://github.com/suno-ai/bark) and [Hugging Face model page](https://huggingface.co/suno/bark-small) for upstream information.

## Acknowledgements

- [Suno Bark](https://github.com/suno-ai/bark)
- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [PyTorch](https://pytorch.org/)
- NVIDIA CUDA

## Author

Srini B

GitHub: @turiya9710

---

**Project:** bark-rtx-gpu