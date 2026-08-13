# Environment Record

- Recorded: 2026-08-13 16:09:36 +05:30
- Operating system: Windows 11, 64-bit
- Python: 3.11.9 in `.venv`
- Device: CPU (`torch.cuda.is_available() == False`)
- Inference dtype: fp32
- GPU: Intel Iris Xe; no CUDA device
- CPU: Intel Core i5-1135G7, 4 cores / 8 logical processors
- Memory: 15.79 GB total; approximately 4 GB free during the initial audit

## Verified packages

| Package | Version |
|---|---:|
| torch | 2.13.0 |
| torchvision | 0.28.0 |
| ultralytics | 8.4.118 |
| transformers | 4.44.0 |
| accelerate | 1.14.0 |
| Pillow | 11.3.0 |
| pi-heif | 1.4.0 |
| gradio | 5.49.1 |
| einops | 0.8.2 |
| huggingface-hub | 0.36.2 |
| safetensors | 0.8.0 |
| sentencepiece | 0.2.2 |
| pytest | 9.1.1 |

All imports completed successfully and `pip check` reported no broken requirements.

## Source prompt

- SHA-256: `CA067315162DEF65B957C0015784A2C7942A4BEFA91645418473F7CA7AF29BA0`
