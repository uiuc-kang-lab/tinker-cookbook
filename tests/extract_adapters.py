import os

adapters = {
    "Qwen3.5-4B-R1": "model_9dcd85d3",
    "Qwen3.5-4B-R2": "model_b9bd1d67",
    "Qwen3.5-4B-R4": "model_eb225491",
    "Qwen3.5-2B-R1": "model_480fb55b",
    "Qwen3.5-2B-R2": "model_9790c804",
    "Qwen3.5-2B-R4": "model_80d4939b",
    "Qwen3.5-0.8B-R1": "model_8b9c6cd6",
    "Qwen3.5-0.8B-R2": "model_70f34e13",
    "Qwen3.5-0.8B-R4": "model_6dbaae77",
}

adapter_path_template = "/tmp/skyrl_checkpoints/{adapter_id}/sampler_weights/0000.tar.gz"

for adapter_name, adapter_id in adapters.items():
    adapter_path = adapter_path_template.format(adapter_id=adapter_id)
    # decompress the tar.gz file to adapters/{adapter_name}.safetensors
    os.system(f"cp {adapter_path} adapters/{adapter_name}.tar.gz")
    os.system(f"tar -xzf adapters/{adapter_name}.tar.gz -C adapters")
    os.system(f"mv adapters/adapter_model.safetensors adapters/{adapter_name}.safetensors")
    os.system(f"rm adapters/{adapter_name}.tar.gz")
