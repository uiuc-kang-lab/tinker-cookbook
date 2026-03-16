# Setup
import tinker
import urllib.request

service_client = tinker.ServiceClient(base_url="http://localhost:8000")
rc = service_client.create_rest_client()

models = ["Qwen/Qwen3.5-0.8B"]
ranks = [1, 2, 4]
for model in models:
    for rank in ranks:
        training_client = service_client.create_lora_training_client(
            base_model=model, rank=rank, train_mlp=False
        )
        
        # Save a checkpoint that you can use for sampling
        sampling_path = training_client.save_weights_for_sampler(name="0000").result().path
        
        print(f"Model: {model}, Rank: {rank}, path: {sampling_path}")
        
        
