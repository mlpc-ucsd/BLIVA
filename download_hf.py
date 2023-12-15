from huggingface_hub import snapshot_download
snapshot_download(repo_id = "liuhaotian/LLaVA-Pretrain", repo_type = "dataset", local_dir = "/workspace/BLIVA/llava_data")