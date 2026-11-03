import os

path = "/kaggle/input/datasets/deepresearch1a10/hf-token-store/hf_token.txt"
print("checking:", path)
if os.path.isfile(path):
    with open(path, "r", encoding="utf-8") as f:
        token = f.read().strip()
    print("token loaded, length:", len(token) if token else 0)
    if token:
        try:
            from huggingface_hub import HfApi
            api = HfApi(token=token)
            info = api.whoami()
            print("whoami ok:", info.get("type"), info.get("name"))
        except Exception as exc:
            print("whoami failed:", repr(exc))
else:
    print("dataset file not found")
print("TEST_DONE")
