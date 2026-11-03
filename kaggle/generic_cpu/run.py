import os
import platform
import socket
import sys

print("host      :", socket.gethostname())
print("python    :", sys.version.split()[0])
print("platform  :", platform.platform())
print("cwd       :", os.getcwd())

token = None
try:
    from kaggle_secrets import UserSecretsClient
    usc = UserSecretsClient()
    token = usc.get_secret("HF_TOKEN")
    print("HF token loaded from Kaggle secret, length:", len(token) if token else 0)
except Exception as exc:
    print("Could not load HF token from Kaggle secret:", repr(exc))
    token = os.environ.get("HF_TOKEN")
    if token:
        print("HF token loaded from environment, length:", len(token))
    else:
        print("No HF token found")

if token:
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=token)
        info = api.whoami()
        print("HF whoami type:", info.get("type"))
        print("HF whoami name:", info.get("name"))
    except Exception as exc:
        print("HF whoami failed:", repr(exc))

    try:
        from huggingface_hub import upload_file
        import tempfile
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("debug upload test from generic cpu")
            tmp = f.name
        upload_file(path_or_fileobj=tmp, path_in_repo="debug_test_cpu.txt", repo_id="w-ahmad/paper-scope-vectors", repo_type="dataset", token=token)
        print("DEBUG UPLOAD OK")
    except Exception as exc:
        print("DEBUG UPLOAD FAILED:", repr(exc))
else:
    print("No token available, skipping HF calls")

print("GENERIC-KERNEL-OK")
