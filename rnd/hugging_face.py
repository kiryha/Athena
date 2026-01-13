"""
Check status of downloaded models

How to download (cmd)
pip install huggingface_hub
huggingface-cli download THUDM/CogVideoX-2b --local-dir E:/Models/CogVideoX-2b --local-dir-use-symlinks False
"""

from huggingface_hub import snapshot_download

# usage: verify_download.py

REPO_ID = "THUDM/CogVideoX-2b"
# MUST match the folder you intend to use in your main app
LOCAL_DIR = "E:/Models/CogVideoX-2b" 

print(f"Verifying files for {REPO_ID} in {LOCAL_DIR}...")

try:
    # This function checks if local files match the remote files.
    # It repairs broken files and downloads missing ones automatically.
    path = snapshot_download(
        repo_id=REPO_ID, 
        local_dir=LOCAL_DIR, 
        local_dir_use_symlinks=False, # Important for Windows
        resume_download=True          # Resumes partial downloads
    )
    print("\n✅ SUCCESS: All files are verified and present.")
    print(f"Model is ready at: {path}")

except Exception as e:
    print(f"\n❌ ERROR: Download incomplete or corrupted.")
    print(str(e))