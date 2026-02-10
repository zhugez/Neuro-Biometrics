import zipfile
import os
import datetime

def zip_weights():
    # Timestamp for versioning
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_name = f"weights_backup_{timestamp}.zip"
    
    # Source folders to check
    dirs_to_check = [
        "experiments/v1_two_stage_snr_0_5_10_20/weights",
        "experiments/v2_mamba_denoiser/weights",
        "weights" # Root weights if any
    ]
    
    print(f"📦 Đang nén file weights vào: {zip_name}...")
    
    count = 0
    with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for folder in dirs_to_check:
            if os.path.exists(folder):
                for root, _, files in os.walk(folder):
                    for file in files:
                        if file.endswith(".pth") or file.endswith(".pt"):
                            file_path = os.path.join(root, file)
                            # Archive name structure: v1/weights/best.pth
                            arcname = os.path.relpath(file_path, start=".")
                            zipf.write(file_path, arcname)
                            count += 1
                            print(f"  + {arcname}")
    
    if count > 0:
        print(f"\n✅ Xong! File nén: {os.path.abspath(zip_name)}")
        print(f"👉 Hãy upload file này lên Google Drive thủ công để backup.")
    else:
        print("\n⚠️ Không tìm thấy file .pth/.pt nào để nén.")
        # Clean up empty zip
        os.remove(zip_name)

if __name__ == "__main__":
    zip_weights()
