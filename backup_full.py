import zipfile
import os
import datetime
import shutil

# --- PHẦN 1: ZIP WEIGHTS ---
def zip_weights():
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_name = f"weights_backup_{timestamp}.zip"
    
    dirs_to_check = [
        "experiments/v1_two_stage_snr_0_5_10_20/weights",
        "experiments/v2_mamba_denoiser/weights",
        "weights"
    ]
    
    print(f"📦 [1/2] Đang nén file weights vào: {zip_name}...")
    count = 0
    with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for folder in dirs_to_check:
            if os.path.exists(folder):
                for root, _, files in os.walk(folder):
                    for file in files:
                        if file.endswith(".pth") or file.endswith(".pt"):
                            file_path = os.path.join(root, file)
                            arcname = os.path.relpath(file_path, start=".")
                            zipf.write(file_path, arcname)
                            count += 1
                            print(f"  + {arcname}")
    
    if count > 0:
        size_mb = os.path.getsize(zip_name) / (1024 * 1024)
        print(f"  ✓ {count} files, {size_mb:.1f} MB")
        return zip_name
    else:
        if os.path.exists(zip_name): os.remove(zip_name)
        return None

# --- PHẦN 2: COPY TO KAGGLE OUTPUT ---
def save_to_output(filepath):
    """Copy zip vào /kaggle/working/ để Kaggle tự lưu khi commit."""
    output_dir = "/kaggle/working"
    dest = os.path.join(output_dir, os.path.basename(filepath))
    
    # Nếu file đã nằm trong /kaggle/working, chỉ cần thông báo
    abs_path = os.path.abspath(filepath)
    if abs_path.startswith(output_dir):
        print(f"\n📂 [2/2] File đã nằm trong Kaggle output:")
        print(f"  📍 {abs_path}")
    else:
        print(f"\n📂 [2/2] Copy vào Kaggle output...")
        shutil.copy2(filepath, dest)
        print(f"  📍 {dest}")
    
    size_mb = os.path.getsize(dest if not abs_path.startswith(output_dir) else abs_path) / (1024 * 1024)
    print(f"\n✅ HOÀN TẤT! ({size_mb:.1f} MB)")
    print("💡 Để tải về: Kaggle Notebook → Output tab → Download")
    print("💡 Hoặc chạy: cp <file> /kaggle/working/ trước khi Save & Run All")

# --- MAIN ---
if __name__ == "__main__":
    zip_file = zip_weights()
    if zip_file:
        save_to_output(zip_file)
    else:
        print("⚠️ Không có file weights nào để backup.")
