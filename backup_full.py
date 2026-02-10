import zipfile
import os
import datetime
import subprocess
import sys

# --- PHẦN 1: ZIP WEIGHTS ---
def zip_weights():
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_name = f"weights_backup_{timestamp}.zip"
    
    dirs_to_check = [
        "experiments/v1_two_stage_snr_0_5_10_20/weights",
        "experiments/v2_mamba_denoiser/weights",
        "weights"
    ]
    
    print(f"📦 [1/3] Đang nén file weights vào: {zip_name}...")
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
        return zip_name
    else:
        if os.path.exists(zip_name): os.remove(zip_name)
        return None

# --- PHẦN 2: UPLOAD GDRIVE ---
def install_gdrive():
    if not os.path.exists("./gdrive"):
        print("⬇️ [2/3] Đang tải tool upload GDrive...")
        cmd = "wget -q -O gdrive https://github.com/glotlabs/gdrive/releases/download/3.1.0/gdrive_linux-x64 && chmod +x gdrive"
        subprocess.run(cmd, shell=True)

def check_login():
    ret = subprocess.run("./gdrive account list", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return ret.returncode == 0

def login():
    print("\n🔐 CẦN ĐĂNG NHẬP (Lần đầu)")
    print("👉 Copy link dưới, dán vào trình duyệt, login rồi copy code về đây:")
    subprocess.run("./gdrive account add", shell=True)

def upload(filepath):
    install_gdrive()
    if not check_login():
        login()
    
    print(f"\n🚀 [3/3] Đang upload {filepath} lên Google Drive...")
    ret = subprocess.run(f"./gdrive files upload \"{filepath}\"", shell=True)
    if ret.returncode == 0:
        print(f"✅ HOÀN TẤT! File đã lên Drive: {filepath}")
    else:
        print("❌ Lỗi upload.")

# --- MAIN ---
if __name__ == "__main__":
    zip_file = zip_weights()
    if zip_file:
        upload(zip_file)
    else:
        print("⚠️ Không có file weights nào để backup.")
