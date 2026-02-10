import os
import subprocess
import sys

def install_gdrive():
    print("⬇️ Đang tải tool upload GDrive (glotlabs/gdrive)...")
    # Tải binary gdrive (Linux x64) - phiên bản ổn định
    cmd = "wget -q -O gdrive https://github.com/glotlabs/gdrive/releases/download/3.1.0/gdrive_linux-x64 && chmod +x gdrive"
    ret = subprocess.run(cmd, shell=True)
    if ret.returncode != 0:
        print("❌ Lỗi tải gdrive. Kiểm tra kết nối mạng.")
        sys.exit(1)
    print("✅ Cài đặt xong ./gdrive")

def check_login():
    # Kiểm tra xem đã login chưa bằng lệnh list
    ret = subprocess.run("./gdrive account list", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return ret.returncode == 0

def login():
    print("\n🔐 CẦN ĐĂNG NHẬP (Chỉ làm 1 lần duy nhất)")
    print("1. Chạy lệnh sau trong terminal: ./gdrive account add")
    print("2. Copy link hiện ra -> Mở bằng trình duyệt trên máy tính của bạn.")
    print("3. Đăng nhập Google -> Copy mã xác thực -> Paste vào terminal.")
    print("\n👉 Đang chạy lệnh đăng nhập cho bạn...")
    subprocess.run("./gdrive account add", shell=True)

def upload(filepath):
    if not check_login():
        login()
    
    print(f"\n🚀 Đang upload {filepath} lên Google Drive...")
    # Upload file
    ret = subprocess.run(f"./gdrive files upload \"{filepath}\"", shell=True)
    if ret.returncode == 0:
        print(f"✅ Upload thành công file: {filepath}")
    else:
        print("❌ Upload thất bại.")

if __name__ == "__main__":
    # 1. Check/Install tool
    if not os.path.exists("./gdrive"):
        install_gdrive()
    
    # 2. Tìm file zip backup mới nhất
    files = [f for f in os.listdir(".") if f.startswith("weights_backup_") and f.endswith(".zip")]
    if files:
        latest_file = max(files, key=os.path.getctime)
        upload(latest_file)
    else:
        print("⚠️ Không tìm thấy file zip backup nào.")
        print("👉 Hãy chạy 'python zip_weights.py' trước để tạo file nén.")
