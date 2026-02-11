import gdown
import zipfile
import os

# ID lấy từ link cậu đưa
DRIVE_ID = "1V6i8Cq-7ldFtIlDq9T-TlIh6eN6AAP_2" 
OUTPUT_FILE = "dataset.zip"
EXTRACT_TO = "./"

def main():
    print(f"⬇️ Đang tải {OUTPUT_FILE} từ Google Drive...")
    # Dùng fuzzy=True để gdown tự tìm link download chính xác
    url = f'https://drive.google.com/uc?id={DRIVE_ID}'
    gdown.download(url, OUTPUT_FILE, quiet=False, fuzzy=True)

    if not os.path.exists(OUTPUT_FILE):
        print("❌ Lỗi: Không tải được file.")
        return

    print(f"\n📦 Đang giải nén vào {EXTRACT_TO}...")
    if not os.path.exists(EXTRACT_TO):
        os.makedirs(EXTRACT_TO)

    try:
        with zipfile.ZipFile(OUTPUT_FILE, 'r') as zip_ref:
            zip_ref.extractall(EXTRACT_TO)
        print("✅ Xong! Dataset đã sẵn sàng.")
    except zipfile.BadZipFile:
        print("❌ Lỗi: File zip bị lỗi (có thể do chưa tải xong hoặc link sai).")

if __name__ == "__main__":
    main()
