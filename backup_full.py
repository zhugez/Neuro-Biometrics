import zipfile
import os
import datetime
import shutil
import json
import argparse

# --- PHẦN 1: ZIP WEIGHTS ---
def zip_weights():
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_name = f"weights_backup_{timestamp}.zip"
    
    dirs_to_check = [
        "experiments/v1_two_stage_snr_0_5_10_20/weights",
        "experiments/v2_mamba_denoiser/weights",
        "weights"
    ]
    
    # Also backup result JSONs and READMEs
    extra_files = [
        "experiments/v2_mamba_denoiser/output_v2_mamba.json",
        "experiments/v2_mamba_denoiser/README.md",
        "README.md",
    ]
    
    print(f"📦 [1/3] Đang nén file weights vào: {zip_name}...")
    count = 0
    with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Zip weight files
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
        
        # Zip extra files (results, READMEs)
        for f in extra_files:
            if os.path.exists(f):
                zipf.write(f, f)
                count += 1
                print(f"  + {f}")
    
    if count > 0:
        size_mb = os.path.getsize(zip_name) / (1024 * 1024)
        print(f"  ✓ {count} files, {size_mb:.1f} MB")
        return zip_name
    else:
        if os.path.exists(zip_name): os.remove(zip_name)
        return None


# --- PHẦN 2: COPY TO KAGGLE OUTPUT ---
def save_to_kaggle(filepath):
    """Copy zip vào /kaggle/working/ để Kaggle tự lưu khi commit."""
    output_dir = "/kaggle/working"
    if not os.path.isdir(output_dir):
        print(f"\n📂 [2/3] Không tìm thấy {output_dir} (không phải Kaggle env). Bỏ qua.")
        return
    
    dest = os.path.join(output_dir, os.path.basename(filepath))
    abs_path = os.path.abspath(filepath)
    
    if abs_path.startswith(output_dir):
        print(f"\n📂 [2/3] File đã nằm trong Kaggle output: {abs_path}")
    else:
        print(f"\n📂 [2/3] Copy vào Kaggle output...")
        shutil.copy2(filepath, dest)
        print(f"  📍 {dest}")


# --- PHẦN 3: UPLOAD TO GOOGLE DRIVE ---
def upload_to_gdrive(filepath, client_secret_path, folder_id=None):
    """Upload file lên Google Drive sử dụng OAuth2 client secret."""
    try:
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import InstalledAppFlow
        from google.auth.transport.requests import Request
        from googleapiclient.discovery import build
        from googleapiclient.http import MediaFileUpload
    except ImportError:
        print("\n☁️  [3/3] Thiếu thư viện Google API. Cài đặt:")
        print("  pip install google-api-python-client google-auth-oauthlib")
        return False
    
    SCOPES = ['https://www.googleapis.com/auth/drive.file']
    TOKEN_PATH = os.path.join(os.path.dirname(client_secret_path), 'token.json')
    
    creds = None
    
    # Load existing token
    if os.path.exists(TOKEN_PATH):
        creds = Credentials.from_authorized_user_file(TOKEN_PATH, SCOPES)
    
    # Refresh or create new token
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            print("  🔄 Refreshing expired token...")
            creds.refresh(Request())
        else:
            print("  🔑 Mở trình duyệt để xác thực Google Drive...")
            print("  (Nếu không có trình duyệt, chạy trên máy local trước rồi copy token.json)")
            flow = InstalledAppFlow.from_client_secrets_file(client_secret_path, SCOPES)
            try:
                creds = flow.run_local_server(port=0, open_browser=True)
            except Exception:
                # Fallback for headless environments (Kaggle, SSH, etc.)
                print("  ⚠️ Không mở được trình duyệt. Dùng console flow...")
                creds = flow.run_console()
        
        # Save token for next time
        with open(TOKEN_PATH, 'w') as token:
            token.write(creds.to_json())
        print(f"  ✓ Token saved to {TOKEN_PATH}")
    
    # Build Drive service
    service = build('drive', 'v3', credentials=creds)
    
    # Upload file
    file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
    filename = os.path.basename(filepath)
    print(f"\n☁️  [3/3] Uploading {filename} ({file_size_mb:.1f} MB) to Google Drive...")
    
    file_metadata = {'name': filename}
    if folder_id:
        file_metadata['parents'] = [folder_id]
    
    media = MediaFileUpload(filepath, resumable=True)
    file = service.files().create(
        body=file_metadata,
        media_body=media,
        fields='id, name, webViewLink'
    ).execute()
    
    print(f"  ✅ Upload thành công!")
    print(f"  📎 File ID: {file.get('id')}")
    if file.get('webViewLink'):
        print(f"  🔗 Link: {file.get('webViewLink')}")
    
    return True


# --- MAIN ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backup weights & upload to Google Drive")
    parser.add_argument("--gdrive", action="store_true", help="Upload to Google Drive")
    parser.add_argument("--client-secret", type=str, 
                        default="client_secret_830574298098-vk4kcodn9jvrdsdh58bcfoccgt73qikg.apps.googleusercontent.com.json",
                        help="Path to Google OAuth client secret JSON")
    parser.add_argument("--folder-id", type=str, default=None,
                        help="Google Drive folder ID to upload to (optional)")
    args = parser.parse_args()
    
    zip_file = zip_weights()
    if zip_file:
        # Step 2: save to Kaggle if available
        save_to_kaggle(zip_file)
        
        # Step 3: upload to Google Drive if requested
        if args.gdrive:
            if not os.path.exists(args.client_secret):
                print(f"\n❌ Không tìm thấy file client secret: {args.client_secret}")
                print("  Đặt file JSON vào thư mục gốc của project.")
            else:
                upload_to_gdrive(zip_file, args.client_secret, args.folder_id)
        else:
            print("\n💡 Để upload lên Google Drive, thêm flag --gdrive:")
            print(f"   python backup_full.py --gdrive")
        
        print(f"\n✅ HOÀN TẤT!")
    else:
        print("⚠️ Không có file weights nào để backup.")
