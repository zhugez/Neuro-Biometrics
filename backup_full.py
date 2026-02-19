import zipfile
import os
import datetime
import shutil
import subprocess
import argparse

# Auto-load .env file (GOG_KEYRING_PASSWORD, etc.)
_env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
if os.path.exists(_env_path):
    with open(_env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                os.environ.setdefault(_k.strip(), _v.strip())

# --- PHẦN 1: ZIP WEIGHTS ---
def zip_weights():
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_name = f"weights_backup_{timestamp}.zip"
    
    dirs_to_check = [
        "experiments/v1_baseline/weights",
        "experiments/v2_mamba/weights",
        "weights"
    ]
    
    extra_files = [
        "experiments/v2_mamba/output_v2_mamba.json",
        "experiments/v2_mamba/README.md",
        "README.md",
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


# --- PHẦN 3: UPLOAD TO GOOGLE DRIVE via gogcli ---
GOG_KEYRING_PASSWORD = os.environ.get("GOG_KEYRING_PASSWORD", "neuro2024")


def _gog_env(account=None):
    """Build env dict for gog subprocess calls."""
    env = {**os.environ, "GOG_KEYRING_PASSWORD": GOG_KEYRING_PASSWORD}
    if account:
        env["GOG_ACCOUNT"] = account
    return env


def _check_gog():
    """Check if gog CLI is installed."""
    try:
        r = subprocess.run(["gog", "--version"], capture_output=True, text=True, 
                          timeout=5, env=_gog_env())
        if r.returncode == 0:
            print(f"  ✓ gogcli: {r.stdout.strip()}")
            return True
    except FileNotFoundError:
        pass
    print("  ❌ gogcli chưa được cài đặt.")
    print("  Cài đặt: https://github.com/steipete/gogcli")
    print("  Linux:   curl -sL https://github.com/steipete/gogcli/releases/latest/download/gogcli_0.11.0_linux_amd64.tar.gz | tar xz -C /usr/local/bin gog")
    return False


def _setup_gog_auth(client_secret_path, account):
    """Setup gogcli credentials and auth if not already done."""
    env = _gog_env(account)
    
    # Step 1: Store credentials
    print(f"  🔧 Nạp credentials từ {os.path.basename(client_secret_path)}...")
    r = subprocess.run(
        ["gog", "auth", "credentials", client_secret_path],
        capture_output=True, text=True, timeout=10, env=env
    )
    if r.returncode != 0:
        print(f"  ⚠️ credentials: {r.stderr.strip()}")
    
    # Step 2: Check if already authenticated for drive
    r = subprocess.run(
        ["gog", "auth", "status"],
        capture_output=True, text=True, timeout=10, env=env
    )
    if r.returncode == 0 and account in (r.stdout + r.stderr):
        print(f"  ✓ Đã xác thực: {account}")
        return True
    
    # Step 3: Auth with manual flow (for headless/remote servers)
    print(f"\n  🔑 Xác thực tài khoản {account}...")
    print("  (Sử dụng manual flow - copy URL vào trình duyệt)\n")
    r = subprocess.run(
        ["gog", "auth", "add", account, "--services", "drive", "--manual"],
        timeout=300, env=env
    )
    return r.returncode == 0


def upload_to_gdrive(filepath, client_secret_path, account, folder_id=None):
    """Upload file lên Google Drive sử dụng gogcli."""
    print(f"\n☁️  [3/3] Google Drive Upload")
    
    if not _check_gog():
        return False
    
    if not _setup_gog_auth(client_secret_path, account):
        print("  ❌ Xác thực thất bại!")
        return False
    
    # Upload
    env = _gog_env(account)
    file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
    filename = os.path.basename(filepath)
    print(f"\n  ⬆️  Uploading {filename} ({file_size_mb:.1f} MB)...")
    
    cmd = ["gog", "drive", "upload", filepath]
    if folder_id:
        cmd.extend(["--parent", folder_id])
    
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600, env=env)
    
    if r.returncode == 0:
        print(f"  ✅ Upload thành công!")
        if r.stdout.strip():
            print(f"  {r.stdout.strip()}")
        return True
    else:
        print(f"  ❌ Upload thất bại: {r.stderr.strip()}")
        return False


# --- MAIN ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backup weights & upload to Google Drive")
    parser.add_argument("--gdrive", action="store_true", help="Upload to Google Drive via gogcli")
    parser.add_argument("--account", type=str, default=None,
                        help="Google account email (e.g. you@gmail.com)")
    parser.add_argument("--client-secret", type=str, 
                        default="client_secret.json",
                        help="Path to Google OAuth client secret JSON")
    parser.add_argument("--folder-id", type=str, default=None,
                        help="Google Drive folder ID to upload to (optional)")
    args = parser.parse_args()
    
    zip_file = zip_weights()
    if zip_file:
        save_to_kaggle(zip_file)
        
        if args.gdrive:
            if not args.account:
                print("\n❌ Cần chỉ định --account (email Google)")
                print("   Ví dụ: python backup_full.py --gdrive --account you@gmail.com")
            elif not os.path.exists(args.client_secret):
                print(f"\n❌ Không tìm thấy: {args.client_secret}")
            else:
                upload_to_gdrive(zip_file, args.client_secret, args.account, args.folder_id)
        else:
            print("\n💡 Để upload lên Google Drive:")
            print(f"   python backup_full.py --gdrive --account you@gmail.com")
        
        print(f"\n✅ HOÀN TẤT!")
    else:
        print("⚠️ Không có file weights nào để backup.")
