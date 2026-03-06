import os
import sys
import subprocess
import argparse
import base64
from pathlib import Path

# Add the parent directory to sys.path to allow importing from app (if needed in future)
sys.path.append(str(Path(__file__).parent.parent))

def get_project_root():
    return Path(__file__).parent.parent

def check_yt_dlp_installed():
    try:
        subprocess.run(["yt-dlp", "--version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def extract_cookies(browser="chrome", output_file="cookies.txt"):
    """
    Extracts cookies from the specified browser using yt-dlp and saves them to output_file.
    """
    print(f"🍪 Attempting to extract cookies from {browser}...")
    
    # yt-dlp command to extract cookies
    # We use --cookies-from-browser which dumps cookies to stdout or file
    # syntax: yt-dlp --cookies-from-browser browser_name --cookies output_file --dump-user-agent --skip-download URL
    # Actually, we just want to export cookies. The most reliable way to JUST get cookies is:
    # yt-dlp --cookies-from-browser chrome --cookies cookies.txt --skip-download https://www.youtube.com
    
    cmd = [
        "yt-dlp",
        "--cookies-from-browser", browser,
        "--cookies", output_file,
        "--skip-download",
        "--flat-playlist",  # Don't resolve all videos in the feed
        "https://www.youtube.com"
    ]

    try:
        # Run the command
        # We don't capture output here so we can see real-time progress and prompts
        result = subprocess.run(cmd, check=False)
        
        if result.returncode != 0:
            print(f"❌ Failed to extract cookies.")
            print("\n⚠️  HINT: If you see 'database is locked', please CLOSE your browser completely and try again.")
            return False
            
        if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
            print(f"✅ Successfully extracted cookies to {output_file}")
            return True
        else:
            print(f"❌ Command finished but {output_file} is missing or empty.")
            return False

    except Exception as e:
        print(f"❌ An error occurred: {e}")
        return False

def update_env_file(cookies_file):
    """
    Reads the cookies file, encodes to base64, and updates the .env file.
    """
    env_path = get_project_root() / ".env"
    
    if not os.path.exists(cookies_file):
        print(f"⚠️  {cookies_file} not found. Skipping env update.")
        return

    print(f"🔄 Updating .env file with new cookies...")
    
    try:
        with open(cookies_file, 'rb') as f:
            content = f.read()
            
        b64_encoded = base64.b64encode(content).decode('utf-8')
        
        # Read existing .env
        lines = []
        if os.path.exists(env_path):
            with open(env_path, 'r') as f:
                lines = f.readlines()
        
        key = "COOKIES_B64"
        found = False
        new_lines = []
        
        for line in lines:
            if line.strip().startswith(f"{key}="):
                new_lines.append(f"{key}={b64_encoded}\n")
                found = True
            else:
                new_lines.append(line)
        
        if not found:
            if new_lines and not new_lines[-1].endswith('\n'):
                new_lines.append('\n')
            new_lines.append(f"{key}={b64_encoded}\n")
            
        with open(env_path, 'w') as f:
            f.writelines(new_lines)
            
        print(f"✅ Updated {key} in {env_path}")
        print(f"ℹ️  If your app is running in Docker, you may need to restart it: docker-compose restart")
        
    except Exception as e:
        print(f"❌ Failed to update .env: {e}")

def main():
    parser = argparse.ArgumentParser(description="Extract YouTube cookies from browser and update project config.")
    parser.add_argument("--browser", default="chrome", help="Browser to extract from (chrome, firefox, chromium, edge, safari, vivaldi, opera, brave). Default: chrome")
    parser.add_argument("--output", default="cookies.txt", help="Output filename. Default: cookies.txt")
    parser.add_argument("--skip-env", action="store_true", help="Skip updating the .env file")
    
    args = parser.parse_args()
    
    output_path = get_project_root() / args.output
    
    # Check if yt-dlp is installed
    if not check_yt_dlp_installed():
        print("❌ yt-dlp is not found in your PATH.")
        print("Please install it locally first: pip install yt-dlp")
        sys.exit(1)
        
    print(f"ℹ️  This script runs on your HOST machine to extract cookies for the Docker container.")
    print(f"ℹ️  Make sure you are logged into YouTube in {args.browser}.")
    
    # Warn about browser locking
    print(f"⚠️  IMPORTANT: Please ensure {args.browser} is CLOSED before proceeding (to avoid database lock errors).")
    # input("Press Enter when ready...") # Disabled for automation/non-interactive run, but good for real usage
    
    success = extract_cookies(args.browser, str(output_path))
    
    if success and not args.skip_env:
        update_env_file(str(output_path))

if __name__ == "__main__":
    main()
