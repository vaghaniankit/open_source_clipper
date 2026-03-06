import sys
import os

print("Script started")

# Add project root to path so we can import from app
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
print(f"Adding to path: {project_root}")
sys.path.insert(0, project_root)

try:
    from app.services.youtube_downloader import download_youtube
    print("Import successful")
except ImportError as e:
    print(f"Import failed: {e}")
    sys.exit(1)

def main():
    # "Me at the zoo" - short, safe for testing
    url = "https://www.youtube.com/watch?v=jNQXAC9IVRw"
    print(f"Testing download with URL: {url}")
    
    # Ensure downloads directory exists (optional, yt-dlp might do it)
    if not os.path.exists("downloads"):
        os.makedirs("downloads")
        
    try:
        download_youtube(url, filename="test_video")
        print("Download successful!")
        
        # Verify file exists
        files = os.listdir("downloads")
        print(f"Files in downloads/: {files}")
        
    except Exception as e:
        print(f"Download failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
