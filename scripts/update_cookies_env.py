import base64
import os
import sys
import argparse

def update_env_file(env_path, key, value):
    """Updates or adds a key-value pair in a .env file."""
    lines = []
    if os.path.exists(env_path):
        with open(env_path, 'r') as f:
            lines = f.readlines()

    found = False
    new_lines = []
    for line in lines:
        if line.strip().startswith(f"{key}="):
            new_lines.append(f"{key}={value}\n")
            found = True
        else:
            new_lines.append(line)
    
    if not found:
        if new_lines and not new_lines[-1].endswith('\n'):
             new_lines.append('\n')
        new_lines.append(f"{key}={value}\n")

    with open(env_path, 'w') as f:
        f.writelines(new_lines)
    print(f"✅ Updated {key} in {env_path}")

def main():
    parser = argparse.ArgumentParser(description="Convert cookies.txt to base64 for COOKIES_B64 environment variable.")
    parser.add_argument("cookies_file", nargs="?", default="cookies.txt", help="Path to the cookies file (default: cookies.txt)")
    parser.add_argument("--env", default=".env", help="Path to .env file to update (default: .env)")
    parser.add_argument("--no-update", action="store_true", help="Do not update .env file, just print base64")

    args = parser.parse_args()

    # Check relative to script location if not found in cwd
    if not os.path.exists(args.cookies_file):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        possible_path = os.path.join(project_root, args.cookies_file)
        if os.path.exists(possible_path):
             args.cookies_file = possible_path
        else:
            print(f"❌ Error: File '{args.cookies_file}' not found.")
            sys.exit(1)

    print(f"Reading cookies from: {args.cookies_file}")
    with open(args.cookies_file, 'rb') as f:
        content = f.read()

    b64_encoded = base64.b64encode(content).decode('utf-8')

    print(f"\n--- COOKIES_B64 ({len(b64_encoded)} chars) ---")
    # Print a truncated version to avoid spamming the console if it's huge, unless piping?
    # Actually user wants the value, so printing it fully is better.
    print(b64_encoded)
    print("-------------------------------------------\n")

    if not args.no_update:
        # Determine .env path
        env_path = args.env
        if not os.path.exists(env_path):
             # Try looking in project root if script is run from scripts/
             script_dir = os.path.dirname(os.path.abspath(__file__))
             project_root = os.path.dirname(script_dir)
             possible_env = os.path.join(project_root, ".env")
             if os.path.exists(possible_env):
                 env_path = possible_env

        if os.path.exists(env_path):
            update_env_file(env_path, "COOKIES_B64", b64_encoded)
        else:
            print(f"⚠️  {args.env} not found. Skipping update.")

if __name__ == "__main__":
    main()
