import os
import sys
import subprocess
import requests

def get_staged_diff():
    try:
        # Force UTF-8 encoding and gracefully replace unmappable bytes to avoid Windows crashes
        diff = subprocess.check_output(
            ["git", "diff", "--cached"], 
            encoding="utf-8", 
            errors="replace"
        )
        return diff
    except Exception as e:
        print(f"Error fetching git diff: {e}")
        sys.exit(1)

def generate_commit_message(diff_content):
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("CRITICAL: GEMINI_API_KEY environment variable is missing.")
        return "chore: incremental update (Gemini API Key missing)"

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3.1-flash-lite:generateContent?key={api_key}"
    
    prompt = f"""
You are an expert Git automation manager. Review the following staged code diff and generate a precise, clear, and highly technical commit message.
Focus on architectural modifications, API schema shifts, or changes to the retrieval pipelines.

Format requirements:
- First line: A concise summary line (max 70 characters) prefixed with a conventional commit type (feat, fix, refactor, chore, docs).
- Blank line.
- Bullet points detailing exactly what changed, what files were affected, and why.

Staged Code Changes:
\"\"\"
{diff_content}
\"\"\"
"""

    payload = {
        "contents": [{
            "parts": [{"text": prompt}]
        }],
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": 600
        }
    }

    try:
        response = requests.post(url, json=payload, headers={"Content-Type": "application/json"}, timeout=15)
        response_json = response.json()
        
        # 🔍 NEW: Catch structural server-side errors immediately
        if "error" in response_json:
            print(f"❌ Google API Server Error ({response.status_code}): {response_json['error'].get('message')}")
            return "chore: automated code checkpoint updates"
            
        if "candidates" not in response_json:
            print(f"⚠️ Unexpected Response Layout: {response_json}")
            return "chore: automated code checkpoint updates"

        commit_text = response_json['candidates'][0]['content']['parts'][0]['text'].strip()
        return commit_text
    except Exception as e:
        print(f"API generation failed: {e}. Falling back to standard message.")
        return "chore: automated code checkpoint updates"

def main():
    if len(sys.argv) < 2:
        sys.exit(1)
        
    commit_msg_filepath = sys.argv[1]
    diff_content = get_staged_diff()
    
    if not diff_content.strip():
        # No changes to document
        return

    print("⚡ Gemini 3.1 Flash Lite is analyzing staged changes...")
    ai_message = generate_commit_message(diff_content)
    
    with open(commit_msg_filepath, "w", encoding="utf-8") as f:
        f.write(ai_message)

if __name__ == "__main__":
    main()