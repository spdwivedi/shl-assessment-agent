import os
import requests
import json

# Local endpoint targeting our active FastAPI application
API_URL = "http://127.0.0.1:8000/chat"
#API_URL = "http://convo-agent.spdwivedi.me:8002/chat"

def simulate_multi_turn_scenario():
    print("🎭 Initiating Multi-Turn Agent Evaluation Replay Simulator...\n")
    
    conversation_steps = [
        "I need an assessment solution for hiring a software engineer.",
        "Mid-level, around 4 years of experience.",
        "Actually, let's make sure it's a technical coding test, not behavioral patterns."
    ]
    
    stateless_messages = []
    
    for turn_idx, user_input in enumerate(conversation_steps, start=1):
        print(f"--- TURN {turn_idx} ---")
        print(f"👤 User: {user_input}")
        
        stateless_messages.append({"role": "user", "content": user_input})
        payload = {"messages": stateless_messages}
        
        try:
            # 🔄 Updated: Timeout extended to 30 seconds to match assignment constraints
            response = requests.post(API_URL, json=payload, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            print(f"🤖 Agent Reply: {data.get('reply')}")
            print(f"📋 Shortlist Recommendations Provided: {json.dumps(data.get('recommendations'), indent=2)}")
            print(f"🛑 End Of Conversation Flag: {data.get('end_of_conversation')}\n")
            
            stateless_messages.append({"role": "assistant", "content": data.get('reply')})
            
        except Exception as e:
            print(f"❌ Network simulation turn failed: {e}")
            break

if __name__ == "__main__":
    simulate_multi_turn_scenario()