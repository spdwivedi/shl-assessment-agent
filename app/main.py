import os
from typing import Optional
from fastapi import FastAPI, Request, Header
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from app.schemas import ChatRequest, ChatResponse
from app.core_agent import SHLConversationAgent

app = FastAPI(title="SHL Conversational Assessment Recommender API")

# Enable global CORS for testing harnesses and frontend calls
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

agent = SHLConversationAgent()

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(payload: ChatRequest, x_gemini_api_key: Optional[str] = Header(None)):
    """
    Stateless conversational router. Automatically intercepts custom X-Gemini-API-Key 
    headers to guarantee concurrent multi-tenant isolation.
    """
    return agent.process_chat(payload.messages, user_api_key=x_gemini_api_key)

@app.get("/", response_class=HTMLResponse)
def serve_dashboard():
    return """
    <!DOCTYPE html>
    <html lang="en" class="dark">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>SHL Assessment Recommender Dashboard</title>
        <script src="https://cdn.tailwindcss.com"></script>
        <script>
            tailwind.config = {
                darkMode: 'class',
                theme: {
                    extend: {
                        colors: {
                            slate: { 950: '#0b0f19' }
                        }
                    }
                }
            }
        </script>
        <style>
            ::-webkit-scrollbar { width: 6px; height: 6px; }
            ::-webkit-scrollbar-track { background: #0f172a; }
            ::-webkit-scrollbar-thumb { background: #334155; border-radius: 4px; }
            ::-webkit-scrollbar-thumb:hover { background: #475569; }
        </style>
    </head>
    <body class="bg-slate-950 text-slate-100 min-h-screen font-sans antialiased">
        <div class="max-w-7xl mx-auto p-4 md:p-8">
            
            <header class="border-b border-slate-800 pb-6 mb-8 flex flex-col md:flex-row md:items-center md:justify-between gap-4">
                <div>
                    <div class="flex items-center gap-3">
                        <span class="px-2.5 py-1 text-xs font-semibold bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 rounded-full flex items-center gap-1.5">
                            <span class="w-1.5 h-1.5 bg-emerald-400 rounded-full animate-pulse"></span> Live Production
                        </span>
                        <span class="text-xs text-slate-400">v2.2 (Header Proxy Enabled)</span>
                    </div>
                    <h1 class="text-2xl md:text-3xl font-bold tracking-tight bg-gradient-to-r from-white via-slate-200 to-slate-400 bg-clip-text text-transparent mt-1">
                        SHL Conversational Assessment Recommender
                    </h1>
                    <p class="text-sm text-slate-400 mt-1">Autonomous Agentic Search & Candidate Screening Optimization Pipeline</p>
                </div>
                <div class="flex items-center gap-3 flex-wrap">
                    <button onclick="toggleAboutModal(true)" class="px-4 py-2 bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-500 hover:to-indigo-500 text-white text-sm font-medium rounded-lg shadow-lg shadow-indigo-600/10 transition duration-150 flex items-center gap-2">
                        ℹ️ About Developer & Tech Stack
                    </button>
                    <a href="/docs" target="_blank" class="px-4 py-2 bg-slate-800 hover:bg-slate-700 text-slate-200 text-sm font-medium rounded-lg border border-slate-700 transition duration-150 flex items-center gap-2">
                        💾 Interactive Swagger API Docs
                    </a>
                </div>
            </header>

            <div class="grid grid-cols-1 lg:grid-cols-12 gap-8">
                
                <div class="lg:col-span-4 space-y-6">
                    <div class="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-xl ring-1 ring-blue-500/10">
                        <h2 class="text-md font-semibold text-white mb-2 flex items-center gap-2">
                            🔑 Multi-Tenant Key Allocation
                        </h2>
                        <p class="text-slate-400 text-[11px] mb-3 leading-relaxed">
                            If the system environment key matches rate-limits or falls back, supply your private token credential below.
                        </p>
                        <div class="space-y-3">
                            <input id="custom-key-input" type="password" placeholder="Paste Gemini Key (AIzaSy...)" class="w-full bg-slate-950 border border-slate-800 focus:border-slate-700 rounded-lg px-3 py-2 text-xs font-mono text-slate-100 outline-none transition duration-150">
                            <div class="grid grid-cols-2 gap-2">
                                <button onclick="commitCustomKey()" class="w-full py-1.5 bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 rounded-md text-[11px] font-medium transition duration-150">
                                    💾 Attach Token
                                </button>
                                <button onclick="flushCustomKey()" class="w-full py-1.5 bg-rose-950/20 hover:bg-rose-950/40 text-rose-400 border border-rose-900/30 rounded-md text-[11px] font-medium transition duration-150">
                                    🗑️ Reset Default
                                </button>
                            </div>
                            <div id="key-status-indicator" class="text-[10px] text-slate-500 font-mono text-center pt-1">
                                Status: Utilizing System Environment Default Key
                            </div>
                        </div>
                    </div>

                    <div class="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-xl">
                        <h2 class="text-md font-semibold text-white mb-4 flex items-center gap-2">
                            ⚙️ Architecture Telemetry
                        </h2>
                        <div class="space-y-3 text-xs">
                            <div class="flex justify-between border-b border-slate-800/50 pb-2">
                                <span class="text-slate-400">Core Engine:</span>
                                <span class="font-mono text-slate-200 font-semibold">Gemini 3.1 Flash Lite</span>
                            </div>
                            <div class="flex justify-between border-b border-slate-800/50 pb-2">
                                <span class="text-slate-400">Vector Embeddings:</span>
                                <span class="font-mono text-slate-200">gemini-embedding-001</span>
                            </div>
                            <div class="flex justify-between border-b border-slate-800/50 pb-2">
                                <span class="text-slate-400">Retrieval Framework:</span>
                                <span class="text-blue-400 font-semibold">Rank-BM25 + RRF Matrix</span>
                            </div>
                            <div class="flex justify-between border-b border-slate-800/50 pb-2">
                                <span class="text-slate-400">Memory State:</span>
                                <span class="text-emerald-400 font-medium">Stateless (Client Anchored)</span>
                            </div>
                            <div class="flex justify-between pb-1">
                                <span class="text-slate-400">Indexed Catalog Pool:</span>
                                <span class="font-mono text-slate-200">377 Valid Entries</span>
                            </div>
                        </div>
                    </div>

                    <div class="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-xl">
                        <h2 class="text-md font-semibold text-white mb-3">🎯 Verified Compliance Probes</h2>
                        <p class="text-xs text-slate-400 mb-4">Pipeline behaviors mapped perfectly to structural assessment guardrails:</p>
                        <ul class="space-y-2.5 text-xs text-slate-300">
                            <li class="flex items-start gap-2">✔️ <span class="text-slate-300"><strong class="text-white">Strict Type Logic:</strong> Dynamic letter resolution (A,S, P,C)</span></li>
                            <li class="flex items-start gap-2">✔️ <span class="text-slate-300"><strong class="text-white">Gated shortlists:</strong> Clear separation on intent shifts</span></li>
                            <li class="flex items-start gap-2">✔️ <span class="text-slate-300"><strong class="text-white">Compliance Protection:</strong> Native blocker for legal requests</span></li>
                        </ul>
                    </div>
                </div>

                <div class="lg:col-span-8 space-y-8">
                    
                    <div class="flex border-b border-slate-800 gap-4">
                        <button id="tab-simulation-btn" onclick="switchTab('simulation')" class="pb-3 text-sm font-semibold border-b-2 border-blue-500 text-blue-400 transition-all duration-150">
                            🖥️ Automated Replay Simulation Log
                        </button>
                        <button id="tab-chat-btn" onclick="switchTab('chat')" class="pb-3 text-sm font-medium border-b-2 border-transparent text-slate-400 hover:text-slate-200 transition-all duration-150">
                            💬 Live Dynamic Chat Agent
                        </button>
                    </div>

                    <div id="panel-simulation" class="space-y-4">
                        <div class="flex items-center justify-between">
                            <p class="text-xs text-slate-400">Trigger multi-turn trace simulation lookups inside your browser container environment:</p>
                            <button onclick="executeValidationTrace()" id="run-trace-btn" class="px-4 py-2 bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-500 hover:to-indigo-500 text-white text-xs font-semibold rounded-lg shadow-lg shadow-indigo-600/10 transition duration-150 transform active:scale-95">
                                🚀 Execute Verification Trace Run
                            </button>
                        </div>
                        <div class="bg-slate-900 border border-slate-800 rounded-xl p-4 shadow-2xl font-mono text-xs overflow-hidden">
                            <div class="flex items-center justify-between border-b border-slate-800 pb-3 mb-3 text-slate-500">
                                <div class="flex items-center gap-1.5">
                                    <span class="w-3 h-3 bg-rose-500/20 rounded-full block border border-rose-500/40"></span>
                                    <span class="w-3 h-3 bg-amber-500/20 rounded-full block border border-amber-500/40"></span>
                                    <span class="w-3 h-3 bg-emerald-500/20 rounded-full block border border-emerald-500/40"></span>
                                </div>
                                <span>Bash Environment Session Terminal</span>
                            </div>
                            <div id="terminal-screen" class="h-96 overflow-y-auto space-y-3 pr-2 text-slate-300 leading-relaxed">
                                <div class="text-slate-500">[System Initialization Hook Ready] Status: Listening... Click 'Execute' to start test sequence payload loops.</div>
                            </div>
                        </div>
                    </div>

                    <div id="panel-chat" class="hidden space-y-4">
                        <div class="bg-slate-900 border border-slate-800 rounded-xl shadow-2xl flex flex-col h-[500px]">
                            <div id="chat-screen" class="flex-1 overflow-y-auto p-4 space-y-4 text-xs">
                                <div class="flex gap-3 bg-slate-800/30 p-3.5 border border-slate-800/60 rounded-xl">
                                    <span class="text-lg">🤖</span>
                                    <div>
                                        <div class="font-semibold text-slate-200 mb-1">SHL Agent Consultant</div>
                                        <p class="text-slate-300 leading-relaxed">Hello, I am your SHL Individual Test Solutions assistant. Could you tell me more about the role and target requirements you are looking to evaluate today?</p>
                                    </div>
                                </div>
                            </div>
                            
                            <div id="chat-loader" class="hidden px-6 py-2 flex items-center gap-2 text-xs text-slate-400">
                                <span class="w-2 h-2 bg-blue-400 rounded-full animate-ping"></span> Realtime model thinking...
                            </div>

                            <div class="border-t border-slate-800 p-3 bg-slate-900/50 rounded-b-xl flex gap-3">
                                <input id="chat-input" type="text" onkeydown="handleChatKey(event)" placeholder="Ask about assessments or input job constraints (e.g., 'Hiring mid-level developer')..." class="flex-1 bg-slate-950 border border-slate-800 focus:border-slate-700 rounded-lg px-4 py-2.5 text-xs text-slate-100 placeholder-slate-500 outline-none transition duration-150">
                                <button onclick="sendChatMessage()" class="px-5 py-2.5 bg-blue-600 hover:bg-blue-500 text-white font-semibold text-xs rounded-lg shadow-md transition duration-150">
                                    Send Payload
                                </button>
                            </div>
                        </div>
                    </div>

                </div>
            </header>
        </div>

        <div id="about-modal" class="hidden fixed inset-0 bg-slate-950/80 backdrop-blur-sm z-50 flex items-center justify-center p-4">
            <div class="bg-slate-900 border border-slate-800 rounded-xl max-w-2xl w-full p-6 shadow-2xl relative overflow-y-auto max-h-[90vh]">
                <button onclick="toggleAboutModal(false)" class="absolute top-4 right-4 text-slate-400 hover:text-white transition text-lg">&times;</button>
                
                <div class="flex items-center gap-4 border-b border-slate-800 pb-4 mb-4">
                    <div class="w-12 h-12 rounded-full bg-gradient-to-tr from-blue-500 to-indigo-600 flex items-center justify-center text-xl font-bold text-white shadow-md">
                        SD
                    </div>
                    <div>
                        <h3 class="text-xl font-bold text-white">Surya Prakash Dwivedi</h3>
                        <p class="text-xs text-blue-400 font-medium">Computer Applications Professional | Lucknow & Mandi, India</p>
                    </div>
                </div>
                
                <div class="space-y-4 text-xs text-slate-300">
                    <p class="leading-relaxed text-slate-400">
                        Passionate engineer specializing in Full-Stack Web Development, Artificial Intelligence, and Machine Learning architecture. Highly focused on enterprise RAG platforms, Large Language Models (LLMs), computer vision, and high-performance pipeline optimization.
                    </p>
                    
                    <div>
                        <h4 class="text-sm font-semibold text-white mb-2 flex items-center gap-1.5">🚀 Technical Competencies & Tooling</h4>
                        <div class="flex flex-wrap gap-1.5">
                            <span class="px-2 py-0.5 rounded bg-slate-800 border border-slate-700 text-slate-300">FastAPI</span>
                            <span class="px-2 py-0.5 rounded bg-slate-800 border border-slate-700 text-slate-300">Python</span>
                            <span class="px-2 py-0.5 rounded bg-slate-800 border border-slate-700 text-slate-300">Gemini Pro/Flash</span>
                            <span class="px-2 py-0.5 rounded bg-slate-800 border border-slate-700 text-slate-300">RAG Frameworks</span>
                            <span class="px-2 py-0.5 rounded bg-slate-800 border border-slate-700 text-slate-300">Dense Vector Semantics</span>
                            <span class="px-2 py-0.5 rounded bg-slate-800 border border-slate-700 text-slate-300">Okapi BM25 Lexical</span>
                            <span class="px-2 py-0.5 rounded bg-slate-800 border border-slate-700 text-slate-300">React</span>
                            <span class="px-2 py-0.5 rounded bg-slate-800 border border-slate-700 text-slate-300">Node.js</span>
                            <span class="px-2 py-0.5 rounded bg-slate-800 border border-slate-700 text-slate-300">Express</span>
                            <span class="px-2 py-0.5 rounded bg-slate-800 border border-slate-700 text-slate-300">Tailwind CSS</span>
                            <span class="px-2 py-0.5 rounded bg-slate-800 border border-slate-700 text-slate-300">OCI Cloud Compute</span>
                        </div>
                    </div>

                    <div>
                        <h4 class="text-sm font-semibold text-white mb-1.5 flex items-center gap-1.5">🎓 Professional Qualifications</h4>
                        <ul class="space-y-1.5 text-slate-400 list-disc list-inside">
                            <li><strong class="text-slate-200">IIT Mandi:</strong> Minor in Data Science and Machine Learning</li>
                            <li><strong class="text-slate-200">SRMCEM:</strong> Master of Computer Applications (MCA)</li>
                            <li><strong class="text-slate-200">University of Lucknow:</strong> Bachelor of Computer Applications (BCA)</li>
                        </ul>
                    </div>
                    
                    <div class="border-t border-slate-800 pt-4 mt-4 grid grid-cols-1 sm:grid-cols-3 gap-3">
                        <a href="https://spdwivedi.me" target="_blank" class="p-2.5 bg-slate-950/60 border border-slate-800 rounded-lg flex items-center justify-center gap-2 text-center font-medium hover:border-slate-700 hover:text-white transition">
                            🌐 Portfolio Website
                        </a>
                        <a href="https://github.com/spdwivedi/shl-assessment-agent" target="_blank" class="p-2.5 bg-slate-950/60 border border-slate-800 rounded-lg flex items-center justify-center gap-2 text-center font-medium hover:border-slate-700 hover:text-white transition">
                            💻 Project Repository
                        </a>
                        <a href="https://www.linkedin.com/in/spdwivedi2001/" target="_blank" class="p-2.5 bg-slate-950/60 border border-slate-800 rounded-lg flex items-center justify-center gap-2 text-center font-medium hover:border-slate-700 hover:text-white transition">
                            💼 LinkedIn Connect
                        </a>
                    </div>
                </div>
            </div>
        </div>

        <script>
            let activeMessages = [];

            // UI Initial State Synchronizer
            document.addEventListener("DOMContentLoaded", () => {
                const activeKey = sessionStorage.getItem("CUSTOM_GEMINI_KEY");
                const indicator = document.getElementById("key-status-indicator");
                const inputField = document.getElementById("custom-key-input");
                
                if (activeKey) {
                    indicator.innerText = "Status: Utilizing Active Custom Session Key 🔒";
                    indicator.className = "text-[10px] text-emerald-400 font-mono text-center pt-1 font-semibold";
                    inputField.value = activeKey;
                }
            });

            function commitCustomKey() {
                const keyVal = document.getElementById("custom-key-input").value.trim();
                const indicator = document.getElementById("key-status-indicator");
    
                // Updated validation: Accepts standard AIzaSy keys, new AQ. keys, and generic secure tokens
                if(!keyVal || (!keyVal.startsWith("AIzaSy") && !keyVal.startsWith("AQ.") && keyVal.length < 20)) {
                    alert("Please insert a valid, authenticated Gemini API Token String.");
                    return;
                }
                sessionStorage.setItem("CUSTOM_GEMINI_KEY", keyVal);
                indicator.innerText = "Status: Utilizing Active Custom Session Key 🔒";
                indicator.className = "text-[10px] text-emerald-400 font-mono text-center pt-1 font-semibold";
                alert("Custom Session Token successfully attached.");
            }

            function flushCustomKey() {
                const indicator = document.getElementById("key-status-indicator");
                document.getElementById("custom-key-input").value = "";
                sessionStorage.removeItem("CUSTOM_GEMINI_KEY");
                indicator.innerText = "Status: Utilizing System Environment Default Key";
                indicator.className = "text-[10px] text-slate-500 font-mono text-center pt-1";
                alert("Session context flushed. Reverted to Server Environment Defaults.");
            }

            function getActiveHeaders() {
                const headers = { 'Content-Type': 'application/json' };
                const customKey = sessionStorage.getItem("CUSTOM_GEMINI_KEY");
                if (customKey) {
                    headers['X-Gemini-API-Key'] = customKey;
                }
                return headers;
            }

            function toggleAboutModal(show) {
                const modal = document.getElementById('about-modal');
                if (show) modal.classList.remove('hidden');
                else modal.classList.add('hidden');
            }

            function switchTab(target) {
                const simBtn = document.getElementById('tab-simulation-btn');
                const chatBtn = document.getElementById('tab-chat-btn');
                const simPanel = document.getElementById('panel-simulation');
                const chatPanel = document.getElementById('panel-chat');

                if (target === 'simulation') {
                    simBtn.className = "pb-3 text-sm font-semibold border-b-2 border-blue-500 text-blue-400 transition-all duration-150";
                    chatBtn.className = "pb-3 text-sm font-medium border-b-2 border-transparent text-slate-400 hover:text-slate-200 transition-all duration-150";
                    simPanel.classList.remove('hidden');
                    chatPanel.classList.add('hidden');
                } else {
                    chatBtn.className = "pb-3 text-sm font-semibold border-b-2 border-blue-500 text-blue-400 transition-all duration-150";
                    simBtn.className = "pb-3 text-sm font-medium border-b-2 border-transparent text-slate-400 hover:text-slate-200 transition-all duration-150";
                    chatPanel.classList.remove('hidden');
                    simPanel.classList.add('hidden');
                }
            }

            function appendTerminalLine(text, type='info') {
                const screen = document.getElementById('terminal-screen');
                const wrapper = document.createElement('div');
                if (type === 'user') wrapper.className = "text-amber-400";
                else if (type === 'agent') wrapper.className = "text-emerald-400 mt-1";
                else if (type === 'recommendation') wrapper.className = "text-blue-300 pl-4 bg-slate-950/40 p-2 rounded border border-slate-800/40 font-semibold space-y-1 my-1";
                else wrapper.className = "text-slate-500 text-[11px] uppercase tracking-wider mt-4 border-b border-slate-800/40 pb-1";
                
                wrapper.innerHTML = text;
                screen.appendChild(wrapper);
                screen.scrollTop = screen.scrollHeight;
            }

            async function executeValidationTrace() {
                const btn = document.getElementById('run-trace-btn');
                btn.disabled = true;
                btn.innerText = "Processing Trace Replay Loop...";
                
                const screen = document.getElementById('terminal-screen');
                screen.innerHTML = "";
                
                const traceTurns = [
                    "I need an assessment solution for hiring a software engineer.",
                    "Mid-level, around 4 years of experience.",
                    "Actually, let's make sure it's a technical coding test, not behavioral patterns."
                ];
                
                let simulatedHistory = [];
                
                for(let i=0; i<traceTurns.length; i++) {
                    appendTerminalLine(`--- SIMULATION TRANSACTION EXECUTION TURN ${i+1} ---`, 'meta');
                    appendTerminalLine(`👤 USER PAYLOAD: "${traceTurns[i]}"`, 'user');
                    
                    simulatedHistory.push({"role": "user", "content": traceTurns[i]});
                    
                    try {
                        let response = await fetch('/chat', {
                            method: 'POST',
                            headers: getActiveHeaders(), // ◄── Appends client-supplied headers if present
                            body: JSON.stringify({ messages: simulatedHistory })
                        });
                        let data = await response.json();
                        
                        appendTerminalLine(`🤖 AGENT ASSISTANT RESPONSE:<br>${data.reply}`, 'agent');
                        
                        if(data.recommendations && data.recommendations.length > 0) {
                            let recHtml = "📋 Shortlist Payload Mapped:<br>";
                            data.recommendations.forEach(r => {
                                recHtml += `▪️ [${r.test_type}] <a href="${r.url}" target="_blank" class="underline text-blue-400 hover:text-blue-300">${r.name}</a><br>`;
                            });
                            appendTerminalLine(recHtml, 'recommendation');
                        } else {
                            appendTerminalLine(`📋 Shortlist Payload Mapped: []`, 'agent');
                        }
                        
                        appendTerminalLine(`🛑 End Conversation Flag: ${data.end_of_conversation}`, 'agent');
                        simulatedHistory.push({"role": "assistant", "content": data.reply});
                        
                    } catch (err) {
                        appendTerminalLine(`❌ API Loop Connection Clashed: ${err}`, 'user');
                    }
                    
                    await new Promise(r => setTimeout(r, 1200));
                }
                
                btn.disabled = false;
                btn.innerText = "Execute Verification Trace Run";
            }

            function handleChatKey(e) {
                if(e.key === 'Enter') sendChatMessage();
            }

            async function sendChatMessage() {
                const input = document.getElementById('chat-input');
                const txt = input.value.trim();
                if(!txt) return;
                
                input.value = "";
                
                const chatScreen = document.getElementById('chat-screen');
                const loader = document.getElementById('chat-loader');
                
                const userDiv = document.createElement('div');
                userDiv.className = "flex gap-3 bg-slate-900 p-3.5 border border-slate-800/40 rounded-xl justify-end text-right";
                userDiv.innerHTML = `<div><div class="font-semibold text-amber-400 mb-1">You (Recruiter)</div><p class="text-slate-300">${txt}</p></div><span class="text-lg">👤</span>`;
                chatScreen.appendChild(userDiv);
                chatScreen.scrollTop = chatScreen.scrollHeight;
                
                activeMessages.push({"role": "user", "content": txt});
                loader.classList.remove('hidden');
                
                try {
                    let response = await fetch('/chat', {
                        method: 'POST',
                        headers: getActiveHeaders(), // ◄── Appends client-supplied headers if present
                        body: JSON.stringify({ messages: activeMessages })
                    });
                    let data = await response.json();
                    
                    loader.classList.add('hidden');
                    
                    const agentDiv = document.createElement('div');
                    agentDiv.className = "flex gap-3 bg-slate-800/20 p-3.5 border border-slate-800/60 rounded-xl";
                    
                    let innerContent = `
                        <span class="text-lg">🤖</span>
                        <div class="w-full">
                            <div class="font-semibold text-slate-200 mb-1">SHL Agent Consultant</div>
                            <p class="text-slate-300 leading-relaxed">${data.reply}</p>
                    `;
                    
                    if(data.recommendations && data.recommendations.length > 0) {
                        innerContent += `<div class="mt-3 grid grid-cols-1 md:grid-cols-2 gap-2 pt-2 border-t border-slate-800/60">`;
                        data.recommendations.forEach(r => {
                            innerContent += `
                                <div class="p-2 bg-slate-950/60 border border-slate-800 rounded flex flex-col justify-between">
                                    <div class="font-medium text-slate-200">${r.name}</div>
                                    <div class="flex items-center justify-between mt-1 text-[10px]">
                                        <span class="px-1.5 py-0.5 rounded bg-blue-500/10 text-blue-400 border border-blue-500/20">Type: ${r.test_type}</span>
                                        <a href="${r.url}" target="_blank" class="text-indigo-400 hover:underline font-semibold">Catalog Link 🔗</a>
                                    </div>
                                </div>
                            `;
                        });
                        innerContent += `</div>`;
                    }
                    
                    innerContent += `</div>`;
                    agentDiv.innerHTML = innerContent;
                    chatScreen.appendChild(agentDiv);
                    activeMessages.push({"role": "assistant", "content": data.reply});
                    
                } catch(err) {
                    loader.classList.add('hidden');
                    const errDiv = document.createElement('div');
                    errDiv.className = "text-rose-400 font-mono text-center py-2";
                    errDiv.innerText = `Network Payload Processing Error: ${err}`;
                    chatScreen.appendChild(errDiv);
                }
                
                chatScreen.scrollTop = chatScreen.scrollHeight;
            }
        </script>
    </body>
    </html>
    """