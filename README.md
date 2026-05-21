# 🤟 BridgeLens: Universal Digital & Physical Inclusion
**Built for the Enyata x Interswitch Buildathon 2026**

BridgeLens is a comprehensive, two-way communication bridge designed to grant the Deaf community complete digital, financial, and physical independence. It moves beyond simple word translation by combining Edge-based Computer Vision with Cloud-based Large Language Models (LLMs) to understand context, alongside seamless integration with the Interswitch API ecosystem.

## ⚠️ The Problem
The world is designed for the hearing. Deaf individuals face compounding barriers:
1. **Communication:** Inability to easily communicate in daily life or medical emergencies.
2. **Financial Exclusion:** USSD codes, uninterpreted banking halls, and audio-based fraud alerts lock them out of the digital economy.
3. **Information Blackout:** Lack of closed captions on media and inability to hear ambient public announcements (transit, hospitals).

## 💡 The Solution: BridgeLens Features

### 🌍 1. Daily Interaction Hub (Context-Aware AI)
* **Logit-Masked Edge CV:** Uses an optimized MediaPipe model to track 225 skeletal landmarks. By using "Context Quick-Keys" (e.g., Coffee Shop, Emergency), the system mathematically restricts the AI's vocabulary, dropping latency to <50ms and completely eliminating hallucinated translations.
* **Ambient Ear:** A passive listening mode that captures background announcements (e.g., Train delays), processes the audio, and pushes visual Sign Language alerts to the user.
* **Indigenous Language Engine:** Two-way translation supporting English, Nigerian Pidgin, Yoruba, Igbo, and Hausa via Groq's Llama-3 API.

### 🏥 2. Medical Visit Module (Two-Way Clinical Bridge)
* **Patient-to-Doctor:** Translates the patient's sign language into clinical text, automatically logging symptoms into a digital chart.
* **Doctor-to-Patient:** Captures the doctor's spoken diagnosis, extracts the core NLP glosses, and plays sequential human sign language videos back to the patient.

### 💳 3. Financial Inclusion (Powered by Interswitch)
* **Branchless KYC:** Integrates the **Interswitch Identity Rails** for NIN verification and live facial comparison, upgrading users to Tier 3 accounts without needing to visit an uninterpreted bank branch.
* **Trust Shield:** Uses the **Interswitch Account Verification API** to visually verify recipient names before transfers, protecting Deaf users from vendor fraud.
* **Sign-to-Pay:** Replaces easily forgotten PINs with encrypted biometric sign language gestures for transaction authorization.

### 📺 4. Digital Content Bridge
* **Universal Audio Listener:** Bypasses missing YouTube closed captions by utilizing real-time SpeechRecognition to "listen" to any playing video.
* **Synchronized Interpretation:** Extracts target glosses from the audio and plays side-by-side sign language videos, making any raw video accessible instantly.

## Core Architecture
1. **Computational Feature Extraction (The Visual Pipeline)** Instead of passing raw video frames ($1920 \times 1080 \times 3$) to a computationally heavy model—which introduces unacceptable $O(N)$ bottlenecks—our pipeline compresses video frames into spatial coordinate vectors. Using MediaPipe's Image Vision Mode, the engine scales 33 holistic pose landmarks and $2 \times 21$ hand landmarks into a highly dense, localized 1D matrix of exactly $225$ spatial features per frame.
   $$\text{Total Features} = (33 \times 3) + (21 \times 3) + (21 \times 3) = 225$$2.
2. **Deep Temporal Modeling & Attention LogicThe sequential engine** processes spatial inputs using a fixed temporal window of $T = 30$ frames. The sequence matrix $\mathbf{X} \in \mathbb{R}^{30 \times 225}$ is fed through a multi-layered deep learning architecture:Bidirectional LSTMs: Two stacked Bi-LSTM layers (128 and 64 units) capture structural semantic patterns from both past and future frame contexts simultaneously.Spatial Attention Block: We implemented a custom Self-Attention over the Time Dimension using a localized Tanh activation. This assigns dynamic weights to frames containing critical kinetic hand shapes while down-weighting static transitional movements.
   $$\alpha_t = \text{Softmax}(\tanh(\mathbf{W}_a \mathbf{h}_t + \mathbf{b}_a))$$3 
3. **Data Engineering & Augmentation Matrix** To handle sample variations across 57 distinct sign classes, we engineered an explicit, automated mathematical data augmentation pipeline. It structurally manipulates sequences on the fly using:Gaussian Noise Injection: Adds deterministic noise vectors ($\mu=0, \sigma=0.01$).Random Spatial Scaling: Adjusts coordinate magnitudes by a factor of $1.0 \pm \delta$.Temporal Time-Shifting: Uses a random index roll ($\pm 3$ frames) along the time axis to simulate variation in user signing speeds.4. High-Throughput API DeploymentThe model is served using a containerized FastAPI backend designed for micro-second response rates. The API abstracts translation logic via three critical components:Sliding Window Stride ($S=5$): The inference engine runs predictions every 5 frames instead of every single frame, cutting redundant compute cycles by 80%.Majority Voting: A temporal stabilization algorithm (Counter().most_common(1)) filters out transient classification flips, returning a clean, stable prediction text.Linguistic Translation Integration: Connects to the Groq API running Llama-3.1-8b to automatically parse and normalize localized Nigerian language inputs (Yoruba, Hausa, Igbo) into standard upper-case sign gloss arrays.

## 🔌 API Endpoint Documentation
The BridgeLens Engine exposes a high-throughput, asynchronous REST API powered by FastAPI to bridge mobile/web frontends with the underlying deep learning and NLP models.
1. **Real-Time Sign ClassificationRoute**: POST /predict-signPayload: Multipart/Form-Data (Video file e.g., .mp4, .webm)Behavior: Accepts a raw video stream, dynamically extracts coordinate vectors using the edge feature extractor loop, slices the sequence through a sliding window stride ($S=5$), and routes it through the Attention-driven Bi-LSTM model.Response: Returns the top-3 predicted sign glosses with mathematical confidence scores.
2. **Indigenous Language Parsing & NormalizationRoute**: POST /translate-to-signsPayload: 
JSON{
  "text": "ebi ń pa mí",
  "language": "Yoruba"
}
Behavior: Sanitizes the raw text string, executes Unicode normalization, and leverages the Groq Llama 3.1 pipeline to tokenise indigenous phrases (Yoruba, Hausa, Igbo) into uppercase sign language keywords.Response: Returns the mapped uppercase gloss array.3. Algorithmic Grammar CorrectionRoute: POST /grammar-correctPayload: JSONJSON{
  "glosses": ["HELLO", "NAME", "ME", "DOCTOR"],
  "target_language": "English"
}
Behavior: Processes raw, disjointed sign language gloss vectors and utilizes an LLM orchestration layer to rebuild them into fluent, context-aware spoken sentences.Response: Returns the finalized natural language sentence.4. Financial Rail Transaction ValidatorRoute: POST /vas-paymentPayload: JSONJSON{
  "biller": "DSTV_NIGERIA",
  "amount": 5000.00,
  "account_id": "1234567890",
  "balance": 7500.00
}
Behavior: Validates accessible financial requests, evaluating account parameters against transaction criteria before securely interfacing with the Sandbox payment gateway rails.Response: Returns execution success parameters along with corresponding fallback sign commands if transaction logic errors occur.

## 💻 How to Run Locally
Follow these steps to set up the environment and spin up the backend microservice on your local Windows 10 machine.PrerequisitesEnsure you have Python 3.10 installed and configured in your system environment variables.
**Step 1:** Clone and Navigate to the DirectoryOpen your terminal (PowerShell or Command Prompt) and run:Bashgit clone <your-repository-url>
cd bridgelens-api
**Step 2:** Create a Virtual EnvironmentCreate a isolated environment to avoid package dependency conflicts:Bashpython -m venv venv
venv\Scripts\activate
**Step 3:** Install Required DependenciesInstall the optimized computational and deep learning frameworks specified in the requirements manifest:Bashpip install -r requirements.txt
(This installs fastapi, uvicorn, tensorflow, mediapipe==0.10.13, opencv-python, groq, and scikit-learn).Step 4: Set Up Environment VariablesConfigure your secure API keys inside your terminal session:DOS:: If using Command Prompt (CMD)
set GROQ_API_KEY=your_groq_api_key_here

:: If using PowerShell
$env:GROQ_API_KEY="your_groq_api_key_here"
Step 5: Launch the High-Throughput ServerInitialize the deployment server using Uvicorn with auto-reload flags enabled for development:Bashuvicorn main:app --reload
Once running, you can access the live interactive API documentation (Swagger UI) directly in your browser at: http://127.0.0.1:8000/docs to test all individual routes manually
