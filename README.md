# 🧠 AI-Powered Code Generation and Enhancement System

A local AI assistant designed to streamline software development through automatic code generation, debugging, and testing using open-source large language models (LLMs) like Meta’s LLaMA 3, DeepSeek, and Mistral.

---

## 🚀 Project Overview

This tool acts as a personal software engineer assistant that:

- Generates, modifies, and tests code using natural language instructions.
- Fine-tunes open-source LLMs on company-specific codebases.
- Automates test case generation and execution.
- Maintains code quality and aligns with company conventions.
- Ensures local, secure, and private AI integration.

---

## 🧩 Key Features

- ✅ Natural language → Production-ready code
- 🔄 Iterative code enhancement with user feedback
- 🧪 Auto test plan generation and execution
- 🔧 AI-assisted debugging
- 📦 Runs locally using quantized LLMs (Q4/Q8 support)
- 🔐 Privacy-first: Proprietary code never leaves your machine
- 🧠 Fine-tuning support with LoRA / QLoRA
- 🔍 RAG-enabled code suggestion using vector DBs like FAISS or ChromaDB

---

## 📂 Project Structure


---

## 🛠 Tech Stack

- 🧠 LLMs: LLaMA 3, DeepSeek, Mistral
- 🔍 Retrieval: FAISS, ChromaDB
- 🧪 Testing: PyTest / JUnit / Selenium
- ⚙️ Containerization: Docker / Firecracker
- 🚀 Backend: Python
- 💻 Local Deployment: CPU/GPU friendly (quantized model support)

---

## 📌 Setup Instructions

> ⚠️ Clone the repo without downloading heavy model files (already ignored in `.gitignore`)

```bash
git clone https://github.com/Vijayhm/ai-codegen-assistant.git
cd ai-codegen-assistant
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

