# AutoStream AI Agent - ServiceHive Assignment

## Project Overview
This project implements a Conversational AI Agent for **AutoStream**, a SaaS platform for automated video editing. The agent is designed to understand user intent, answer product questions using RAG (Retrieval-Augmented Generation), and identify high-intent leads to capture their details for the sales team.

## Features
* **Intent Recognition:** Classifies users into "Casual Greeting", "Product Inquiry", or "High-Intent Lead".
* **RAG-Powered Q&A:** Answers queries about pricing (Basic/Pro plans) and policies using a local knowledge base.
* **Lead Capture:** intelligently collects user details (Name, Email, Platform) when high intent is detected.
* **State Management:** Retains context across conversation turns using LangGraph.

---

## 🚀 How to Run Locally

### Prerequisites
* Python 3.9+
* OpenAI API Key

### Installation Steps

1.  **Clone the Repository**
    ```bash
    git clone <repository_url>
    cd servicehive_assignment
    ```

2.  **Create a Virtual Environment**
    ```bash
    python -m venv .venv
    # Windows
    .venv\Scripts\activate
    # Mac/Linux
    source .venv/bin/activate
    ```

3.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up Environment Variables**
    Create a `.env` file in the root directory and add your OpenAI API key:
    ```env
    OPENAI_API_KEY=your_sk_key_here
    ```

5.  **Run the Application**
    Start the Streamlit interface:
    ```bash
    streamlit run app.py
    ```
    The application will open in your browser at `http://localhost:8501`.

---

## 🏗️ Architecture Explanation

### Choice of Framework: LangGraph
I chose **LangGraph** over AutoGen or standard LangChain chains because this agent requires a cyclic, stateful workflow rather than a linear DAG. The conversation flow involves loops (e.g., repeatedly asking for missing lead details) and conditional branching (routing based on intent), which LangGraph handles natively through its graph-based architecture.

### State Management
State is managed using a strictly typed `UserState` dictionary that persists across the conversation session.
* **Shared State:** The graph passes a `UserState` object containing the message history (`messages`), current classification (`intent`), and lead capture progress (`lead_data`).
* **Persistence:** I utilized `MemorySaver` to checkpoint the state. This allows the bot to "remember" previous turns (like the user's name provided 3 messages ago) by passing a `thread_id` with every configuration.
* **Routing:** The `Classify Intent` node acts as the central router, directing the flow to specific sub-graphs (Greeting, Inquiry, or Lead Handling) based on the latest user input, ensuring separation of concerns.

---

## 📂 Project Structure
* `app.py`: Streamlit frontend for chatting with the agent.
* `chatbot.py`: Core logic containing the LangGraph definition, RAG setup, and tool nodes.
* `data.md`: Knowledge base containing pricing and policy information.
* `requirements.txt`: Project dependencies.
