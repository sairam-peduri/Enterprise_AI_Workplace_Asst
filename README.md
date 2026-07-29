# Enterprise AI Workplace Assistant

Enterprise AI Workplace Assistant is a multi-agent AI application designed to help employees with common workplace tasks through a single conversational interface.

Instead of navigating through different systems for HR, Finance, IT, Travel, and company policies, employees can interact with the assistant using natural language. The application identifies the type of request and routes it to the appropriate specialized agent.

The project also includes proactive assistance, allowing the system to surface useful notifications and recommendations based on employee-related events.

---

## Features

### Multi-Agent Assistance

The application contains specialized agents for different workplace domains:

- HR Agent
- Finance Agent
- IT Support Agent
- Travel Agent
- Knowledge Agent

A supervisor routes each employee request to the appropriate agent.

### Finance Support

Employees can:

- Submit expense claims
- Confirm expenses before final submission
- Check reimbursement status
- Track expense-related information

Expense data is validated before being stored.

### HR Support

The HR module supports employee-related queries and actions such as:

- Checking leave information
- Viewing leave balance
- Handling leave-related requests
- Accessing employee information

### IT Support

The IT agent helps employees with common workplace IT issues such as:

- Password-related problems
- Account issues
- Laptop and system problems
- IT support requests

### Travel Support

The Travel agent handles workplace travel-related requests and queries.

### Knowledge Assistant

The Knowledge Agent uses Retrieval-Augmented Generation (RAG) to answer questions using company documents and policies.

The knowledge base includes documents such as:

- Company Policies
- Employee Handbook
- Remote Work Policy
- Security Guidelines

ChromaDB is used as the vector database for document retrieval.

### Proactive Assistance

The application also contains a proactive assistance layer that can identify relevant employee events and generate useful recommendations or notifications.

Examples include:

- Leave request updates
- Pending approvals
- Important workplace reminders
- Context-aware recommendations

---

## System Architecture

The application follows a multi-agent architecture.

```text
                        User
                          │
                          ▼
                 Streamlit Interface
                          │
                          ▼
                  LangGraph Workflow
                          │
                          ▼
                     Supervisor
                          │
          ┌───────────────┼───────────────┐
          │               │               │
          ▼               ▼               ▼
       HR Agent       Finance Agent    IT Agent
          │               │               │
          └───────────┬───┴───────────────┘
                      │
               ┌──────┴──────┐
               ▼             ▼
          Travel Agent   Knowledge Agent
                              │
                              ▼
                         RAG / ChromaDB
```

Each agent has access only to the tools required for its domain.

---

## Technology Stack

**Frontend**
- Streamlit

**Backend**
- FastAPI
- Python

**AI and Agent Framework**
- LangChain
- LangGraph
- Ollama
- Llama 3.2

**Knowledge Retrieval**
- Retrieval-Augmented Generation (RAG)
- ChromaDB

**Data Storage**
- JSON-based enterprise mock data

**Testing**
- Pytest

---

## Project Structure

```text
Enterprise_AI_Workplace_Asst/
│
├── data/
│   ├── chroma_db/
│   ├── finance/
│   │   └── expenses.json
│   ├── hr/
│   │   ├── employees.json
│   │   ├── holidays.json
│   │   └── leave.json
│   ├── it/
│   ├── policies/
│   ├── travel/
│   ├── Company_Policies.pdf
│   ├── Employee_Handbook.pdf
│   ├── Remote_Work_Policy.pdf
│   ├── Security_Guidelines.pdf
│   ├── expenses.json
│   └── sessions.json
│
├── logs/
│   └── enterprise_ai.log
│
├── src/
│   │
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── finance_agent.py
│   │   ├── hr_agent.py
│   │   ├── it_agent.py
│   │   ├── knowledge_agent.py
│   │   └── travel_agent.py
│   │
│   ├── auth/
│   │
│   ├── graph/
│   │   ├── __init__.py
│   │   ├── supervisor.py
│   │   ├── tool_dispatch.py
│   │   └── workflow.py
│   │
│   ├── proactive/
│   │   ├── approval_manager.py
│   │   ├── context_manager.py
│   │   ├── event_collector.py
│   │   ├── event_correlator.py
│   │   ├── event_models.py
│   │   ├── memory_manager.py
│   │   ├── notification_manager.py
│   │   ├── priority_engine.py
│   │   ├── proactive_engine.py
│   │   └── recommendation_engine.py
│   │
│   ├── prompts/
│   │   ├── __init__.py
│   │   ├── finance_prompt.py
│   │   ├── hr_prompt.py
│   │   ├── it_prompt.py
│   │   └── travel_prompt.py
│   │
│   ├── state/
│   │   ├── __init__.py
│   │   └── state.py
│   │
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── finance_tools.py
│   │   ├── hr_tools.py
│   │   ├── it_tools.py
│   │   ├── knowledge_tools.py
│   │   └── travel_tools.py
│   │
│   ├── ui/
│   │   └── __init__.py
│   │
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── file_handler.py
│   │   ├── finance_confirmation.py
│   │   ├── logging_config.py
│   │   ├── models.py
│   │   ├── paths.py
│   │   └── session_persistence.py
│   │
│   └── __init__.py
│
├── tests/
│   ├── test_finance_agent.py
│   ├── test_finance_confirmation.py
│   ├── test_finance_submission.py
│   ├── test_finance_workflow.py
│   ├── test_finance.py
│   ├── test_hr_agent.py
│   ├── test_hr_tools.py
│   ├── test_it_agent.py
│   ├── test_it_tools.py
│   ├── test_proactive.py
│   ├── test_RAG.py
│   ├── test_workflow.py
│   ├── travel_test_agent.py
│   └── travel_test_tools.py
│
├── .gitignore
├── app.py
├── doc
├── fastapi_server.py
├── pytest.ini
├── README.md
└── requirements.txt
```

---

## How the Application Works

When an employee sends a message, the request first enters the LangGraph workflow.

The supervisor analyzes the request and determines which agent should handle it.

For example:

```text
"I want to submit an expense"
        ↓
Supervisor
        ↓
Finance Agent
        ↓
Finance Tools
        ↓
Expense preparation
        ↓
User confirmation
        ↓
Expense submission
```

Similarly:

```text
"I forgot my password"
        ↓
Supervisor
        ↓
IT Agent
        ↓
IT Tools
```

For questions related to company policies:

```text
"What is the company's remote work policy?"
        ↓
Supervisor
        ↓
Knowledge Agent
        ↓
RAG Retrieval
        ↓
Relevant Company Documents
        ↓
Generated Answer
```

---

## Finance Expense Workflow

Expense submission follows a confirmation-based workflow to prevent accidental submissions.

The Finance Agent first collects the required details:

- Employee ID
- Expense amount
- Category

The expense is then prepared and shown to the employee for confirmation.

Only after confirmation is the expense submitted and stored.

Example:

```text
Employee:
I want to submit an expense.

Assistant:
Please provide the amount and category

Employee:
Amount: ₹2450
Category: Local Travel

Assistant:
Expense Summary
...
Please confirm the submission.
```
yes or no
---

## Proactive Assistance

The proactive module allows the application to go beyond responding only when the user asks a question.

It contains components responsible for:

```text
Event Collection
      ↓
Event Correlation
      ↓
Context Analysis
      ↓
Priority Evaluation
      ↓
Recommendation Generation
      ↓
Notification
```

This makes it possible to provide contextual workplace recommendations and notifications.

---

## Running the Project

### 1. Clone the Repository

```bash
git clone <repository-url>
cd Enterprise_AI_Workplace_Asst
```

### 2. Create a Virtual Environment

```bash
py -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
py -m pip install -r requirements.txt
```

### 4. Start Ollama

Make sure Ollama is installed and running.

Pull the required model if necessary:

```bash
ollama pull llama3.2
```

### 5. Start the FastAPI Backend

```bash
py -m uvicorn fastapi_server:app --reload --port 8000
```

The backend will run on port `8000`.

### 6. Start the Streamlit Application

Open another terminal and run:

```bash
py -m streamlit run app.py
```

The Streamlit application will normally open at:

```text
localhost:8501
```

## Login Credentials

After launching the application, log in using the appropriate credentials based on the selected mode.

### Employee Mode
- Use a valid Employee ID and its corresponding password.
- The default password is the lowercase version of the Employee ID.
- Example: `EMP002` → `emp002`

### HR Mode
- Use only authorized HR employee credentials.
- The default password is the lowercase version of the HR Employee ID.
- Example: `EMP001` → `emp001`

### General Mode
- General Mode provides access to all general EnterpriseAssist AI features without requiring employee authentication.
- Users can interact with the AI assistant, access enterprise knowledge, and use available modules.

> **Note:** Proactive recommendations for a specific employee are available only after logging in as an employee or HR user, as they require personalized employee context. Default passwords are provided for demonstration purposes only.

---

## Testing

The project includes a pytest-based test suite covering the major components of the system.

Tests are available for:

- Finance Agent
- Expense preparation and submission
- Finance confirmation workflow
- HR Agent and tools
- IT Agent and tools
- Travel Agent and tools
- LangGraph workflow
- Proactive assistance
- RAG-based knowledge retrieval

Run all tests using:

```bash
pytest
```

For more detailed output:

```bash
pytest -v
```

---

## Example Queries

Here are a few requests that can be used to test the assistant:

```text
I want to submit an expense.

Check the reimbursement status of my expense.

How many leaves do I have remaining?

I forgot my password.

My account is locked.

I need help with a travel request.

What is the company's remote work policy?

What are the company's security guidelines?
```

---

## Key Design Principles

The application was designed around a few important principles:

**Separation of responsibilities**  
Each agent handles a specific workplace domain.

**Tool-based enterprise access**  
Agents use dedicated tools instead of directly modifying enterprise data.

**Confirmation before important actions**  
State-changing operations such as expense submission require confirmation.

**Context-aware assistance**  
The proactive layer uses employee context and events to provide relevant recommendations.

**Modular architecture**  
Agents, prompts, tools, workflow logic, utilities, and tests are separated into independent modules, making the project easier to maintain and extend.

---

## Future Improvements

Possible extensions to the project include:

- Database integration instead of JSON storage
- Enterprise SSO authentication
- Role-based access control
- Email and workplace notification integration
- Improved employee personalization
- More proactive workplace recommendations
- Additional enterprise agents
- Cloud deployment
- Production-grade monitoring and logging

---

## Conclusion

Enterprise AI Workplace Assistant demonstrates how a multi-agent AI system can bring different workplace services into one conversational interface.

By combining LangGraph-based agent orchestration, specialized enterprise tools, RAG-based policy retrieval, confirmation workflows, proactive recommendations, and a Streamlit interface, the project provides a foundation for building an intelligent workplace assistant that can both respond to employee requests and proactively surface relevant information.

---

## Output


<img width="1916" height="888" alt="Screenshot 2026-07-29 132636" src="https://github.com/user-attachments/assets/faf18a58-bfe0-4130-a5c7-0095eea3e17a" />


<img width="1912" height="894" alt="Screenshot 2026-07-29 132737" src="https://github.com/user-attachments/assets/7d2317e3-3b1a-4d0f-a00f-b15a83dc0a91" />

## Pro active recommendations

<img width="1912" height="894" alt="Screenshot 2026-07-29 132737" src="https://github.com/user-attachments/assets/5d02b635-d0f9-461a-9fc7-ed28aacb4c79" />


<img width="1884" height="885" alt="Screenshot 2026-07-29 132834" src="https://github.com/user-attachments/assets/62a79618-f9bf-492a-bf3e-c2b5de163d1a" />

---

## Live Link

https://udmsh7hhcsuzk9tz7gkj3o.streamlit.app/




