from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage, AIMessage
from langgraph.prebuilt import ToolNode, tools_condition
from pydantic import BaseModel, Field
from typing import TypedDict, Literal, Annotated, Optional
from langgraph.checkpoint.memory import MemorySaver
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
import operator
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from dotenv import load_dotenv
load_dotenv()


system_template = """You are AutoStream's conversational AI agent.

Your role is to engage users naturally while performing three core tasks:
1. Understand and classify user intent (greeting, product inquiry, or high-intent lead).
2. Answer product and pricing questions strictly using the provided knowledge base.
3. When a user shows clear intent to sign up, professionally collect their name, email, and creator platform, asking for only one missing detail at a time.

Behavior rules:
- Be concise, friendly, and professional.
- Do not hallucinate features, pricing, or policies.
- Do not ask multiple questions in a single message.
- Do not execute any lead-capture action until all required details are collected.
- If information is missing, ask only for the next required detail.
- Each response should move the conversation forward naturally.

Assume you are part of a stateful chatbot and respond based only on the current user message and conversation context.
"""


llm = ChatOpenAI(model="gpt-4o-mini")

# Schemas
class IntentSchema(BaseModel):
    intent: Literal['greeting', 'inquiry', 'high_intent'] = Field(description="The intent of the User's query")
    user_is_lead: bool = Field(description="Return true if the user is high-intent, false otherwise. Make sure to return true when the user is asked to give his credentials")


class LeadCheck(BaseModel):
    all_vals_parsed: Literal['true', 'false'] = Field(description="Return true if the values name, email and platform are present, and return False otherwise")

class ParseLead(BaseModel):
    name: str = Field(description="Name of the user")
    email: str = Field(description="Email of the user")
    platform: str = Field(description="Desired Platform of the user")

sys_prompt = ChatPromptTemplate(
    messages=[
        ('system', system_template),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

llm_chain = sys_prompt | llm
lead_check_llm = llm.with_structured_output(LeadCheck)
intent_llm = llm.with_structured_output(IntentSchema)
parse_lead_llm = llm.with_structured_output(ParseLead)

# Graph states
class LeadData(BaseModel):
    name: Optional[str] = None
    email: Optional[str] = None
    platform: Optional[str] = None

class UserState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    intent: Annotated[Literal['greeting', 'inquiry', 'high_intent'], operator.add]
    lead_data: LeadData
    lead_status: Optional[str]    
    user_is_lead: bool


# RAG
loader = UnstructuredMarkdownLoader(file_path="data.md")
docs = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)

chunks = splitter.split_documents(docs)

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

vectorstore = FAISS.from_documents(chunks, embedding=embeddings)

retriever = vectorstore.as_retriever(search_type='similarity', search_kwargs={'k':4})

@tool
def rag_tool(query):
    """Retrieve relevant information from the company document
    Use this tool whenever the user asks anything related to the company or pricing questions
    that might be answered from the stored document."""

    result = retriever.invoke(query)

    context = [doc.page_content for doc in result]

    return {
        'query':query,
        'context':context
    }

tools = [rag_tool]
llm_with_tool = llm.bind_tools(tools)

# Routers
def route_intent(state: UserState):
    if state["intent"][-1] == "greeting":
        return "greet"
    elif state["intent"][-1] == "inquiry":
        return "inquiry"
    else:
        return "lead"

def route_lead(state: UserState):
    if state['lead_status'] == "true":
        return "valid"
    else:
        return "invalid"
    

# Graph Nodes
def classify_intent(state: UserState):
    last_message = state["messages"][-1].content
    response = intent_llm.invoke(
        f"Classify the intent of this message:{last_message}"
    )
    return {"intent": [response.intent]}
    

def handle_greeting(state: UserState):
    response = llm_chain.invoke({"messages": state["messages"]})
    return {"messages": [response]}

rag_node = ToolNode(tools)


def handle_inquiry(state: UserState):
    """Handle product inquiries using RAG tool"""
    response = llm_with_tool.invoke(state["messages"])
    return {"messages": [response]}


def handle_lead(state: UserState):
    messages = "\n".join(msg.content for msg in state['messages'])
    """Ask for the next missing piece of lead information OR save user's response"""
    response = llm.invoke(f"Ask the user for whatever field is not there in the given message history out of name, email and platform \n\n message_history: {messages}")
    return {"messages": [response], "user_is_lead": True}

def parse_lead(state: UserState):
    messages = "\n".join(msg.content for msg in state['messages'])
    prompt = f"Parse the name, email, and the platform from the given message history: {messages}"
    lead_data = parse_lead_llm.invoke(prompt)

    lead_check_prompt = f"Check if the given values are valid or not \n name: {lead_data.name}, \n email: {lead_data.email}, \n platform: {lead_data.platform}"

    response = lead_check_llm.invoke(lead_check_prompt)

    return {"lead_data": lead_data, "lead_status": response.all_vals_parsed}

def mock_lead_capture(state: UserState):
    lead = state["lead_data"]
    print(
        f"Lead captured successfully: {lead.name}, {lead.email}, {lead.platform}"
    )
    return {
        "messages": [
            AIMessage(
                content="🎉 Your details have been captured! Our team will reach out shortly."
            )
        ]
    }


graph = StateGraph(UserState)

checkpoint = MemorySaver()
graph.add_node("Classify Intent", classify_intent)
graph.add_node("Greeting", handle_greeting)
graph.add_node("Handle Lead", handle_lead)
graph.add_node("Parse Lead", parse_lead)
graph.add_node("Inquiry", handle_inquiry)
graph.add_node("tools", rag_node)
graph.add_node("Capture Lead", mock_lead_capture)



graph.add_edge(START, "Classify Intent")

graph.add_conditional_edges(
    "Classify Intent",
    route_intent,
    {
        "greet": "Greeting",
        "inquiry": "Inquiry",
        "lead": "Handle Lead",
    },
)

graph.add_edge("Greeting", END)


graph.add_conditional_edges("Inquiry", tools_condition)
graph.add_edge("tools", "Inquiry")
graph.add_edge("Inquiry", END)

graph.add_edge("Handle Lead", "Parse Lead")
graph.add_conditional_edges("Parse Lead",route_lead, {"valid": "Capture Lead", "invalid": END})
graph.add_edge("Capture Lead", END)

chatbot = graph.compile(checkpointer=checkpoint)