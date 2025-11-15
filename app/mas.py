from typing import TypedDict, Annotated, List, Dict, Any
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import Tool
import requests
from bs4 import BeautifulSoup
import os
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(
    model="Qwen/Qwen3-Next-80B-A3B-Instruct",
    base_url="https://foundation-models.api.cloud.ru/v1",
    api_key=os.getenv("MAIN_LLM_KEY"),
    temperature=0
)

deepseek_llm = ChatOpenAI(
    model="deepseek-chat",
    base_url="https://api.deepseek.com",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    temperature=0.7,
    max_tokens=4096
)

search_tool = DuckDuckGoSearchRun()
scrape_tool = Tool(
    name="scrape_page",
    func=lambda url: BeautifulSoup(requests.get(url).text, 'html.parser').get_text()[:2000],
    description="Scrapes content from a URL"
)

tools = [search_tool, scrape_tool]

class AgentState(TypedDict):
    query: str
    sub_queries: Annotated[List[str], "add"]
    data: Annotated[Dict[str, Any], "add"]
    verified_facts: Annotated[Dict[str, Any], "add"]
    counter_arguments: Annotated[Dict[str, Any], "add"]
    final_answer: str
    messages: Annotated[List[BaseMessage], "add"]

def create_runnable(llm, tools, system_prompt):
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])
    if tools:
        return prompt | llm.bind_tools(tools)
    else:
        return prompt | llm


router_runnable = create_runnable(
    llm,
    [],
    "You are a router. Classify the query as 'simple' or 'pro'. Output only the mode."
)

def router_node(state: AgentState) -> AgentState:
    result = router_runnable.invoke({"input": state["query"], "agent_scratchpad": []})
    mode = result.content.lower()
    state["messages"] += [AIMessage(content=mode)]
    return state

simple_runnable = create_runnable(
    llm,
    [search_tool],
    "You are a simple search assistant. Provide a quick answer using search."
)

def simple_node(state: AgentState) -> AgentState:
    result = simple_runnable.invoke({"input": state["query"], "agent_scratchpad": state["messages"]})
    if hasattr(result, 'tool_calls') and result.tool_calls:
        tool_outputs = []
        for tc in result.tool_calls:
            tool_name = tc['name']
            tool_args = tc['args']
            if tool_name == 'duckduckgo_search':
                output = search_tool.run(tool_args['query'])
            tool_outputs.append(output)
        state["final_answer"] = "\n".join(tool_outputs)
    else:
        state["final_answer"] = result.content
    return state

analyzer_runnable = create_runnable(
    llm,
    [],
    "Break the query into sub-queries for multi-hop reasoning."
)

def analyzer_node(state: AgentState) -> AgentState:
    result = analyzer_runnable.invoke({"input": state["query"], "agent_scratchpad": []})
    sub_queries = [q.strip() for q in result.content.split("\n") if q.strip()]  # Improved parsing
    state["sub_queries"] += sub_queries
    state["messages"] += [AIMessage(content=result.content)]
    return state

retriever_runnable = create_runnable(
    llm,
    [search_tool, scrape_tool],
    "Retrieve data from multiple sources using search and scrape. Rerank results semantically."
)

def retriever_node(state: AgentState) -> AgentState:
    data = {}
    for sq in state["sub_queries"]:
        result = retriever_runnable.invoke({
            "input": sq,
            "agent_scratchpad": state["messages"]
        })

        collected = []
        if hasattr(result, 'tool_calls') and result.tool_calls:
            for tc in result.tool_calls:
                tool_name = tc['name']
                tool_args = tc['args']
                if tool_name == 'duckduckgo_search':
                    collected.append(search_tool.run(tool_args['query']))
                elif tool_name == 'scrape_page':
                    collected.append(scrape_tool.func(tool_args['url']))

        else:
            collected.append(result.content)
        data[sq] = collected
    state["data"].update(data)
    state["messages"] += [AIMessage(content="Data retrieved")]
    return state

checker_runnable = create_runnable(
    llm,
    [search_tool],
    "Verify facts by cross-checking sources. If gaps, return 'needs_more'."
)

def checker_node(state: AgentState) -> AgentState:
    result = checker_runnable.invoke({"input": f"Verify: {state['data']}", "agent_scratchpad": state["messages"]})
    if hasattr(result, 'tool_calls') and result.tool_calls:
        tool_outputs = []
        for tc in result.tool_calls:
            if tc['name'] == 'duckduckgo_search':
                output = search_tool.run(tc['args']['query'])
                tool_outputs.append(output)
        verification = "\n".join(tool_outputs)
    else:
        verification = result.content
    if "needs_more" in verification.lower():
        state["messages"] += [AIMessage(content="Needs more data")]
    else:
        state["verified_facts"].update({"facts": verification})
        state["messages"] += [AIMessage(content=verification)]
    return state

counter_argument_runnable = create_runnable(
    llm,
    [search_tool, scrape_tool],
    """You are a Counter-Argument Agent. Your role is to:
    1. Find opposing viewpoints and alternative perspectives on the topic
    2. Search for critical analysis and dissenting opinions
    3. Identify potential biases in the collected data
    4. Look for contradictory evidence or interpretations
    5. Present balanced counter-arguments to reduce bias
    
    Always search for phrases like: 'criticism of', 'opposing view', 'alternative perspective', 'debate about', 'controversy'.
    """
)

def counter_argument_node(state: AgentState) -> AgentState:
    # Формируем запросы для поиска контраргументов
    query = state["query"]
    facts = state["verified_facts"]
    counter_queries = [
        f"criticism of {query}",
        f"opposing views {query}",
        f"alternative perspectives {query}",
        f"debate controversy {query}",
        f"counter-arguments for {facts}",
    ]

    counter_data = {}
    for cq in counter_queries:
        result = counter_argument_runnable.invoke({"input": cq, "agent_scratchpad": state["messages"]})
        if hasattr(result, 'tool_calls') and result.tool_calls:
            for tc in result.tool_calls:
                tool_name = tc['name']
                tool_args = tc['args']
                if tool_name == 'duckduckgo_search':
                    output = search_tool.run(tool_args['query'])
                elif tool_name == 'scrape_page':
                    output = scrape_tool.func(tool_args['url'])
                counter_data.setdefault(cq, []).append(output)
        else:
            counter_data[cq] = result.content

    state["counter_arguments"].update(counter_data)
    state["messages"] += [AIMessage(content="Counter-arguments collected")]
    return state

synthesizer_runnable = create_runnable(
    llm,
    [],
    """Synthesize the final answer with reasoning. Include:
    1. Main findings from verified facts
    2. Counter-arguments and opposing viewpoints
    3. Balanced analysis considering all perspectives
    4. 'Explain your reasoning' section
    5. Sources and references
    6. Acknowledgment of limitations and biases
    """
)

def synthesizer_node(state: AgentState) -> AgentState:
    synthesis_input = f"""
    Verified Facts: {state['verified_facts']}
    Counter-Arguments: {state['counter_arguments']}
    
    Provide a balanced answer that considers both the verified facts and counter-arguments.
    """
    result = synthesizer_runnable.invoke({"input": synthesis_input, "agent_scratchpad": state["messages"]})
    state["final_answer"] = result.content
    return state

workflow = StateGraph(state_schema=AgentState)

workflow.add_node("router", router_node)
workflow.add_node("simple", simple_node)
workflow.add_node("analyzer", analyzer_node)
workflow.add_node("retriever", retriever_node)
workflow.add_node("checker", checker_node)
workflow.add_node("counter_argument", counter_argument_node)
workflow.add_node("synthesizer", synthesizer_node)

workflow.set_entry_point("router")

def route_mode(state: AgentState):
    mode = state["messages"][-1].content
    if "simple" in mode.lower():
        return "simple"
    else:
        return "analyzer"

workflow.add_conditional_edges("router", route_mode, {"simple": "simple", "analyzer": "analyzer"})

workflow.add_edge("simple", END)
workflow.add_edge("analyzer", "retriever")
workflow.add_edge("retriever", "checker")

def check_condition(state: AgentState):
    if any("needs_more" in msg.content.lower() for msg in state["messages"][-5:]):  # Check recent
        return "retriever"
    else:
        return "counter_argument"

workflow.add_conditional_edges("checker", check_condition, {"retriever": "retriever", "counter_argument": "counter_argument"})

workflow.add_edge("counter_argument", "synthesizer")
workflow.add_edge("synthesizer", END)

checkpointer = MemorySaver()
graph = workflow.compile(checkpointer=checkpointer)

if __name__ == "__main__":
    initial_state = {"query": "Время жизни самой известной женщины-программиста", "sub_queries": [], "data": {}, "verified_facts": {}, "counter_arguments": {}, "messages": []}
    config = {"configurable": {"thread_id": "example_thread"}}
    result = graph.invoke(initial_state, config=config)
    print(result["final_answer"])