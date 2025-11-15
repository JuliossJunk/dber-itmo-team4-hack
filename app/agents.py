from typing import TypedDict, Annotated, List, Dict, Any
from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import Tool
from vectorstore import cache_get, cache_set, cache_search
import requests
from bs4 import BeautifulSoup
import os
from dotenv import load_dotenv

from prompts import runnable_prompt, simple_prompt, analyzer_prompt, retriever_prompt, checker_prompt, \
    counter_prompt, synthesizer_prompt

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
    runnable_prompt
)


def router_node(state: AgentState) -> AgentState:
    result = router_runnable.invoke({"input": state["query"], "agent_scratchpad": []})
    mode = result.content.lower()
    state["messages"] += [AIMessage(content=mode)]
    return state


simple_runnable = create_runnable(
    llm,
    [search_tool],
    simple_prompt
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
    analyzer_prompt
)


def analyzer_node(state: AgentState) -> AgentState:
    result = analyzer_runnable.invoke({"input": state["query"], "agent_scratchpad": []})
    sub_queries = [q.strip() for q in result.content.split("\n") if q.strip()]
    state["sub_queries"] += sub_queries
    state["messages"] += [AIMessage(content=result.content)]
    return state


retriever_runnable = create_runnable(
    llm,
    [search_tool, scrape_tool],
    retriever_prompt
)


def retriever_node(state: AgentState) -> AgentState:
    data = {}
    for sq in state["sub_queries"]:

        collected = []
        cached_results = cache_search(sq)

        if cached_results:
            formatted = [
                f"[CACHED FACT] {r['facts']} (source: {r['source']}, date: {r['date']})"
                for r in cached_results
            ]
            collected.extend(formatted)
        result = retriever_runnable.invoke({
            "input": sq,
            "agent_scratchpad": state["messages"]
        })

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
    checker_prompt
)


def checker_node(state: AgentState) -> AgentState:
    query = state["query"]

    cached = cache_get(query)
    if cached:
        state["verified_facts"] = {"facts": cached["facts"], "source": cached["source"], "cached": True}
        state["messages"] += [AIMessage(content=f"Cached fact used from {cached['date']}")]
        return state

    result = checker_runnable.invoke(
        {"input": f"Verify: {state['data']}", "agent_scratchpad": state["messages"]}
    )

    if hasattr(result, 'tool_calls') and result.tool_calls:
        tool_outputs = []
        used_source = ""
        for tc in result.tool_calls:
            if tc['name'] == 'duckduckgo_search':
                q = tc['args']['query']
                output = search_tool.run(q)
                tool_outputs.append(output)
                used_source = q
        verification = "\n".join(tool_outputs)
    else:
        verification = result.content
        used_source = "unknown"

    if "needs_more" in verification.lower():
        state["messages"] += [AIMessage(content="Needs more data")]
        return state

    state["verified_facts"].update({"facts": verification, "source": used_source})
    state["messages"] += [AIMessage(content=verification)]

    cache_set(query, verification, used_source)

    return state


counter_argument_runnable = create_runnable(
    llm,
    [search_tool, scrape_tool],
    counter_prompt
)


def counter_argument_node(state: AgentState) -> AgentState:
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
    synthesizer_prompt
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