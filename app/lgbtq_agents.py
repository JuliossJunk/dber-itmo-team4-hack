"""Dummy agents"""

from langchain.tools import Tool
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, AgentType
from langgraph.graph import StateGraph, END

import requests
from bs4 import BeautifulSoup
from typing import Dict, List, Any


TRUSTED_SOURCES: Dict[str, List[str]] = {
    "scientific": [
        "https://arxiv.org",
        "https://scholar.google.com",
        "https://pubmed.ncbi.nlm.nih.gov",
    ],
    "financial": [
        "https://www.sec.gov",
        "https://www.imf.org",
        "https://www.worldbank.org",
    ],
    "social": [
        "https://ourworldindata.org",
        "https://www.un.org",
        "https://www.oecd.org",
    ],
}


def fetch_from_source(url: str, query: str) -> str:
    """Simple placeholder search request to a trusted source URL.

    Performs a raw GET request and tries to find the query text.
    Used only as a stub, should be replaced with real search providers.

    Args:
        url (str): Target trusted URL.
        query (str): Search query string.

    Returns:
        str: Extracted text preview or a placeholder result.
    """
    try:
        response = requests.get(url, timeout=10)
        if response.status_code != 200:
            return f"Ошибка при доступе: {url}"

        soup = BeautifulSoup(response.text, "html.parser")
        text = soup.get_text()

        if query.lower() in text.lower():
            return (
                f"🔎 Найдено на {url}:\n"
                f"{text[:800]}\n...\n(Обрезано, заглушка)"
            )

        return f"Ничего не найдено на {url}"

    except Exception as exc:
        return f"Ошибка: {str(exc)}"


def generate_search_tools() -> List[Tool]:
    """Generate LangChain search tools for all trusted source categories.

    Creates a Tool for each URL in each category. Functions are minimal
    and only provide placeholder functionality.

    Returns:
        List[Tool]: List of LangChain tools.
    """
    tools = []

    for category, urls in TRUSTED_SOURCES.items():
        for url in urls:
            sanitized = url.replace("https://", "").replace(".", "_")

            tool = Tool(
                name=f"search_{category}_{sanitized}",
                func=lambda q, url=url: fetch_from_source(url, q),
                description=f"Stub search tool for category '{category}' using URL: {url}."
            )
            tools.append(tool)

    return tools


def init_search_agent() -> Any:
    """Initialize a simple LangChain search agent (stub).

    The agent uses ReAct (ZERO_SHOT_REACT_DESCRIPTION) and a set of
    placeholder search tools. This is not a production search agent.

    Returns:
        Any: Initialized agent instance.
    """
    llm = ChatOpenAI(model="gpt-4.1", temperature=0)
    tools = generate_search_tools()

    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
    )

    return agent


class SearchState(dict):
    """State object for LangGraph search pipeline.

    Keys:
        query (str): Search query string.
        result (str): Result of executing the search agent.
    """
    query: str
    result: str


def search_step(state: SearchState) -> SearchState:
    """Perform a single search step using the search agent.

    This is a stub function, intended to represent a pipeline step
    in a LangGraph graph.

    Args:
        state (SearchState): Graph state containing user query.

    Returns:
        SearchState: Updated state with a search result.
    """
    agent = init_search_agent()
    result = agent.run(state["query"])
    return {"result": result}


def build_graph() -> Any:
    """Create a one-step LangGraph search pipeline.

    The graph contains a single processing node ("search").
    This is only a template meant for later extension.

    Returns:
        Any: Compiled LangGraph graph instance.
    """
    graph = StateGraph(SearchState)
    graph.add_node("search", search_step)
    graph.set_entry_point("search")
    graph.add_edge("search", END)

    return graph.compile()


if __name__ == "__main__":
    graph = build_graph()
    out = graph.invoke({"query": "evolution swarm"})
    print("\n===== РЕЗУЛЬТАТ =====")
    print(out.get("result"))
