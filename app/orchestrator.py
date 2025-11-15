from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from app.agents import AgentState, router_node, simple_node, analyzer_node, retriever_node, checker_node, \
    counter_argument_node, synthesizer_node

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