from fastmcp import FastMCP

from .api.dependencies import create_services


mcp = FastMCP("Copter Reasoning MCP")

services = create_services()


@mcp.tool()
async def ask_copter(
    question: str,
) -> dict:
    """
    Ask a question about the copter configuration,
    failsafe mechanisms, parameters and flight modes.

    Retrieves relevant graph information, selects rules,
    generates facts, executes Prolog and returns an answer.
    """
    return await services.chat.query(question)


@mcp.tool()
async def search_graph(
    query: str,
) -> dict:
    """
    Search the FalkorDB knowledge graph for relevant
    copter entities and relationships.
    """
    return await services.graph.search(query)


@mcp.tool()
async def select_rules(
    question: str,
) -> dict:
    """
    Select relevant Prolog rule artifacts and return
    their predicate signatures.
    """
    artifacts, signatures = services.rules.select_rules(question)

    return {
        "artifacts": [
            {
                "artifact_id": item.artifact_id,
                "domain": item.domain,
                "intents": list(item.intents),
            }
            for item in artifacts
        ],
        "signatures": signatures,
    }


@mcp.tool()
async def generate_prolog_facts(
    question: str,
    graph_state: str,
) -> str:
    """
    Generate validated Prolog ground facts from
    the supplied graph state and selected rules.
    """
    artifacts, _ = services.rules.select_rules(question)

    return await services.facts.generate(
        question=question,
        graph_state=graph_state,
        artifacts=artifacts,
    )


if __name__ == "__main__":
    mcp.run()
