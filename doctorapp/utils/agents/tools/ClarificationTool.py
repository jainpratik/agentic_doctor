from langchain.tools import Tool

clarifier_tool = Tool(
    name="Clarifier",
    func=lambda q: "I need more information. Can you clarify what symptoms are most severe or when they started?",
    description="Used when the input is ambiguous or lacks key clinical details.",
)
