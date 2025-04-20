from langchain.tools import Tool

rag_tool = Tool(
    name="MedlineRAG",
    func=lambda query: "\n".join(
        [doc.page_content for doc in retriever.get_relevant_documents(query)]
    ),
    description="Use this tool to answer medical questions from literature. Input should be a question like: 'How is pneumonia diagnosed?'",
)
