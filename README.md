I developed an Agentic Graph RAG (Retrieval-Augmented Generation) system that intelligently retrieves, evaluates, refines, and generates responses using a structured state graph workflow. By integrating LangChain, OpenAI, and Neo4j, this system enhances context-aware AI applications.

The workflow begins with an LLM-driven agent that decides whether to retrieve relevant documents from a Neo4j vector store or refine the query for better results. Retrieved documents are then evaluated for relevance using an AI grader, ensuring only the most useful information is considered. If relevance is low, the system rewrites the query before proceeding. Finally, the LLM generates responses based on the validated information.

Key technologies include LangChain tools, Neo4j knowledge graphs, OpenAI embeddings, and LangGraph for decision-making workflows. This structured approach enhances retrieval accuracy and ensures responses are contextually rich, precise, and trustworthy.

This system is ideal for knowledge-based AI assistants, research applications, and intelligent document retrieval systems. By leveraging graph-based reasoning and retrieval evaluation, it transforms traditional RAG pipelines into adaptive, agent-driven AI solutions.
