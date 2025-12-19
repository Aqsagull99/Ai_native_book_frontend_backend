# """
# Multi-Agent System using OpenAI Agents Python SDK
# Following the documentation at https://openai.github.io/openai-agents-python
# Implements specialized agents with handoff capabilities
# """

# import os
# import logging
# from typing import Dict, Any, List
# from datetime import datetime

# from agents import Agent, Runner, handoff, function_tool
# from pydantic import BaseModel

# from src.rag_agent.models import AgentRequest, AgentResponse, ContentChunk
# from src.rag_agent.qdrant_service import search_vectors
# from src.content_embedding.retrieval_service import create_query_embedding
# from src.rag_agent.config import Config
# from src.rag_agent.llm_service import GeminiClientService

# # Set up comprehensive logging
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
# )
# logger = logging.getLogger(__name__)


# @function_tool
# def vector_retrieval_tool(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
#     """
#     Retrieve relevant content chunks from the vector database based on semantic similarity.

#     Args:
#         query: The search query to find relevant content
#         top_k: Number of results to retrieve (default: 5)

#     Returns:
#         List of retrieved content chunks with their metadata
#     """
#     try:
#         logger.info(f"Executing vector retrieval for query: '{query[:50]}{'...' if len(query) > 50 else ''}'")

#         # Validate inputs
#         if not query or not query.strip():
#             raise ValueError("Query cannot be empty")

#         if top_k <= 0 or top_k > 100:
#             raise ValueError("top_k must be between 1 and 100")

#         # Create embedding for the query text
#         query_vector = create_query_embedding(query)

#         # Perform semantic search using Qdrant
#         search_results = search_vectors(
#             query_vector=query_vector,
#             top_k=top_k,
#             metadata_filter=None,  # No additional filters for now
#             collection_name=Config.QDRANT_COLLECTION_NAME
#         )

#         # Format results
#         formatted_results = []
#         for i, result in enumerate(search_results):
#             content_chunk = {
#                 "content": result["content"],
#                 "similarity_score": result["similarity_score"],
#                 "metadata": result["metadata"],
#                 "rank": i + 1
#             }
#             formatted_results.append(content_chunk)

#         logger.info(f"Retrieved {len(formatted_results)} results for query: '{query[:30]}{'...' if len(query) > 30 else ''}'")
#         return formatted_results

#     except Exception as e:
#         logger.error(f"Error in vector retrieval tool: {str(e)}")
#         raise


# class EscalationData(BaseModel):
#     """Data model for handoff operations"""
#     reason: str
#     original_query: str


# # Create specialized agents
# research_agent = Agent(
#     name="Research Agent",
#     instructions="You are a research specialist. Use the vector_retrieval tool to find relevant information from the book content. Focus on factual accuracy and provide detailed citations.",
#     tools=[vector_retrieval_tool]
# )

# analysis_agent = Agent(
#     name="Analysis Agent",
#     instructions="You are an analysis specialist. Synthesize information from research, identify patterns, and provide comprehensive answers to user questions based on retrieved content."
# )

# summary_agent = Agent(
#     name="Summary Agent",
#     instructions="You are a summarization specialist. Create concise, well-structured summaries of complex information while preserving key details and citations."
# )


# # Create the main triage agent with handoffs
# triage_agent = Agent(
#     name="Triage Agent",
#     instructions="You are the main interface agent. Assess user queries and delegate to specialized agents when appropriate. For factual questions about book content, use the research agent. For analysis tasks, use the analysis agent. For summary requests, use the summary agent.",
#     tools=[vector_retrieval_tool],
#     handoffs=[
#         handoff(
#             agent=research_agent,
#             tool_name_override="research_assistant",
#             tool_description_override="Delegate to the research specialist for factual information retrieval"
#         ),
#         handoff(
#             agent=analysis_agent,
#             tool_name_override="analysis_assistant",
#             tool_description_override="Delegate to the analysis specialist for synthesis and analysis"
#         ),
#         handoff(
#             agent=summary_agent,
#             tool_name_override="summary_assistant",
#             tool_description_override="Delegate to the summary specialist for concise summaries"
#         )
#     ]
# )


# class MultiAgentRAGSystem:
#     """
#     Multi-Agent RAG System using OpenAI Agents Python SDK with handoff capabilities.
#     Implements specialized agents for different aspects of the RAG process.
#     """

#     def __init__(self):
#         """Initialize the multi-agent system."""
#         # Validate configuration
#         is_valid, error_msg = Config.validate()
#         if not is_valid:
#             raise ValueError(f"Invalid configuration: {error_msg}")

#         # Initialize the Gemini client service
#         self.gemini_client = GeminiClientService()

#         # Store the agents
#         self.triage_agent = triage_agent
#         self.research_agent = research_agent
#         self.analysis_agent = analysis_agent
#         self.summary_agent = summary_agent

#         logger.info("✅ Multi-Agent RAG System initialized successfully with handoff capabilities")

#     def process_query(self, query: str, top_k: int = 5) -> AgentResponse:
#         """
#         Process a user query through the multi-agent system.

#         Args:
#             query: The user's query
#             top_k: Number of results to retrieve

#         Returns:
#             AgentResponse with the answer and metadata
#         """
#         start_time = datetime.now()
#         logger.info(f"Starting query processing with Multi-Agent System for: '{query[:50]}{'...' if len(query) > 50 else ''}' (top_k={top_k})")

#         try:
#             # Run the triage agent with the user query
#             # The triage agent will determine if handoffs are needed
#             result = Runner.run_sync(
#                 self.triage_agent,
#                 f"Query: {query}\nPlease provide a comprehensive answer. If the query requires research, analysis, or summarization, delegate to the appropriate specialized agent."
#             )

#             # Extract the final output from the agent result
#             answer = result.final_output if result.final_output else "I couldn't find sufficient information to answer your question."

#             # Get the content for response formatting by calling the retrieval tool directly
#             tool_results = vector_retrieval_tool(query, top_k)

#             # Format the retrieved chunks
#             retrieved_chunks = []
#             for result in tool_results:
#                 chunk = ContentChunk(
#                     content=result["content"],
#                     similarity_score=result["similarity_score"],
#                     metadata=result["metadata"],
#                     rank=result["rank"]
#                 )
#                 retrieved_chunks.append(chunk)

#             # Calculate confidence score as average similarity of retrieved chunks
#             confidence_score = sum(c.similarity_score for c in retrieved_chunks) / len(retrieved_chunks) if retrieved_chunks else 0.0

#             total_time = (datetime.now() - start_time).total_seconds()

#             # Create the response
#             response = AgentResponse(
#                 query_text=query,
#                 answer=answer,
#                 retrieved_chunks=retrieved_chunks,
#                 confidence_score=confidence_score,
#                 execution_time=total_time,
#                 timestamp=datetime.now()
#             )

#             logger.info(f"Query processed successfully: '{query[:30]}{'...' if len(query) > 30 else ''}' "
#                        f"in {total_time:.4f}s with {len(retrieved_chunks)} retrieved chunks, confidence: {confidence_score:.3f}")

#             return response

#         except Exception as e:
#             total_time = (datetime.now() - start_time).total_seconds()
#             logger.error(f"Error processing query with Multi-Agent System: {str(e)}")

#             # Return an error response
#             response = AgentResponse(
#                 query_text=query,
#                 answer="Sorry, I encountered an error while processing your query.",
#                 retrieved_chunks=[],
#                 confidence_score=0.0,
#                 execution_time=total_time,
#                 timestamp=datetime.now()
#             )

#             return response

#     def run_research_only(self, query: str, top_k: int = 5) -> AgentResponse:
#         """
#         Run only the research agent for direct information retrieval.

#         Args:
#             query: The research query
#             top_k: Number of results to retrieve

#         Returns:
#             AgentResponse with the answer and metadata
#         """
#         start_time = datetime.now()
#         logger.info(f"Starting research-only query for: '{query[:50]}{'...' if len(query) > 50 else ''}' (top_k={top_k})")

#         try:
#             # Run the research agent directly
#             result = Runner.run_sync(
#                 self.research_agent,
#                 f"Research Query: {query}\nPlease find and provide relevant information from the book content. Retrieve {top_k} relevant chunks."
#             )

#             # Extract the final output
#             answer = result.final_output if result.final_output else "I couldn't find relevant information."

#             # Get the content for response formatting
#             tool_results = vector_retrieval_tool(query, top_k)

#             # Format the retrieved chunks
#             retrieved_chunks = []
#             for result in tool_results:
#                 chunk = ContentChunk(
#                     content=result["content"],
#                     similarity_score=result["similarity_score"],
#                     metadata=result["metadata"],
#                     rank=result["rank"]
#                 )
#                 retrieved_chunks.append(chunk)

#             # Calculate confidence score
#             confidence_score = sum(c.similarity_score for c in retrieved_chunks) / len(retrieved_chunks) if retrieved_chunks else 0.0

#             total_time = (datetime.now() - start_time).total_seconds()

#             # Create the response
#             response = AgentResponse(
#                 query_text=query,
#                 answer=answer,
#                 retrieved_chunks=retrieved_chunks,
#                 confidence_score=confidence_score,
#                 execution_time=total_time,
#                 timestamp=datetime.now()
#             )

#             logger.info(f"Research query completed: '{query[:30]}{'...' if len(query) > 30 else ''}' "
#                        f"in {total_time:.4f}s with {len(retrieved_chunks)} retrieved chunks")

#             return response

#         except Exception as e:
#             total_time = (datetime.now() - start_time).total_seconds()
#             logger.error(f"Error in research-only query: {str(e)}")

#             # Return an error response
#             response = AgentResponse(
#                 query_text=query,
#                 answer="Sorry, I encountered an error while processing your research query.",
#                 retrieved_chunks=[],
#                 confidence_score=0.0,
#                 execution_time=total_time,
#                 timestamp=datetime.now()
#             )

#             return response


# def create_multi_agent_rag_system() -> MultiAgentRAGSystem:
#     """
#     Create and return a configured Multi-Agent RAG System instance.

#     Returns:
#         Configured MultiAgentRAGSystem instance
#     """
#     return MultiAgentRAGSystem()


# def process_agent_request_with_multi_agent(agent_request: AgentRequest) -> AgentResponse:
#     """
#     Process an agent request using the Multi-Agent System and return the response.

#     Args:
#         agent_request: The agent request with query and parameters

#     Returns:
#         AgentResponse with the answer and metadata
#     """
#     agent = create_multi_agent_rag_system()

#     return agent.process_query(
#         query=agent_request.query_text,
#         top_k=agent_request.top_k
#     )




import logging
from datetime import datetime

from agents import Agent, Runner, handoff, function_tool
from agents.run import RunConfig
from agents import AsyncOpenAI, OpenAIChatCompletionsModel

from src.rag_agent.config import Config
from src.rag_agent.qdrant_service import search_vectors
from src.content_embedding.retrieval_service import create_query_embedding
from src.rag_agent.models import AgentResponse, ContentChunk


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# 🔹 GEMINI CLIENT
client = AsyncOpenAI(
    api_key=Config.GEMINI_API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

run_config = RunConfig(
    model=OpenAIChatCompletionsModel(
        model=Config.GEMINI_MODEL_NAME,
        openai_client=client
    ),
    model_provider=client
)


# 🔥 STRING-ONLY TOOL (Gemini compatible)
@function_tool
def retrieve_book_context(query: str, top_k: int = 5) -> str:
    query_vector = create_query_embedding(query)
    results = search_vectors(
        query_vector=query_vector,
        top_k=top_k,
        collection_name=Config.QDRANT_COLLECTION_NAME
    )

    if not results:
        return "No relevant book context found."

    context = ""
    for i, r in enumerate(results, 1):
        context += f"""
Source {i}:
Title: {r['metadata'].get('title')}
URL: {r['metadata'].get('url')}
Content:
{r['content']}
---
"""
    return context.strip()


# 🔹 AGENTS
research_agent = Agent(
    name="ResearchAgent",
    instructions="Retrieve factual information from book context.",
    tools=[retrieve_book_context]
)

analysis_agent = Agent(
    name="AnalysisAgent",
    instructions="Analyze and synthesize provided research context."
)

summary_agent = Agent(
    name="SummaryAgent",
    instructions="Summarize the provided information clearly and concisely."
)

triage_agent = Agent(
    name="TriageAgent",
    instructions="""
Decide how to answer:
- Use research for factual lookup
- Use analysis for explanation
- Use summary for concise answers
""",
    tools=[retrieve_book_context],
    handoffs=[
        handoff(research_agent),
        handoff(analysis_agent),
        handoff(summary_agent)
    ]
)


class MultiAgentRAGSystem:

    def process_query(self, query: str, top_k: int = 5) -> AgentResponse:
        start = datetime.now()

        result = Runner.run_sync(
            triage_agent,
            input=query,
            run_config=run_config
        )

        answer = result.final_output or "No answer generated."

        execution_time = (datetime.now() - start).total_seconds()

        return AgentResponse(
            query_text=query,
            answer=answer,
            retrieved_chunks=[],  # tool already injected context
            confidence_score=0.0,
            execution_time=execution_time,
            timestamp=datetime.now()
        )
