"""Tools for the Analyst agent: web search, knowledge base RAG, and FRED economic data."""

from __future__ import annotations

from pathlib import Path

import boto3
from fredapi import Fred
from langchain_aws import BedrockEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.tools import tool

from financial_advisory_agent.config import settings

_fred: Fred | None = None


def _get_fred() -> Fred:
    global _fred
    if _fred is None:
        _fred = Fred(api_key=settings.fred_api_key)
    return _fred

_vectorstore: FAISS | None = None


@tool
def search_web(query: str) -> str:
    """Search the internet for current financial data or market news."""
    try:
        from tavily import TavilyClient

        client = TavilyClient(api_key=settings.tavily_api_key)
        resp = client.search(query=query, max_results=3)
        results = [
            f"**{r['title']}**\n{r['content']}\nSource: {r['url']}"
            for r in resp.get("results", [])
        ]
        return "\n\n---\n\n".join(results) if results else "No results found."
    except Exception as e:
        return f"Web search failed: {e}"


def _build_vectorstore() -> FAISS | None:
    """Build FAISS index from knowledge directory (cached)."""
    global _vectorstore
    if _vectorstore is not None:
        return _vectorstore

    kb_path = Path(settings.knowledge_dir)
    if not kb_path.exists():
        return None

    docs = [
        Document(page_content=f.read_text(), metadata={"source": f.name})
        for f in sorted(kb_path.glob("*.txt"))
    ]
    if not docs:
        return None

    _vectorstore = FAISS.from_documents(
        docs, BedrockEmbeddings(model_id=settings.embedding_model, region_name=settings.aws_region)
    )
    return _vectorstore


@tool
def search_knowledge_base(query: str) -> str:
    """Search curated financial planning documents (asset allocation, tax accounts, risk, etc.)."""
    store = _build_vectorstore()
    if store is None:
        return "Knowledge base is empty."

    results = store.similarity_search(query, k=3)
    return "\n\n".join(
        f"[{i}] ({doc.metadata.get('source', '?')})\n{doc.page_content}"
        for i, doc in enumerate(results, 1)
    )


@tool
def get_fred_data(series_id: str, observation_start: str = "", num_observations: int = 12) -> str:
    """Fetch economic data from the Federal Reserve (FRED).

    Use this for interest rates, inflation (CPI/PCE), GDP, unemployment,
    Treasury yields, VIX, and other macro indicators.

    Args:
        series_id: The FRED series ID (e.g., 'DGS10' for 10-Year Treasury,
                   'CPIAUCSL' for CPI, 'FEDFUNDS' for Fed Funds Rate).
        observation_start: Start date in YYYY-MM-DD format. Defaults to
                          last 12 observations if empty.
        num_observations: Number of most recent data points to return.

    Returns:
        Formatted time series data with dates and values.
    """
    kwargs = {}
    if observation_start:
        kwargs["observation_start"] = observation_start

    fred = _get_fred()
    data = fred.get_series(series_id, **kwargs)
    recent = data.dropna().tail(num_observations)

    try:
        info = fred.get_series_info(series_id)
        title = info["title"]
        units = info["units"]
        freq = info["frequency"]
        header = f"FRED Series: {title} ({series_id})\nUnits: {units} | Frequency: {freq}"
    except Exception:
        header = f"FRED Series: {series_id}"

    lines = [header, f"{'─' * 40}"]
    for date, value in recent.items():
        lines.append(f"  {date.strftime('%Y-%m-%d')}  {value:.2f}")
    lines.append(f"{'─' * 40}")
    lines.append(f"Latest: {recent.iloc[-1]:.2f} ({recent.index[-1].strftime('%Y-%m-%d')})")
    return "\n".join(lines)


@tool
def search_fred(query: str, limit: int = 5) -> str:
    """Search FRED for economic data series by keyword.

    Args:
        query: Search terms (e.g., 'inflation', 'treasury yield', 'GDP').
        limit: Max results to return.

    Returns:
        List of matching series with IDs, titles, and frequencies.
    """
    results = _get_fred().search(query).head(limit)
    lines = [f"FRED Search: '{query}' — top {len(results)} results", ""]
    for _, row in results.iterrows():
        lines.append(f"  {row['id']:20s}  {row['title'][:60]}")
        lines.append(f"  {'':20s}  Freq: {row['frequency']}  |  Last updated: {row['last_updated']}")
        lines.append("")
    return "\n".join(lines)


@tool
def search_bedrock_kb(query: str) -> str:
    """Search the Bedrock-managed knowledge base for financial planning principles and strategies.

    Use this instead of search_knowledge_base when a Bedrock KB ID is configured.
    Covers asset allocation, tax-advantaged accounts, risk diversification, and common mistakes.
    """
    if not settings.bedrock_kb_id:
        return "Bedrock knowledge base is not configured (BEDROCK_KB_ID not set)."

    bedrock = boto3.client("bedrock-agent-runtime", region_name=settings.aws_region)
    response = bedrock.retrieve(
        knowledgeBaseId=settings.bedrock_kb_id,
        retrievalQuery={"text": query},
        retrievalConfiguration={"vectorSearchConfiguration": {"numberOfResults": 3}},
    )
    results = response.get("retrievalResults", [])
    if not results:
        return "No results found in Bedrock knowledge base."

    return "\n\n".join(
        f"[{i}] (score={r['score']:.3f}) {r['location']['s3Location']['uri']}\n{r['content']['text']}"
        for i, r in enumerate(results, 1)
    )


analyst_tools = [search_web, search_knowledge_base, search_bedrock_kb, get_fred_data, search_fred]
