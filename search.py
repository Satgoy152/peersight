import arxiv
import pandas as pd
from datetime import datetime
from typing import List, Dict

MOCK_PAPERS = [
  {
    "id": 1,
    "title": "Masked Diffusion Language Models are Fast and Privacy-Preserving",
    "authors": "Shi et al.",
    "venue": "NeurIPS 2024",
    "year": 2024,
    "relevance": 97,
    "citations_mock": 311,
    "tags": ["MDLM", "training", "privacy"],
    "summary": "We show that masked diffusion outperforms autoregressive models in low-data regimes by reducing exposure to out-of-distribution tokens during training. The work connects exposure bias to discrete flow matching models.",
    "url": "#"
  },
  {
    "id": 2,
    "title": "SEDD: Score Entropy Discrete Diffusion",
    "authors": "Lou et al.",
    "venue": "ICML 2024",
    "year": 2024,
    "relevance": 91,
    "citations_mock": 203,
    "tags": ["score entropy", "discrete diffusion"],
    "summary": "Score entropy provides a principled objective for discrete diffusion that avoids the train-test mismatch inherent in ELBO-based objectives. It allows training diffusion models on language with strong results.",
    "url": "#"
  },
  {
    "id": 3,
    "title": "LLaDA: Large Language Diffusion with mAsking",
    "authors": "Nie et al.",
    "venue": "arXiv 2025",
    "year": 2025,
    "relevance": 88,
    "citations_mock": 54,
    "tags": ["MDLM", "scaling", "instruction tuning"],
    "summary": "LLaDA scales masked diffusion to 8B parameters and demonstrates competitive performance with GPT-4 on several benchmarks. Using novel masking techniques.",
    "url": "#"
  },
  {
    "id": 4,
    "title": "Discrete Flow Matching for Language Generation",
    "authors": "Gat et al.",
    "venue": "NeurIPS 2024",
    "year": 2024,
    "relevance": 79,
    "citations_mock": 89,
    "tags": ["flow matching", "discrete"],
    "summary": "Flow matching provides a unified view of masked diffusion and autoregressive models, enabling interpolation between training paradigms. Introduces new training techniques.",
    "url": "#"
  }
]

def search_arxiv(query: str, max_results: int = 10) -> List[Dict]:
    """Search arxiv for papers in the cs.CL category."""
    client = arxiv.Client()
    
    try:
        # Enforce search within cs.CL (NLP)
        search = arxiv.Search(
            query=f'all:"{query}" AND cat:cs.CL',
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance
        )
        
        results = list(client.results(search))
        
        papers = []
        for r in results:
            papers.append({
                "id": r.get_short_id(),
                "title": r.title,
                "authors": ", ".join([a.name for a in r.authors]),
                "year": r.published.year,
                "published_date": r.published.strftime("%Y-%m-%d"),
                "summary": r.summary,
                "url": r.pdf_url,
                "citations_mock": 0
            })
            
        return papers
    except Exception as e:
        print(f"Error fetching from arxiv: {e}")
        return []
