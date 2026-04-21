import os
import streamlit as st
import google.generativeai as genai
from typing import List, Dict
import json

def init_gemini():
    api_key = st.secrets["GEMINI_API_KEY"]
    if api_key:
        genai.configure(api_key=api_key)

def generate_search_plan(query: str, allowed_categories: List[str]) -> Dict:
    """Uses Gemini to decide if it needs clarification or can generate searches/categories."""
    print(f"[SYSTEM] Generating search plan for query: '{query}'")
    try:
        model = genai.GenerativeModel("gemini-3-flash-preview")
        
        prompt = f"""
        You are PeerSight, an expert AI Research Assistant.
        The user has given a research query: "{query}"

        Allowed Source Categories for arXiv:
        {allowed_categories}

        Your task is to either:
        A) Provide 1 to 3 specific keyword queries and the best allowed categories to search to find the most relevant papers.
        B) If the user query is completely incomprehensible or too vague, ask for clarification.

        Return a JSON object that strictly matches this format:
        {{
            "status": "execute_search" or "needs_clarification",
            "clarifying_question": "<string, only if needs_clarification>",
            "queries": ["<string, specific arxiv keyword query>", "<string>"],
            "categories": ["<category code from allowed_categories>"]
        }}
        """
        
        response = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json"
            )
        )
        plan = json.loads(response.text)
        print(f"[SYSTEM] Search plan generated: {plan['status']}")
        return plan
    except Exception as e:
        print(f"[ERROR] Error in search plan: {e}")
        return {"status": "needs_clarification", "clarifying_question": "Sorry, I had an error analyzing your query. Can you please rephrase?"}

def get_synthesized_answer(query: str, papers: List[Dict], chat_history: List[Dict] = None) -> str:
    """Uses Gemini 3 Flash Preview to synthesize an answer based on papers and chat history."""
    print(f"[SYSTEM] Synthesizing answer using {len(papers)} papers...")
    try:
        model = genai.GenerativeModel("gemini-3-flash-preview")
        
        context = ""
        for i, p in enumerate(papers):
            context += f"Paper {i+1}:\nTitle: {p['title']}\nAuthors: {p['authors']}\nSummary: {p['summary']}\n\n"
            
        history_context = ""
        if chat_history:
            history_context = "Conversation history:\n"
            for msg in chat_history:
                history_context += f"{msg['role']}: {msg['text']}\n"
                
        prompt = f"""
        You are PeerSight, an expert AI Research Assistant specializing in NLP.
        The user's core research topic is: "{query}"
        
        Using the provided context papers and the conversation history, synthesize an evolving literature review.
        
        Guidelines for your synthesis:
        1. The Core is Focal: Keep the user's original query as the central theme.
        2. Evolving Thought: Treat the conversation history as an evolution of ideas. Add answers to new follow-ups by appending insights without eagerly deleting the original context.
        3. Compress, Don't Drop: If you need to make room for new discoveries, compress older points rather than removing them entirely. Ensure previously established context remains available unless explicitly discarded by the user.
        4. Length Limit: Your final synthesis must be at most 3 paragraphs long.
        5. Citations: Always cite the papers in your text using the [Author Year] format.
        
        Context Papers:
        {context}
        
        {history_context}
        
        Provide a succinct synthesized overview focusing on consensus, conflicting findings, and identifying gaps while adhering to the guidelines.
        """
        
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error connecting to Gemini API: {e}\n\nPlease make sure your GEMINI_API_KEY is correctly set in the .env file."

def rate_papers(query: str, papers: List[Dict]) -> List[Dict]:
    """Evaluates and rates a list of papers based on relevance, quality, and recency."""
    if not papers:
        print("[SYSTEM] No papers to rate.")
        return []
    print(f"[SYSTEM] Rating {len(papers)} papers based on query: '{query}'")
    try:
        model = genai.GenerativeModel("gemini-3-flash-preview")
        
        context = ""
        for i, p in enumerate(papers):
            year = p.get('year', 'Unknown')
            venue = p.get('venue', 'Unknown')
            context += f"Paper ID: {p.get('id', i)}\nTitle: {p['title']}\nAuthors: {p['authors']}\nYear: {year}\nVenue: {venue}\nSummary: {p['summary']}\n\n"

        prompt = f"""
        You are PeerSight, an expert AI Research Assistant.
        The user has requested research on: "{query}"
        
        Based on the retrieved papers below, rate each paper from 0-100 based on:
        1. Relevance to the query.
        2. Quality (e.g., reputable authors and venues if known).
        3. Recency (newer papers generally score higher unless older ones are highly relevant seminal works).
        
        Papers:
        {context}
        
        Return a JSON list of objects matching this exact format:
        [
            {{
                "id": "<paper id from input>",
                "score": <0-100 integer>,
                "justification": "<1-2 sentence explanation of the score based on relevance, quality, and recency>"
            }}
        ]
        """
        
        response = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json"
            )
        )
        ratings = json.loads(response.text)
        print(f"[SYSTEM] Successfully rated {len(ratings)} papers.")
        return ratings
    except Exception as e:
        print(f"[ERROR] Error rating papers: {e}")
        return []

def process_chat_message(query: str, active_papers: List[Dict], chat_history: List[Dict], allowed_categories: List[str] = None) -> Dict:
    """Process a conversation message and returns a structured action plan."""
    print(f"[SYSTEM] Processing chat message. Active papers: {len(active_papers)}")
    try:
        model = genai.GenerativeModel("gemini-3-flash-preview")
        
        context = ""
        for i, p in enumerate(active_papers):
            context += f"Paper {i+1}: {p['title']} ({str(p.get('summary', ''))[:150]}...)\n"

        history = "\n".join([f"{m['role']}: {m['text']}" for m in chat_history[-5:]])
        
        prompt = f"""
        You are PeerSight, an AI Research Assistant interacting via a chat thread.
        
        Current context papers:
        {context}
        
        Allowed Source Categories for arXiv searches (if needed):
        {allowed_categories if allowed_categories else []}
        
        Recent conversation:
        {history}
        
        User's newest message: {query}
        
        Determine the best action to take:
        1. 'reply_only': The user just wants to chat, ask a simple question, or there's no need to update the formal synthesized document or search for new papers.
        2. 'update_synthesis': The user requested a change in formatting, scope, or focus that requires rewriting the main Synthesized Answer based on EXISTING papers.
        3. 'search': The user is asking about new concepts, asking to expand the scope, or asking questions that require retrieving NEW papers from arXiv.

        Return a JSON object that strictly matches this format:
        {{
            "message_to_user": "<Concise 1-3 sentence Assistant response to the user's message>",
            "action": "reply_only", "update_synthesis", or "search",
            "new_search_queries": ["<string, specific arxiv keyword query>", "<string>"],
            "new_search_categories": ["<category code from allowed_categories>"]
        }}
        """
        
        response = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json"
            )
        )
        action_plan = json.loads(response.text)
        print(f"[SYSTEM] Chat processed. Suggested action: {action_plan.get('action')}")
        return action_plan
    except Exception as e:
        print(f"[ERROR] Error processing chat message: {e}")
        return {
            "message_to_user": f"Error connecting to Gemini: {e}",
            "action": "reply_only",
            "new_search_queries": [],
            "new_search_categories": []
        }
