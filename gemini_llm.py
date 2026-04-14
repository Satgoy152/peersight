import os
import google.generativeai as genai
from typing import List, Dict
import json

def init_gemini():
    api_key = os.getenv("GEMINI_API_KEY")
    if api_key:
        genai.configure(api_key=api_key)

def get_synthesized_answer(query: str, papers: List[Dict], chat_history: List[Dict] = None) -> str:
    """Uses Gemini 1.5 Flash to synthesize an answer based on papers and chat history."""
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
        Synthesize a literature review answer to the following query: "{query}"
        
        Use mainly the provided paper summaries to inform your answer. 
        Cite the papers in your text using the [Author Year] format.
        Focus on answering how it compares to existing work.
        Keep it concise (1-2 paragraphs).
        
        Context Papers:
        {context}
        
        {history_context}
        
        Provide a succinct synthesized overview focusing on consensus, conflicting findings, and identifying gaps.
        """
        
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error connecting to Gemini API: {e}\n\nPlease make sure your GEMINI_API_KEY is correctly set in the .env file."

def evaluate_novelty(idea: str, papers: List[Dict]) -> Dict:
    """Evaluates the novelty of a given idea compared to retrieved papers."""
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        
        context = ""
        for i, p in enumerate(papers):
            context += f"Paper {i+1}:\nTitle: {p['title']}\nAuthors: {p['authors']}\nSummary: {p['summary']}\n\n"

        prompt = f"""
        You are PeerSight, an expert AI Research Assistant assessing research novelty.
        The user has proposed the following research idea: "{idea}"
        
        Based on the following existing literature:
        {context}
        
        Evaluate the novelty of the user's idea.
        Return a JSON response strictly exactly matching this format. Your output must start with {{ and end with }}:
        {{
            "score": <0-100 integer representing novelty percentage>,
            "overlap_description": "<1-2 sentence description of what overlaps and what is novel>",
            "closest_papers": [
                {{"title": "<title of closest paper>", "overlap_percentage": <integer>, "reason": "<short reason>"}}
            ]
        }}
        """
        
        response = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json"
            )
        )
        return json.loads(response.text)
    except Exception as e:
        return {
            "score": 0,
            "overlap_description": f"Error connected to Gemini: {e}",
            "closest_papers": []
        }

def process_chat_message(query: str, active_papers: List[Dict], chat_history: List[Dict]) -> str:
    """Process a conversation message."""
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        
        context = ""
        for i, p in enumerate(active_papers):
            context += f"Paper {i+1}: {p['title']} ({str(p.get('summary', ''))[:150]}...)\n"

        history = "\n".join([f"{m['role']}: {m['text']}" for m in chat_history[-5:]])
        
        prompt = f"""
        You are PeerSight, an AI Research Assistant interacting via a chat thread.
        
        Current context papers:
        {context}
        
        Recent conversation:
        {history}
        
        User's newest message: {query}
        
        Respond clearly, briefly, and professionally. If the user asks to filter or change scope, acknowledge it and say you'll update the synthesis document.
        Keep the response very short (1-3 sentences max).
        """
        
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error connecting to Gemini: {e}"
