import streamlit as st
import os
from dotenv import load_dotenv
from search import execute_agent_search, MOCK_PAPERS
from gemini_llm import init_gemini, get_synthesized_answer, process_chat_message, generate_search_plan, rate_papers
from datetime import datetime
import difflib

# Load env variables
load_dotenv()
init_gemini()

# Setup Streamlit Config
st.set_page_config(page_title="PeerSight", layout="wide", initial_sidebar_state="expanded")

# --- Custom CSS for Styling ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Outfit:wght@500;600;700&display=swap');

html, body, [class*="css"]  {
    font-family: 'Inter', sans-serif;
}
h1, h2, h3, h4, h5, .st-metric-label {
    font-family: 'Outfit', sans-serif;
}

/* Make headers look sharp */
header[data-testid="stHeader"] {
    background: transparent;
}
div[data-testid="stButton"] > button:first-child {
    background-color: #0f172a;
    color: white;
    border-radius: 6px;
    font-weight: 500;
}
div[data-testid="stButton"] > button:first-child:hover {
    background-color: #1e293b;
    border-color: #1e293b;
}

/* Version badge / Synthesis UI */
.version-badge {
    background-color: #0f172a;
    color: white;
    padding: 2px 8px;
    border-radius: 12px;
    font-size: 0.8em;
    font-family: monospace;
}
</style>
""", unsafe_allow_html=True)


# --- State Initialization ---
if "query" not in st.session_state:
    st.session_state.query = ""
if "papers" not in st.session_state:
    st.session_state.papers = []
if "selected_sources" not in st.session_state:
    st.session_state.selected_sources = ["Computation and Language (NLP)"]
if "agent_queries" not in st.session_state:
    st.session_state.agent_queries = []
if "agent_categories" not in st.session_state:
    st.session_state.agent_categories = []
if "synthesis" not in st.session_state:
    st.session_state.synthesis = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

AVAILABLE_CATEGORIES = {
    "Computation and Language (NLP)": "cs.CL",
    "Artificial Intelligence": "cs.AI",
    "Machine Learning": "cs.LG",
    "Computer Vision": "cs.CV",
    "Quantitative Biology (Biomolecules)": "q-bio.BM",
    "Genomics": "q-bio.GN",
    "Physics": "physics.comp-ph"
}

# --- Main Functions ---

def trigger_search():
    print(f"\n[ACTION] Triggering search for query: '{st.session_state.query}'")
    selected_cats = [AVAILABLE_CATEGORIES[k] for k in st.session_state.selected_sources]

    with st.spinner("Agent planning search..."):
        plan = generate_search_plan(st.session_state.query, selected_cats)
        
        if plan.get("status") == "needs_clarification":
            print("[INFO] Agent requested clarification.")
            st.session_state.chat_history.append({
                "role": "assistant", 
                "text": plan.get("clarifying_question", "Can you please clarify your request?"), 
                "timestamp": "now"
            })
            st.warning("Agent requires clarification. Please reply in the Research Thread.")
            return

        queries = plan.get("queries", [st.session_state.query])
        categories = plan.get("categories", selected_cats)
        
        st.session_state.agent_queries = queries
        st.session_state.agent_categories = categories

    with st.spinner(f"Executing targeted searches across {', '.join(categories)}..."):
        print(f"[ACTION] Executing agent search across queries: {queries}")
        results = execute_agent_search(queries, categories, max_results_per_query=3)
        if results:
            print(f"[INFO] Found {len(results)} overall papers.")
            with st.spinner("Rating retrieved papers based on relevance, quality, and recency..."):
                ratings = rate_papers(st.session_state.query, results)
                rating_map = {str(r.get("id")): r for r in ratings}
                for p in results:
                    pid = str(p.get("id"))
                    p["score"] = rating_map.get(pid, {}).get("score", "N/A")
                    p["justification"] = rating_map.get(pid, {}).get("justification", "No rating available.")
                
                # Sort papers by score descending if available
                try:
                    results.sort(key=lambda x: int(x.get("score", 0)) if str(x.get("score")).isdigit() else 0, reverse=True)
                except ValueError:
                    pass

            print("[INFO] Papers successfully rated and sorted.")
            st.session_state.papers = results
            trigger_synthesis(trigger_reason="Agentic Context Search")
        else:
            print("[INFO] No papers found during search.")
            st.warning("No results from arXiv databases. Try a different query.")
            st.session_state.papers = []

def trigger_synthesis(trigger_reason="Initial Context Search"):
    print(f"\n[ACTION] Triggering synthesis. Reason: '{trigger_reason}'")
    with st.spinner("Synthesizing answer with Gemini..."):
        answer = get_synthesized_answer(st.session_state.query, st.session_state.papers, st.session_state.chat_history)
        
        # Save version
        version = {
            "version": len(st.session_state.synthesis) + 1,
            "sourceCount": len(st.session_state.papers),
            "timestamp": datetime.now().strftime("%I:%M %p"),
            "trigger": trigger_reason,
            "text": answer
        }
        st.session_state.synthesis.append(version)
        print(f"[INFO] Syntehsis completed. Version {version['version']} saved.")


# --- UI Layout ---

# SIDEBAR (Process Panel)
with st.sidebar:
    st.markdown("### Process Panel")
    if not st.session_state.agent_queries:
        st.info("Awaiting agent actions...")
    else:
        st.markdown(f"**Selected Categories:** {', '.join(st.session_state.agent_categories)}")
        st.caption("Agent Queries Executed")
        for i, q in enumerate(st.session_state.agent_queries):
            st.code(f"{i+1}. search(\"{q}\")", language="text")
        
        cat_str = '", "'.join(st.session_state.agent_categories)
        if cat_str:
            st.code(f"filter(category IN [\"{cat_str}\"])", language="text")
        
        st.divider()
        st.markdown("### Databases")
        st.markdown(f"🟢 **arXiv Automated** ({len(st.session_state.papers)} sources fetched)")

        st.divider()
        st.markdown("### Agent Reasoning Trace")
        if not st.session_state.papers:
            st.info("Run a search to see the agent reasoning trace.")
        else:
            st.markdown(f"""
            1. **Query Decomposition**: Extracted concepts from '{st.session_state.query}'.
            2. **Source Selection**: Filtered for {', '.join(st.session_state.agent_categories)}.
            3. **Retrieval & Rating**: Retrieved {len(st.session_state.papers)} papers, scoring by relevance/quality/recency.
            4. **Synthesis Engine**: Passed rated results to Gemini for summarization, enforcing citation inclusion.
            """)
            st.warning("Uncertainty Flag: Automated synthesis may occasionally hallucinate citations. Please verify against provided snippets.")

# MAIN AREA

# Top Header
st.markdown("## PeerSight 🔍")
st.caption("AI RESEARCH ASSISTANT")
st.divider()

# Search Bar Area
with st.container():
    col1, col2, col3 = st.columns([6, 3, 1])
    with col1:
        st.text_input("Ask a research question...", key="query", label_visibility="collapsed", placeholder="e.g. Tell me about how MDLMs are used for molecule generation.")
    with col2:
        st.multiselect(
            "Data Sources",
            options=list(AVAILABLE_CATEGORIES.keys()),
            key="selected_sources",
            label_visibility="collapsed"
        )
    with col3:
        if st.button("Search", use_container_width=True):
            trigger_search()
            
st.markdown("<br>", unsafe_allow_html=True)


def get_inline_diff(old_text: str, new_text: str) -> str:
    matcher = difflib.SequenceMatcher(None, old_text.split(), new_text.split())
    result = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'replace':
            result.append(f'<del style="background-color: #ffcccc; color: #990000; text-decoration: line-through;">{" ".join(old_text.split()[i1:i2])}</del>')
            result.append(f'<ins style="background-color: #ccffcc; color: #006600; text-decoration: none;">{" ".join(new_text.split()[j1:j2])}</ins>')
        elif tag == 'delete':
            result.append(f'<del style="background-color: #ffcccc; color: #990000; text-decoration: line-through;">{" ".join(old_text.split()[i1:i2])}</del>')
        elif tag == 'insert':
            result.append(f'<ins style="background-color: #ccffcc; color: #006600; text-decoration: none;">{" ".join(new_text.split()[j1:j2])}</ins>')
        elif tag == 'equal':
            result.append(" ".join(old_text.split()[i1:i2]))
    return " ".join(result)

# Main Content: Split into Left and Right Panes
main_col, chat_col = st.columns([2, 1], gap="large")

with main_col:
    # Synthesized Answer Section
    if len(st.session_state.synthesis) > 0:
        st.markdown("### Synthesized Answer")
        latest_v = st.session_state.synthesis[-1]
        st.markdown(f"**Version {latest_v['version']}** &mdash; {latest_v['sourceCount']} sources &mdash; Prompted by: *{latest_v['trigger']}*")
        st.info(latest_v['text'])
        
        if len(st.session_state.synthesis) > 1:
            with st.expander("Compare Versions (Diff)"):
                v_opts = [str(v['version']) for v in st.session_state.synthesis]
                col_a, col_b = st.columns(2)
                with col_a:
                    base_v = st.selectbox("Base Version", v_opts, index=len(v_opts)-2)
                with col_b:
                    comp_v = st.selectbox("Comparison Version", v_opts, index=len(v_opts)-1)
                
                base_text = next(v['text'] for v in st.session_state.synthesis if str(v['version']) == base_v)
                comp_text = next(v['text'] for v in st.session_state.synthesis if str(v['version']) == comp_v)
                
                diff_html = get_inline_diff(base_text, comp_text)
                st.markdown(diff_html, unsafe_allow_html=True)
        
        with st.expander("View Version History"):
            for v in reversed(st.session_state.synthesis[:-1]):
                st.markdown(f"**v{v['version']}** ({v['trigger']}) - {v['text']}")
                st.divider()
    else:
        st.info("Run a search to generate a synthesized literature review!")
        
    st.markdown("### Retrieved Papers")
    for paper in st.session_state.papers:
        with st.container(border=True):
            url = paper.get('url', '#')
            st.markdown(f"**[{paper['title']}]({url})**")
            authors = paper.get('authors', '')
            year = paper.get('year', '')
            cites = paper.get('citations_mock', 'N/A')
            score = paper.get('score', 'N/A')
            st.caption(f"Authors: {authors} | Year: {year} | Score: **{score}/100**")
            justification = paper.get('justification', '')
            if justification:
                st.caption(f"*Justification: {justification}*")
            with st.expander("Show snippet"):
                st.write(paper['summary'])

with chat_col:
    st.markdown("### Research Thread")
    st.caption("Steer the search and ask follow-ups.")
    
    # Render chat history
    chat_container = st.container(height=500)
    with chat_container:
        for msg in st.session_state.chat_history:
            if msg["role"] == "assistant":
                st.chat_message("assistant").write(msg["text"])
            else:
                st.chat_message("user").write(msg["text"])
                
    # Chat Input
    if user_input := st.chat_input("Type here to steer the agent... (e.g. 'Narrow scope to 2024')"):
        print(f"\n[ACTION] New chat message from user: '{user_input}'")
        # Add User Message
        st.session_state.chat_history.append({"role": "user", "text": user_input, "timestamp": "now"})
        chat_container.chat_message("user").write(user_input)
        
        # Process via Gemini
        with chat_container.chat_message("assistant"):
            with st.spinner("Thinking..."):
                allowed_cats = list(AVAILABLE_CATEGORIES.values())
                response_dict = process_chat_message(user_input, st.session_state.papers, st.session_state.chat_history, allowed_cats)
                
                response_text = response_dict.get("message_to_user", "I've processed your request.")
                action = response_dict.get("action", "reply_only")
                print(f"[INFO] Chat response parsed. Action determined: {action}")
            
            st.write(response_text)
            
        st.session_state.chat_history.append({"role": "assistant", "text": response_text, "timestamp": "now"})
        
        # Agent execution flow
        if action == "search":
            new_queries = response_dict.get("new_search_queries", [])
            new_cats = response_dict.get("new_search_categories", [])
            print(f"[ACTION] Agent initiating follow-up search. Queries: {new_queries}, Categories: {new_cats}")
            
            if new_queries:
                with st.spinner("Agent running additional search..."):
                    st.session_state.agent_queries.extend(new_queries)
                    cat_set = set(st.session_state.agent_categories + new_cats)
                    st.session_state.agent_categories = list(cat_set)
                    print("[INFO] Executing follow-up agent search logic.")
                    
                    new_results = execute_agent_search(new_queries, new_cats, max_results_per_query=2)
                    
                    if new_results:
                        print(f"[INFO] Follow-up search returned {len(new_results)} papers. Rating now...")
                        ratings = rate_papers(user_input, new_results)
                        rating_map = {str(r.get("id")): r for r in ratings}
                        
                        existing_ids = {str(p.get("id")) for p in st.session_state.papers}
                        
                        for p in new_results:
                            pid = str(p.get("id"))
                            if pid not in existing_ids:
                                p["score"] = rating_map.get(pid, {}).get("score", "N/A")
                        print(f"[INFO] Merged new parsed context with existing pool. Total sources now: {len(st.session_state.papers)}")
                                
                        try:
                            st.session_state.papers.sort(key=lambda x: int(x.get("score", 0)) if str(x.get("score")).isdigit() else 0, reverse=True)
                        except ValueError:
                            pass
                    else:
                        print("[INFO] Agent follow-up search did not find any unique papers.")
                
            trigger_synthesis(trigger_reason=f"Agent search on: '{user_input[:20]}...'")
        
        elif action == "update_synthesis":
            print("[ACTION] Updating current synthesis configuration.")
            trigger_synthesis(trigger_reason=f"User feedback update: '{user_input[:20]}...'")
            
        else:
            print("[INFO] No action needed except text reply.")
            
        st.rerun()
