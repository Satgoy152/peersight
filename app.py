import streamlit as st
import os
from dotenv import load_dotenv
from search import search_arxiv, MOCK_PAPERS
from gemini_llm import init_gemini, get_synthesized_answer, evaluate_novelty, process_chat_message
from datetime import datetime

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
if "use_real_search" not in st.session_state:
    st.session_state.use_real_search = False
if "synthesis" not in st.session_state:
    st.session_state.synthesis = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "novelty_result" not in st.session_state:
    st.session_state.novelty_result = None

# --- Main Functions ---

def trigger_search():
    with st.spinner("Searching for related papers..."):
        if st.session_state.use_real_search:
            results = search_arxiv(st.session_state.query, max_results=5)
            if results:
                st.session_state.papers = results
            else:
                st.warning("No results from arXiv. Falling back to mock data.")
                st.session_state.papers = MOCK_PAPERS
        else:
            st.session_state.papers = MOCK_PAPERS
            
    # Trigger Synthesis automatically
    trigger_synthesis(trigger_reason="Initial Context Search")

def trigger_synthesis(trigger_reason="Initial Context Search"):
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


# --- UI Layout ---

# SIDEBAR (Process Panel)
with st.sidebar:
    st.markdown("### Process Panel")
    if not st.session_state.papers:
        st.info("Awaiting user query...")
    else:
        st.caption("Agent Queries")
        st.code(f"1. search(\"{st.session_state.query}\")", language="text")
        st.code("2. filter(subject IN [\"cs.CL\"])", language="text")
        
        st.divider()
        st.markdown("### Databases")
        st.markdown(f"🟢 **arXiv cs.CL** ({len(st.session_state.papers)} sources fetched)")

# MAIN AREA

# Top Header
st.markdown("## PeerSight 🔍")
st.caption("AI RESEARCH ASSISTANT")
st.divider()

# Search Bar Area
with st.container():
    col1, col2, col3 = st.columns([6, 2, 1])
    with col1:
        st.session_state.query = st.text_input("Ask a research question...", value=st.session_state.query, label_visibility="collapsed")
    with col2:
        st.session_state.use_real_search = st.toggle("Live arXiv Search", value=st.session_state.use_real_search, help="Toggle to fetch real data from arXiv instead of using mock data.")
    with col3:
        if st.button("Search", use_container_width=True):
            trigger_search()
            
st.markdown("<br>", unsafe_allow_html=True)


# Main Content: Split into Left and Right Panes
main_col, chat_col = st.columns([2, 1], gap="large")

with main_col:
    # Tabs for main info
    tab_results, tab_reasoning, tab_novelty = st.tabs(["📑 Results & Synthesis", "🧠 Reasoning Trace", "💡 Novelty Check"])
    
    with tab_results:
        # Synthesized Answer Section
        if len(st.session_state.synthesis) > 0:
            st.markdown("#### Synthesized Answer")
            latest_v = st.session_state.synthesis[-1]
            st.markdown(f"**Version {latest_v['version']}** &mdash; {latest_v['sourceCount']} sources &mdash; Prompted by: *{latest_v['trigger']}*")
            st.info(latest_v['text'])
            
            with st.expander("View Version History"):
                for v in reversed(st.session_state.synthesis[:-1]):
                    st.markdown(f"**v{v['version']}** ({v['trigger']}) - {v['text']}")
                    st.divider()
        else:
            st.info("Run a search to generate a synthesized literature review!")
            
        st.markdown("#### Retrieved Papers")
        for paper in st.session_state.papers:
            with st.container(border=True):
                st.markdown(f"**{paper['title']}**")
                authors = paper.get('authors', '')
                year = paper.get('year', '')
                cites = paper.get('citations_mock', 'N/A')
                st.caption(f"Authors: {authors} | Year: {year} | Citations (mock): {cites}")
                with st.expander("Show snippet"):
                    st.write(paper['summary'])
    
    with tab_reasoning:
        st.markdown("#### Agent Reasoning Trace")
        if not st.session_state.papers:
            st.info("Run a search to see the agent reasoning trace.")
        else:
            st.markdown(f"""
            1. **Query Decomposition**: Extracted concepts from '{st.session_state.query}'.
            2. **Source Selection**: Filtered for arXiv cs.CL.
            3. **Synthesis Engine**: Passed {len(st.session_state.papers)} results to Gemini 1.5 Flash for summarization, enforcing citation inclusion.
            """)
            st.warning("Uncertainty Flag: Automated synthesis may occasionally hallucinate citations. Please verify against provided snippets.")
        
    with tab_novelty:
        st.markdown("#### Idea Novelty Assessment")
        novelty_query = st.text_area("Propose a new idea to check its overlap with current literature:", placeholder="e.g. A two-pass remasking approach for training continuous diffusion...")
        if st.button("Evaluate Originality"):
            if novelty_query:
                with st.spinner("Assessing vs Literature..."):
                    res = evaluate_novelty(novelty_query, st.session_state.papers)
                    st.session_state.novelty_result = res
            else:
                st.error("Please enter an idea.")
                
        if st.session_state.novelty_result is not None:
            res = st.session_state.novelty_result
            score = res.get("score", 0)
            st.metric(label="Novelty Score", value=f"{score}/100")
            st.write(res.get("overlap_description", ""))
            
            st.markdown("##### Closest Overlapping Work")
            for cp in res.get("closest_papers", []):
                st.markdown(f"- **{cp.get('title')}** (Overlap: {cp.get('overlap_percentage', 0)}%)")
                st.caption(cp.get("reason", ""))

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
        # Add User Message
        st.session_state.chat_history.append({"role": "user", "text": user_input, "timestamp": "now"})
        chat_container.chat_message("user").write(user_input)
        
        # Process via Gemini
        with chat_container.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response_text = process_chat_message(user_input, st.session_state.papers, st.session_state.chat_history)
            st.write(response_text)
            
        st.session_state.chat_history.append({"role": "assistant", "text": response_text, "timestamp": "now"})
        
        # Automatically update synthesis on follow up
        trigger_synthesis(trigger_reason=f"User feedback: '{user_input[:20]}...'")
        st.rerun()
