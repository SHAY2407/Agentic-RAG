import streamlit as st
import os
from dotenv import load_dotenv
import pandas as pd
from llama_index.llms.google_genai import GoogleGenAI
import google.generativeai as genai
from llama_index.experimental.query_engine.pandas import PandasQueryEngine
from note_engine import note_engine
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent import ReActAgent
from prompts import context
from pdf import multi_pdf_engine, pdf_files

# Page configuration
st.set_page_config(
    page_title="Agentic RAG System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
}
.sub-header {
    font-size: 1.5rem;
    color: #333;
    margin-bottom: 1rem;
}
.info-box {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 0.5rem;
    margin: 1rem 0;
}
.tool-info {
    background-color: #e8f4fd;
    padding: 0.5rem;
    border-radius: 0.3rem;
    margin: 0.5rem 0;
    border-left: 4px solid #1f77b4;
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def initialize_agent():
    """Initialize the RAG agent with all tools."""
    # Load .env and get API key
    load_dotenv()
    api_key = os.getenv("API_KEY")
    
    if not api_key:
        st.error("API_KEY not found in .env file. Please add your Gemini API key.")
        st.stop()
    
    genai.configure(api_key=api_key)
    llm = GoogleGenAI(model="gemini-2.0-flash", api_key=api_key)
    
    # Read population CSV
    population_path = os.path.join("data", "population.csv")
    if os.path.exists(population_path):
        population_df = pd.read_csv(population_path)
        population_query_engine = PandasQueryEngine(df=population_df, llm=llm, verbose=True)
    else:
        st.error(f"Population data file not found: {population_path}")
        st.stop()
    
    # Create tools
    tools = [
        note_engine,
        QueryEngineTool(
            query_engine=population_query_engine, 
            metadata=ToolMetadata(
                name="population_data",
                description="this gives information at the world population and demographics",
            ),
        ),
        QueryEngineTool(
            query_engine=multi_pdf_engine, 
            metadata=ToolMetadata(
                name="pdf_data",
                description=(
                "This gives detailed information about all PDFs in the data folder: "
                "Circular1.pdf, Circular2.pdf, Circular5.pdf, Circular6.pdf, "
                "Circular7.pdf, Circular8.pdf, Circular9.pdf, Circular10.pdf. "
                "Use this tool to answer questions about highway construction, "
                "NHAI directives, toll plaza management, maintenance issues, "
                "authorized signatories, project details, and any content from these circulars. "
                f"The available PDFs are: {', '.join(pdf_files)}"
            ),
            ),
        ),
    ]
    
    # Create agent
    agent = ReActAgent.from_tools(tools, llm=llm, verbose=True, context=context)
    return agent, population_df, pdf_files

def main():
    # Header
    st.markdown('<h1 class="main-header">🤖 Agentic RAG System</h1>', unsafe_allow_html=True)
    
    # Initialize agent
    try:
        agent, population_df, pdf_files = initialize_agent()
    except Exception as e:
        st.error(f"Error initializing agent: {str(e)}")
        st.stop()
    
    # Sidebar with information
    with st.sidebar:
        st.markdown('<h2 class="sub-header">📊 Available Data Sources</h2>', unsafe_allow_html=True)
        
        # Population data info
        st.markdown('<div class="tool-info"><b>Population Data</b><br>CSV with world population statistics and demographics</div>', unsafe_allow_html=True)
        
        # PDF files info
        st.markdown('<div class="tool-info"><b>PDF Documents</b></div>', unsafe_allow_html=True)
        for pdf in pdf_files:
            st.markdown(f"• {pdf}")
        
        # Notes info
        st.markdown('<div class="tool-info"><b>Note Saver</b><br>Save important information to notes.txt</div>', unsafe_allow_html=True)
        
        # Show sample population data
        if st.checkbox("Show sample population data"):
            st.subheader("Population Data Preview")
            st.dataframe(population_df.head())
    
    # Main interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<h2 class="sub-header">💬 Chat Interface</h2>', unsafe_allow_html=True)
        
        # Query input
        user_query = st.text_area(
            "Enter your question:",
            placeholder="Ask about population data, PDF documents, or request to save notes...",
            height=100
        )
        
        # Submit button
        if st.button("🚀 Submit Query", type="primary"):
            if user_query.strip():
                with st.spinner("Processing your query..."):
                    try:
                        response = agent.query(user_query)
                        st.success("Query completed!")
                        st.markdown("### Response:")
                        st.markdown(f"**{response}**")
                    except Exception as e:
                        st.error(f"Error processing query: {str(e)}")
            else:
                st.warning("Please enter a question first.")
    
    with col2:
        st.markdown('<h2 class="sub-header">💡 Example Queries</h2>', unsafe_allow_html=True)
        
        examples = [
            "What is the population of India?",
            "Show me countries with population over 100 million",
            "What information is available in the PDF documents?",
            "Save a note: Meeting scheduled for tomorrow",
            "Compare population growth rates between countries",
            "What are the key points in the Guidelines.pdf?"
        ]
        
        for i, example in enumerate(examples, 1):
            if st.button(f"Example {i}", key=f"example_{i}"):
                st.session_state.example_query = example
        
        # If an example was clicked, show it in the text area
        if 'example_query' in st.session_state:
            st.info(f"Example query: {st.session_state.example_query}")
    
    # Chat history section
    st.markdown('<h2 class="sub-header">📝 Recent Notes</h2>', unsafe_allow_html=True)
    
    notes_file = os.path.join("data", "notes.txt")
    if os.path.exists(notes_file):
        try:
            with open(notes_file, "r") as f:
                notes_content = f.read()
            if notes_content.strip():
                st.text_area("Saved Notes:", value=notes_content, height=150, disabled=True)
            else:
                st.info("No notes saved yet.")
        except Exception as e:
            st.error(f"Error reading notes: {str(e)}")
    else:
        st.info("No notes file found. Notes will be created when you save your first note.")
    
    # Footer
    st.markdown("---")
    st.markdown(
        '<div class="info-box">'
        '<b>How to use:</b><br>'
        '• Ask questions about world population data<br>'
        '• Query information from PDF documents<br>'
        '• Save important notes by asking to "save a note: your content"<br>'
        '• Use natural language - the AI agent will determine which tool to use'
        '</div>',
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()

