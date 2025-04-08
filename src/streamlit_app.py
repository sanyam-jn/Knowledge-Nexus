import streamlit as st
import networkx as nx
try:
    import pyvis.network as net
except ImportError:
    st.error("Installing required packages...")
    import subprocess
    subprocess.check_call(["pip", "install", "pyvis"])
    import pyvis.network as net
from nexus import NEXUS
import tempfile
import os

# Initialize NEXUS
@st.cache_resource
def get_nexus():
    return NEXUS()

nexus = get_nexus()

# Set page config
st.set_page_config(
    page_title="Knowledge Nexus",
    page_icon="🧠",
    layout="wide"
)

# Title
st.title("Knowledge Nexus")
st.markdown("An intelligent system for building and querying knowledge graphs")

# Create two columns
col1, col2 = st.columns(2)

# Input Document Section
with col1:
    st.header("Input Document")
    text_input = st.text_area(
        "Enter your text here",
        height=200,
        placeholder="Enter text to process..."
    )
    if st.button("Process Text", type="primary"):
        if text_input:
            with st.spinner("Processing text..."):
                # Extract entities and relationships
                triplets = nexus.extract_entities_relationships(text_input)
                
                # Build knowledge graph
                nexus.build_knowledge_graph(triplets)
                
                # Create visualization
                graph = net.Network(height="400px", width="100%", bgcolor="#ffffff", font_color="black")
                
                # Add nodes and edges
                for node in nexus.knowledge_graph.nodes():
                    graph.add_node(node, label=node)
                
                for u, v, data in nexus.knowledge_graph.edges(data=True):
                    graph.add_edge(u, v, label=data["relation"])
                
                # Save and display the graph
                with tempfile.NamedTemporaryFile(delete=False, suffix='.html') as tmpfile:
                    graph.save_graph(tmpfile.name)
                    with open(tmpfile.name, 'r', encoding='utf-8') as f:
                        html_data = f.read()
                    os.unlink(tmpfile.name)  # Delete the temporary file
                
                # Display success message
                st.success("Text processed successfully!")
                
                # Store the graph HTML in session state
                st.session_state['graph_html'] = html_data
                st.session_state['triplets'] = triplets
        else:
            st.error("Please enter some text to process.")

# Query Section
with col2:
    st.header("Query")
    query_input = st.text_input("Enter your question", placeholder="Ask a question about the text...")
    if st.button("Ask", type="primary"):
        if query_input:
            if 'triplets' in st.session_state:
                with st.spinner("Processing query..."):
                    # Process query
                    result = nexus.process_query(query_input)
                    
                    # Display results in an expander
                    with st.expander("Query Results", expanded=True):
                        st.markdown(f"**Question:** {query_input}")
                        st.markdown(f"**Answer:** {result.get('answer', 'No answer found.')}")
                        if result.get('sources'):
                            st.markdown("**Sources:**")
                            for source in result['sources']:
                                st.markdown(f"- {source}")
            else:
                st.warning("Please process some text first before asking questions.")
        else:
            st.error("Please enter a question.")

# Display Knowledge Graph
st.header("Knowledge Graph")
if 'graph_html' in st.session_state:
    st.components.v1.html(st.session_state['graph_html'], height=600)
else:
    st.info("Process some text to see the knowledge graph visualization.")
