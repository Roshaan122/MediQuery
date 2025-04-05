import os
import json
import time
import numpy as np
import re
import warnings
import streamlit as st
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import pandas as pd
# LangChain imports
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Set up Streamlit page
st.set_page_config(page_title="DiReCT Medical RAG", page_icon="🏥", layout="wide")

# Custom CSS
st.markdown("""
<style>
    .main-title {
        font-size: 36px;
        font-weight: bold;
        color: #1E88E5;
        margin-bottom: 20px;
    }
    .source-box {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 10px;
    }
    .source-title {
        font-weight: bold;
        margin-bottom: 5px;
    }
    .response-time {
        font-size: 14px;
        color: #555;
        margin-top: 5px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1E88E5;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# --- DATA PROCESSING UTILITIES ---

def cal_a_json(file_path):
    """
    Process an annotated JSON file and reconstruct with tree structure.
    Based on DiReCT GitHub repository implementation.
    
    Args:
        file_path: Path to JSON file
    
    Returns:
        record_node: Dict for all nodes with node index as key
        input_content: Original clinical notes
        chain: List of diagnostic procedure in order
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        
        record_node = {}  # A dictionary for all nodes in the annotation
        input_content = {}  # Original clinical notes from input1-6
        chain = []  # Diagnostic procedure in order
        node_id = 0
        
        # Extract input content (original clinical notes)
        for key, value in data.items():
            if key.startswith("input"):
                input_content[key] = value
        
        # Process all other nodes
        for key, value in data.items():
            if not key.startswith("input") and not key.startswith("_"):
                # This is a diagnosis node
                parts = key.split("$")
                if len(parts) == 2:
                    node_name = parts[0]
                    node_type = parts[1]
                    
                    record_node[node_id] = {
                        "content": node_name,
                        "type": node_type,
                        "connection": [],
                        "upper": None
                    }
                    
                    # Process children nodes
                    if isinstance(value, dict):
                        process_node(value, record_node, node_id, node_id+1)
                    
                    # Add to diagnostic chain if it's an Intermediate node
                    if node_type.startswith("Intermedia"):
                        chain.append(node_id)
                        
                    node_id += 1
        
        return record_node, input_content, chain
    
    except Exception as e:
        st.error(f"Error processing file {file_path}: {e}")
        return {}, {}, []

def process_node(data, record_node, parent_id, current_id):
    """
    Recursively process nodes in the diagnostic tree.
    
    Args:
        data: Dictionary containing node data
        record_node: Dictionary to store all nodes
        parent_id: ID of parent node
        current_id: Starting ID for new nodes
        
    Returns:
        Next available node ID
    """
    next_id = current_id
    
    for key, value in data.items():
        parts = key.split("$")
        if len(parts) == 2:
            node_name = parts[0]
            node_type = parts[1]
            
            # Add node to record
            record_node[next_id] = {
                "content": node_name,
                "type": node_type,
                "connection": [],
                "upper": parent_id
            }
            
            # Add to parent's connections
            if parent_id in record_node:
                record_node[parent_id]["connection"].append(next_id)
            
            # Process children recursively
            if isinstance(value, dict) and value:
                next_id = process_node(value, record_node, next_id, next_id+1)
            else:
                next_id += 1
    
    return next_id

def deduction_assemble(record_node):
    """
    Organize all nodes and return deductions.
    
    Args:
        record_node: Dict for all nodes
        
    Returns:
        Dictionary of deductions {observation: [rationale, source, diagnosis]}
    """
    deductions = {}
    
    # Find all leaf nodes (Input type)
    for node_id, node in record_node.items():
        if node["type"].startswith("Input"):
            # This is an observation from clinical notes
            observation = node["content"]
            source = node["type"]  # input1-6
            
            # Find parent (rationale)
            if node["upper"] is not None and node["upper"] in record_node:
                parent = record_node[node["upper"]]
                rationale = parent["content"]
                
                # Find ultimate diagnosis by traversing up
                diagnosis = find_diagnosis(record_node, node["upper"])
                
                # Store deduction
                deductions[observation] = [rationale, source, diagnosis]
    
    return deductions

def find_diagnosis(record_node, node_id):
    """Find the top-level diagnosis for a node by traversing up the tree"""
    current_id = node_id
    visited = set()  # Prevent infinite loops
    
    while current_id is not None and current_id not in visited:
        visited.add(current_id)
        node = record_node.get(current_id)
        
        if not node:
            break
        
        # If this is a top-level diagnostic node
        if node["type"].startswith("Intermedia") and node["upper"] is None:
            return node["content"]
        
        current_id = node["upper"]
    
    return "Unknown"

# --- KNOWLEDGE GRAPH PROCESSING ---

def load_knowledge_graph(file_path):
    """
    Load knowledge graph from JSON file and create documents.
    This function correctly processes the knowledge graph structure 
    described in the README with semicolon-separated items.
    
    Args:
        file_path: Path to knowledge graph JSON
        
    Returns:
        List of Document objects
    """
    documents = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            kg = json.load(f)
        
        disease_category = os.path.basename(file_path).replace('.json', '')
        
        # 1. Process the diagnostic structure
        if "diagnostic" in kg:
            diagnostic_docs = process_diagnostic_structure(kg["diagnostic"], disease_category)
            documents.extend(diagnostic_docs)
        
        # 2. Process the knowledge structure (most important for answering questions)
        if "knowledge" in kg:
            knowledge_docs = process_knowledge_structure(kg["knowledge"], disease_category)
            documents.extend(knowledge_docs)
        
        return documents
    
    except Exception as e:
        st.error(f"Error loading knowledge graph {file_path}: {e}")
    
    return documents

def process_diagnostic_structure(diagnostic_data, disease_category, prefix=""):
    """Process the diagnostic structure recursively"""
    documents = []
    
    if not isinstance(diagnostic_data, dict):
        return documents
        
    for diagnosis, children in diagnostic_data.items():
        # Create current path
        current_path = f"{prefix} > {diagnosis}" if prefix else diagnosis
        
        # Create document for this diagnostic node
        documents.append(Document(
            page_content=f"In {disease_category}, {diagnosis} is a diagnostic entity in the pathway: {current_path}",
            metadata={
                "source": "knowledge_graph",
                "type": "diagnostic_pathway",
                "disease_category": disease_category,
                "diagnosis": diagnosis,
                "path": current_path
            }
        ))
        
        # Process children recursively
        if isinstance(children, dict):
            child_docs = process_diagnostic_structure(children, disease_category, current_path)
            documents.extend(child_docs)
    
    return documents

def process_knowledge_structure(knowledge_data, disease_category):
    """
    Process the knowledge section from a knowledge graph.
    This handles semicolon-separated items as described in the README.
    
    Args:
        knowledge_data: Knowledge section from the knowledge graph
        disease_category: Name of the disease category
        
    Returns:
        List of Document objects
    """
    documents = []
    
    for diagnosis, info in knowledge_data.items():
        # Check if this is a dict with categories (symptoms, risk factors, etc.)
        if isinstance(info, dict):
            for category, details in info.items():
                # Split by semicolons if present (as per README)
                if isinstance(details, str) and ";" in details:
                    # First create a document with all items for this category
                    documents.append(Document(
                        page_content=f"In {disease_category}, {diagnosis} has the following {category.lower()}: {details}",
                        metadata={
                            "source": "knowledge_graph",
                            "type": "knowledge_comprehensive",
                            "disease_category": disease_category,
                            "diagnosis": diagnosis,
                            "category": category
                        }
                    ))
                    
                    # Then create individual documents for each item
                    items = [item.strip() for item in details.split(";")]
                    for item in items:
                        if item and not item.endswith("etc."):  # Skip empty items or "etc."
                            documents.append(Document(
                                page_content=f"In {disease_category}, {diagnosis} has this {category.lower()}: {item}",
                                metadata={
                                    "source": "knowledge_graph",
                                    "type": "knowledge_item",
                                    "disease_category": disease_category,
                                    "diagnosis": diagnosis,
                                    "category": category,
                                    "item": item
                                }
                            ))
                else:
                    # Direct knowledge without semicolons
                    documents.append(Document(
                        page_content=f"In {disease_category}, {diagnosis} has {category.lower()}: {details}",
                        metadata={
                            "source": "knowledge_graph",
                            "type": "knowledge",
                            "disease_category": disease_category,
                            "diagnosis": diagnosis,
                            "category": category
                        }
                    ))
        else:
            # Direct information
            documents.append(Document(
                page_content=f"In {disease_category}, {diagnosis} characteristics: {info}",
                metadata={
                    "source": "knowledge_graph",
                    "type": "knowledge_direct",
                    "disease_category": disease_category,
                    "diagnosis": diagnosis
                }
            ))
    
    return documents

# --- QUERY REFORMULATION ---

class QueryReformulator:
    """Class for query reformulation to improve retrieval performance"""
    
    def __init__(self):
        self.clinical_categories = [
            "symptom", "symptoms", "sign", "signs", "risk factor", "risk factors",
            "diagnostic criteria", "diagnosis", "treatment", "management", 
            "prognosis", "complications", "epidemiology", "pathophysiology"
        ]
        
        self.prevention_terms = [
            "prevent", "prevention", "avoid", "reduce risk", "lower risk",
            "manage risk", "preventive", "preventative", "protect"
        ]
    
    def reformulate(self, query):
        """
        Reformulate the query to improve retrieval.
        
        Args:
            query: Original query
            
        Returns:
            Reformulated query
        """
        query_lower = query.lower()
        
        # Check for prevention questions
        for term in self.prevention_terms:
            if term in query_lower:
                return f"prevention risk factors modifiable lifestyle changes {query}"
        
        # Emphasized disease search - always prioritize disease-specific queries
        diseases = ["heart failure", "hfref", "hfpef", "hfmref", "stroke", "hemorrhagic stroke", 
                    "ischemic stroke", "myocardial infarction", "nstemi", "stemi", "acs", 
                    "acute coronary syndrome", "heart disease", "heart problem", "cardiac"]
        
        for disease in diseases:
            if disease in query_lower:
                # Format query to emphasize disease
                if "symptom" in query_lower or "sign" in query_lower:
                    return f"{disease} symptoms signs clinical manifestations {query}"
                elif "diagnos" in query_lower:
                    return f"{disease} diagnosis diagnostic criteria assessment {query}"
                elif "risk factor" in query_lower:
                    return f"{disease} risk factors etiology causes {query}"
                elif "treatment" in query_lower or "management" in query_lower:
                    return f"{disease} treatment management therapy {query}"
                elif "difference" in query_lower or "compare" in query_lower:
                    return f"difference comparison between {query}"
                elif "what is" in query_lower or "definition" in query_lower:
                    return f"{disease} definition description overview {query}"
                else:
                    return f"{disease} information {query}"
        
        # General clinical category reformulation
        for category in self.clinical_categories:
            if category in query_lower:
                if category in ["symptom", "symptoms", "sign", "signs"]:
                    return f"{query} clinical manifestations presentation"
                elif category in ["risk factor", "risk factors"]:
                    return f"{query} predisposing factors causes etiology"
                elif "diagnos" in category:
                    return f"{query} diagnostic criteria clinical assessment tests"
                elif category in ["treatment", "management"]:
                    return f"{query} therapy intervention management approach"
        
        # Default reformulation for medical queries
        if any(word in query_lower for word in ["what", "how", "why", "when", "where", "which", "who"]):
            return f"{query} (medical context, clinical information)"
        
        return query

# --- DIRECT ANSWER EXTRACTOR ---

class DirectAnswerExtractor:
    """
    Class to extract direct answers from retrieved documents when the LLM fails
    """
    
    def extract_answer(self, question, documents):
        """
        Extract a direct answer from retrieved documents when possible.
        
        Args:
            question: User's question
            documents: Retrieved documents
            
        Returns:
            Direct answer if possible, otherwise None
        """
        question_lower = question.lower()
        
        # Check for symptoms questions
        if "symptom" in question_lower or "sign" in question_lower:
            # Look for symptom documents
            for doc in documents:
                meta = doc.metadata
                if meta.get("source") == "knowledge_graph" and meta.get("category", "").lower() in ["symptoms", "signs"]:
                    content = doc.page_content
                    # Extract the part after "following symptoms:" or "this symptoms:"
                    match = re.search(r"following symptoms?:(.+?)(?:$|\.)", content)
                    if match:
                        return match.group(1).strip()
                    match = re.search(r"this symptoms?:(.+?)(?:$|\.)", content)
                    if match:
                        return match.group(1).strip()
                    match = re.search(r"has symptoms?:(.+?)(?:$|\.)", content)
                    if match:
                        return match.group(1).strip()
        
        # Check for risk factors questions
        if "risk factor" in question_lower or "cause" in question_lower:
            for doc in documents:
                meta = doc.metadata
                if meta.get("source") == "knowledge_graph" and meta.get("category", "").lower() == "risk factors":
                    content = doc.page_content
                    match = re.search(r"following risk factors:(.+?)(?:$|\.)", content)
                    if match:
                        return match.group(1).strip()
                    match = re.search(r"this risk factors:(.+?)(?:$|\.)", content)
                    if match:
                        return match.group(1).strip()
                    match = re.search(r"has risk factors:(.+?)(?:$|\.)", content)
                    if match:
                        return match.group(1).strip()
        
        # Check for "what is" questions
        if "what is" in question_lower or "what are" in question_lower or "definition" in question_lower:
            for doc in documents:
                if doc.metadata.get("source") == "knowledge_graph":
                    content = doc.page_content
                    if "characteristics:" in content:
                        match = re.search(r"characteristics:(.+?)(?:$|\.)", content)
                        if match:
                            return match.group(1).strip()
                    if "is a diagnostic entity" in content:
                        # For disease definitions, combine with another document if possible
                        disease = doc.metadata.get("diagnosis", "")
                        for other_doc in documents:
                            if other_doc.metadata.get("diagnosis") == disease and "characteristics:" in other_doc.page_content:
                                match = re.search(r"characteristics:(.+?)(?:$|\.)", other_doc.page_content)
                                if match:
                                    return f"{disease} is a medical condition. {match.group(1).strip()}"
                        return f"{disease} is a medical condition in the {doc.metadata.get('disease_category', '')} category."
        
        # Prevention/avoidance questions
        if any(term in question_lower for term in ["prevent", "avoid", "reduce risk"]):
            for doc in documents:
                meta = doc.metadata
                if meta.get("source") == "knowledge_graph" and meta.get("category", "").lower() == "risk factors":
                    risk_factors = doc.page_content
                    match = re.search(r"risk factors:(.+?)(?:$|\.)", risk_factors)
                    if match:
                        factors = match.group(1).strip()
                        return f"To reduce risk, address these modifiable factors: {factors} Lifestyle changes include healthy diet, regular exercise, avoiding smoking, limiting alcohol consumption, and managing stress."
        
        # No direct answer extraction was possible
        return None

# --- CLINICAL RAG SYSTEM ---

class ClinicalRAG:
    """Complete RAG system for clinical question answering"""
    
    def __init__(self, samples_path, kg_path):
        """
        Initialize the RAG system.
        
        Args:
            samples_path: Path to annotated samples directory
            kg_path: Path to knowledge graphs directory
        """
        self.samples_path = samples_path
        self.kg_path = kg_path
        self.vector_store = None
        self.llm = None
        self.qa_chain = None
        self.reformulator = QueryReformulator()
        self.answer_extractor = DirectAnswerExtractor()
        
        # Check paths
        if not os.path.exists(samples_path):
            st.warning(f"Samples path does not exist: {samples_path}")
        if not os.path.exists(kg_path):
            st.warning(f"Knowledge graph path does not exist: {kg_path}")
    
    def explore_dataset(self):
        """
        Explore dataset structure and generate statistics.
        
        Returns:
            Statistics dictionary
        """
        stats = {
            "disease_categories": [],
            "pdds": {},
            "files": {},
            "total_files": 0
        }
        
        # Process samples directory
        if os.path.exists(self.samples_path):
            for disease_category in os.listdir(self.samples_path):
                disease_path = os.path.join(self.samples_path, disease_category)
                if os.path.isdir(disease_path):
                    stats["disease_categories"].append(disease_category)
                    stats["pdds"][disease_category] = []
                    stats["files"][disease_category] = {}
                    
                    for pdd in os.listdir(disease_path):
                        pdd_path = os.path.join(disease_path, pdd)
                        if os.path.isdir(pdd_path):
                            stats["pdds"][disease_category].append(pdd)
                            file_count = len([f for f in os.listdir(pdd_path) if f.endswith('.json')])
                            stats["files"][disease_category][pdd] = file_count
                            stats["total_files"] += file_count
        
        # Process knowledge graphs directory
        stats["knowledge_graphs"] = []
        if os.path.exists(self.kg_path):
            for file in os.listdir(self.kg_path):
                if file.endswith('.json'):
                    stats["knowledge_graphs"].append(file.replace('.json', ''))
        
        return stats
    
    def process_dataset(self, max_files=None, progress_bar=None):
        """
        Process the entire dataset and create documents.
        
        Args:
            max_files: Optional limit on number of files to process
            progress_bar: Optional streamlit progress bar
            
        Returns:
            List of Document objects
        """
        documents = []
        
        # 1. Process knowledge graphs FIRST - they're critical for answering general questions
        if os.path.exists(self.kg_path):
            kg_files = [f for f in os.listdir(self.kg_path) if f.endswith('.json')]
            
            if progress_bar:
                progress_bar.progress(0.1, text="Processing knowledge graphs...")
            
            for i, file in enumerate(kg_files):
                kg_path = os.path.join(self.kg_path, file)
                kg_docs = load_knowledge_graph(kg_path)
                documents.extend(kg_docs)
                
                if progress_bar:
                    progress = 0.1 + 0.4 * ((i + 1) / len(kg_files))
                    progress_bar.progress(progress, text=f"Processing knowledge graph: {file}")
        
        # 2. Process annotated clinical notes
        file_count = 0
        
        if os.path.exists(self.samples_path):
            disease_categories = os.listdir(self.samples_path)
            
            for d_idx, disease_category in enumerate(disease_categories):
                disease_path = os.path.join(self.samples_path, disease_category)
                if os.path.isdir(disease_path):
                    pdds = os.listdir(disease_path)
                    
                    for p_idx, pdd in enumerate(pdds):
                        pdd_path = os.path.join(self.samples_path, disease_category, pdd)
                        if os.path.isdir(pdd_path):
                            files = [f for f in os.listdir(pdd_path) if f.endswith('.json')]
                            
                            for f_idx, file in enumerate(files):
                                file_path = os.path.join(pdd_path, file)
                                
                                if progress_bar:
                                    total_progress = 0.5 + 0.5 * (
                                        (d_idx / len(disease_categories)) + 
                                        (p_idx / (len(disease_categories) * len(pdds))) +
                                        (f_idx / (len(disease_categories) * len(pdds) * len(files)))
                                    )
                                    progress_bar.progress(min(0.99, total_progress), text=f"Processing {file}...")
                                
                                # Process clinical note
                                try:
                                    record_node, input_content, chain = cal_a_json(file_path)
                                    
                                    # Add raw clinical notes to documents
                                    for input_key, input_text in input_content.items():
                                        if input_text and len(str(input_text).strip()) > 0:
                                            documents.append(Document(
                                                page_content=f"{input_key}: {input_text}",
                                                metadata={
                                                    "source": "clinical_note",
                                                    "disease_category": disease_category,
                                                    "pdd": pdd,
                                                    "file": file,
                                                    "input_type": input_key
                                                }
                                            ))
                                    
                                    # Process deductions and add to documents
                                    deductions = deduction_assemble(record_node)
                                    for obs, values in deductions.items():
                                        if len(values) >= 3 and obs and values[0]:
                                            rationale, source, diagnosis = values
                                            doc_text = f"Clinical Observation: {obs}\nRationale: {rationale}\nSource: {source}\nDiagnosis: {diagnosis}"
                                            documents.append(Document(
                                                page_content=doc_text,
                                                metadata={
                                                    "source": "deduction",
                                                    "disease_category": disease_category,
                                                    "pdd": pdd,
                                                    "file": file,
                                                    "diagnosis": diagnosis
                                                }
                                            ))
                                    
                                    file_count += 1
                                    if max_files and file_count >= max_files:
                                        break
                                except Exception as e:
                                    st.error(f"Error processing {file}: {e}")
                            
                            if max_files and file_count >= max_files:
                                break
                    
                    if max_files and file_count >= max_files:
                        break
        
        # Verify we have knowledge graph documents
        kg_docs = [doc for doc in documents if doc.metadata.get("source") == "knowledge_graph"]
        if not kg_docs:
            st.warning("WARNING: No knowledge graph documents were processed!")
            
        if progress_bar:
            progress_bar.progress(1.0, text="Processing complete!")
            
        return documents
    
    def build_vector_store(self, progress_bar=None):
        """
        Build the vector store from processed documents.
        
        Args:
            progress_bar: Optional streamlit progress bar
            
        Returns:
            FAISS vector store
        """
        # Process the dataset
        documents = self.process_dataset(progress_bar=progress_bar)
        
        if not documents:
            st.error("No documents to process!")
            return None
        
        # Create chunks for embedding
        if progress_bar:
            progress_bar.progress(0.7, text="Chunking documents...")
            
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=512,
            chunk_overlap=100
        )
        chunks = text_splitter.split_documents(documents)
        
        # Handle empty chunks case
        if not chunks:
            st.warning("WARNING: No chunks created. Adding fallback content.")
            chunks = [Document(
                page_content="Heart Failure symptoms include shortness of breath, fatigue, and edema.",
                metadata={"source": "fallback"}
            )]
        
        # Create embeddings and vector store
        if progress_bar:
            progress_bar.progress(0.8, text="Creating embeddings...")
            
        embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        if progress_bar:
            progress_bar.progress(0.9, text="Creating vector store...")
            
        self.vector_store = FAISS.from_documents(chunks, embedding)
        
        # Save vector store
        if progress_bar:
            progress_bar.progress(0.95, text="Saving vector store...")
            
        os.makedirs("vector_db", exist_ok=True)
        self.vector_store.save_local("vector_db/direct_medical_index")
        
        if progress_bar:
            progress_bar.progress(1.0, text="Vector store built successfully!")
            
        return self.vector_store
    
    def load_vector_store(self, path="vector_db/direct_medical_index", progress_bar=None):
        """
        Load a saved vector store.
        
        Args:
            path: Path to saved vector store
            progress_bar: Optional streamlit progress bar
            
        Returns:
            FAISS vector store
        """
        if not os.path.exists(path):
            st.warning(f"Vector store path {path} doesn't exist")
            return None
        
        try:
            if progress_bar:
                progress_bar.progress(0.5, text=f"Loading vector store...")
                
            embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            self.vector_store = FAISS.load_local(
                path, 
                embeddings=embedding,
                allow_dangerous_deserialization=True  # Required for loading saved vector stores
            )
            
            if progress_bar:
                progress_bar.progress(1.0, text="Vector store loaded successfully!")
                
            return self.vector_store
        except Exception as e:
            st.error(f"Error loading vector store: {e}")
            return None
    
    def setup_llm(self):
        """
        Set up the language model for rule-based answering
        
        Returns:
            Simple rule-based function
        """
        # Since we're not using an actual LLM, set up a simpler rule-based mechanism
        self.llm = "rule-based"
        return self.llm
    
    def setup_qa_chain(self):
        """
        Set up the question-answering chain.
        
        Returns:
            QA chain
        """
        if not self.vector_store:
            st.warning("Vector store not initialized")
            return None
        
        # Create a simple rule-based QA system
        self.qa_chain = "rule-based"
        return self.qa_chain
    
    def answer_question(self, question):
        """
        Answer a clinical question.
        
        Args:
            question: User question
            
        Returns:
            Dictionary with answer and sources
        """
        start_time = time.time()
        
        # Check if system is initialized
        if not self.vector_store:
            return {
                "question": question,
                "answer": "System not initialized. Please build or load vector store first.",
                "sources": [],
                "time": time.time() - start_time
            }
        
        # Reformulate query for better retrieval
        reformulated = self.reformulator.reformulate(question)
        
        # Retrieve documents 
        sources = self.vector_store.similarity_search(reformulated, k=5)
        
        # Try direct answer extraction first (our main mechanism)
        direct_answer = self.answer_extractor.extract_answer(question, sources)
        if direct_answer:
            return {
                "question": question,
                "reformulated": reformulated,
                "answer": direct_answer,
                "sources": sources,
                "time": time.time() - start_time
            }
        
        # If no direct answer, concatenate information from sources
        answer_parts = []
        for doc in sources[:3]:  # Use top 3 sources
            if doc.metadata.get("source") == "knowledge_graph":
                # Extract the relevant part of the content
                content = doc.page_content
                
                # Remove the "In Category, " prefix if present
                content = re.sub(r"^In [^,]+, ", "", content)
                
                # Add to answer parts
                answer_parts.append(content)
        
        # Combine answer parts
        if answer_parts:
            answer = "\n".join(answer_parts)
        else:
            answer = "I don't have enough information to answer this question fully."
        
        return {
            "question": question,
            "reformulated": reformulated,
            "answer": answer,
            "sources": sources,
            "time": time.time() - start_time
        }

    def ethics_report(self):
        """Generate an ethics report."""
        report = """
# Ethics Report: Clinical RAG System

## Data Privacy Considerations
- All data used is de-identified as part of the MIMIC-IV-Ext collection
- No attempt is made to re-identify patients
- The system does not store user queries that could contain PHI

## Clinical Limitations
- This system is for research/educational purposes only
- Generated responses should not replace clinical judgment
- The system may contain biases inherent in the training data
- Results should be verified against current medical literature

## Fairness and Bias
- Care has been taken to include diverse medical conditions
- Technical limitations may impact accuracy for rare conditions
- Response quality may vary across different demographic groups

## Transparency
- Source documents are displayed alongside generated answers
- System limitations are explicitly documented
- Users are informed that this is an AI system without medical licensing
"""
        return report


# --- STREAMLIT APPLICATION ---

# Initialize session state variables
if 'rag_system' not in st.session_state:
    st.session_state.rag_system = None
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
if 'stats' not in st.session_state:
    st.session_state.stats = None
if 'kg_preview' not in st.session_state:
    st.session_state.kg_preview = None

# Main title
st.markdown("<div class='main-title'>DiReCT Medical RAG System</div>", unsafe_allow_html=True)

# Main app tabs
tab1, tab2, tab3, tab4 = st.tabs(["💬 Ask Questions", "🛠️ System Setup", "📊 Dataset Explorer", "ℹ️ About"])

# Tab 1: Ask Questions
with tab1:
    if st.session_state.initialized:
        st.write("## Ask a Medical Question")
        st.write("Enter your medical question below to get information from the DiReCT dataset.")
        
        # Example questions
        with st.expander("Example Questions"):
            st.markdown("""
            - What are the symptoms of Heart Failure?
            - How is NSTEMI diagnosed?
            - What are the risk factors for hemorrhagic stroke?
            - What is the difference between HFrEF and HFpEF?
            - What ECG findings are associated with myocardial infarction?
            """)
        
        # Query input
        query = st.text_input("Your question:", key="query_input")
        
        if st.button("Submit"):
            if query:
                with st.spinner("Processing your question..."):
                    # Format the query for display
                    st.subheader("Your Query:")
                    st.markdown(f"**{query}**")
                    
                    # Show reformulated query
                    reformulated = st.session_state.rag_system.reformulator.reformulate(query)
                    st.markdown("**Reformulated query:**")
                    st.markdown(f"_{reformulated}_")
                    
                    # Get answer
                    start_time = time.time()
                    result = st.session_state.rag_system.answer_question(query)
                    response_time = time.time() - start_time
                    
                    # Display answer
                    st.subheader("Answer:")
                    st.markdown(result["answer"])
                    
                    # Display response time
                    st.markdown(f"<div class='response-time'>Response time: {response_time:.2f} seconds</div>", unsafe_allow_html=True)
                    
                    # Display sources
                    st.subheader("Sources:")
                    for i, doc in enumerate(result["sources"][:3]):  # Show top 3 sources
                        source_type = doc.metadata.get("source", "Unknown")
                        category = doc.metadata.get("disease_category", doc.metadata.get("disease", "Unknown"))
                        
                        with st.container():
                            st.markdown(f"<div class='source-box'>", unsafe_allow_html=True)
                            st.markdown(f"<div class='source-title'>Source {i+1}: {source_type} - {category}</div>", unsafe_allow_html=True)
                            
                            if "category" in doc.metadata:
                                st.markdown(f"**Category:** {doc.metadata['category']}")
                            
                            # Show content
                            st.markdown("**Content:**")
                            st.text(doc.page_content)
                            st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.warning("Please enter a question.")
    else:
        st.info("Please initialize the system first in the 'System Setup' tab.")
        
# Tab 2: System Setup
with tab2:
    st.subheader("Setup the DiReCT Medical RAG System")
    
    # Data paths
    kg_path = st.text_input("Knowledge Graph Path", 
                           value="Diagnosis_flowchart",
                           help="Path to the folder containing knowledge graph JSON files")
    
    samples_path = st.text_input("Clinical Notes Path", 
                                value="Finished",
                                help="Path to the folder containing annotated clinical notes")
    
    # Initialize system button
    init_col1, init_col2 = st.columns([1, 1])
    
    with init_col1:
        if st.button("Initialize New System"):
            try:
                # Create progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                progress_bar.progress(0, text="Creating RAG System...")
                
                # Initialize the system
                st.session_state.rag_system = ClinicalRAG(samples_path, kg_path)
                
                # Build vector store
                status_text.text("Building vector store (this may take a while)...")
                vector_store = st.session_state.rag_system.build_vector_store(progress_bar)
                
                # Setup LLM
                status_text.text("Setting up answering system...")
                progress_bar.progress(0.95)
                st.session_state.rag_system.setup_llm()
                
                # Set system as initialized
                st.session_state.initialized = True
                
                # Get stats
                st.session_state.stats = st.session_state.rag_system.explore_dataset()
                
                # Complete
                progress_bar.progress(1.0, text="System initialization complete!")
                status_text.text("System ready!")
                
                st.success("RAG System initialized successfully!")
                
            except Exception as e:
                st.error(f"Failed to initialize RAG System: {str(e)}")
    
    with init_col2:
        if st.button("Load Existing System"):
            try:
                # Create progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                progress_bar.progress(0, text="Creating RAG System...")
                
                # Initialize the system
                st.session_state.rag_system = ClinicalRAG(samples_path, kg_path)
                
                # Load vector store
                status_text.text("Loading vector store...")
                vector_store_path = "vector_db/direct_medical_index"
                vector_store = st.session_state.rag_system.load_vector_store(vector_store_path, progress_bar)
                
                if vector_store:
                    # Setup LLM
                    status_text.text("Setting up answering system...")
                    progress_bar.progress(0.9)
                    st.session_state.rag_system.setup_llm()
                    
                    # Set system as initialized
                    st.session_state.initialized = True
                    
                    # Get stats
                    st.session_state.stats = st.session_state.rag_system.explore_dataset()
                    
                    # Complete
                    progress_bar.progress(1.0, text="System initialization complete!")
                    status_text.text("System ready!")
                    
                    st.success("RAG System loaded successfully!")
                else:
                    st.error("Failed to load vector store. Please initialize a new system.")
                
            except Exception as e:
                st.error(f"Failed to load RAG System: {str(e)}")
    
    # Display system status
    st.subheader("System Status")
    if st.session_state.initialized:
        st.markdown("✅ **RAG System is initialized and ready to use**")
        
        # Display vector store statistics if available
        if hasattr(st.session_state.rag_system, 'vector_store') and st.session_state.rag_system.vector_store:
            vector_store_size = len(st.session_state.rag_system.vector_store.index_to_docstore_id)
            st.markdown(f"📑 Vector store contains **{vector_store_size}** document chunks")
    else:
        st.markdown("❌ **RAG System is not initialized**")

# Tab 3: Dataset Explorer
with tab3:
    st.subheader("DiReCT Dataset Explorer")
    
    if st.session_state.stats:
        # Display dataset statistics
        stats = st.session_state.stats
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Disease Categories", len(stats["disease_categories"]))
        with col2:
            st.metric("Knowledge Graphs", len(stats["knowledge_graphs"]))
        with col3:
            st.metric("Total Clinical Notes", stats["total_files"])
        
        # Display disease categories
        st.subheader("Disease Categories")
        if stats["disease_categories"]:
            disease_categories = stats["disease_categories"]
            selected_category = st.selectbox("Select a disease category", disease_categories)
            
            if selected_category in stats["pdds"]:
                st.write(f"**PDDs in {selected_category}:**")
                pdds = stats["pdds"][selected_category]
                st.write(", ".join(pdds))
                
                # Display file counts
                if selected_category in stats["files"]:
                    files_dict = stats["files"][selected_category]
                    file_counts = {pdd: files_dict.get(pdd, 0) for pdd in pdds}
                    
                    # Create bar chart
                    df = pd.DataFrame(list(file_counts.items()), columns=['PDD', 'File Count'])
                    st.bar_chart(df.set_index('PDD'))
        
        # Knowledge Graph Explorer
        st.subheader("Knowledge Graph Explorer")
        kg_select = st.selectbox("Select a knowledge graph", stats["knowledge_graphs"])
        
        if kg_select:
            kg_path = os.path.join(st.session_state.rag_system.kg_path, f"{kg_select}.json")
            
            try:
                with open(kg_path, 'r', encoding='utf-8') as f:
                    kg_data = json.load(f)
                    
                # Cache the KG data
                st.session_state.kg_preview = kg_data
                
                # Display KG data
                kg_tab1, kg_tab2 = st.tabs(["Knowledge", "Diagnostic Structure"])
                
                with kg_tab1:
                    if "knowledge" in kg_data:
                        knowledge_data = kg_data["knowledge"]
                        
                        # Select a diagnosis
                        diagnosis = st.selectbox("Select diagnosis", list(knowledge_data.keys()))
                        
                        if diagnosis and diagnosis in knowledge_data:
                            diag_data = knowledge_data[diagnosis]
                            
                            if isinstance(diag_data, dict):
                                # Display categories (symptoms, risk factors, etc.)
                                for category, details in diag_data.items():
                                    with st.expander(f"{category}"):
                                        if isinstance(details, str) and ";" in details:
                                            # Split by semicolon and show as bullet points
                                            items = [item.strip() for item in details.split(";")]
                                            for item in items:
                                                if item and not item.endswith("etc."):
                                                    st.markdown(f"- {item}")
                                        else:
                                            st.write(details)
                            else:
                                st.write(diag_data)
                    else:
                        st.warning("No knowledge data found in this knowledge graph.")
                
                with kg_tab2:
                    if "diagnostic" in kg_data:
                        diagnostic_data = kg_data["diagnostic"]
                        st.json(diagnostic_data)
                    else:
                        st.warning("No diagnostic structure found in this knowledge graph.")
                
            except Exception as e:
                st.error(f"Error loading knowledge graph: {e}")
    else:
        st.info("Please initialize the system first to explore the dataset.")

# Tab 4: About
with tab4:
    st.subheader("About DiReCT Medical RAG System")
    
    st.markdown("""
    This application provides a Retrieval-Augmented Generation (RAG) system for medical question answering
    based on the MIMIC-IV-Ext-DiReCT dataset.
    
    ### Features
    
    - **Question Answering**: Ask medical questions about heart failure, stroke, and other conditions
    - **Knowledge Graph Access**: Explore structured medical knowledge from the DiReCT dataset
    - **Clinical Notes**: Access relevant information from annotated clinical notes
    
    ### Data Source
    
    The DiReCT dataset is organized into two main components:
    
    1. **Knowledge Graphs**: JSON files containing structured medical knowledge
    2. **Annotated Clinical Notes**: 511 annotated notes organized by disease categories
    
    ### About DiReCT
    
    DiReCT (Diagnostic Research in Clinical Text) is a publicly available dataset for medical diagnostic reasoning.
    It contains structured knowledge graphs and annotated clinical notes.
    
    For more information, visit: [DiReCT GitHub Repository](https://github.com/wbw520/DiReCT)
    """)
    
    st.subheader("Ethics Report")
    st.markdown(ClinicalRAG("", "").ethics_report())

# Footer
st.markdown("---")
st.markdown("DiReCT Medical RAG System • Built with Streamlit")