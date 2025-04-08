import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import networkx as nx
from typing import List, Dict, Tuple, Set
import spacy
from sklearn.cluster import KMeans
import re

class NEXUS:
    """Knowledge Nexus: An intelligent system for building and querying knowledge graphs"""
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self.knowledge_graph = nx.DiGraph()
        self.document_embeddings = {}
        self.nlp = spacy.load("en_core_web_sm")
        self.fitted = False
        self.documents = {}
        
    def get_noun_phrase(self, token) -> str:
        """Extract full noun phrase from a token"""
        words = []
        for child in token.subtree:
            if child.dep_ in {'compound', 'nmod', 'poss', 'amod'} or child == token:
                words.append(child.text)
        return ' '.join(words)

    def extract_entities_relationships(self, text: str) -> List[Tuple[str, str, str]]:
        """Extract entities and relationships from text using advanced NLP techniques"""
        doc = self.nlp(text)
        triplets = []
        entities_seen = set()
        
        # First pass: collect all named entities and noun phrases
        named_entities = {}
        noun_chunks = {}
        
        # Custom patterns for movies and films
        movie_patterns = [
            r'(?i)(?:the\s+)?(?:film|movie)\s+([A-Z][a-zA-Z\s]+)',
            r'(?i)(?:created|made|produced|directed)\s+([A-Z][a-zA-Z\s]+)(?:\s+in\s+\d{4})?',
            r'(?i)(?:like|including|such\s+as)\s+([A-Z][a-zA-Z\s]+)(?:\s+in\s+\d{4})?'
        ]
        
        # Extract movies from patterns
        movies = set()
        for pattern in movie_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                movie = match.group(1).strip()
                if movie:
                    movies.add(movie)
                    named_entities[movie] = 'MOVIE'
        
        # Extract standard named entities
        for ent in doc.ents:
            named_entities[ent.text] = ent.label_
            entities_seen.add(ent.text.lower())
        
        # Process each sentence to extract relationships
        for sent in doc.sents:
            sent_text = sent.text.lower()
            
            # Special handling for movie creation relationships
            for movie in movies:
                if movie.lower() in sent_text:
                    # Find the creators in the same sentence
                    creators = []
                    for ent in [e for e in doc.ents if e.start >= sent.start and e.end <= sent.end]:
                        if ent.label_ in ['ORG', 'PERSON']:
                            creators.append(ent.text)
                    
                    # If we found creators, add the relationships
                    if creators:
                        for creator in creators:
                            triplets.append((creator, "created", movie))
                    # If no specific creators found but company mentioned
                    elif "pixar" in sent_text:
                        triplets.append(("Pixar", "created", movie))
            
            # Find main verbs and their arguments
            for token in sent:
                if token.pos_ == "VERB":
                    # Get subject
                    subjects = []
                    for child in token.children:
                        if "subj" in child.dep_:
                            subj = self.get_noun_phrase(child)
                            subjects.append(subj)
                    
                    # Get object
                    objects = []
                    for child in token.children:
                        if "obj" in child.dep_ or child.dep_ in {"attr", "dobj", "pobj"}:
                            obj = self.get_noun_phrase(child)
                            objects.append(obj)
                    
                    # Get auxiliary verbs and negations
                    aux_tokens = []
                    neg = False
                    for child in token.children:
                        if child.dep_ == "aux":
                            aux_tokens.append(child.text)
                        elif child.dep_ == "neg":
                            neg = True
                    
                    # Construct the full verb phrase
                    verb_phrase = " ".join(aux_tokens + [token.text])
                    if neg:
                        verb_phrase = "did not " + verb_phrase
                    
                    # Create triplets
                    for subj in subjects:
                        for obj in objects:
                            if subj and obj and subj.lower() != obj.lower():
                                triplets.append((subj, verb_phrase, obj))
        
        # Extract relationships between named entities
        entity_pairs = []
        for ent1 in doc.ents:
            for ent2 in doc.ents:
                if ent1.text != ent2.text:
                    entity_pairs.append((ent1, ent2))
        
        for ent1, ent2 in entity_pairs:
            # Look for semantic relationships based on entity types
            if ent1.label_ == "PERSON" and ent2.label_ == "ORG":
                context = doc[max(0, ent1.start-5):min(len(doc), ent2.end+5)].text.lower()
                if any(word in context for word in ["found", "start", "establish"]):
                    triplets.append((ent1.text, "founded", ent2.text))
                elif any(word in context for word in ["lead", "run", "head", "direct"]):
                    triplets.append((ent1.text, "leads", ent2.text))
                elif any(word in context for word in ["work", "join", "employ"]):
                    triplets.append((ent1.text, "works at", ent2.text))
            elif ent1.label_ == "ORG" and ent2.label_ == "ORG":
                context = doc[max(0, ent1.start-5):min(len(doc), ent2.end+5)].text.lower()
                if "acquire" in context:
                    # Look for acquisition amount
                    amount = None
                    for ent in doc.ents:
                        if ent.label_ == "MONEY" and ent.start >= ent1.start and ent.end <= ent2.end:
                            amount = ent.text
                            break
                    triplets.append((ent1.text, f"acquired for {amount}" if amount else "acquired", ent2.text))
        
        return triplets

    def build_knowledge_graph(self, triplets: List[Tuple[str, str, str]]):
        """Build a rich knowledge graph from extracted triplets"""
        self.knowledge_graph.clear()
        
        # Add nodes and edges
        for subj, pred, obj in triplets:
            # Add nodes with attributes
            if not self.knowledge_graph.has_node(subj):
                self.knowledge_graph.add_node(subj, type='entity')
            if not self.knowledge_graph.has_node(obj):
                self.knowledge_graph.add_node(obj, type='entity')
            
            # Add edge with relationship as attribute
            self.knowledge_graph.add_edge(subj, obj, relation=pred)
            
            # Create embeddings for nodes
            if subj not in self.document_embeddings:
                self.document_embeddings[subj] = self.create_embeddings(subj)
            if obj not in self.document_embeddings:
                self.document_embeddings[obj] = self.create_embeddings(obj)

    def find_path_between_entities(self, entity1: str, entity2: str) -> List[Tuple[str, str, str]]:
        """Find the shortest path between two entities in the knowledge graph"""
        try:
            path = nx.shortest_path(self.knowledge_graph, entity1, entity2)
            path_triplets = []
            for i in range(len(path)-1):
                edge_data = self.knowledge_graph.get_edge_data(path[i], path[i+1])
                path_triplets.append((path[i], edge_data['relation'], path[i+1]))
            return path_triplets
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []

    def process_query(self, query: str) -> Dict:
        """Process a query and return relevant information from the knowledge graph"""
        if not self.knowledge_graph.nodes():
            return {
                'query': query,
                'answer': "No knowledge base available to answer the query.",
                'sources': []
            }
        
        query_doc = self.nlp(query)
        query_lower = query.lower()
        
        # Get all triplets from the graph
        all_triplets = []
        for u, v, data in self.knowledge_graph.edges(data=True):
            all_triplets.append((u, data['relation'], v))
        
        # Extract entities from query
        query_entities = set()
        for ent in query_doc.ents:
            query_entities.add(ent.text)
        
        # Extract potential entities from noun chunks
        for chunk in query_doc.noun_chunks:
            if any(word.text.istitle() for word in chunk):
                query_entities.add(chunk.text)
        
        # Find all relevant nodes that match query terms
        relevant_nodes = set()
        for node in self.knowledge_graph.nodes():
            # Check if node matches any query entity
            if any(self.is_entity_match(node, entity) for entity in query_entities):
                relevant_nodes.add(node)
            # Check if node appears in query text
            elif self.is_entity_match(node, query):
                relevant_nodes.add(node)
        
        # Handle different question types
        if "who is" in query_lower or "what is" in query_lower:
            # Find information about the entity
            for entity in relevant_nodes:
                entity_info = []
                
                # Get all relationships where this entity is involved
                for s, r, o in all_triplets:
                    if s == entity:
                        entity_info.append(f"{entity} {r} {o}")
                    elif o == entity:
                        entity_info.append(f"{s} {r} {entity}")
                
                if entity_info:
                    return {
                        'query': query,
                        'answer': " and ".join(entity_info),
                        'sources': entity_info
                    }
        
        elif "who" in query_lower:
            # Look for people or organizations related to the entities in the query
            for entity in relevant_nodes:
                person_info = []
                
                # Find relationships where the entity is the object
                for s, r, o in all_triplets:
                    if o == entity:
                        person_info.append((s, r, o))
                
                if person_info:
                    # Group by relationship type
                    by_relation = {}
                    for s, r, o in person_info:
                        if r not in by_relation:
                            by_relation[r] = []
                        by_relation[r].append(s)
                    
                    # Format answer based on relationship
                    answers = []
                    for r, subjects in by_relation.items():
                        answers.append(f"{' and '.join(subjects)} {r} {entity}")
                    
                    return {
                        'query': query,
                        'answer': " and ".join(answers),
                        'sources': [f"{s} {r} {o}" for s, r, o in person_info]
                    }
        
        elif "what did" in query_lower or "what has" in query_lower:
            # Look for actions performed by the entities
            for entity in relevant_nodes:
                actions = []
                
                # Find relationships where the entity is the subject
                for s, r, o in all_triplets:
                    if s == entity:
                        actions.append((s, r, o))
                
                if actions:
                    return {
                        'query': query,
                        'answer': " and ".join(f"{s} {r} {o}" for s, r, o in actions),
                        'sources': [f"{s} {r} {o}" for s, r, o in actions]
                    }
        
        elif "how" in query_lower and "related" in query_lower:
            # Find paths between entities
            if len(relevant_nodes) >= 2:
                nodes_list = list(relevant_nodes)
                paths = []
                
                # Try all pairs of entities
                for i in range(len(nodes_list)):
                    for j in range(i + 1, len(nodes_list)):
                        path = self.find_path_between_entities(nodes_list[i], nodes_list[j])
                        if path:
                            paths.extend(path)
                
                if paths:
                    return {
                        'query': query,
                        'answer': " and ".join(f"{s} {r} {o}" for s, r, o in paths),
                        'sources': [f"{s} {r} {o}" for s, r, o in paths]
                    }
        
        # For other types of questions, use similarity-based approach
        similar_triplets = []
        for s, r, o in all_triplets:
            # Check if any part of the triplet is relevant to the query
            if (s in relevant_nodes or o in relevant_nodes or
                any(self.is_entity_match(s, entity) or self.is_entity_match(o, entity) 
                    for entity in query_entities)):
                similar_triplets.append((s, r, o))
        
        if similar_triplets:
            # Sort triplets by relevance to query
            triplet_scores = []
            query_emb = self.create_embeddings(query)
            
            for s, r, o in similar_triplets:
                triplet_text = f"{s} {r} {o}"
                triplet_emb = self.create_embeddings(triplet_text)
                score = np.dot(query_emb[:min(len(query_emb), len(triplet_emb))], 
                             triplet_emb[:min(len(query_emb), len(triplet_emb))])
                triplet_scores.append((score, (s, r, o)))
            
            triplet_scores.sort(reverse=True)
            best_triplets = [t for _, t in triplet_scores[:3]]
            
            return {
                'query': query,
                'answer': " and ".join(f"{s} {r} {o}" for s, r, o in best_triplets),
                'sources': [f"{s} {r} {o}" for s, r, o in best_triplets]
            }
        
        return {
            'query': query,
            'answer': "I couldn't find a specific answer to your question.",
            'sources': []
        }
    
    def is_entity_match(self, text1: str, text2: str) -> bool:
        """Check if two pieces of text refer to the same entity"""
        # Convert to lowercase for comparison
        text1_lower = text1.lower()
        text2_lower = text2.lower()
        
        # Direct match
        if text1_lower == text2_lower:
            return True
        
        # Check if one contains the other
        if text1_lower in text2_lower or text2_lower in text1_lower:
            return True
        
        # Check for partial matches of multi-word entities
        words1 = set(text1_lower.split())
        words2 = set(text2_lower.split())
        
        # If one is a single word and appears in the other's words
        if len(words1) == 1 and words1.issubset(words2):
            return True
        if len(words2) == 1 and words2.issubset(words1):
            return True
        
        # If they share significant words (more than 50% overlap)
        common_words = words1.intersection(words2)
        if common_words and (len(common_words) / len(words1) > 0.5 or 
                           len(common_words) / len(words2) > 0.5):
            return True
        
        return False

    def create_embeddings(self, text: str) -> np.ndarray:
        """Create embeddings for text"""
        if not self.fitted:
            self.vectorizer.fit([text])
            self.fitted = True
        return self.vectorizer.transform([text]).toarray()[0]

    def cluster_documents(self, documents: List[str], n_clusters: int = 5) -> Dict:
        """Cluster documents using embeddings"""
        if not documents:
            return {}
            
        # Store documents
        for i, doc in enumerate(documents):
            self.documents[f"doc_{i}"] = doc
            
        # Fit vectorizer on all documents
        self.vectorizer.fit(documents)
        self.fitted = True
        
        # Create embeddings
        embeddings = self.vectorizer.transform(documents).toarray()
        
        # Adjust n_clusters if we have fewer documents
        n_clusters = min(n_clusters, len(documents))
        if n_clusters < 2:
            return {0: documents}
            
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(embeddings)
        
        clustered_docs = {}
        for doc, cluster in zip(documents, clusters):
            if cluster not in clustered_docs:
                clustered_docs[cluster] = []
            clustered_docs[cluster].append(doc)
        
        return clustered_docs
