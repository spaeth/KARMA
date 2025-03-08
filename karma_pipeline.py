# karma_pipeline.py
"""
KARMA: Leveraging Multi-Agent LLMs for Automated Knowledge Graph Enrichment

This module implements a pipeline of specialized agents that collaborate to:
1. Extract knowledge from text sources
2. Structure it into a knowledge graph
3. Evaluate and integrate new knowledge

Author: Yuxing Lu
"""

import os
import logging
import time
import json
from typing import List, Dict, Tuple, Union, Set, Optional
from dataclasses import dataclass, field, asdict
import PyPDF2
from openai import OpenAI

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("KARMA")

##############################################################################
# Data Structures
##############################################################################

@dataclass
class KnowledgeTriple:
    """
    Data class representing a single knowledge triple in the biomedical domain.
    
    Attributes:
        head: The subject entity
        relation: The relationship type
        tail: The object entity
        confidence: Model confidence score [0-1]
        source: Origin of the triple
        relevance: Domain relevance score [0-1]
        clarity: Linguistic clarity score [0-1]
    """
    head: str
    relation: str
    tail: str
    confidence: float = 0.0
    source: str = "unknown"
    relevance: float = 0.0
    clarity: float = 0.0
    
    def __str__(self) -> str:
        """String representation of the knowledge triple."""
        return f"({self.head}) -[{self.relation}]-> ({self.tail})"

@dataclass
class KGEntity:
    """
    Data class representing a canonical entity in the knowledge graph.
    
    Attributes:
        entity_id: Unique identifier
        entity_type: Semantic type (e.g. Drug, Disease)
        name: Display name
        normalized_id: Reference to standard ontology (e.g., UMLS:C0004238)
    """
    entity_id: str
    entity_type: str = "Unknown"
    name: str = ""
    normalized_id: str = "N/A"
    
    def __str__(self) -> str:
        """String representation of the entity."""
        return f"{self.name} ({self.entity_type})"

@dataclass
class IntermediateOutput:
    """
    Data class for storing intermediate outputs from each agent.
    
    Tracks the full pipeline state including raw inputs, 
    intermediate results, and final outputs.
    """
    raw_text: str = ""
    segments: List[Dict] = field(default_factory=list)
    relevant_segments: List[Dict] = field(default_factory=list)
    summaries: List[Dict] = field(default_factory=list)
    entities: List[KGEntity] = field(default_factory=list)
    relationships: List[KnowledgeTriple] = field(default_factory=list)
    aligned_entities: List[KGEntity] = field(default_factory=list)
    aligned_triples: List[KnowledgeTriple] = field(default_factory=list)
    final_triples: List[KnowledgeTriple] = field(default_factory=list)
    integrated_triples: List[KnowledgeTriple] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    processing_time: float = 0.0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        result = {}
        for key, value in asdict(self).items():
            if isinstance(value, list) and value and hasattr(value[0], '__dict__'):
                result[key] = [item.__dict__ for item in value]
            else:
                result[key] = value
        return result

##############################################################################
# Multi-Agent Classes
##############################################################################

class IngestionAgent:
    """
    Ingestion Agent (IA):
    1) Retrieves and standardizes raw documents (PDF, text)
    2) Extracts minimal metadata if available
    """
    def __init__(self, client: OpenAI, model_name: str):
        """
        Initialize the Ingestion Agent.
        
        Args:
            client: OpenAI client instance
            model_name: LLM model identifier
        """
        self.client = client
        self.model_name = model_name
        self.system_prompt = """You are the Ingestion Agent. Your responsibility is to:
1. Retrieve raw publications from designated sources
2. Convert various file formats into a consistent normalized text format
3. Extract metadata such as the title, authors, journal name, publication date, and identifiers"""

    def ingest_document(self, raw_text: str) -> Dict:
        """
        Standardize the raw text into a structured format with metadata.
        
        Args:
            raw_text: Input text to process
            
        Returns:
            Dict containing metadata and content
        """
        prompt = f"""
        Please analyze this document and extract the following metadata if available:
        - Title
        - Authors
        - Journal or source
        - Publication date
        - DOI or other identifiers
        
        If any field cannot be determined, mark it as "Unknown" or "N/A".
        
        Document:
        {raw_text[:5000]}  # Truncate for efficiency
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1
            )
            
            extracted_text = response.choices[0].message.content
            
            # Extract metadata from the response
            metadata = {
                "title": "Unknown Title",
                "authors": [],
                "journal": "Unknown Journal", 
                "pub_date": "N/A",
                "doi": "N/A",
                "pmid": "N/A"
            }
            
            # Simple parsing of the LLM response
            for line in extracted_text.split('\n'):
                line = line.strip()
                if line.startswith("Title:"):
                    metadata["title"] = line[6:].strip()
                elif line.startswith("Authors:"):
                    authors_text = line[8:].strip()
                    metadata["authors"] = [a.strip() for a in authors_text.split(',') if a.strip()]
                elif line.startswith("Journal:"):
                    metadata["journal"] = line[8:].strip()
                elif line.startswith("Publication date:"):
                    metadata["pub_date"] = line[17:].strip()
                elif line.startswith("DOI:"):
                    metadata["doi"] = line[4:].strip()
                elif line.startswith("PMID:"):
                    metadata["pmid"] = line[5:].strip()
            
            return {
                "metadata": metadata,
                "content": raw_text
            }
        except Exception as e:
            logger.error(f"Ingestion failed: {str(e)}")
            # Return a default structure on error
            return {
                "metadata": {
                    "title": "Unknown Title",
                    "authors": [],
                    "journal": "Unknown Journal", 
                    "pub_date": "N/A",
                    "doi": "N/A",
                    "pmid": "N/A",
                    "error": str(e)
                },
                "content": raw_text
            }

class ReaderAgent:
    """
    Reader Agent (RA):
    1) Splits the document into segments or sections
    2) Assigns a relevance score to each segment
    """
    def __init__(self, client: OpenAI, model_name: str):
        """
        Initialize the Reader Agent.
        
        Args:
            client: OpenAI client instance
            model_name: LLM model identifier
        """
        self.client = client
        self.model_name = model_name
        self.system_prompt = """You are the Reader Agent. Your goal is to parse text and generate logical segments 
that are likely to contain relevant knowledge. Each segment must be accompanied by a 
numeric Relevance Score indicating its importance for downstream extraction tasks."""
    
    def split_into_segments(self, content: str) -> List[Dict]:
        """
        Split content into logical segments.
        
        Args:
            content: Text to segment
            
        Returns:
            List of segment dictionaries with text and estimated score
        """
        # First do a basic split on paragraph breaks
        raw_segments = content.split("\n\n")
        segments = [{"text": seg.strip(), "score": 0.0} for seg in raw_segments if seg.strip()]
        
        # Process in batches to avoid context limitations
        batch_size = 5
        processed_segments = []
        
        for i in range(0, len(segments), batch_size):
            batch = segments[i:i+batch_size]
            batch_texts = [f"Segment {j+1}:\n{seg['text']}" for j, seg in enumerate(batch)]
            
            scores = self._batch_score_relevance(batch_texts)
            
            for j, seg in enumerate(batch):
                if j < len(scores):
                    seg["score"] = scores[j]
                processed_segments.append(seg)
        
        return processed_segments

    def _batch_score_relevance(self, segments: List[str]) -> List[float]:
        """
        Query the LLM for domain-specific relevance scores for multiple segments.
        
        Args:
            segments: List of text segments to score
            
        Returns:
            List of relevance scores
        """
        prompt = f"""
        You are a biomedical text relevance scorer.
        Rate how relevant each of the following segments is (0 to 1) for extracting
        new biomedical knowledge (e.g., relationships between diseases, drugs, genes).
        
        Consider:
        - Sections with experiments, results, or discussions usually have higher relevance
        - Methodology sections without findings have lower relevance
        - References have very low relevance
        
        For each segment, return only a single float value between 0.0 and 1.0, with no other text.
        
        {chr(10).join(segments)}
        
        Return one score per line, with no labels:
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1
            )
            
            lines = response.choices[0].message.content.strip().split('\n')
            scores = []
            
            for line in lines:
                try:
                    # Extract number from line
                    import re
                    match = re.search(r"[-+]?\d*\.\d+|\d+", line)
                    if match:
                        score = float(match.group())
                        # Ensure score is in range [0,1]
                        score = max(0.0, min(1.0, score))
                        scores.append(score)
                    else:
                        scores.append(0.5)  # Default if no number found
                except:
                    scores.append(0.5)  # Default on error
                    
            return scores
        except Exception as e:
            logger.warning(f"Failed to score segments. Error: {e}")
            return [0.5] * len(segments)  # Default on overall failure

    def score_relevance(self, segment: str) -> Tuple[float, int, int, float]:
        """
        Query the LLM for a domain-specific relevance score for a single segment.
        
        Args:
            segment: Text segment to score
            
        Returns:
            Tuple of (relevance_score, prompt_tokens, completion_tokens, processing_time)
        """
        prompt = f"""
        You are a biomedical text relevance scorer.
        Rate how relevant the following text is (0 to 1) for extracting
        new biomedical knowledge (e.g., relationships between diseases, drugs, genes):

        Text:
        {segment}

        Return only a single float value between 0.0 and 1.0, with no other text.
        Example valid responses:
        0.75
        0.3
        """
        start_time = time.time()
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1
            )
            score_str = response.choices[0].message.content.strip()
            # Extract first float value found in response
            import re
            float_matches = re.findall(r"[-+]?\d*\.\d+|\d+", score_str)
            if float_matches:
                score_val = float(float_matches[0])
            else:
                score_val = 0.5  # default if no float found
            
            prompt_tokens = response.usage.prompt_tokens
            completion_tokens = response.usage.completion_tokens
            processing_time = time.time() - start_time

            return max(0.0, min(1.0, score_val)), prompt_tokens, completion_tokens, processing_time
        except Exception as e:
            logger.warning(f"Failed to parse relevance for segment. Error: {e}")
            return 0.5, 0, 0, time.time() - start_time  # default

class SummarizerAgent:
    """
    Summarizer Agent (SA):
    1) Takes relevant text segments
    2) Produces concise summaries that preserve domain-specific terms
    """
    def __init__(self, client: OpenAI, model_name: str):
        """
        Initialize the Summarizer Agent.
        
        Args:
            client: OpenAI client instance
            model_name: LLM model identifier
        """
        self.client = client
        self.model_name = model_name
        self.system_prompt = """You are the Summarizer Agent. Your task is to convert text segments into concise summaries 
while retaining technical detail such as gene symbols, chemical names, or numeric data that may be crucial 
for entity/relationship extraction."""
    
    def summarize_segment(self, segment: str) -> Tuple[str, int, int, float]:
        """
        Summarize a single text segment using an LLM prompt.
        
        Args:
            segment: Text to summarize
            
        Returns:
            Tuple of (summary, prompt_tokens, completion_tokens, processing_time)
        """
        prompt = f"""
        Summarize the following biomedical text in 2-4 sentences, 
        retaining key domain terms (genes, proteins, drugs, diseases, etc.).
        Preserve any numeric data or statistical findings that indicate relationships.
        Keep the summary under 100 words.
        Provide only the summary with no additional text or formatting.
        
        Text:
        {segment}
        """
        start_time = time.time()
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2
            )
            summary = response.choices[0].message.content.strip()
            # Handle empty or invalid responses
            if not summary:
                summary = segment[:200] + "..."  # fallback to truncated original
                
            prompt_tokens = response.usage.prompt_tokens
            completion_tokens = response.usage.completion_tokens
            processing_time = time.time() - start_time
            
            return summary, prompt_tokens, completion_tokens, processing_time
        except Exception as e:
            logger.warning(f"Summarization failed. Error: {e}")
            return segment[:200] + "...", 0, 0, time.time() - start_time  # fallback to truncated original

class EntityExtractionAgent:
    """
    Entity Extraction Agent (EEA):
    1) Identifies biomedical entities in the text
    2) Optionally normalizes them to canonical IDs
    """
    def __init__(self, client: OpenAI, model_name: str):
        """
        Initialize the Entity Extraction Agent.
        
        Args:
            client: OpenAI client instance
            model_name: LLM model identifier
        """
        self.client = client
        self.model_name = model_name
        self.system_prompt = """You are the Entity Extraction Agent. Your objective is to identify biomedical entities 
(Disease, Drug, Gene, Protein, Chemical, etc.) and link each mention to a canonical ontology reference where possible."""

    def extract_entities(self, text: str) -> Tuple[List[KGEntity], int, int, float]:
        """
        Query the LLM to identify entities.
        
        Args:
            text: Text to extract entities from
            
        Returns:
            Tuple of (entity_list, prompt_tokens, completion_tokens, processing_time)
        """
        prompt = f"""
        Extract biomedical entities from the text below.
        Include potential diseases, drugs, genes, proteins, chemicals, etc.

        For each entity:
        1. Identify the exact mention in the text
        2. Assign an entity type (Disease, Drug, Gene, Protein, Chemical, etc.)
        3. Provide a normalized ID if possible (e.g., UMLS:C0018681 for headache)
        
        Format each entity as JSON: {{"mention": "...", "type": "...", "normalized_id": "..."}}
        If no suitable ontology reference is found, set normalized_id to "N/A"

        Text:
        {text}
        """
        start_time = time.time()
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1
            )
            
            content = response.choices[0].message.content.strip()
            entity_list = []
            
            # Try to parse JSON entities from the response
            try:
                # Check if response contains JSON array
                if "[" in content and "]" in content:
                    json_str = content[content.find("["):content.rfind("]")+1]
                    entities_data = json.loads(json_str)
                    for ent_data in entities_data:
                        if isinstance(ent_data, dict) and "mention" in ent_data:
                            entity_list.append(KGEntity(
                                entity_id=ent_data.get("mention", ""),
                                entity_type=ent_data.get("type", "Unknown"),
                                name=ent_data.get("mention", ""),
                                normalized_id=ent_data.get("normalized_id", "N/A")
                            ))
                else:
                    # Fallback: extract entities line by line
                    for line in content.split('\n'):
                        if ':' in line and len(line) > 5:  # Simple heuristic for entity lines
                            # Basic extraction
                            mention = line.split(':')[0].strip()
                            entity_list.append(KGEntity(
                                entity_id=mention,
                                name=mention
                            ))
            except json.JSONDecodeError:
                # Fallback for line-by-line extraction if JSON parsing fails
                for line in content.split('\n'):
                    line = line.strip()
                    if line and len(line) > 2:  # Skip empty lines
                        entity_list.append(KGEntity(
                            entity_id=line,
                            name=line
                        ))
                    
            prompt_tokens = response.usage.prompt_tokens
            completion_tokens = response.usage.completion_tokens
            processing_time = time.time() - start_time
            
            return entity_list, prompt_tokens, completion_tokens, processing_time
        except Exception as e:
            logger.warning(f"Entity extraction failed. Error: {e}")
            return [], 0, 0, time.time() - start_time

class RelationshipExtractionAgent:
    """
    Relationship Extraction Agent (REA):
    1) Given a set of recognized entities within a text,
    2) Extract potential relationships among them (triplets)
    """
    def __init__(self, client: OpenAI, model_name: str):
        """
        Initialize the Relationship Extraction Agent.
        
        Args:
            client: OpenAI client instance
            model_name: LLM model identifier
        """
        self.client = client
        self.model_name = model_name
        self.system_prompt = """You are the Relationship Extraction Agent. Given a text snippet plus a set of recognized entities, 
your mission is to detect possible relationships (e.g., treats, causes, interactsWith, inhibits) among them."""

    def extract_relationships(self, text: str, entities: List[KGEntity]) -> Tuple[List[KnowledgeTriple], int, int, float]:
        """
        Query the LLM to identify relationships among provided entities.
        
        Args:
            text: Source text
            entities: List of entities to find relationships between
            
        Returns:
            Tuple of (relationship_list, prompt_tokens, completion_tokens, processing_time)
        """
        if not entities:
            return [], 0, 0, 0.0

        # Format the entity list for the LLM
        entity_bullets = "\n".join(f"- {ent.name} (Type: {ent.entity_type})" for ent in entities)
        
        prompt = f"""
        We have these entities of interest:
        {entity_bullets}

        From the text below, identify direct relationships between these entities.
        Use standardized relationships such as 'treats', 'causes', 'inhibits', 'activates', 'interacts_with', 'associated_with', etc.
        
        For each relationship, provide:
        1. Head entity (subject)
        2. Relation type
        3. Tail entity (object)
        4. A confidence score (0-1) for how certain the relationship is expressed in the text
        
        Format as JSON: {{"head": "...", "relation": "...", "tail": "...", "confidence": 0.X}}
        
        If no relationships are found, return an empty array.

        Text:
        {text}
        """

        start_time = time.time()
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1
            )
            
            content = response.choices[0].message.content.strip()
            triples: List[KnowledgeTriple] = []
            
            # Try to parse JSON from the response
            try:
                if "[" in content and "]" in content:
                    json_str = content[content.find("["):content.rfind("]")+1]
                    relations_data = json.loads(json_str)
                    
                    for rel in relations_data:
                        if isinstance(rel, dict) and "head" in rel and "relation" in rel and "tail" in rel:
                            # Basic data validation
                            head = rel.get("head", "").strip()
                            relation = rel.get("relation", "").strip()
                            tail = rel.get("tail", "").strip()
                            
                            if head and relation and tail:
                                confidence = float(rel.get("confidence", 0.5))
                                # Get clarity and relevance scores
                                clarity, clarity_tokens = self._estimate_metric(head, relation, tail, "clarity")
                                relevance, relevance_tokens = self._estimate_metric(head, relation, tail, "relevance")
                                
                                triples.append(KnowledgeTriple(
                                    head=head,
                                    relation=relation,
                                    tail=tail,
                                    confidence=confidence,
                                    clarity=clarity,
                                    relevance=relevance,
                                    source="relationship_extraction"
                                ))
                else:
                    # Fallback: extract relationships line by line
                    for line in content.split('\n'):
                        if '->' in line:
                            parts = [p.strip() for p in line.split('->')]
                            if len(parts) == 3:
                                head, relation, tail = parts
                                triples.append(KnowledgeTriple(
                                    head=head,
                                    relation=relation,
                                    tail=tail,
                                    confidence=0.5,  # Default confidence
                                    clarity=0.5,     # Default clarity
                                    relevance=0.5,   # Default relevance
                                    source="relationship_extraction"
                                ))
            except json.JSONDecodeError:
                # Simple line-by-line fallback
                for line in content.split('\n'):
                    if '->' in line:
                        parts = [p.strip() for p in line.split('->')]
                        if len(parts) == 3:
                            head, relation, tail = parts
                            triples.append(KnowledgeTriple(
                                head=head,
                                relation=relation,
                                tail=tail,
                                confidence=0.5,
                                clarity=0.5,
                                relevance=0.5,
                                source="relationship_extraction"
                            ))
                        
            prompt_tokens = response.usage.prompt_tokens
            completion_tokens = response.usage.completion_tokens 
            processing_time = time.time() - start_time
            
            return triples, prompt_tokens, completion_tokens, processing_time
        except Exception as e:
            logger.warning(f"Relationship extraction failed. Error: {e}")
            return [], 0, 0, time.time() - start_time

    def _estimate_metric(self, head: str, relation: str, tail: str, metric_type: str) -> Tuple[float, int]:
        """
        Estimate clarity or relevance for a triple.
        
        Args:
            head: Subject entity
            relation: Relationship type
            tail: Object entity
            metric_type: Either "clarity" or "relevance"
            
        Returns:
            Tuple of (metric_score, tokens_used)
        """
        prompt = f"""
        Evaluate the {metric_type} of this relationship triple:
        "{head} -> {relation} -> {tail}"

        For clarity, consider:
        - Whether the terms are specific and unambiguous
        - Whether the relationship is well-defined
        
        For relevance, consider:
        - How important this relationship is in the biomedical domain
        - Whether it captures significant knowledge
        
        Return a single float between 0.01 and 0.99 representing the {metric_type} score.
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": f"You evaluate {metric_type} of biomedical relationships."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=50
            )
            
            content = response.choices[0].message.content.strip()
            tokens_used = response.usage.prompt_tokens + response.usage.completion_tokens
            
            # Extract the score
            import re
            matches = re.findall(r"[-+]?\d*\.\d+|\d+", content)
            if matches:
                score = float(matches[0])
                # Ensure score is in range [0.01, 0.99]
                score = max(0.01, min(0.99, score))
                return score, tokens_used
            else:
                return 0.5, tokens_used  # Default
        except Exception as e:
            logger.warning(f"Failed to estimate {metric_type}. Error: {e}")
            return 0.5, 0  # Default on error

class SchemaAlignmentAgent:
    """
    Schema Alignment Agent (SAA):
    1) Maps extracted entities/relationships to existing KG schema types
    2) Potentially flags new classes or relation types if unknown
    """
    def __init__(self, client: OpenAI, model_name: str):
        """
        Initialize the Schema Alignment Agent.
        
        Args:
            client: OpenAI client instance
            model_name: LLM model identifier
        """
        self.client = client
        self.model_name = model_name
        self.system_prompt = """You are the Schema Alignment Agent. Newly extracted entities or relationships may not match 
existing knowledge graph classes or relation types. Your job is to determine how they should map onto the existing 
ontology or schema."""

    def align_entities(self, entities: List[KGEntity]) -> Tuple[List[KGEntity], int, int, float]:
        """
        Attempt to classify each entity type via LLM or external ontology.
        
        Args:
            entities: List of entities to classify
            
        Returns:
            Tuple of (aligned_entities, prompt_tokens, completion_tokens, processing_time)
        """
        start_time = time.time()
        total_prompt_tokens = 0
        total_completion_tokens = 0
        
        aligned = []
        
        # Process entities in batches to be more efficient
        batch_size = 10
        for i in range(0, len(entities), batch_size):
            batch = entities[i:i+batch_size]
            batch_results, pt, ct = self._batch_classify_entity_types(batch)
            
            total_prompt_tokens += pt
            total_completion_tokens += ct
            
            for original_entity, classification in zip(batch, batch_results):
                if classification:
                    original_entity.entity_type = classification
                aligned.append(original_entity)
            
        processing_time = time.time() - start_time
        return aligned, total_prompt_tokens, total_completion_tokens, processing_time

    def _batch_classify_entity_types(self, entities: List[KGEntity]) -> Tuple[List[str], int, int]:
        """
        Classify multiple entities at once for efficiency.
        
        Args:
            entities: List of entities to classify
            
        Returns:
            Tuple of (classifications, prompt_tokens, completion_tokens)
        """
        entity_names = [ent.name for ent in entities]
        entities_text = "\n".join([f"{i+1}. {name}" for i, name in enumerate(entity_names)])
        
        prompt = f"""
        Classify each entity by its biomedical type: Drug, Disease, Gene, Protein, Chemical, RNA, Pathway, Cell, or Other.
        
        Entities:
        {entities_text}
        
        Return classifications one per line, numbered to match the input:
        1. [Type]
        2. [Type]
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0
            )
            
            lines = response.choices[0].message.content.strip().split('\n')
            classifications = []
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                    
                # Parse numbered responses
                if '.' in line:
                    parts = line.split('.', 1)
                    if len(parts) == 2 and parts[1].strip():
                        classifications.append(parts[1].strip())
                else:
                    # If response isn't numbered, just use the line
                    classifications.append(line)
            
            # Pad with "Unknown" if some entities didn't get classified
            while len(classifications) < len(entities):
                classifications.append("Unknown")
                
            # Truncate if we somehow got more classifications than entities
            classifications = classifications[:len(entities)]
            
            prompt_tokens = response.usage.prompt_tokens
            completion_tokens = response.usage.completion_tokens
            
            return classifications, prompt_tokens, completion_tokens
        except Exception as e:
            logger.warning(f"Entity type classification failed. Error: {e}")
            return ["Unknown"] * len(entities), 0, 0

    def align_relationships(self, triples: List[KnowledgeTriple]) -> List[KnowledgeTriple]:
        """
        Unify relationship labels to a standard set.
        
        Args:
            triples: List of triples to normalize relations for
            
        Returns:
            List of triples with normalized relations
        """
        aligned_triples = []
        for t in triples:
            normalized_rel = self._normalize_relation(t.relation)
            t.relation = normalized_rel
            aligned_triples.append(t)
        return aligned_triples

    def _normalize_relation(self, relation: str) -> str:
        """
        Standardize relation labels to canonical forms.
        
        Args:
            relation: Relation label to normalize
            
        Returns:
            Normalized relation label
        """
        # Standard mapping of similar relations, just an example
        synonyms = {
            "inhibit": "inhibits",
            "inhibited": "inhibits",
            "inhibits": "inhibits",
            "treat": "treats",
            "treated": "treats",
            "treats": "treats",
            "cause": "causes",
            "caused": "causes",
            "causes": "causes",
            "activate": "activates",
            "activates": "activates",
            "regulates": "regulates",
            "regulate": "regulates",
            "regulated": "regulates",
            "associated with": "associated_with",
            "associatedwith": "associated_with",
            "associated_with": "associated_with",
            "interacts with": "interacts_with",
            "interactswith": "interacts_with",
            "interacts_with": "interacts_with",
            "binds to": "binds_to",
            "bindsto": "binds_to",
            "binds_to": "binds_to"
        }
        
        base = relation.lower().strip()
        # Return the mapped version or the original if no mapping exists
        return synonyms.get(base, base)

class ConflictResolutionAgent:
    """
    Conflict Resolution Agent (CRA):
    1) Checks if newly extracted triplets conflict with existing knowledge
    2) Decides whether to keep, discard, or flag them for manual review
    """
    def __init__(self, client: OpenAI, model_name: str):
        """
        Initialize the Conflict Resolution Agent.
        
        Args:
            client: OpenAI client instance
            model_name: LLM model identifier
        """
        self.client = client
        self.model_name = model_name
        self.system_prompt = """You are the Conflict Resolution Agent. Sometimes new triplets are detected that contradict 
existing knowledge. Your role is to classify these contradictions and decide whether the new triplet should be 
discarded, flagged for expert review, or integrated with caution."""

    def resolve_conflicts(
        self, 
        new_triples: List[KnowledgeTriple],
        existing_triples: List[KnowledgeTriple]
    ) -> Tuple[List[KnowledgeTriple], int, int, float]:
        """
        Compare each new triple against existing ones for direct contradictions.
        
        Args:
            new_triples: Newly extracted triples
            existing_triples: Existing knowledge graph triples
            
        Returns:
            Tuple of (final_triples, prompt_tokens, completion_tokens, processing_time)
        """
        start_time = time.time()
        total_prompt_tokens = 0
        total_completion_tokens = 0
        
        final_triples = []
        for nt in new_triples:
            # Check if new triple directly contradicts any existing triple
            conflicting_triple = self._find_contradiction(nt, existing_triples)
            
            if conflicting_triple:
                # Use LLM to decide between conflicting triples
                keep, pt, ct, _ = self._resolve_contradiction(nt, conflicting_triple)
                total_prompt_tokens += pt
                total_completion_tokens += ct
                
                if keep:
                    final_triples.append(nt)
            else:
                # No conflict, keep the triple
                final_triples.append(nt)
                
        processing_time = time.time() - start_time
        return final_triples, total_prompt_tokens, total_completion_tokens, processing_time

    def _find_contradiction(
        self, 
        new_t: KnowledgeTriple, 
        existing_list: List[KnowledgeTriple]
    ) -> Optional[KnowledgeTriple]:
        """
        Check if a new triple contradicts any existing triples.
        
        Args:
            new_t: New triple to check
            existing_list: List of existing triples
            
        Returns:
            The conflicting triple if found, None otherwise
        """
        # Define opposite relation pairs (bidirectional)
        opposite_relations = {
            ("treats", "causes"),
            ("inhibits", "activates"),
            ("increases", "decreases"),
            ("upregulates", "downregulates")
        }
        
        # Create a set of relation pairs that are opposites
        contradiction_pairs = set()
        for a, b in opposite_relations:
            contradiction_pairs.add((a, b))
            contradiction_pairs.add((b, a))  # Add reverse pair too

        for ex in existing_list:
            # Check for triples about the same entities but with potentially opposing relations
            if (ex.head.lower() == new_t.head.lower() and 
                ex.tail.lower() == new_t.tail.lower()):
                # Check if relation pair is an opposite
                if (ex.relation, new_t.relation) in contradiction_pairs:
                    return ex
        return None

    def _resolve_contradiction(self, new_triple: KnowledgeTriple, existing_triple: KnowledgeTriple) -> Tuple[bool, int, int, float]:
        """
        LLM-based evaluation to decide which of two contradicting triples to keep.
        
        Args:
            new_triple: New triple being evaluated
            existing_triple: Existing conflicting triple
            
        Returns:
            Tuple of (keep_decision, prompt_tokens, completion_tokens, processing_time)
        """
        prompt = f"""
        We have two potentially contradicting biomedical statements:
        
        NEW: {new_triple.head} -> {new_triple.relation} -> {new_triple.tail}
        EXISTING: {existing_triple.head} -> {existing_triple.relation} -> {existing_triple.tail}
        
        Analyze these statements and decide:
        1. Do they truly contradict each other, or could both be valid in different contexts?
        2. Which statement appears more credible based on biological knowledge?
        
        Return one of:
        - "KEEP_NEW" if the new statement should replace or complement the existing one
        - "KEEP_EXISTING" if the existing statement is more reliable
        - "KEEP_BOTH" if they can coexist (different contexts, conditions, etc.)
        - "REVIEW" if expert human review is needed due to high uncertainty
        
        Provide only the decision code with no additional text.
        """
        
        start_time = time.time()
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=256,
                temperature=0.1
            )
            
            decision = response.choices[0].message.content.strip().upper()
            prompt_tokens = response.usage.prompt_tokens
            completion_tokens = response.usage.completion_tokens
            processing_time = time.time() - start_time
            
            # Parse the decision
            keep_new = decision in ["KEEP_NEW", "KEEP_BOTH"]
            
            return keep_new, prompt_tokens, completion_tokens, processing_time
            
        except Exception as e:
            logger.warning(f"Conflict resolution failed. Error: {e}")
            # Default to keeping existing knowledge when uncertain
            return False, 0, 0, time.time() - start_time

class EvaluatorAgent:
    """
    Evaluator Agent (EA):
    1) Aggregates various confidence signals
    2) Produces a final integration score for each triple
    3) Evaluates clarity and relevance in addition to confidence
    """
    def __init__(self, client: OpenAI, model_name: str, integrate_threshold=0.6):
        """
        Initialize the Evaluator Agent.
        
        Args:
            client: OpenAI client instance
            model_name: LLM model identifier
            integrate_threshold: Minimum score to integrate knowledge
        """
        self.client = client
        self.model_name = model_name
        self.integrate_threshold = integrate_threshold
        self.system_prompt = """You are the Evaluator Agent. Your duty is to aggregate confidence signals 
into final scores and decide which knowledge to integrate into the knowledge graph."""

    def finalize_triples(self, candidate_triples: List[KnowledgeTriple]) -> Tuple[List[KnowledgeTriple], int, int, float]:
        """
        Evaluate and filter triples based on confidence, clarity, and relevance.
        
        Args:
            candidate_triples: Triples to evaluate
            
        Returns:
            Tuple of (integrated_triples, prompt_tokens, completion_tokens, processing_time)
        """
        start_time = time.time()
        total_prompt_tokens = 0
        total_completion_tokens = 0
        
        integrated_triples = []
        for i, triple in enumerate(candidate_triples):
            # Evaluate confidence, clarity, and relevance if missing
            if triple.confidence < 0.01:
                conf, conf_pt, conf_ct, _ = self._evaluate_confidence(triple)
                triple.confidence = conf
                total_prompt_tokens += conf_pt
                total_completion_tokens += conf_ct
                
            if triple.clarity < 0.01:
                clarity, clar_pt, clar_ct, _ = self._evaluate_clarity(triple)
                triple.clarity = clarity
                total_prompt_tokens += clar_pt
                total_completion_tokens += clar_ct
                
            if triple.relevance < 0.01:
                relevance, rel_pt, rel_ct, _ = self._evaluate_relevance(triple)
                triple.relevance = relevance
                total_prompt_tokens += rel_pt
                total_completion_tokens += rel_ct
            
            # Compute final integration score
            integration_score = self._aggregate_scores(triple)
            
            # Keep triple if score meets threshold
            if integration_score >= self.integrate_threshold:
                integrated_triples.append(triple)
        
        processing_time = time.time() - start_time
        return integrated_triples, total_prompt_tokens, total_completion_tokens, processing_time

    def _aggregate_scores(self, triple: KnowledgeTriple) -> float:
        """
        Combine confidence metrics into final score.
        
        Args:
            triple: Triple to score
            
        Returns:
            Aggregated confidence score
        """
        # Weighting factors for different metrics (customizable)
        w_conf, w_clarity, w_rel = 0.5, 0.25, 0.25
        
        # Ensure all scores are in [0.0, 1.0] range
        conf = max(0.0, min(1.0, triple.confidence))
        clarity = max(0.0, min(1.0, triple.clarity))
        relevance = max(0.0, min(1.0, triple.relevance))
        
        # Weighted average
        return (
            w_conf * conf +
            w_clarity * clarity +
            w_rel * relevance
        )

    def _evaluate_confidence(self, triple: KnowledgeTriple) -> Tuple[float, int, int, float]:
        """
        LLM-based evaluation of triple confidence.
        
        Args:
            triple: Triple to evaluate
            
        Returns:
            Tuple of (confidence_score, prompt_tokens, completion_tokens, processing_time)
        """
        prompt = f"""
        Evaluate the factual confidence of this biomedical statement:
        "{triple.head} -> {triple.relation} -> {triple.tail}"

        Based on established biomedical knowledge, how confident are we that this relationship is accurate?
        
        Rate from 0.0 (completely uncertain) to 1.0 (extremely confident).
        Return only a single float value between 0.0 and 1.0.
        
        Example outputs:
          0.67
          0.34

        Invalid outputs:
          1.0
          0.0
          0.99 is acceptable, 0.01 is acceptable.

        Return only the numeric value.
        """
        
        start_time = time.time()
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=256,
                temperature=0.0
            )
            
            content = response.choices[0].message.content.strip()
            prompt_tokens = response.usage.prompt_tokens
            completion_tokens = response.usage.completion_tokens
            processing_time = time.time() - start_time
            
            # Extract float value
            score = self._extract_float_score(content)
            return score, prompt_tokens, completion_tokens, processing_time
            
        except Exception as e:
            logger.warning(f"Confidence evaluation failed. Error: {e}")
            return 0.5, 0, 0, time.time() - start_time

    def _evaluate_clarity(self, triple: KnowledgeTriple) -> Tuple[float, int, int, float]:
        """
        LLM-based evaluation of triple linguistic clarity.
        
        Args:
            triple: Triple to evaluate
            
        Returns:
            Tuple of (clarity_score, prompt_tokens, completion_tokens, processing_time)
        """
        prompt = f"""
        Evaluate the clarity and specificity of this biomedical relationship:
        "{triple.head} -> {triple.relation} -> {triple.tail}"

        Consider:
        1. Are the entities precise and unambiguous?
        2. Is the relationship type specific and well-defined?
        3. Would biomedical experts interpret this statement consistently?
        
        Rate clarity from 0.0 (very ambiguous) to 1.0 (perfectly clear).
        Return only a single float value between 0.0 and 1.0.
        Example outputs:
          0.67
          0.34

        Invalid outputs:
          1.0
          0.0
          0.99 is acceptable, 0.01 is acceptable.

        Return only the numeric value.
        """
        
        start_time = time.time()
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You evaluate clarity of biomedical relationships."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=256,
                temperature=0.0
            )
            
            content = response.choices[0].message.content.strip()
            prompt_tokens = response.usage.prompt_tokens
            completion_tokens = response.usage.completion_tokens
            processing_time = time.time() - start_time
            
            # Extract float value
            score = self._extract_float_score(content)
            return score, prompt_tokens, completion_tokens, processing_time
            
        except Exception as e:
            logger.warning(f"Clarity evaluation failed. Error: {e}")
            return 0.5, 0, 0, time.time() - start_time

    def _evaluate_relevance(self, triple: KnowledgeTriple, domain="biomedical") -> Tuple[float, int, int, float]:
        """
        LLM-based evaluation of triple domain relevance.
        
        Args:
            triple: Triple to evaluate
            domain: Domain topic for relevance
            
        Returns:
            Tuple of (relevance_score, prompt_tokens, completion_tokens, processing_time)
        """
        prompt = f"""
        Evaluate how relevant this relationship is to the {domain} domain:
        "{triple.head} -> {triple.relation} -> {triple.tail}"

        Consider:
        1. Is this directly relevant to {domain} research or practice?
        2. Would this information be valuable to include in a {domain} knowledge graph?
        3. Is this specialized knowledge rather than general knowledge?
        
        Rate relevance from 0.0 (completely irrelevant) to 1.0 (highly relevant).
        Return only a single float value between 0.0 and 1.0.
        Example outputs:
          0.67
          0.34

        Invalid outputs:
          1.0
          0.0
          0.99 is acceptable, 0.01 is acceptable.

        Return only the numeric value.
        """
        
        start_time = time.time()
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": f"You evaluate {domain} relevance of relationships."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=256,
                temperature=0.0
            )
            
            content = response.choices[0].message.content.strip()
            prompt_tokens = response.usage.prompt_tokens
            completion_tokens = response.usage.completion_tokens
            processing_time = time.time() - start_time
            
            # Extract float value
            score = self._extract_float_score(content)
            return score, prompt_tokens, completion_tokens, processing_time
            
        except Exception as e:
            logger.warning(f"Relevance evaluation failed. Error: {e}")
            return 0.5, 0, 0, time.time() - start_time

    def _extract_float_score(self, content: str) -> float:
        """
        Extract a float value from LLM response.
        
        Args:
            content: Text containing float value(s)
            
        Returns:
            Parsed and normalized float value in [0.0, 1.0]
        """
        # Look for float numbers in the content
        import re
        float_matches = re.findall(r"[-+]?\d*\.\d+|\d+", content)
        
        if float_matches:
            try:
                score = float(float_matches[0])
                # Clamp to [0.0, 1.0]
                return max(0.0, min(1.0, score))
            except ValueError:
                pass
        
        # Default fallback
        return 0.5

##############################################################################
# The KARMA Pipeline Class
##############################################################################

class KARMA:
    """
    KARMA: Knowledge Agent for Relationship and Meta-data Acquisition
    
    A multi-agent pipeline that orchestrates:
    1) Ingestion -> 2) Reading -> 3) Summarization -> 4) Entity Extraction
    -> 5) Relationship Extraction -> 6) Schema Alignment
    -> 7) Conflict Resolution -> 8) Evaluation -> Integration
    """
    def __init__(self, api_key: str, base_url: str = None, model_name: str = "gpt-4o"):
        """
        Initialize KARMA pipeline with API credentials.
        
        Args:
            api_key: OpenAI API key
            base_url: Optional API base URL for Azure or custom endpoints
            model_name: Model identifier
        """
        # Initialize OpenAI client
        client_kwargs = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
        
        self.client = OpenAI(**client_kwargs)
        self.model_name = model_name
        
        # Initialize all agent classes
        self.ingestion_agent = IngestionAgent(self.client, model_name)
        self.reader_agent = ReaderAgent(self.client, model_name)
        self.summarizer_agent = SummarizerAgent(self.client, model_name)
        self.entity_ex_agent = EntityExtractionAgent(self.client, model_name)
        self.relation_ex_agent = RelationshipExtractionAgent(self.client, model_name)
        self.schema_agent = SchemaAlignmentAgent(self.client, model_name)
        self.conflict_agent = ConflictResolutionAgent(self.client, model_name)
        self.evaluator_agent = EvaluatorAgent(self.client, model_name, integrate_threshold=0.6)

        # Internally stored knowledge graph representation
        self.knowledge_graph = {
            "entities": set(),       # set of string IDs or entity names
            "triples": []            # list of KnowledgeTriple objects
        }

        # Logging and tracking
        self.output_log: List[str] = []
        self.intermediate = IntermediateOutput()

    ##########################################################################
    # Helper: PDF reading
    ##########################################################################
    def _read_pdf(self, pdf_path: str) -> str:
        """
        Extract raw text from a PDF file.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Extracted text content
        """
        try:
            text = ""
            with open(pdf_path, 'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
            return text
        except Exception as e:
            logger.error(f"Failed to read PDF: {pdf_path}, error: {str(e)}")
            return ""

    def _log(self, message: str):
        """
        Simple logger for storing pipeline messages.
        
        Args:
            message: Log message to store
        """
        logger.info(message)
        self.output_log.append(message)

    ##########################################################################
    # The Main Pipeline
    ##########################################################################
    def process_document(self, source: Union[str, os.PathLike], domain: str = "biomedical") -> List[KnowledgeTriple]:
        """
        Pipeline entry point to process a document and extract knowledge.
        
        If `source` ends with '.pdf', we assume it is a PDF file path;
        otherwise, treat `source` as raw text input.
        
        Args:
            source: Text content or path to PDF file
            domain: Domain context for relevance scoring
            
        Returns:
            List of integrated KnowledgeTriple objects
        """
        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_time = 0
        pipeline_start_time = time.time()
        
        # Handle various input types
        if isinstance(source, str) and source.lower().endswith('.pdf'):
            raw_text = self._read_pdf(source)
        elif isinstance(source, os.PathLike) and str(source).lower().endswith('.pdf'):
            raw_text = self._read_pdf(str(source))
        else:
            # treat source directly as text
            raw_text = source
            
        self.intermediate.raw_text = raw_text
        
        # === (1) Ingestion ===
        step_start = time.time()
        doc_dict = self.ingestion_agent.ingest_document(raw_text)
        step_time = time.time() - step_start
        total_time += step_time
        self._log(f"[1] Ingestion completed in {step_time:.2f}s. Document standardized.")

        # === (2) Reader: Segment + Score Relevance ===
        step_start = time.time()
        segments = self.reader_agent.split_into_segments(doc_dict["content"])
        self.intermediate.segments = segments
        
        relevant_content = []
        for seg in segments:
            score, prompt_tokens, completion_tokens, processing_time = self.reader_agent.score_relevance(seg)
            total_prompt_tokens += prompt_tokens
            total_completion_tokens += completion_tokens
            total_time += processing_time
            if score > 0.2:  # Keep only segments with relevance above threshold
                relevant_content.append(seg)
        
        self.intermediate.relevant_segments = relevant_content
        step_time = time.time() - step_start
        self._log(f"[2] Reader completed in {step_time:.2f}s. Total segments: {len(segments)}, relevant: {len(relevant_content)}")

        # === (3) Summarizer ===
        step_start = time.time()
        summaries = []
        for seg in relevant_content:
            summary, prompt_tokens, completion_tokens, processing_time = self.summarizer_agent.summarize_segment(seg)
            total_prompt_tokens += prompt_tokens
            total_completion_tokens += completion_tokens
            total_time += processing_time
            summaries.append(summary)
        
        self.intermediate.summaries = summaries
        step_time = time.time() - step_start
        self._log(f"[3] Summarizer completed in {step_time:.2f}s. Summaries produced: {len(summaries)}")

        # === (4) Entity Extraction ===
        step_start = time.time()
        all_entities: List[KGEntity] = []
        for summary in summaries:
            extracted, prompt_tokens, completion_tokens, processing_time = self.entity_ex_agent.extract_entities(summary)
            total_prompt_tokens += prompt_tokens
            total_completion_tokens += completion_tokens
            total_time += processing_time
            all_entities.extend(extracted)
        
        # Deduplicate entities by name (case-insensitive)
        unique_entities_map = {}
        for ent in all_entities:
            unique_entities_map[ent.name.lower()] = ent
        all_entities = list(unique_entities_map.values())
        
        self.intermediate.entities = all_entities
        step_time = time.time() - step_start
        self._log(f"[4] Entity Extraction completed in {step_time:.2f}s. Unique entities found: {len(all_entities)}")

        # === (5) Relationship Extraction ===
        step_start = time.time()
        all_relationships: List[KnowledgeTriple] = []
        for summary in summaries:
            new_trips, prompt_tokens, completion_tokens, processing_time = self.relation_ex_agent.extract_relationships(summary, all_entities)
            total_prompt_tokens += prompt_tokens
            total_completion_tokens += completion_tokens
            total_time += processing_time
            all_relationships.extend(new_trips)
            
        # Deduplicate relationships (exact matches only)
        unique_rel_map = {}
        for rel in all_relationships:
            key = f"{rel.head.lower()}__{rel.relation.lower()}__{rel.tail.lower()}"
            # Keep the one with higher confidence if duplicates exist
            if key not in unique_rel_map or rel.confidence > unique_rel_map[key].confidence:
                unique_rel_map[key] = rel
        all_relationships = list(unique_rel_map.values())
        
        self.intermediate.relationships = all_relationships
        step_time = time.time() - step_start
        self._log(f"[5] Relationship Extraction completed in {step_time:.2f}s. Relationships found: {len(all_relationships)}")

        # === (6) Schema Alignment ===
        step_start = time.time()
        # Align entities to standard types (e.g., Drug, Disease, Gene)
        aligned_entities, prompt_tokens, completion_tokens, processing_time = self.schema_agent.align_entities(all_entities)
        total_prompt_tokens += prompt_tokens
        total_completion_tokens += completion_tokens
        total_time += processing_time
        
        # Align relationships to standard forms
        aligned_triples = self.schema_agent.align_relationships(all_relationships)
        
        self.intermediate.aligned_entities = aligned_entities
        self.intermediate.aligned_triples = aligned_triples
        step_time = time.time() - step_start
        self._log(f"[6] Schema Alignment completed in {step_time:.2f}s. Entities and relationships aligned to schema.")

        # === (7) Conflict Resolution ===
        step_start = time.time()
        # Check new triples against existing knowledge graph for contradictions
        non_conflicting_triples, prompt_tokens, completion_tokens, processing_time = self.conflict_agent.resolve_conflicts(
            aligned_triples, self.knowledge_graph["triples"]
        )
        total_prompt_tokens += prompt_tokens
        total_completion_tokens += completion_tokens
        total_time += processing_time
        
        step_time = time.time() - step_start
        self._log(f"[7] Conflict Resolution completed in {step_time:.2f}s. Non-conflicting triples: {len(non_conflicting_triples)}/{len(aligned_triples)}")

        # === (8) Evaluation ===
        step_start = time.time()
        # Score and filter triples by confidence, clarity, and relevance
        integrated_triples, prompt_tokens, completion_tokens, processing_time = self.evaluator_agent.finalize_triples(non_conflicting_triples)
        total_prompt_tokens += prompt_tokens
        total_completion_tokens += completion_tokens
        total_time += processing_time
        
        self.intermediate.integrated_triples = integrated_triples
        step_time = time.time() - step_start
        self._log(f"[8] Evaluation completed in {step_time:.2f}s. Final integrated triples: {len(integrated_triples)}/{len(non_conflicting_triples)}")

        # === Integration into KG ===
        # Add new entities to the knowledge graph
        for entity in aligned_entities:
            self.knowledge_graph["entities"].add(entity.name)
            
        # Add new triples to the knowledge graph
        for triple in integrated_triples:
            self.knowledge_graph["triples"].append(triple)
            
        # Update tracking metrics
        self.intermediate.prompt_tokens = total_prompt_tokens
        self.intermediate.completion_tokens = total_completion_tokens
        self.intermediate.processing_time = total_time
        
        total_pipeline_time = time.time() - pipeline_start_time
        self._log(f"KARMA pipeline completed in {total_pipeline_time:.2f}s. Added {len(integrated_triples)} new knowledge triples.")
        
        return integrated_triples
    
    ##########################################################################
    # Utility Methods
    ##########################################################################
    def export_knowledge_graph(self, output_path: str = None) -> Dict:
        """
        Export the knowledge graph as a dictionary or save to a file.
        
        Args:
            output_path: Optional file path to save the knowledge graph
            
        Returns:
            Dictionary representation of the knowledge graph
        """
        # Convert the KG to a serializable format
        kg_export = {
            "entities": list(self.knowledge_graph["entities"]),
            "triples": [asdict(triple) for triple in self.knowledge_graph["triples"]]
        }
        
        # Save to file if path is provided
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(kg_export, f, indent=2)
                
        return kg_export
    
    def save_intermediate_results(self, output_path: str):
        """
        Save all intermediate results from the pipeline for analysis.
        
        Args:
            output_path: Path to save the results
        """
        try:
            with open(output_path, 'w') as f:
                json.dump(self.intermediate.to_dict(), f, indent=2)
            logger.info(f"Intermediate results saved to {output_path}")
        except Exception as e:
            logger.error(f"Failed to save intermediate results: {str(e)}")
    
    def clear_knowledge_graph(self):
        """Reset the knowledge graph to empty state."""
        self.knowledge_graph = {
            "entities": set(),
            "triples": []
        }
        logger.info("Knowledge graph cleared")

    def print_statistics(self):
        """Print statistics about the current knowledge graph."""
        entity_count = len(self.knowledge_graph["entities"])
        triple_count = len(self.knowledge_graph["triples"])
        
        # Calculate types distribution
        entity_types = {}
        relation_types = {}
        
        for triple in self.knowledge_graph["triples"]:
            relation_types[triple.relation] = relation_types.get(triple.relation, 0) + 1
        
        # Print summary
        print(f"Knowledge Graph Statistics:")
        print(f"  - Entities: {entity_count}")
        print(f"  - Relationships: {triple_count}")
        print(f"  - Unique relation types: {len(relation_types)}")
        
        if relation_types:
            print("\nTop relation types:")
            for rel, count in sorted(relation_types.items(), key=lambda x: x[1], reverse=True)[:5]:
                print(f"  - {rel}: {count}")

##############################################################################
# Example Usage
##############################################################################

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run KARMA pipeline on a document')
    parser.add_argument('--input', required=True, help='Path to input PDF or text file')
    parser.add_argument('--api_key', required=True, help='API key')
    parser.add_argument('--base_url', default=None, help='Optional API base URL')
    parser.add_argument('--model', default='deepseek-chat', help='model name')
    parser.add_argument('--output', default='karma_output.json', help='Output file path')
    
    args = parser.parse_args()
    
    # Initialize KARMA
    karma = KARMA(api_key=args.api_key, base_url=args.base_url, model_name=args.model)
    
    # Process document
    print(f"Processing document: {args.input}")
    triples = karma.process_document(args.input)
    
    # Export results
    karma.export_knowledge_graph(args.output)
    karma.save_intermediate_results(f"intermediate_{args.output}")
    
    print(f"\nExtracted {len(triples)} knowledge triples.")
    print(f"Results saved to {args.output}")