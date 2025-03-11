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
        self.system_prompt = """You are the Ingestion Agent (IA). Your responsibility is to:
1. Retrieve raw publications from designated sources (e.g., PubMed, internal repositories).
2. Convert various file formats (PDF, HTML, XML) into a consistent normalized text format.
3. Extract metadata such as the title, authors, journal/conference name, publication date, and unique identifiers (DOI, PubMed ID).

Key Requirements:
- Handle OCR artifacts if the PDF is scanned (e.g., correct typical OCR errors where possible).
- Normalize non-ASCII characters (Greek letters, special symbols) to ASCII or minimal LaTeX markup when relevant (e.g., \\alpha).
- If certain fields cannot be extracted, leave them as empty or "N/A" but do not remove the key from the JSON.

Error Handling:
- In case of partial or unreadable text, mark the corrupted portions with placeholders (e.g., "[UNREADABLE]").
- If the document is locked or inaccessible, set an error flag in the output JSON.

POSITIVE EXAMPLE:
Input: A complex PDF with LaTeX symbols and tables about IL-6 inhibition in rheumatoid arthritis
Output: {
  "metadata": {
    "title": "Effects of IL-6 Inhibition on Inflammatory Markers in Rheumatoid Arthritis",
    "authors": ["Jane Smith", "Robert Johnson"],
    "journal": "Journal of Immunology",
    "pub_date": "2021-05-15",
    "doi": "10.1234/jimmunol.2021.05.123",
    "pmid": "33123456"
  },
  "content": "Introduction\\nInterleukin-6 (IL-6) is a key cytokine in the pathogenesis of rheumatoid arthritis (RA)...Methods\\nPatients (n=120) were randomized to receive either IL-6 inhibitor (n=60) or placebo (n=60)..."
}

NEGATIVE EXAMPLE:
Input: A complex PDF with LaTeX symbols and tables about IL-6 inhibition
Bad Output: {
  "title": "Effects of IL-6 Inhibition",
  "text": "Interleukin-6 inhibition showed p<0.05 significance..."
}
This is incorrect because it doesn't use the expected metadata/content structure and omits required metadata fields.
"""

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
    1) Segments normalized text into logical chunks
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
        self.system_prompt = """You are the Reader Agent (RA). Your goal is to parse the normalized text and generate logical segments (e.g., paragraph-level chunks) that are likely to contain relevant knowledge. Each segment must be accompanied by a numeric Relevance Score indicating its importance for downstream extraction tasks.

Scoring Heuristics:
- Use domain knowledge (e.g., presence of known keywords, synonyms, or known entity patterns) to increase the score.
- Use structural cues (e.g., headings like "Results", "Discussion" might have higher relevance for new discoveries).
- If a segment is purely methodological (e.g., protocols or references to equipment) with no new knowledge, assign a lower score.

Edge Cases:
- Very short segments (<30 characters) or references sections might be assigned a minimal score.
- If certain sections are incomplete or corrupted, still generate a segment but label it with "score": 0.0.

POSITIVE EXAMPLE:
Input: {
  "metadata": {"title": "Antimicrobial Study"...},
  "content": "Abstract\n We tested new...\n Methods\n The protocol was...\n Results\n The compound inhibited growth of S. aureus with MIC of 0.5 μg/mL\n"
}
Output: {
  "segments": [
    {"text": "Abstract We tested new...", "score": 0.85},
    {"text": "Methods The protocol was...", "score": 0.30},
    {"text": "Results The compound inhibited growth of S. aureus with MIC of 0.5 μg/mL", "score": 0.95}
  ]
}

NEGATIVE EXAMPLE:
Input: {
  "metadata": {"title": "Antimicrobial Study"...},
  "content": "Abstract\n We tested new...\n Methods\n The protocol was...\n Results\n The compound inhibited growth of S. aureus with MIC of 0.5 μg/mL\n"
}
Bad Output: {
  "segments": [
    {"text": "The entire paper discusses antimicrobial compounds", "score": 0.5}
  ]
}
This is incorrect because it doesn't segment the text properly into logical chunks and doesn't assign differentiated relevance scores.
"""
    
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
                score = float(float_matches[0])
            else:
                score = 0.5  # default if no float found
            
            prompt_tokens = response.usage.prompt_tokens
            completion_tokens = response.usage.completion_tokens
            processing_time = time.time() - start_time

            return max(0.0, min(1.0, score)), prompt_tokens, completion_tokens, processing_time
        except Exception as e:
            logger.warning(f"Failed to parse relevance for segment. Error: {e}")
            return 0.5, 0, 0, time.time() - start_time  # default

class SummarizerAgent:
    """
    Summarizer Agent (SA):
    1) Converts high-relevance segments into concise summaries
    2) Preserves technical details important for knowledge extraction
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
        self.system_prompt = """You are the Summarizer Agent (SA). Your task is to convert high-relevance segments into concise summaries while retaining technical detail such as gene symbols, chemical names, or numeric data that may be crucial for entity/relationship extraction.

Summarization Rules:
- Avoid discarding domain-specific terms that could indicate potential relationships. For example, retain "IL-6" or "p53" references precisely.
- If numeric data is relevant (e.g., concentrations, p-values), incorporate them verbatim if possible.
- Keep the summary length under 100 words to reduce computational overhead for downstream agents.

Handling Irrelevant Segments:
- If the Relevance Score is below a threshold (e.g., 0.2), you may skip or heavily compress the summary.
- Mark extremely low relevance segments with "summary": "[OMITTED]" if not summarizable.

POSITIVE EXAMPLE:
Input: {
  "segments": [
    {"text": "In this study, IL-6 blockade with tocilizumab led to significant reduction in DAS28 scores (p<0.001) compared to placebo. Patients receiving 8mg/kg showed the greatest improvement (mean reduction 3.2 points).", "score": 0.90},
    {"text": "The control group had p=0.01 in the secondary analysis.", "score": 0.75}
  ]
}
Output: {
  "summaries": [
    {"original_text": "In this study, IL-6 blockade with tocilizumab led to significant reduction in DAS28 scores (p<0.001) compared to placebo. Patients receiving 8mg/kg showed the greatest improvement (mean reduction 3.2 points).",
     "summary": "IL-6 blockade with tocilizumab significantly reduced DAS28 scores (p<0.001) vs placebo. The 8mg/kg dose had the best results (mean reduction 3.2 points).",
     "score": 0.90},
    {"original_text": "The control group had p=0.01 in the secondary analysis.",
     "summary": "The control group showed statistical significance (p=0.01) in secondary analysis.",
     "score": 0.75}
  ]
}

NEGATIVE EXAMPLE:
Input: {
  "segments": [
    {"text": "In this study, IL-6 blockade with tocilizumab led to significant reduction in DAS28 scores (p<0.001) compared to placebo. Patients receiving 8mg/kg showed the greatest improvement (mean reduction 3.2 points).", "score": 0.90}
  ]
}
Bad Output: {
  "summaries": [
    {"original_text": "In this study, IL-6 blockade with tocilizumab led to significant reduction in DAS28 scores (p<0.001) compared to placebo. Patients receiving 8mg/kg showed the greatest improvement (mean reduction 3.2 points).",
     "summary": "A drug helped patients improve their condition.",
     "score": 0.90}
  ]
}
This is incorrect because it discarded crucial technical details (IL-6, tocilizumab, DAS28 scores, p-value, dosage).
"""
    
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
    1) Identifies biomedical entities in summarized text
    2) Classifies entity types and links to ontologies where possible
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
        self.system_prompt = """You are the Entity Extraction Agent (EEA). Based on summarized text, your objective is to:
1. Identify biomedical entities (Disease, Drug, Gene, Protein, Chemical, etc.).
2. Link each mention to a canonical ontology reference (e.g., UMLS, MeSH, SNOMED CT).

LLM-driven NER:
- Use domain-specific knowledge to identify synonyms ("acetylsalicylic acid" → Aspirin).
- Include multi-word expressions ("breast cancer" as a single mention).

Handling Ambiguity:
- If multiple ontology matches are possible, list the top candidate plus a short reason or partial mention of the second-best match.
- If no suitable ontology reference is found, set "normalized_id": "N/A" and keep the raw mention.

POSITIVE EXAMPLE:
Input: {
  "summary": "We tested Aspirin and ibuprofen for headache relief. Aspirin (100mg) was more effective for migraine, while ibuprofen (400mg) worked better for tension headaches. PTGS2 inhibition was the proposed mechanism."
}
Output: {
  "entities": [
    {"mention": "Aspirin", "type": "Drug", "normalized_id": "MESH:D001241"},
    {"mention": "ibuprofen", "type": "Drug", "normalized_id": "MESH:D007052"},
    {"mention": "headache", "type": "Symptom", "normalized_id": "MESH:D006261"},
    {"mention": "migraine", "type": "Disease", "normalized_id": "MESH:D008881"},
    {"mention": "tension headaches", "type": "Disease", "normalized_id": "MESH:D013313"},
    {"mention": "PTGS2", "type": "Gene", "normalized_id": "NCBI:5743"}
  ]
}

NEGATIVE EXAMPLE:
Input: {
  "summary": "We tested Aspirin for headache relief at a dosage of 100 mg."
}
Bad Output: {
  "entities": [
    {"mention": "Aspirin for headache relief", "type": "Medication", "normalized_id": "unknown"},
    {"mention": "100", "type": "Measurement", "normalized_id": "N/A"},
    {"mention": "mg", "type": "Unit", "normalized_id": "N/A"}
  ]
}
This is incorrect because it didn't properly separate entities (Aspirin and headache should be separate) and created overly granular entities for dosage information.
"""

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
    1) Identifies relationships between extracted entities
    2) Classifies relationship types
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
        self.system_prompt = """You are the Relationship Extraction Agent (REA). Given a text snippet plus a set of recognized entities, your mission is to detect possible relationships (e.g., treats, causes, interactsWith, inhibits).

LLM-based Relation Classification:
- Consider grammar structures (e.g., "X was observed to inhibit Y") and domain patterns ("X reduces expression of Y").
- Allow multiple relationship candidates if the text is ambiguous or suggests multiple interactions.

Negative Relation Handling:
- If the text says "Aspirin does not treat migraine," the relationship (Aspirin, treats, migraine) is negative. Output no relationship in such cases.
- Recognize negation cues ("no effect", "absence of association").

POSITIVE EXAMPLE:
Input: {
  "summary": "Aspirin was shown to reduce headaches by inhibiting prostaglandin synthesis. It has no effect on hypertension.",
  "entities": [
    {"mention": "Aspirin", "type": "Drug", "normalized_id": "MESH:D001241"},
    {"mention": "headaches", "type": "Disease", "normalized_id": "MESH:D006261"},
    {"mention": "prostaglandin", "type": "Chemical", "normalized_id": "MESH:D011453"},
    {"mention": "hypertension", "type": "Disease", "normalized_id": "MESH:D006973"}
  ]
}
Output: {
  "relationships": [
    {"head": "Aspirin", "relation": "treats", "tail": "headaches", "confidence": 0.95},
    {"head": "Aspirin", "relation": "inhibits", "tail": "prostaglandin", "confidence": 0.90}
  ]
}
Note: No relationship is extracted between Aspirin and hypertension due to the negation.

NEGATIVE EXAMPLE:
Input: {
  "summary": "Aspirin was shown to reduce headaches by inhibiting prostaglandin synthesis.",
  "entities": [
    {"mention": "Aspirin", "type": "Drug", "normalized_id": "MESH:D001241"},
    {"mention": "headaches", "type": "Disease", "normalized_id": "MESH:D006261"},
    {"mention": "prostaglandin", "type": "Chemical", "normalized_id": "MESH:D011453"}
  ]
}
Bad Output: {
  "relationships": [
    {"head": "prostaglandin", "relation": "causes", "tail": "headaches", "confidence": 0.8}
  ]
}
This is incorrect because the text doesn't explicitly state this relationship. While it might be inferred, we should only extract relationships directly supported by the text.
"""

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
                max_tokens=4096
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
    1) Maps extracted entities to standard schema types
    2) Normalizes relationship labels
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
        self.system_prompt = """You are the Schema Alignment Agent (SAA). Newly extracted entities or relationships may not match existing KG classes or relation types. Your job is to determine how they should map onto the existing ontology or schema.

Ontology Reference:
- For each unknown entity, propose a parent type from {Drug, Disease, Gene, Chemical, Protein, Pathway, Symptom, ...} if not in the KG.
- For each unknown relation, map it to an existing relation if semantically close. Otherwise, propose a new label.

Confidence Computation:
- Consider lexical similarity, embedding distance, or domain rules (e.g., if an entity ends with "-in" or "-ase", it might be a protein or enzyme).
- Provide a final numeric score for how certain you are of the proposed alignment.

POSITIVE EXAMPLE:
Input: {
  "unknown_entities": ["TNF-alpha", "miR-21", "PDE4", "blood-brain barrier"],
  "unknown_relations": ["overexpresses", "disrupts"]
}
Output: {
  "alignments": [
    {"id": "TNF-alpha", "proposed_type": "Protein", "status": "mapped", "confidence": 0.95},
    {"id": "miR-21", "proposed_type": "RNA", "status": "new", "confidence": 0.90},
    {"id": "PDE4", "proposed_type": "Enzyme", "status": "mapped", "confidence": 0.85},
    {"id": "blood-brain barrier", "proposed_type": "Anatomical_Structure", "status": "mapped", "confidence": 0.95}
  ],
  "new_relations": [
    {"relation": "overexpresses", "closest_match": "upregulates", "status": "mapped", "confidence": 0.85},
    {"relation": "disrupts", "closest_match": "damages", "status": "new", "confidence": 0.70}
  ]
}

NEGATIVE EXAMPLE:
Input: {
  "unknown_entities": ["TNF-alpha", "miR-21"],
  "unknown_relations": ["overexpresses"]
}
Bad Output: {
  "alignments": [
    {"id": "TNF-alpha", "proposed_type": "Unknown", "status": "unknown"},
    {"id": "miR-21", "proposed_type": "Unknown", "status": "unknown"}
  ],
  "new_relations": [
    {"relation": "overexpresses", "closest_match": "unknown", "status": "unknown"}
  ]
}
This is incorrect because the agent should use domain knowledge to propose appropriate entity types and relation mappings, rather than marking everything as unknown.
"""

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
        self.system_prompt = """You are the Conflict Resolution Agent (CRA). Sometimes new triplets are detected that contradict existing knowledge (e.g., (DrugX, causes, DiseaseY) vs. (DrugX, treats, DiseaseY)). Your role is to classify these into Contradict, Agree, or Ambiguous, and decide whether the new triplet should be discarded, flagged for expert review, or integrated with caution.

LLM-based Debate:
- Use domain knowledge to see if relationships can coexist (e.g., inhibits vs. activates are typically contradictory for the same target).
- Consider partial contexts, e.g., different dosages or subpopulations.

Escalation Criteria:
- If the new triplet has high confidence but conflicts with old data that has lower confidence, consider overriding or review.
- If both are high confidence, label Contradict, prompt manual verification.

POSITIVE EXAMPLE:
Input: {
  "t_new": {"head": "Aspirin", "relation": "treats", "tail": "Headache", "confidence": 0.95},
  "t_existing": {"head": "Aspirin", "relation": "causes", "tail": "Headache", "confidence": 0.70}
}
Output: {
  "decision": "Contradict",
  "resolution": {
    "action": "review",
    "rationale": "These represent opposite effects. However, Aspirin can both treat existing headaches and cause headaches as a side effect in some individuals. Expert validation needed to clarify contexts."
  }
}

NEGATIVE EXAMPLE:
Input: {
  "t_new": {"head": "DrugX", "relation": "treats", "tail": "DiseaseY", "confidence": 0.95},
  "t_existing": {"head": "DrugX", "relation": "causes", "tail": "DiseaseY", "confidence": 0.40}
}
Bad Output: {
  "decision": "Agree",
  "resolution": {"action": "integrate", "rationale": "Both can be true."}
}
This is incorrect because it fails to recognize the direct contradiction between treats and causes, and doesn't provide a sufficiently detailed rationale.
"""

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
                max_tokens=4096,
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
        self.system_prompt = """You are the Evaluator Agent (EA). After extraction, alignment, and conflict resolution, each candidate triplet has multiple verification scores. Your duty is to aggregate these signals into final confidence, clarity, and relevance scores and decide whether to integrate each triplet into the KG.

For CONFIDENCE evaluation:
- Assess the factual correctness of the triple based on biomedical knowledge
- Consider the source reliability and extraction confidence
- Account for any conflict resolution outcomes

For CLARITY evaluation:
- Determine how unambiguous and well-defined the entities and relation are
- Check for vague terms or imprecise relationship descriptions
- Assess whether the triple would be interpretable to domain experts

For RELEVANCE evaluation:
- Evaluate how important and appropriate the triple is for the knowledge graph
- Consider whether it aligns with the domain focus
- Assess its potential utility for downstream applications

POSITIVE EXAMPLE:
Input: {
  "triple": {"head": "Metformin", "relation": "decreases", "tail": "blood glucose levels"}
}
Output: {
  "confidence": 0.95,
  "clarity": 0.90,
  "relevance": 0.85,
  "rationale": "Well-established mechanism of action for this first-line antidiabetic drug. The entities and relationship are clearly defined. Highly relevant for a biomedical knowledge graph."
}

NEGATIVE EXAMPLE:
Input: {
  "triple": {"head": "Drug X", "relation": "may influence", "tail": "some cellular processes"}
}
Output: {
  "confidence": 0.85,
  "clarity": 0.40,
  "relevance": 0.30,
  "rationale": "The relationship is vaguely defined with uncertain terms. The entities lack specificity. Limited utility in a biomedical knowledge graph."
}
"""

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

        I need you to assess how confident we can be that this relationship is scientifically accurate based on established biomedical knowledge.
        
        Consider factors like:
        1. Is this a well-established scientific fact?
        2. Is there substantial evidence in the literature supporting this claim?
        3. Would biomedical experts broadly agree with this statement?
        4. Does it align with current scientific understanding?
        
        Rate your confidence from 0.0 (completely uncertain) to 1.0 (extremely confident).
        Return only a single float value between 0.0 and 1.0.
        
        Example ratings:
        - 0.95-0.99: Well-established scientific facts with overwhelming evidence
        - 0.80-0.94: Strong scientific consensus with substantial supporting evidence
        - 0.60-0.79: Reasonably supported claims with some evidence base
        - 0.40-0.59: Mixed evidence or emerging hypotheses
        - 0.20-0.39: Limited evidence, preliminary findings
        - 0.01-0.19: Very weak evidence, highly speculative

        POSITIVE EXAMPLES:
        - "Metformin -> decreases -> blood glucose levels" = 0.97
        - "Statins -> inhibit -> HMG-CoA reductase" = 0.95
        - "Insulin resistance -> contributes to -> type 2 diabetes" = 0.93

        NEGATIVE EXAMPLES:
        - "Vitamin C -> cures -> cancer" = 0.12
        - "Unknown compound -> may affect -> some pathways" = 0.25
        
        Return only the numeric value as a float between 0.0 and 1.0.
        """
        
        start_time = time.time()
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=4096,
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

        Assess how clear, specific, and unambiguous this statement is to biomedical experts.
        
        Consider these aspects:
        1. Are the entities precise and unambiguous? (e.g., "ACE inhibitors" is clearer than "some drugs")
        2. Is the relationship type specific and well-defined? (e.g., "inhibits" is clearer than "affects")
        3. Would biomedical experts interpret this statement consistently?
        4. Are there any vague terms or imprecise language that reduce clarity?
        5. Does the statement avoid unnecessary hedging words (e.g., "may", "possibly", "might")?
        
        Rate clarity from 0.0 (very ambiguous) to 1.0 (perfectly clear).
        Return only a single float value between 0.0 and 1.0.
        
        Example ratings:
        - 0.95-0.99: Crystal clear, highly specific statements with no ambiguity
        - 0.80-0.94: Very clear statements with minor room for interpretation
        - 0.60-0.79: Mostly clear statements with some potential ambiguity
        - 0.40-0.59: Statements with moderate ambiguity or vagueness
        - 0.20-0.39: Significantly vague or ambiguous statements
        - 0.01-0.19: Extremely vague statements with minimal specificity

        POSITIVE EXAMPLES:
        - "Atorvastatin -> inhibits -> HMG-CoA reductase" = 0.95
        - "Aspirin -> irreversibly inhibits -> cyclooxygenase-1" = 0.97
        - "TNF-alpha -> induces -> apoptosis in tumor cells" = 0.85

        NEGATIVE EXAMPLES:
        - "Some medicines -> may affect -> various biological processes" = 0.20
        - "Drug X -> possibly influences -> certain cellular pathways" = 0.35
        
        Return only the numeric value as a float between 0.0 and 1.0.
        """
        
        start_time = time.time()
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You evaluate clarity of biomedical relationships."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=4096,
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

        Assess how important and appropriate this statement is for inclusion in a specialized {domain} knowledge graph.
        
        Consider these aspects:
        1. Is this directly relevant to {domain} research or practice?
        2. Does this provide valuable information to {domain} experts?
        3. Would this knowledge be useful for {domain} applications (research, clinical practice, drug discovery, etc.)?
        4. Is this specialized knowledge rather than general world knowledge?
        5. Does it align with the core interests of the {domain} field?
        
        Rate relevance from 0.0 (completely irrelevant) to 1.0 (highly relevant).
        Return only a single float value between 0.0 and 1.0.
        
        Example ratings:
        - 0.95-0.99: Core {domain} knowledge essential to the field
        - 0.80-0.94: Highly relevant information for {domain} specialists
        - 0.60-0.79: Moderately relevant information with clear applications
        - 0.40-0.59: Somewhat relevant but not central to the domain
        - 0.20-0.39: Marginally relevant information
        - 0.01-0.19: Largely irrelevant to the {domain} domain

        POSITIVE EXAMPLES:
        - "Metformin -> decreases -> insulin resistance" = 0.95
        - "BRCA1 mutation -> increases risk of -> breast cancer" = 0.97
        - "Tumor necrosis factor -> stimulates -> inflammatory response" = 0.90

        NEGATIVE EXAMPLES:
        - "William Shakespeare -> wrote -> Hamlet" = 0.05 (literature knowledge, not biomedical)
        - "Earth -> orbits -> Sun" = 0.01 (general knowledge, not specialized)
        - "Company X -> manufactures -> medical devices" = 0.30 (business information, not core biomedical knowledge)
        
        Return only the numeric value as a float between 0.0 and 1.0.
        """
        
        start_time = time.time()
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": f"You evaluate {domain} relevance of relationships."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=4096,
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
