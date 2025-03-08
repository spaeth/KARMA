# KARMA: Leveraging Multi-Agent LLMs for Automated Knowledge Graph Enrichment 🤖 

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/) [![Arxiv](https://img.shields.io/badge/arxiv-2502.06472)](https://arxiv.org/abs/2502.06472)

KARMA is a natural language processing framework that leverages a coordinated multi-agent system to automatically extract, validate, and integrate scientific knowledge into structured knowledge graphs. By employing specialized Large Language Model (LLM) agents, KARMA ensures high-quality knowledge extraction while maintaining semantic consistency.

![Overall](https://github.com/user-attachments/assets/477485dc-8d56-4b05-95a4-77547e5ceb39)

> [!NOTE]
> A primitive version of KARMA has released, we are still working on updating the pipeline. Feel free to provide suggestions.

## 🌟 Highlights

- **Multi-Agent Architecture**: Coordinated system of specialized agents for robust knowledge extraction
- **Quality-Focused**: Multi-stage validation with confidence, clarity, and relevance scoring
- **Domain-Adaptive**: Specialized for scientific literature processing
- **Scalable**: Handles both single documents and large-scale batch processing

## 🎯 Core Capabilities

### 1. Document Processing
- PDF and text parsing
- Context-aware content segmentation
- Content summarization

### 2. Knowledge Extraction
- Entity recognition
- Relationship identification
- Semantic triple formation
- Conflict Resolution

### 3. Quality Assurance
- Multi-dimensional scoring system
- Domain relevance validation

## 🚀 Quick Start

To quickly get started with KARMA, you can run the provided script 

```bash
python karma_pipeline.py \
--input path/to/document.pdf \
--api_key your-openai-api-key \
--model gpt-4o \
--output results.json
```

## 📊 Output Format

KARMA generates knowledge triples with quality metrics:

```json
{
   "results": [
    {
      "head": "KARMA",
      "relation": "uses",
      "tail": "Multi-Agent LLMs",
      "confidence": 0.85,
      "clarity": 0.92,
      "relevance": 0.78
    }, 
      ... 
    ]
}
```

## 🛠️ Technical Requirements

- Python 3.8+
- Dependencies:
  - `openai>=1.0.0`: LLM integration
  - `PyPDF2>=3.0.0`: PDF processing
  - `spacy>=3.0.0`: NLP processing
  - `networkx>=2.6.0`: Knowledge graph operations
  - `typing-extensions>=4.0.0`: Type hints

## 🤝 Contributing

We welcome contributions!

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📬 Contact

For questions and feedback:
- Open an issue
- Email: yxlu0613@gmail.com

## 🔖 Cite Information

@article{lu2025karma,
  title={KARMA: Leveraging Multi-Agent LLMs for Automated Knowledge Graph Enrichment},
  author={Lu, Yuxing and Wang, Jinzhuo},
  journal={arXiv preprint arXiv:2502.06472},
  year={2025}
}

## 🙏 Acknowledgments

- All LLMs we ues in our experiments
- PubMed
- All contributors and users
