# LLM Practical Portfolio

PUSL3205 Large Language Models and Generative AI, practical portfolio of Oshani Hewavitharane.

This repository has my work from the six lab sessions, in order. They start with a sentiment classifier built from word counts, then a Transformer written from scratch, and end with an agent that uses tools, memory, a RAG search and Google Calendar. Each lab has its own folder, and each folder's README follows the same six sections: laboratory session, objective, tasks and implementation, results and evidence, observations, and learning outcomes.

What I took from the module as a whole is in [Final-Reflection.md](Final-Reflection.md).

## Labs

| Lab | What I did | Main tools | Result |
|---|---|---|---|
| [01 Machine learning](Lab-01-Machine-Learning) | Cleaned 3,000 review sentences, turned them into Bag of Words counts and trained two sentiment classifiers | pandas, NLTK, scikit-learn | 82.33% accuracy with Logistic Regression; "not bad" still came out negative |
| [02 LLMs in everyday life](Lab-02-LLM-Everyday-Life) | Used an LLM as a product-development partner for Group 1, The Forgetful Student | Microsoft Copilot, Claude | PackSync, a timetable app with Bluetooth tags, with a product photo and app screens |
| [03 Tiny Transformer](Lab-03-Tiny-Transformer) | Wrote a character-level Transformer from its parts and trained it on TinyStories | PyTorch | 113,096 parameters, perplexity 4.14; layer and head experiments |
| [04 Fine-tuning](Lab-04-Fine-Tune) | Fine-tuned Qwen2.5-0.5B-Instruct with QLoRA on 30 student-support Q&A pairs | transformers, PEFT, TRL, bitsandbytes (Colab T4) | 0.22% of the parameters trained in 42 seconds; answers went from about 100 words to one sentence |
| [05 RAG pipeline](Lab-05-RAG-Pipeline) | Built a local assistant that answers from university PDFs and PDFs the user uploads, then compared three chunking methods | Ollama, ChromaDB, Streamlit | Answers from the NSBM guide; parent-child retrieval was the only method that got all 15 modules into the context |
| [06 AI agents](Lab-06-AI-Agents) | Built an agent with Windows tools, SQLite memory, the Lab 5 RAG as a tool and Google Calendar (parts A to D) | LangChain, Ollama, SQLite, Google Calendar API | Right tool for all six test queries; remembers facts after a restart |

## Repository layout

```
LLM-Practical-Portfolio/
├── README.md
├── Final-Reflection.md
├── Lab-01-Machine-Learning/
│   ├── README.md
│   ├── data/
│   ├── notebook/
│   └── screenshots/
├── Lab-02-LLM-Everyday-Life/
│   ├── README.md
│   ├── activity-outcomes.md
│   ├── llm-process.md
│   └── screenshots/
├── Lab-03-Tiny-Transformer/
│   ├── README.md
│   ├── notebooks/
│   └── results/
├── Lab-04-Fine-Tune/
│   ├── README.md
│   ├── notebooks/
│   └── screenshots/
├── Lab-05-RAG-Pipeline/
│   ├── README.md
│   ├── ingest.py, rag_pipeline.py, app.py
│   ├── data/
│   ├── group-activity/
│   └── screenshots/
└── Lab-06-AI-Agents/
    ├── README.md
    ├── calculator_agent.py, multi_tool_agent.py, memory_agent.py, rag_agent.py, calendar_agent.py
    └── screenshots/
```

I did Lab 2 on my own, so it has no `group-activity` folder. Its outcomes are in `activity-outcomes.md`, and the prompts are in `llm-process.md`.

## Running the code

- Labs 1, 3, 5 and 6 ran on my Windows 11 laptop in VS Code, with Python 3.14.6 in a `.venv` at the top of the repository. Labs 3, 5 and 6 ran on the CPU.
- Lab 4 runs in Google Colab on a T4 GPU, because 4-bit loading with `bitsandbytes` needs an NVIDIA GPU.
- Labs 5 and 6 need [Ollama](https://ollama.com) with `llama3.2:3b` and `qwen3-embedding:0.6b`, and their scripts are run from the top folder of the repository. Each of these folders has a `requirements.txt`.

The commands for each lab are in its README.

## Not in this repository

- `credentials.json` and `token.json` for Google Calendar (Lab 6), because they would give access to my account.
- `student_memory.db` (Lab 6), which holds personal facts the agent saved.
- `chroma_db` (Lab 5), which `ingest.py` builds again from the PDF.
- The `.venv` folder.
