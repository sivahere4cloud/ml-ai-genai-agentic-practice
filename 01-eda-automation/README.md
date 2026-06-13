# 01 — EDA Automation with Mistral, Ollama & Gradio

A locally-hosted tool that automates exploratory data analysis on tabular datasets. The user uploads a CSV file through a browser interface, the application generates descriptive statistics using Pandas, and the **Mistral 7B** language model (running locally via **Ollama**) produces plain-English insights derived from the dataset's statistical profile.

The entire pipeline runs offline after model setup — no API keys, no external calls, no data leaves the user's machine.

## 🎯 What It Does

1. User uploads a CSV file via a Gradio web interface
2. The application reads the dataset into a Pandas DataFrame
3. Descriptive statistics (`mean`, `std`, `min`, `max`, quartiles) are computed using `df.describe()`
4. The statistical summary is passed to Mistral 7B as a prompt
5. The LLM returns natural-language interpretations of the data
6. Insights are displayed in the browser

## 🛠️ Tech Stack

| Component | Purpose |
|---|---|
| **Python** | Core language |
| **Pandas** | Data loading and statistical summarisation |
| **Ollama** | Local LLM runtime |
| **Mistral 7B** | Language model generating dataset insights |
| **Gradio** | Web-based user interface |

## 🚀 How to Run

### Prerequisites

- Python 3.9+
- [Ollama](https://ollama.com/download) installed locally
- Mistral model pulled:
```bash
  ollama pull mistral
```

### Setup

```bash
pip install pandas ollama gradio
```

### Launch

```bash
python app.py
```

The Gradio interface launches in the browser. Upload any CSV and the LLM-generated insights appear after processing.

## 🔮 Planned Enhancements

- Missing value analysis and imputation suggestions
- Correlation matrix and statistical visualizations
- Outlier detection
- Conversational query interface over the dataset
- Export results as a downloadable report
- Support for Excel and Parquet input formats

## 📝 Notes

A minimal-architecture project demonstrating the integration pattern of local LLMs, data analysis tooling, and interactive web UIs. The current implementation focuses on the core pipeline; extended EDA features are planned in future iterations.
