# 📈 Financial News Sentiment & Entity Tracker

An end-to-end **NLP-based Financial News Analyzer** that extracts **market sentiment** and identifies **key entities** from financial news headlines.  
Built with **Streamlit**, it provides an interactive dashboard where users can upload custom datasets and instantly visualize insights.

---

## 🚀 Key Features

- **Sentiment Classification:** Uses **FinBERT** to classify financial text as *Positive*, *Negative*, or *Neutral*.
- **Entity Extraction:** Employs **spaCy** to identify company names and other financial entities.
- **Interactive Web App:** Streamlit dashboard for real-time visualization and CSV uploads.
- **Data Insights:** Displays sentiment distribution, entity frequency, and confidence scores using **Plotly** charts.

---

## 🛠️ Technology Stack

| Category | Tool / Library | Purpose |
| :--- | :--- | :--- |
| **Language** | Python 3.10+ | Core programming language |
| **Model** | ProsusAI/FinBERT | Financial sentiment classification |
| **Entity Extraction** | spaCy (`en_core_web_sm`) | Named Entity Recognition (NER) |
| **Framework** | Streamlit | Web-based interactive dashboard |
| **Data Handling** | Pandas, NumPy | Data ingestion and manipulation |
| **Visualization** | Plotly Express | Charts and KPIs |
| **Backend Engine** | PyTorch / Transformers | Running FinBERT |

---

## 📂 Repository Structure

| File | Description |
| :--- | :--- |
| `app.py` | Main Streamlit app — runs the NLP pipeline and dashboard |
| `requirements.txt` | Python dependencies |
| `notebook.ipynb` | Jupyter notebook for model testing and data exploration |
| `final_web_data.csv` | Sample processed dataset for demo |
| `all-data.csv` | Raw financial news dataset |

---

## ⚙️ Running the Project Locally

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/<your-username>/finnews-nlp.git
cd finnews-nlp
