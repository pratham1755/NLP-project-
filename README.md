# 🎯 NLP Mini Project: Resume and Job Description Matcher
> **AI Job Matching Analyzer: Score Your Resume Against Any Job | Resume Optimizer Pro**  
> *Department of Data Science & Engineering · Natural Language Processing · Python*

---

## 📌 1. Project Overview & Problem Statement

Recruiters regularly receive hundreds of resumes for a single job opening, making manual candidate screening inefficient, time-consuming, and prone to human bias. At the same time, job applicants often struggle to understand whether their resumes adequately align with job descriptions or which specific technical skills they are missing.

This project delivers a full-stack Natural Language Processing (NLP) web application that automates text comparison between a candidate's resume and a target job description. The system parses, cleans, vectorizes, and analyzes text to deliver:
- **Numerical Text Similarity Score (0 - 100%)** via TF-IDF and Cosine Similarity
- **Skill Gap Analysis** (Matched Skills vs. Missing Skills vs. Bonus Candidate Skills)
- **Categorical Skill Breakdown** (Data Science, Cloud, Programming, BI, etc.)
- **TF-IDF Keyword Relevance** and high-weight missing vocabulary
- **Personalized Recommendations** for tailoring bullet points and ATS optimization
- **Downloadable Audit Report** in formatted plain text
- **Interactive College Viva & Architecture Defense Guide** built directly into the UI

---

## 🛠️ 2. Technology Stack

| Technology | Purpose | Why We Use It |
|---|---|---|
| **Python 3.10+** | Core Language | Industry standard for Data Science and NLP engineering |
| **Streamlit** | Web Application UI | Fast, reactive web dashboard with native Python data binding |
| **spaCy (`en_core_web_sm`)** | NLP Pipeline | Industrial-strength tokenization, stop-word filtering, and lemmatization |
| **Scikit-learn** | Machine Learning | `TfidfVectorizer` (unigrams + bigrams) and `cosine_similarity` computation |
| **PyMuPDF / pdfminer** | PDF Parsing | Robust extraction of raw text from multi-page PDF resumes |
| **python-docx** | DOCX Parsing | Extracts text, headings, and tables from Word documents |
| **Pandas** | Data Wrangling | Organizes keyword weights and category comparisons into tabular structures |
| **Plotly** | Data Visualization | Interactive modern gauge meters and grouped skill comparison bar charts |

---

## 🏗️ 3. System Architecture & Workflow

```
 [User Uploads Resume (PDF/DOCX/TXT) & Pastes Job Description]
                           │
                           ▼
 ┌─────────────────────────────────────────────────────────────┐
 │                Module 1: Document Parser                    │
 │       (Extracts text from PDF via PyMuPDF/pdfminer,         │
 │            DOCX via python-docx, or plain TXT)              │
 └──────────────────────────────┬──────────────────────────────┘
                                │
                                ▼
 ┌─────────────────────────────────────────────────────────────┐
 │                Module 2: NLP Preprocessing                  │
 │    (Lowercasing · Regex Cleaning · spaCy Tokenization ·     │
 │            Stop-word Removal · Lemmatization)               │
 └──────────────┬───────────────────────────────┬──────────────┘
                │                               │
                ▼                               ▼
 ┌─────────────────────────────┐ ┌─────────────────────────────┐
 │ Module 3: Similarity Engine │ │  Module 4: Skill Extractor  │
 │  • TfidfVectorizer (1-2)    │ │  • Domain Skill Taxonomy    │
 │  • Cosine Similarity Metric │ │  • Word Boundary Regex      │
 │  • Shared & Missing N-grams │ │  • Matched & Missing Skills │
 └──────────────┬──────────────┘ └──────────────┬──────────────┘
                │                               │
                └───────────────┬───────────────┘
                                │
                                ▼
 ┌─────────────────────────────────────────────────────────────┐
 │             Module 5: Results & Reporting                   │
 │   (Interactive Gauge & Bar Charts · Recommendations ·       │
 │                Downloadable Audit Report)                   │
 └─────────────────────────────────────────────────────────────┘
```

---

## 📂 4. Project Directory Structure

```
nlp project/
│
├── app.py                      # Main Streamlit web application & UI dashboard
├── requirements.txt            # Python dependencies
├── test_system.py              # Automated unit tests for all 5 modules
├── create_sample_documents.py  # Script generating sample PDF & DOCX resumes
├── README.md                   # Installation, usage, and quickstart documentation
├── PROJECT_REPORT_VIVA_GUIDE.md# Complete college project report & viva question bank
│
├── modules/                    # Modular Python backend architecture
│   ├── __init__.py
│   ├── document_parser.py      # Module 1: Resume text extraction (PDF/DOCX/TXT)
│   ├── nlp_preprocessor.py     # Module 2: Text cleaning, tokenization & lemmatization
│   ├── similarity_engine.py    # Module 3: TF-IDF vectorization & Cosine Similarity
│   ├── skill_extractor.py      # Module 4: Categorized taxonomy & skill gap matching
│   └── report_generator.py     # Module 5: Tailored recommendations & text report
│
└── sample_data/                # Preset files for testing and viva demonstrations
    ├── sample_data_analyst_resume.txt
    ├── sample_data_analyst_resume.pdf
    ├── sample_data_analyst_resume.docx
    ├── sample_job_description_data_analyst.txt
    ├── sample_swe_resume.txt
    └── sample_job_description_swe.txt
```

---

## ⚡ 5. Installation & Setup Guide (Windows / Mac / Linux)

### Step 1: Clone / Open Directory
Open terminal or Command Prompt in the project folder:
```bash
cd "c:\Users\Prathmesh Doke\nlp project"
```

### Step 2: Create and Activate Virtual Environment (Optional but Recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Required Libraries
```bash
pip install -r requirements.txt
```

### Step 4: Download spaCy English Language Model
```bash
python -m spacy download en_core_web_sm
```

### Step 5: Run Automated Verification Tests
```bash
python test_system.py
```
*You should see `ALL 5 MODULE TESTS PASSED PERFECTLY!`*

### Step 6: Launch the Streamlit Web Application
```bash
streamlit run app.py
```
The app will automatically open in your default browser at:  
👉 **`http://localhost:8501`**

---

## 🧪 6. Testing & Demonstration Guide

### Quick Demonstration via Presets:
1. Open the app at `http://localhost:8501`.
2. Look at the left sidebar under **⚡ 1-Click Viva Presets**.
3. Click **📊 Data Analyst**.
4. Both the sample resume and job description will be populated automatically and analyzed.
5. Review the **Match Score Gauge**, **Matched Skills**, **Missing Skills**, and **Download Report**.

### Manual Testing with Sample Files:
- Upload `sample_data/sample_data_analyst_resume.pdf` or `sample_data/sample_data_analyst_resume.docx`.
- Paste the content of `sample_data/sample_job_description_data_analyst.txt` into the Job Description box.
- Click **🚀 Analyze Resume & Match Job**.

### Expected Test Results:
- **Similarity Score**: ~51.7% (Moderate Match)
- **Required Skill Coverage**: ~66.7%
- **Matched Skills**: Python, SQL, Pandas, NumPy, Excel, Scikit-Learn, Machine Learning, Data Cleaning, Statistics, Communication
- **Missing Skills**: Power BI, AWS, Data Visualization, Collaboration, Data Science

---

## 🎓 7. College Viva Quick Cheat Sheet

| Question | Short Answer |
|---|---|
| **What is NLP?** | A subfield of AI enabling computers to process, understand, and extract meaning from human language. |
| **Why TF-IDF?** | Converts text into numbers while downweighting ubiquitous words and prioritizing distinctive technical keywords. |
| **Why Cosine Similarity over Euclidean?** | Cosine similarity evaluates the angle between vectors rather than length, making it invariant to resume page length. |
| **Why Lemmatization over Stemming?** | Stemming cuts word ends heuristically (e.g., "studying" -> "study"), whereas lemmatization uses morphological analysis to return valid dictionary root words (e.g., "better" -> "good"). |
| **Is this deep learning?** | No, it uses traditional, interpretable NLP and ML (TF-IDF vector space model). This ensures fast execution and zero GPU requirement. |
