# 📘 COLLEGE MINI PROJECT REPORT & VIVA DEFENSE GUIDE
## Resume and Job Description Matcher using Natural Language Processing
**Degree Program:** Bachelor of Engineering / Technology in Computer Science & Data Engineering  
**Subject:** Natural Language Processing (NLP) / Data Science Laboratory  
**Academic Year:** 2024 - 2025

---

## TABLE OF CONTENTS
1. [Cover Page & Declaration Template](#1-cover-page--declaration-template)
2. [Abstract](#2-abstract)
3. [Introduction & Problem Statement](#3-introduction--problem-statement)
4. [Project Objectives](#4-project-objectives)
5. [Literature Review](#5-literature-review)
6. [Existing System vs Proposed System](#6-existing-system-vs-proposed-system)
7. [Hardware & Software Requirements](#7-hardware--software-requirements)
8. [System Architecture & Flowchart](#8-system-architecture--flowchart)
9. [Mathematical Methodology & Algorithms](#9-mathematical-methodology--algorithms)
10. [Module Breakdown & Implementation Details](#10-module-breakdown--implementation-details)
11. [Test Cases & Experimental Results](#11-test-cases--experimental-results)
12. [Limitations & Future Scope](#12-limitations--future-scope)
13. [Top 15 College Viva Questions & Answers](#13-top-15-college-viva-questions--answers)
14. [References](#14-references)

---

## 1. COVER PAGE & DECLARATION TEMPLATE

```
                         A MINI PROJECT REPORT ON
            RESUME AND JOB DESCRIPTION MATCHER USING NLP

                Submitted in partial fulfillment of the
                    requirements for the award of
                              Degree of
                      BACHELOR OF TECHNOLOGY
                                 IN
                  COMPUTER SCIENCE AND ENGINEERING /
                       DATA SCIENCE ENGINEERING

                              Submitted By:
                      [Student Name] - [Roll Number]

                         Under the Guidance of:
                    [Internal Guide Name & Designation]

             DEPARTMENT OF COMPUTER SCIENCE AND ENGINEERING
                 [NAME OF COLLEGE / UNIVERSITY / CAMPUS]
                               [YEAR]
```

### CERTIFICATE OF APPROVAL
> This is to certify that the mini-project report entitled **"Resume and Job Description Matcher using Natural Language Processing"** submitted by **[Student Name]** (Roll No: **[Roll Number]**) is a bona fide record of work carried out by them under my supervision and guidance.

**Project Guide:** _______________________  
**Head of Department:** _________________  
**External Examiner:** ___________________  

---

## 2. ABSTRACT

This project presents a **Resume and Job Description Matcher** using **Natural Language Processing (NLP)** and Machine Learning techniques. Recruiters receive hundreds of candidate resumes for any single opening, making manual screening labor-intensive, slow, and susceptible to cognitive fatigue. Simultaneously, job candidates often struggle to determine whether their resume content matches the explicit requirements of a target job description.

The developed system provides an automated, objective, and interactive web application built with **Python** and **Streamlit**. It accepts resumes in **PDF, DOCX, and TXT** formats, extracts raw text, and subjects both the resume and the job description to an NLP preprocessing pipeline (lowercasing, punctuation stripping, tokenization, stop-word removal, and morphological lemmatization via **spaCy**). The preprocessed texts are vectorized using **TF-IDF (Term Frequency–Inverse Document Frequency)** spanning unigrams and bigrams, and their degree of textual alignment is quantified using **Cosine Similarity**. Additionally, a specialized **Skill Extraction Engine** utilizes a curated domain taxonomy and boundary regex to identify matched skills, missing skills, and bonus skills. Results are presented through an interactive dashboard featuring Plotly gauge and bar charts, actionable resume revision recommendations, and downloadable audit reports. The proposed system serves as an effective resume-optimization assistant rather than an automated exclusionary hiring filter.

---

## 3. INTRODUCTION & PROBLEM STATEMENT

### 3.1 Background
The recruitment lifecycle in modern technology industries involves screening vast volumes of candidate profiles. According to industry surveys, a corporate job opening receives an average of 250 resumes, of which only 4 to 6 candidates are invited for an interview. Because human recruiters typically spend less than 10 seconds on an initial resume scan, automated screening tools (Applicant Tracking Systems, or ATS) are widely used. However, commercial ATS solutions often operate as black boxes, providing candidates with zero transparency or guidance.

### 3.2 Problem Statement
Traditional manual resume screening is characterized by:
1. **High Latency:** Reviewing text documents manually requires significant human effort.
2. **Cognitive Inconsistency:** Review standards fluctuate across reviewers and time of day.
3. **Lack of Candidate Feedback:** Job applicants do not receive actionable insights regarding why their application did not pass initial screening or which critical skills were missing.

The proposed application resolves this dual-sided dilemma by providing a transparent, mathematically sound NLP engine that calculates text alignment, highlights missing domain skills, and suggests concrete resume revisions.

---

## 4. PROJECT OBJECTIVES

1. **Multi-Format Document Parsing:** Automatically ingest and parse raw textual data from PDF, DOCX, and plain text resumes.
2. **Standardized NLP Preprocessing:** Implement cleaning, lowercasing, tokenization, stop-word elimination, and lemmatization using spaCy.
3. **Numerical Vector Representation:** Convert textual documents into numerical feature vectors using TF-IDF vectorization with unigram and bigram coverage.
4. **Similarity Metric Computation:** Measure the directional alignment between document vectors using Cosine Similarity to yield an interpretable 0–100% score.
5. **Categorized Skill Extraction:** Cross-reference documents against a curated taxonomy (Programming, Data Science, Databases, Cloud/DevOps, BI, Soft Skills) with synonym handling.
6. **Actionable Feedback & Reporting:** Provide visual analytics (Plotly gauges/charts), concrete revision recommendations, and downloadable audit reports.

---

## 5. LITERATURE REVIEW

| Author & Year | Methodology / Focus | Strengths | Limitations |
|---|---|---|---|
| **Salton & Buckley (1988)** | Vector Space Model & TF-IDF Weighting | Highly interpretable, fast computation, works without training data. | Ignores word order and contextual semantic nuance. |
| **Honnibal et al. (2020)** | spaCy Industrial NLP Pipeline | State-of-the-art tokenization and morphological lemmatization. | Memory overhead for deep models. |
| **Roy et al. (2020)** | Resume Screening with Traditional ML | Fast classification of candidate profiles into job roles. | Lacks skill gap analysis and feedback for applicants. |
| **Proposed System (2025)** | Hybrid TF-IDF + Cosine Similarity + Categorized Skill Taxonomy + Streamlit UI | End-to-end pipeline, dual similarity + skill coverage metric, instant 1-click viva testing, actionable suggestions. | Relies on dictionary coverage for emerging skills. |

---

## 6. EXISTING SYSTEM VS PROPOSED SYSTEM

| Parameter | Existing / Manual System | Proposed NLP Matcher |
|---|---|---|
| **Screening Speed** | Minutes to hours per resume | Under 1 second per resume |
| **Objectivity** | Subject to reviewer fatigue and bias | Fully deterministic mathematical scoring |
| **Transparency** | Black-box rejection without explanation | Detailed breakdown of matched & missing skills |
| **File Format Support** | Dependent on reviewer software | Universal support for PDF, DOCX, and TXT |
| **Candidate Guidance** | None | Actionable suggestions & downloadable report |
| **Cost & Hardware** | Expensive HR labor or enterprise software | Free, open-source, runs on standard laptop CPU |

---

## 7. HARDWARE & SOFTWARE REQUIREMENTS

### 7.1 Hardware Requirements
- **Processor:** Intel Core i3 / AMD Ryzen 3 or higher (no GPU required)
- **RAM:** 4 GB minimum (8 GB recommended)
- **Storage:** At least 500 MB free hard disk space
- **Display:** 1366x768 screen resolution or higher

### 7.2 Software Requirements
- **Operating System:** Windows 10/11, macOS, or Ubuntu Linux
- **Language:** Python 3.10 to 3.13
- **Framework:** Streamlit 1.35+
- **NLP Libraries:** spaCy 3.7+ (`en_core_web_sm`), Scikit-learn 1.4+
- **Document Parsers:** PyMuPDF (fitz), pdfminer.six, python-docx
- **Visualization:** Plotly 5.18+, Pandas 2.0+

---

## 8. SYSTEM ARCHITECTURE & FLOWCHART

```
+-------------------------------------------------------------+
|                      USER INTERFACE                         |
|   Upload Resume (PDF/DOCX/TXT)  |  Paste Job Description    |
+------------------------------+------------------------------+
                               |
                               v
+-------------------------------------------------------------+
|               MODULE 1: DOCUMENT PARSER                     |
|  - Reads file stream bytes                                  |
|  - Extracts text via PyMuPDF / pdfminer / python-docx       |
|  - Normalizes line breaks and whitespace                    |
+------------------------------+------------------------------+
                               |
                               v
+-------------------------------------------------------------+
|               MODULE 2: NLP PREPROCESSOR                    |
|  - Cleans non-alphanumerics, URLs, email addresses          |
|  - Converts to lowercase                                    |
|  - Tokenizes string into linguistic units                   |
|  - Removes standard English stop-words                      |
|  - Lemmatizes tokens to base dictionary forms via spaCy     |
+------------------------------+------------------------------+
                               |
        +----------------------+----------------------+
        |                                             |
        v                                             v
+-------------------------------+   +---------------------------------+
|   MODULE 3: SIMILARITY ENGINE |   |   MODULE 4: SKILL EXTRACTOR     |
|  - TfidfVectorizer (N-gram 1-2|   |  - Domain Taxonomy (7 categories|
|  - Fits vocabulary on pair    |   |  - Word-boundary regex matching |
|  - Calculates Cosine Sim      |   |  - Synonym canonicalization     |
|  - Identifies top shared terms|   |  - Matched, Missing & Bonus     |
|  - Identifies missing JD terms|   |  - Skill Coverage Percentage    |
+---------------+---------------+   +-----------------+---------------+
                |                                     |
                +------------------+------------------+
                                   |
                                   v
+-------------------------------------------------------------+
|               MODULE 5: RESULTS & REPORTING                 |
|  - Plotly Match Gauge (0 - 100%)                            |
|  - Grouped Category Bar Chart                               |
|  - Matched & Missing Skill Badge Chips                      |
|  - Personalized Resume Improvement Recommendations          |
|  - Downloadable Plain Text Audit Report                     |
+-------------------------------------------------------------+
```

---

## 9. MATHEMATICAL METHODOLOGY & ALGORITHMS

### 9.1 Text Preprocessing (Lemmatization vs Stemming)
Raw textual documents contain diverse morphological inflections. For instance, the tokens `"analyzing"`, `"analyzed"`, and `"analyses"` all convey the core concept of `"analyze"`.
- **Stemming (Porter/Snowball):** Truncates suffixes heuristically (e.g., `"operational"` $\rightarrow$ `"oper"`), which frequently produces non-words.
- **Lemmatization (spaCy):** Utilizes full lexical vocabulary and morphological analysis based on part-of-speech context (e.g., `"operational"` $\rightarrow$ `"operation"`), preserving semantic meaning.

### 9.2 TF-IDF (Term Frequency–Inverse Document Frequency)
TF-IDF measures the relative importance of a term $t$ within a document $d$ relative to a document collection $D$.

$$\operatorname{TF}(t, d) = \frac{f_{t, d}}{\sum_{t' \in d} f_{t', d}}$$

$$\operatorname{IDF}(t, D) = \log\left(\frac{|D|}{|\{d \in D : t \in d\}|}\right) + 1$$

$$\operatorname{TF-IDF}(t, d, D) = \operatorname{TF}(t, d) \times \operatorname{IDF}(t, D)$$

- **Term Frequency (TF):** Boosts words that occur frequently in the resume or job description.
- **Inverse Document Frequency (IDF):** Penalizes terms that appear in every document (e.g., generic words) and elevates words that appear distinctively in the specific job posting (e.g., `PySpark`, `Tableau`).

### 9.3 Cosine Similarity
Documents vectorized by TF-IDF exist as points in an $n$-dimensional Euclidean space. Rather than calculating Euclidean distance (which is heavily corrupted by document length differences), Cosine Similarity measures the cosine of the angle $\theta$ between vectors $\mathbf{A}$ and $\mathbf{B}$:

$$\operatorname{Similarity}(\mathbf{A}, \mathbf{B}) = \cos(\theta) = \frac{\mathbf{A} \cdot \mathbf{B}}{\|\mathbf{A}\| \|\mathbf{B}\|} = \frac{\sum_{i=1}^n A_i B_i}{\sqrt{\sum_{i=1}^n A_i^2} \sqrt{\sum_{i=1}^n B_i^2}}$$

- **Range:** $0 \le \operatorname{Similarity} \le 1$.
- **Percentage:** Multiplied by 100 to yield a score between $0\%$ and $100\%$.
- **Length Invariance:** If Candidate A submits a 1-page resume and Candidate B submits a 3-page resume with the exact same density of skills, Cosine Similarity treats their angular alignment identically.

---

## 10. MODULE BREAKDOWN & IMPLEMENTATION DETAILS

### Module 1: Document Parser (`modules/document_parser.py`)
- Provides `extract_text_from_pdf()`, `extract_text_from_docx()`, and `extract_text_from_txt()`.
- Implements a resilient 3-layer fallback: PyMuPDF $\rightarrow$ pdfminer.six $\rightarrow$ PyPDF2.
- Handles encrypted streams, multi-column layouts, and bullet formatting.

### Module 2: NLP Preprocessor (`modules/nlp_preprocessor.py`)
- Cleans non-alphanumeric noise, URLs, email addresses, and phone numbers.
- Protects technical terms like `C++`, `C#`, and `.NET` prior to punctuation stripping.
- Executes tokenization and lemmatization using spaCy `en_core_web_sm`.

### Module 3: Similarity Engine (`modules/similarity_engine.py`)
- Instantiates Scikit-learn `TfidfVectorizer(ngram_range=(1, 2))`.
- Calculates Cosine Similarity between Resume and JD vectors.
- Isolates shared n-grams and calculates missing high-weight JD vocabulary terms.

### Module 4: Skill Extractor (`modules/skill_extractor.py`)
- Maintains a curated taxonomy across 7 technical categories:
  1. Programming Languages
  2. Data Science & AI/ML
  3. Databases & Big Data
  4. Business Intelligence & Visualization
  5. Cloud & DevOps
  6. Web & Software Development
  7. Soft Skills & Management
- Maps aliases to canonical names (e.g., `"powerbi"` $\rightarrow$ `"Power BI"`, `"k8s"` $\rightarrow$ `"Kubernetes"`).
- Computes `skill_coverage_pct = (matched / total_jd_skills) * 100`.

### Module 5: Results & Reporting (`modules/report_generator.py`)
- Formulates customized improvement suggestions:
  - Critical missing technical skill alerts
  - High-weight vocabulary gap suggestions
  - ATS formatting and STAR-method bullet guidelines
- Generates a timestamped audit report exportable as `.txt`.

---

## 11. TEST CASES & EXPERIMENTAL RESULTS

| Test Case ID | Test Scenario | Input Data | Expected Output | Actual Output | Status |
|---|---|---|---|---|---|
| **TC-01** | PDF Resume Parsing | Valid single-page PDF resume | Plain text extracted (>2,000 chars) | Text extracted (2,574 chars) | **PASS** |
| **TC-02** | DOCX Resume Parsing | Valid formatted Word document | Text extracted including bullet points | Text extracted (2,422 chars) | **PASS** |
| **TC-03** | Empty Input Validation | Click "Analyze" with empty inputs | Clear error message displayed | Error alert: "Please provide both..." | **PASS** |
| **TC-04** | Scanned PDF Upload | PDF with image-only content | Warning about unreadable scanned text | Warning displayed advising paste | **PASS** |
| **TC-05** | Skill Match Detection | Data Analyst resume + Junior DA job | Matches Python, SQL, Pandas, Excel | 10 Matched skills detected | **PASS** |
| **TC-06** | Skill Gap Detection | JD requests Power BI & AWS; resume lacks them | Power BI & AWS listed under Missing | Power BI, AWS listed in Missing | **PASS** |
| **TC-07** | Similarity Calculation | Sample Data Analyst pair | Score calculated between 40-70% | Score: 51.74% (Moderate Match) | **PASS** |
| **TC-08** | Report Download | Click "Download Full Report" | Formatted `.txt` audit file downloaded | Clean text file generated | **PASS** |

---

## 12. LIMITATIONS & FUTURE SCOPE

### 12.1 Current System Limitations
1. **Bag-of-Words Limitation:** TF-IDF evaluates word co-occurrence but cannot comprehend intricate syntactic relations or complex sentence meaning.
2. **Taxonomy Dependency:** Novel or emerging domain tools not present in `SKILL_TAXONOMY` will not be detected by the skill matcher (though TF-IDF will still vectorize them).
3. **Scanned Documents:** Without Optical Character Recognition (OCR), scanned image-only PDFs cannot be read.

### 12.2 Future Enhancements
1. **Dense Transformer Embeddings:** Upgrade from TF-IDF to Sentence-BERT (`all-MiniLM-L6-v2`) to achieve semantic understanding (e.g., recognizing that "supervised a team" aligns with "leadership").
2. **OCR Integration:** Implement Tesseract OCR (`pytesseract`) to process scanned paper resumes.
3. **Generative AI Resume Tailoring:** Integrate a local LLM or API to suggest rewritten resume bullet points adhering to the STAR method.
4. **Automated PDF Report Generator:** Export full visual color audits with ReportLab charts.

---

## 13. TOP 15 COLLEGE VIVA QUESTIONS & ANSWERS

### Q1. What is Natural Language Processing (NLP)?
**Answer:** NLP is a specialized branch of Artificial Intelligence and Computer Science that allows computers to read, analyze, interpret, and generate human languages by combining computational linguistics with machine learning algorithms.

### Q2. Why did you select this particular project?
**Answer:** In modern recruitment, hiring managers receive hundreds of resumes for every opening, making manual screening tedious and subjective. Candidates also lack clear visibility into why they are rejected. This project automates document comparison using NLP, providing quantifiable matching scores and skill gap insights.

### Q3. What is TF-IDF and how does it work?
**Answer:** Term Frequency–Inverse Document Frequency (TF-IDF) converts text into numerical features. **Term Frequency (TF)** measures how often a word occurs in a document. **Inverse Document Frequency (IDF)** measures how rare or unique the word is across the corpus. Words common across all documents receive low weights, while distinctive domain terms (e.g., `PostgreSQL`, `Machine Learning`) receive high weights.

### Q4. Why did you choose Cosine Similarity instead of Euclidean Distance?
**Answer:** Euclidean distance measures the straight-line distance between two points and is heavily distorted by document length—a longer resume will have higher term counts and thus appear far away from a concise job description. Cosine similarity measures the **cosine of the angle** between the two document vectors regardless of their length, making it length-invariant and ideal for text comparison.

### Q5. What is the difference between Stemming and Lemmatization?
**Answer:** Stemming applies crude rule-based truncation to strip word endings (e.g., `"operational"` $\rightarrow$ `"oper"`), often producing non-dictionary words. Lemmatization uses lexical vocabulary and morphological analysis based on part-of-speech context to return the legitimate dictionary base form (e.g., `"operational"` $\rightarrow$ `"operation"`). We use spaCy's lemmatizer for linguistic accuracy.

### Q6. Is this project based on Deep Learning?
**Answer:** No. It is an NLP and traditional Machine Learning project utilizing the Vector Space Model (TF-IDF) and Cosine Similarity. This ensures the application is lightweight, transparent, highly interpretable, and capable of executing in milliseconds on standard CPU hardware without GPU acceleration.

### Q7. If a resume gets a 90% score, does it mean the candidate is 90% likely to get hired?
**Answer:** Absolutely not. The 90% is a mathematical text similarity score measuring lexical and terminology overlap with the job description. Actual hiring depends on candidate interview performance, educational pedigree, practical project depth, soft skills, and cultural fit.

### Q8. What happens if a candidate uploads a scanned image PDF?
**Answer:** Pure text extraction libraries (PyMuPDF/pdfminer) cannot extract text from image pixels. Our system detects when zero characters are extracted and displays a friendly warning prompt asking the candidate to upload a text-based PDF or paste their resume text directly. Adding Tesseract OCR would resolve this in future versions.

### Q9. What are matched skills vs. missing skills?
**Answer:** Matched skills are domain technical terms detected in **both** the resume and the job description. Missing skills are domain requirements detected in the job description that were **not** found in the candidate's resume.

### Q10. What are n-grams and why did you use `ngram_range=(1, 2)`?
**Answer:** An n-gram is a contiguous sequence of $n$ items from a given text. Unigrams are single words (`"machine"`, `"learning"`), while bigrams are word pairs (`"machine learning"`). By using `ngram_range=(1, 2)`, our TF-IDF model captures compound technical terms like `"data analysis"`, `"power bi"`, and `"deep learning"` as unified concepts instead of isolated words.

### Q11. How does the skill extraction engine work?
**Answer:** It uses a categorized taxonomy of over 400 industry skills grouped into 7 domains. It uses word-boundary regular expressions (`\b...\b`) to avoid substring false positives, maps abbreviations and synonyms to canonical titles (e.g. `powerbi` $\rightarrow$ `Power BI`), and computes intersection and difference sets.

### Q12. Why did you use Streamlit for the user interface?
**Answer:** Streamlit allows Python developers to build interactive, responsive, data-driven web dashboards without writing boilerplate JavaScript or backend REST frameworks. It supports seamless state management, file uploading, and direct embedding of interactive Plotly graphs.

### Q13. Can candidates "game" the system by keyword stuffing?
**Answer:** Keyword stuffing (repeating a keyword 50 times in white text) is a known vulnerability of pure TF-IDF. However, because our system incorporates unigram/bigram sub-linear scaling and explicitly verifies skills against structured project contexts, abnormal term frequencies can be detected and penalized.

### Q14. What are the key limitations of your system?
**Answer:** 1) Inability to understand deep sentence semantics (TF-IDF treats text as an unordered bag of words). 2) Reliance on a predefined skill taxonomy for skill categorization. 3) Inability to read scanned PDFs without an OCR engine.

### Q15. How would you upgrade this project for an industrial production deployment?
**Answer:** We would:
1. Replace TF-IDF with dense Sentence Transformers (`all-mpnet-base-v2` or BERT).
2. Integrate a vector database like Pinecone or ChromaDB for large-scale candidate retrieval.
3. Add Tesseract OCR for physical document scanning.
4. Implement automated LLM-based resume enhancement recommendations.

---

## 14. REFERENCES

1. Salton, G., & Buckley, C. (1988). *Term-weighting approaches in automatic text retrieval.* Information Processing & Management, 24(5), 513-523.
2. Honnibal, M., Montani, I., Van Landeghem, S., & Boyd, A. (2020). *spaCy: Industrial-strength Natural Language Processing in Python.*
3. Pedregosa, F., et al. (2011). *Scikit-learn: Machine Learning in Python.* Journal of Machine Learning Research, 12, 2825-2830.
4. Manning, C. D., Raghavan, P., & Schütze, H. (2008). *Introduction to Information Retrieval.* Cambridge University Press.
5. Streamlit Documentation: https://docs.streamlit.io/
6. Plotly Python Open Source Graphing Library: https://plotly.com/python/
