"""
AI Job Matching Analyzer: Score Your Resume Against Any Job | Resume Optimizer Pro
NLP Mini Project: Resume and Job Description Matcher
Data Science Engineering · Natural Language Processing · Python
"""

import os
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# Import custom modular NLP components
from modules.document_parser import parse_resume_file
from modules.nlp_preprocessor import preprocess_text, get_preprocessing_demonstration, _SPACY_LOADED
from modules.similarity_engine import compute_similarity
from modules.skill_extractor import compare_skills, extract_skills_from_text
from modules.report_generator import generate_recommendations, generate_text_report

# ---------------------------------------------------------
# Streamlit Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="AI Resume & Job Matcher | NLP Analyzer",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# Custom Modern CSS Styling
# ---------------------------------------------------------
st.markdown("""
<style>
    /* Main container enhancements */
    .main-header {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        padding: 1.8rem 2.2rem;
        border-radius: 16px;
        margin-bottom: 1.8rem;
        border: 1px solid #334155;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.25);
    }
    .main-title {
        color: #f8fafc;
        font-size: 2.2rem;
        font-weight: 800;
        margin-bottom: 0.3rem;
        letter-spacing: -0.5px;
    }
    .main-subtitle {
        color: #94a3b8;
        font-size: 1.05rem;
        font-weight: 400;
        margin: 0;
    }
    .badge-nlp {
        background-color: #3b82f6;
        color: white;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 6px;
        display: inline-block;
    }
    
    /* Metric Card styling */
    .metric-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        backdrop-filter: blur(8px);
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        transition: transform 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
    }
    .metric-val {
        font-size: 2.2rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
    }
    .metric-lbl {
        color: #94a3b8;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        font-weight: 600;
    }

    /* Skill Badges */
    .badge-container {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-top: 10px;
        margin-bottom: 15px;
    }
    .skill-matched {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.18), rgba(16, 185, 129, 0.08));
        color: #10b981;
        border: 1px solid #10b981;
        padding: 5px 12px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-flex;
        align-items: center;
    }
    .skill-missing {
        background: linear-gradient(135deg, rgba(244, 63, 94, 0.18), rgba(244, 63, 94, 0.08));
        color: #f43f5e;
        border: 1px solid #f43f5e;
        padding: 5px 12px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-flex;
        align-items: center;
    }
    .skill-extra {
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.18), rgba(59, 130, 246, 0.08));
        color: #60a5fa;
        border: 1px solid #3b82f6;
        padding: 5px 12px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-flex;
        align-items: center;
    }
    
    /* Recommendation box */
    .rec-box {
        border-left: 4px solid #3b82f6;
        background: rgba(30, 41, 59, 0.5);
        border-radius: 0 8px 8px 0;
        padding: 14px 18px;
        margin-bottom: 12px;
        border-top: 1px solid #334155;
        border-right: 1px solid #334155;
        border-bottom: 1px solid #334155;
    }
    .rec-high { border-left-color: #f43f5e; }
    .rec-medium { border-left-color: #f59e0b; }
    .rec-success { border-left-color: #10b981; }
    .rec-info { border-left-color: #3b82f6; }
    
    .qa-box {
        background: rgba(30, 41, 59, 0.4);
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 16px;
        margin-bottom: 14px;
    }
    .qa-q {
        color: #38bdf8;
        font-weight: 700;
        font-size: 1.05rem;
        margin-bottom: 6px;
    }
    .qa-a {
        color: #e2e8f0;
        font-size: 0.95rem;
        line-height: 1.5;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# Preset Loader Helper Functions
# ---------------------------------------------------------
def load_sample_file(rel_path: str) -> str:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    target = os.path.join(base_dir, rel_path)
    if os.path.exists(target):
        with open(target, "r", encoding="utf-8") as f:
            return f.read()
    return ""


# Initialize session state for text inputs if not set
if "resume_input" not in st.session_state:
    st.session_state["resume_input"] = ""
if "jd_input" not in st.session_state:
    st.session_state["jd_input"] = ""
if "analysis_performed" not in st.session_state:
    st.session_state["analysis_performed"] = False


# ---------------------------------------------------------
# Sidebar: Controls, Presets & Project Info
# ---------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/clouds/200/resume.png", width=110)
    st.title("Resume Matcher Pro")
    st.caption("Data Science & NLP Engineering")

    st.markdown("---")
    st.subheader("⚡ 1-Click Viva Presets")
    st.write("Load pre-configured test cases instantly:")

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("📊 Data Analyst", use_container_width=True, help="Load Junior Data Analyst test case"):
            st.session_state["resume_input"] = load_sample_file("sample_data/sample_data_analyst_resume.txt")
            st.session_state["jd_input"] = load_sample_file("sample_data/sample_job_description_data_analyst.txt")
            st.session_state["analysis_performed"] = True
            st.rerun()

    with col_btn2:
        if st.button("💻 Full Stack Dev", use_container_width=True, help="Load Software Engineer test case"):
            st.session_state["resume_input"] = load_sample_file("sample_data/sample_swe_resume.txt")
            st.session_state["jd_input"] = load_sample_file("sample_data/sample_job_description_swe.txt")
            st.session_state["analysis_performed"] = True
            st.rerun()

    if st.button("🧹 Reset / Clear Inputs", use_container_width=True):
        st.session_state["resume_input"] = ""
        st.session_state["jd_input"] = ""
        st.session_state["analysis_performed"] = False
        st.rerun()

    st.markdown("---")
    st.subheader("⚙️ System Status")
    if _SPACY_LOADED:
        st.success("✅ spaCy (`en_core_web_sm`) Active")
    else:
        st.warning("⚠️ spaCy Basic Mode (Rule-based lemmatizer active)")

    st.markdown("---")
    st.markdown("""
    **Project Scope:**
    - Document Parsing (PDF / DOCX)
    - NLP Preprocessing (spaCy)
    - TF-IDF Vectorization
    - Cosine Similarity Metric
    - Skill Gap Analytics
    - Automated Audit Report
    """)
    st.caption("College Mini Project · Antigravity AI")


# ---------------------------------------------------------
# Main Page Header
# ---------------------------------------------------------
st.markdown("""
<div class="main-header">
    <div style="margin-bottom: 8px;">
        <span class="badge-nlp">Natural Language Processing</span>
        <span class="badge-nlp" style="background:#059669;">TF-IDF & Cosine Similarity</span>
        <span class="badge-nlp" style="background:#7c3aed;">spaCy Lemmatizer</span>
    </div>
    <h1 class="main-title">AI Job Matching Analyzer</h1>
    <p class="main-subtitle">Score your resume against any job description, pinpoint missing technical skills, and optimize for recruiter screening.</p>
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# Main Tabs: Analyzer vs Viva & Architecture
# ---------------------------------------------------------
tab_analyzer, tab_viva, tab_prep_demo = st.tabs([
    "🎯 Resume & Job Analyzer",
    "🎓 College Viva & Architecture",
    "🔬 NLP Preprocessing Lab"
])


# =========================================================
# TAB 1: RESUME & JOB ANALYZER
# =========================================================
with tab_analyzer:
    col_left, col_right = st.columns(2, gap="large")

    # Column 1: Resume Upload / Input
    with col_left:
        st.subheader("1. Candidate Resume")
        uploaded_file = st.file_uploader(
            "Upload Resume (PDF, DOCX, or TXT)",
            type=["pdf", "docx", "txt"],
            help="Extracts text automatically using PyMuPDF / pdfminer / python-docx"
        )

        extracted_text = ""
        extract_warning = None

        if uploaded_file is not None:
            with st.spinner("Extracting text from resume file..."):
                extracted_text, extract_warning = parse_resume_file(uploaded_file)
                if extract_warning:
                    st.warning(f"⚠️ {extract_warning}")
                if extracted_text:
                    st.success(f"Extracted {len(extracted_text.split())} words from `{uploaded_file.name}`")
                    st.session_state["resume_input"] = extracted_text

        resume_text_area = st.text_area(
            "Resume Text (Review or paste directly):",
            value=st.session_state["resume_input"],
            height=280,
            placeholder="Paste raw resume text here, or upload your PDF/DOCX file above..."
        )
        st.session_state["resume_input"] = resume_text_area

        if resume_text_area:
            st.caption(f"📝 Word count: {len(resume_text_area.split())} words | {len(resume_text_area)} characters")

    # Column 2: Job Description Input
    with col_right:
        st.subheader("2. Target Job Description")
        st.caption("Paste the requirements, qualifications, and responsibilities for the target job.")

        jd_text_area = st.text_area(
            "Job Description Text:",
            value=st.session_state["jd_input"],
            height=345,
            placeholder="e.g. We are looking for a Junior Data Analyst with knowledge of Python, SQL, Excel, Power BI, Pandas, statistics, data cleaning, data visualization..."
        )
        st.session_state["jd_input"] = jd_text_area

        if jd_text_area:
            st.caption(f"📋 Word count: {len(jd_text_area.split())} words | {len(jd_text_area)} characters")

    # Action Button
    st.markdown("<br>", unsafe_allow_html=True)
    analyze_btn = st.button("🚀 Analyze Resume & Match Job", type="primary", use_container_width=True)

    if analyze_btn:
        st.session_state["analysis_performed"] = True

    # Run Analysis if triggered or state is active
    if st.session_state["analysis_performed"]:
        resume_content = st.session_state["resume_input"].strip()
        jd_content = st.session_state["jd_input"].strip()

        # Validation Checks
        if not resume_content and not jd_content:
            st.error("❌ Validation Error: Please provide both a Resume and a Job Description to run the analysis.")
        elif not resume_content:
            st.error("❌ Validation Error: Resume is empty! Please upload a file or paste resume text.")
        elif not jd_content:
            st.error("❌ Validation Error: Job description is empty! Please paste the job description text.")
        else:
            with st.spinner("Processing NLP pipelines (Cleaning · Lemmatization · TF-IDF · Skill Extraction)..."):
                # Step 1: NLP Preprocessing
                clean_resume = preprocess_text(resume_content)
                clean_jd = preprocess_text(jd_content)

                # Step 2: TF-IDF & Cosine Similarity
                sim_results = compute_similarity(clean_resume, clean_jd)

                # Step 3: Skill Extraction & Comparison
                skill_results = compare_skills(resume_content, jd_content)

                # Step 4: Recommendations
                recs = generate_recommendations(sim_results, skill_results)

                # Step 5: Full Report
                full_report = generate_text_report(sim_results, skill_results, resume_content, jd_content)

            st.markdown("---")
            st.markdown("## 📊 Analysis Dashboard & Results")

            # KPI Summary Cards
            col_kpi1, col_kpi2, col_kpi3, col_kpi4, col_kpi5 = st.columns(5)

            match_pct = sim_results["match_percentage"]
            color_theme = sim_results["match_color"]

            with col_kpi1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-val" style="color: {color_theme};">{match_pct}%</div>
                    <div class="metric-lbl">Text Similarity</div>
                </div>
                """, unsafe_allow_html=True)

            with col_kpi2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-val" style="color: #38bdf8;">{skill_results['skill_coverage_pct']}%</div>
                    <div class="metric-lbl">Skill Coverage</div>
                </div>
                """, unsafe_allow_html=True)

            with col_kpi3:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-val" style="color: #10b981;">{skill_results['matched_skills_count']}</div>
                    <div class="metric-lbl">Matched Skills</div>
                </div>
                """, unsafe_allow_html=True)

            with col_kpi4:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-val" style="color: #f43f5e;">{skill_results['missing_skills_count']}</div>
                    <div class="metric-lbl">Missing Skills</div>
                </div>
                """, unsafe_allow_html=True)

            with col_kpi5:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-val" style="color: #a855f7;">{len(skill_results['extra_resume_skills'])}</div>
                    <div class="metric-lbl">Bonus Skills</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Visual Charts Section
            chart_col1, chart_col2 = st.columns([1, 1.3], gap="medium")

            with chart_col1:
                st.markdown("### 🎯 Match Score Gauge")
                # Plotly Gauge Chart
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=match_pct,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': f"Overall Fit: {sim_results['match_level']}", 'font': {'size': 18, 'color': '#f8fafc'}},
                    number={'suffix': "%", 'font': {'size': 38, 'color': '#ffffff'}},
                    gauge={
                        'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#94a3b8"},
                        'bar': {'color': color_theme, 'thickness': 0.28},
                        'bgcolor': "rgba(15, 23, 42, 0.6)",
                        'borderwidth': 1,
                        'bordercolor': "#334155",
                        'steps': [
                            {'range': [0, 40], 'color': "rgba(239, 68, 68, 0.2)"},
                            {'range': [40, 70], 'color': "rgba(245, 158, 11, 0.2)"},
                            {'range': [70, 100], 'color': "rgba(16, 185, 129, 0.2)"}
                        ],
                        'threshold': {
                            'line': {'color': "#ffffff", 'width': 3},
                            'thickness': 0.75,
                            'value': match_pct
                        }
                    }
                ))
                fig_gauge.update_layout(
                    height=300,
                    margin=dict(l=25, r=25, t=50, b=25),
                    paper_bgcolor="rgba(0,0,0,0)",
                    font={'family': "Inter, sans-serif"}
                )
                st.plotly_chart(fig_gauge, use_container_width=True)

            with chart_col2:
                st.markdown("### 📊 Skill Category Breakdown")
                breakdown = skill_results["category_breakdown"]
                if breakdown:
                    cats = [b["category"] for b in breakdown]
                    matched_counts = [b["matched_count"] for b in breakdown]
                    missing_counts = [b["missing_count"] for b in breakdown]

                    fig_bar = go.Figure(data=[
                        go.Bar(name='Matched in Resume', x=cats, y=matched_counts, marker_color='#10b981'),
                        go.Bar(name='Missing in Resume', x=cats, y=missing_counts, marker_color='#f43f5e')
                    ])
                    fig_bar.update_layout(
                        barmode='group',
                        height=300,
                        margin=dict(l=20, r=20, t=30, b=40),
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                        xaxis=dict(tickangle=-25, color="#94a3b8"),
                        yaxis=dict(color="#94a3b8", title="Skill Count"),
                        font={'family': "Inter, sans-serif"}
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)
                else:
                    st.info("No categorical skills detected for bar chart breakdown.")

            # Skills Pill Cards
            st.markdown("---")
            st.markdown("### 🔍 Detailed Skill Gap Breakdown")

            skill_tab1, skill_tab2, skill_tab3 = st.tabs([
                f"✅ Matched Skills ({len(skill_results['matched_skills'])})",
                f"❌ Missing Skills ({len(skill_results['missing_skills'])})",
                f"🌟 Additional Resume Skills ({len(skill_results['extra_resume_skills'])})"
            ])

            with skill_tab1:
                matched = skill_results['matched_skills']
                if matched:
                    st.markdown("These skills were found in **both** your resume and the job description:")
                    badge_html = "".join([f'<span class="skill-matched">✓ {s}</span>' for s in matched])
                    st.markdown(f'<div class="badge-container">{badge_html}</div>', unsafe_allow_html=True)
                else:
                    st.warning("No direct skill matches detected. Review the job description requirements below.")

            with skill_tab2:
                missing = skill_results['missing_skills']
                if missing:
                    st.markdown("These skills are mentioned in the **Job Description** but were **not found** in your resume:")
                    badge_html = "".join([f'<span class="skill-missing">✗ {s}</span>' for s in missing])
                    st.markdown(f'<div class="badge-container">{badge_html}</div>', unsafe_allow_html=True)
                    st.info("💡 **Tip:** If you have practical experience or certifications in any of these skills, add them to your resume to increase your match score.")
                else:
                    st.success("🎉 Outstanding! Your resume covers all key skills identified in the job description!")

            with skill_tab3:
                extras = skill_results['extra_resume_skills']
                if extras:
                    st.markdown("These skills are highlighted in your **resume** but were **not strictly required** in the job posting (Bonus points):")
                    badge_html = "".join([f'<span class="skill-extra">+ {s}</span>' for s in extras])
                    st.markdown(f'<div class="badge-container">{badge_html}</div>', unsafe_allow_html=True)
                else:
                    st.caption("No additional taxonomy skills detected outside the job description.")

            # TF-IDF Keywords Section
            st.markdown("---")
            st.markdown("### 🔤 TF-IDF Keyword Relevance & Missing Terms")
            kw_col1, kw_col2 = st.columns(2)

            with kw_col1:
                st.markdown("#### Top Shared Terms (Highest TF-IDF Relevance)")
                shared_kw = sim_results.get("top_shared_keywords", [])
                if shared_kw:
                    df_shared = pd.DataFrame(shared_kw)
                    df_shared.columns = ["Keyword / Phrase", "Resume Weight", "Job Description Weight", "Relevance Product"]
                    st.dataframe(df_shared, use_container_width=True, hide_index=True)
                else:
                    st.write("No strong shared n-grams found.")

            with kw_col2:
                st.markdown("#### High-Importance Missing Job Terms")
                missing_kw = sim_results.get("missing_keywords", [])
                if missing_kw:
                    df_missing = pd.DataFrame(missing_kw)
                    df_missing.columns = ["Missing Term", "Job Description Importance"]
                    st.dataframe(df_missing, use_container_width=True, hide_index=True)
                else:
                    st.write("No significant missing vocabulary terms found.")

            # Actionable Suggestions
            st.markdown("---")
            st.markdown("### 💡 Tailored Resume Improvement Recommendations")
            for rec in recs:
                css_class = f"rec-box rec-{rec['type']}"
                st.markdown(f"""
                <div class="{css_class}">
                    <strong style="font-size: 1.05rem; color: #f8fafc;">{rec['title']}</strong><br>
                    <span style="color: #cbd5e1; font-size: 0.95rem;">{rec['text']}</span>
                </div>
                """, unsafe_allow_html=True)

            # Download Report Option
            st.markdown("---")
            st.markdown("### 📥 Download Analysis Audit Report")
            st.download_button(
                label="📄 Download Full Analysis Report (.txt)",
                data=full_report,
                file_name=f"resume_job_match_report.txt",
                mime="text/plain",
                use_container_width=True
            )


# =========================================================
# TAB 2: COLLEGE VIVA & ARCHITECTURE GUIDE
# =========================================================
with tab_viva:
    st.markdown("## 🎓 College Viva & Project Defense Guide")
    st.caption("Comprehensive technical breakdown, mathematical formulas, and viva questions for your presentation.")

    st.markdown("### 1. System Architecture & Flowchart")
    st.markdown("""
```
 ┌─────────────────────────────────────────────────────────────┐
 │                Streamlit User Interface                     │
 │          (Resume Upload + Job Description Paste)            │
 └──────────────────────────────┬──────────────────────────────┘
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
    """)

    st.markdown("---")
    st.markdown("### 2. Mathematical Algorithms Explained")

    col_algo1, col_algo2 = st.columns(2)

    with col_algo1:
        st.markdown("#### A. TF-IDF (Term Frequency - Inverse Document Frequency)")
        st.latex(r"\operatorname{TF}(t, d) = \frac{\text{Count of term } t \text{ in document } d}{\text{Total terms in document } d}")
        st.latex(r"\operatorname{IDF}(t, D) = \log\left(\frac{N}{|\{d \in D : t \in d\}|}\right) + 1")
        st.latex(r"\operatorname{TF-IDF}(t, d, D) = \operatorname{TF}(t, d) \times \operatorname{IDF}(t, D)")
        st.markdown("""
        **Viva Explanation:**
        - **TF** measures how frequently a word appears in the resume or job description.
        - **IDF** downweights generic words appearing everywhere and boosts distinctive technical terms (e.g. `PySpark`, `Docker`).
        """)

    with col_algo2:
        st.markdown("#### B. Cosine Similarity Formula")
        st.latex(r"\operatorname{Similarity}(\mathbf{A}, \mathbf{B}) = \cos(\theta) = \frac{\mathbf{A} \cdot \mathbf{B}}{\|\mathbf{A}\| \|\mathbf{B}\|} = \frac{\sum_{i=1}^n A_i B_i}{\sqrt{\sum_{i=1}^n A_i^2} \sqrt{\sum_{i=1}^n B_i^2}}")
        st.markdown("""
        **Viva Explanation:**
        - Treats the documents as vectors in high-dimensional TF-IDF space.
        - Measures the **cosine of the angle** $\\theta$ between them.
        - Unaffected by document length differences (a 1-page resume vs a 3-page resume).
        - Value lies between 0 (completely orthogonal/unrelated) and 1 (identical vector directions).
        """)

    st.markdown("---")
    st.markdown("### 3. Top 10 College Viva Questions & Answers")

    viva_qa = [
        ("Q1. What is Natural Language Processing (NLP)?",
         "NLP is a subfield of Artificial Intelligence and Computer Science focused on enabling computers to understand, interpret, manipulate, and generate human language."),
        ("Q2. Why did you choose this project?",
         "Recruiters receive hundreds of resumes per opening, making manual review slow. Candidates also struggle to tailor resumes. This tool automates relevance checking and highlights skill gaps objectively using NLP."),
        ("Q3. What is the role of TF-IDF in this system?",
         "TF-IDF converts unstructured text into numerical vector representations. Words unique to the job description and resume receive high weights, while common English words receive low weights."),
        ("Q4. Why use Cosine Similarity instead of Euclidean Distance?",
         "Euclidean distance is sensitive to document length (a long resume has higher word counts and larger distance from a short JD). Cosine similarity measures the orientation/angle between vectors rather than magnitude, making it length-invariant."),
        ("Q5. Why is spaCy used over basic regex?",
         "spaCy provides an industrial-grade linguistic pipeline including accurate part-of-speech tagging, stopword dictionaries, and true lemmatization (reducing words like 'analyzed', 'analyzing', 'analysis' to base forms)."),
        ("Q6. Is this application a Deep Learning model?",
         "No. It uses traditional, interpretable Natural Language Processing and Machine Learning (TF-IDF vector space model and Cosine Similarity), which is fast, lightweight, and does not require GPU training."),
        ("Q7. Does a 90% score guarantee an interview or job?",
         "No. It is a lexical and semantic text similarity score. Real hiring depends on portfolio quality, work authorization, interview performance, and problem-solving skills."),
        ("Q8. What are the key limitations of this system?",
         "1) Pure TF-IDF misses deep semantic context (e.g. 'supervised a team' vs 'team leadership'). 2) Scanned image PDFs without OCR cannot be read. 3) Novel emerging skills outside the taxonomy require dictionary expansion."),
        ("Q9. What is the difference between matched and missing skills?",
         "Matched skills exist in both the resume and the job description. Missing skills are explicitly requested in the job description but absent from the candidate's resume."),
        ("Q10. How can this project be enhanced in the future?",
         "By integrating Dense Transformer Embeddings (BERT/Sentence-BERT), Tesseract OCR for scanned resumes, automated resume rewriting suggestions via LLMs, and PDF audit download.")
    ]

    for q, a in viva_qa:
        st.markdown(f"""
        <div class="qa-box">
            <div class="qa-q">{q}</div>
            <div class="qa-a">{a}</div>
        </div>
        """, unsafe_allow_html=True)


# =========================================================
# TAB 3: NLP PREPROCESSING LAB (INTERACTIVE DEMO)
# =========================================================
with tab_prep_demo:
    st.markdown("## 🔬 Interactive NLP Preprocessing Laboratory")
    st.caption("Demonstrate how raw human text gets converted step-by-step into clean tokens for vectorization.")

    demo_text = st.text_input(
        "Enter a sample sentence to inspect NLP transformation:",
        value="The candidate is working with Python and developing advanced data analysis projects in SQL!"
    )

    if demo_text:
        demo_results = get_preprocessing_demonstration(demo_text)

        col_d1, col_d2 = st.columns(2)
        with col_d1:
            st.markdown("**Step 1: Original Raw Text**")
            st.code(demo_results["raw_text"], language="text")

            st.markdown("**Step 2: After Cleaning & Lowercasing**")
            st.code(demo_results["cleaned_text"], language="text")

            st.markdown("**Step 3: Initial Tokens**")
            st.write(demo_results["all_tokens"])

        with col_d2:
            st.markdown("**Step 4: After Stop-word Removal**")
            st.info(", ".join(demo_results["without_stopwords"]))

            st.markdown("**Step 5: After Lemmatization (Base Forms)**")
            st.success(", ".join(demo_results["lemmatized_tokens"]))

            st.markdown("**spaCy Pipeline Status:**")
            if demo_results["spacy_model_active"]:
                st.caption("🟢 spaCy English model `en_core_web_sm` is performing morphological analysis.")
            else:
                st.caption("🟡 Fallback rule-based lemmatizer is active.")
