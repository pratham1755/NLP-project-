"""
Module 5: Results and Reporting
Generates actionable resume improvement recommendations and formatted, downloadable
analysis reports (Markdown & Plain Text).
"""

from datetime import datetime
from typing import Dict, Any, List


def generate_recommendations(similarity_data: Dict[str, Any], skill_data: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Produces smart, actionable, personalized resume improvement tips
    based on the similarity score and skill gap analysis.
    """
    recommendations = []
    match_pct = similarity_data.get("match_percentage", 0.0)
    missing_skills = skill_data.get("missing_skills", [])
    missing_kws = [item["keyword"] for item in similarity_data.get("missing_keywords", [])[:5]]
    coverage_pct = skill_data.get("skill_coverage_pct", 0.0)

    # 1. Critical Missing Skills
    if missing_skills:
        top_missing = ", ".join(missing_skills[:5])
        recommendations.append({
            "title": "Incorporate Key Missing Technical Skills",
            "type": "high",
            "text": f"The job description highlights requirements that are not found in your resume: **{top_missing}**. "
                    "If you have hands-on experience or coursework in these areas, add them explicitly under your 'Skills' "
                    "and 'Projects' sections."
        })

    # 2. Vocabulary & Keyword Alignment (TF-IDF gap)
    if missing_kws:
        kw_str = ", ".join(missing_kws)
        recommendations.append({
            "title": "Bridge High-Importance Vocabulary Gaps",
            "type": "medium",
            "text": f"Important domain keywords with high TF-IDF weighting in the job posting are missing: **{kw_str}**. "
                    "Integrate these terms naturally into your project bullet points and experience summary."
        })

    # 3. Overall Match Score advice
    if match_pct < 50:
        recommendations.append({
            "title": "Tailor Summary and Project Descriptions",
            "type": "high",
            "text": "Your textual similarity score is below 50%. Rewrite your professional summary and project headings "
                    "to mirror the terminology used in this specific job specification."
        })
    elif match_pct >= 75:
        recommendations.append({
            "title": "Strong Match - Focus on Impact and Metrics",
            "type": "success",
            "text": "Your resume has a strong alignment with this job role! To stand out further, quantify your "
                    "achievements using numerical metrics (e.g., 'Improved data processing speed by 35%', 'Built dashboard used by 50+ analysts')."
        })

    # 4. ATS Formatting & Section Organization
    recommendations.append({
        "title": "Optimize for ATS (Applicant Tracking Systems)",
        "type": "info",
        "text": "Ensure your resume uses clean, single-column layouts with standard headers ('Work Experience', "
                "'Technical Skills', 'Education'). Avoid embedding crucial text inside graphical tables or text boxes "
                "which automated parsers may skip."
    })

    # 5. STAR Method for bullet points
    recommendations.append({
        "title": "Structure Bullet Points with the STAR Technique",
        "type": "info",
        "text": "Frame project accomplishments using Situation, Task, Action, and Result. Start each bullet point "
                "with an active strong verb (e.g., 'Engineered', 'Optimized', 'Architected', 'Spearheaded')."
    })

    return recommendations


def generate_text_report(
    similarity_data: Dict[str, Any],
    skill_data: Dict[str, Any],
    resume_preview: str,
    jd_preview: str
) -> str:
    """
    Builds a complete, formatted Plain Text / Markdown report suitable for downloading.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    match_pct = similarity_data.get("match_percentage", 0.0)
    match_level = similarity_data.get("match_level", "N/A")
    coverage_pct = skill_data.get("skill_coverage_pct", 0.0)

    matched_skills = skill_data.get("matched_skills", [])
    missing_skills = skill_data.get("missing_skills", [])
    extra_skills = skill_data.get("extra_resume_skills", [])
    missing_kws = similarity_data.get("missing_keywords", [])
    top_shared = similarity_data.get("top_shared_keywords", [])

    recommendations = generate_recommendations(similarity_data, skill_data)

    report_lines = [
        "=" * 70,
        " AI RESUME & JOB DESCRIPTION MATCHER - AUDIT REPORT",
        " Natural Language Processing & Machine Learning Mini Project",
        f" Generated On: {timestamp}",
        "=" * 70,
        "",
        "1. EXECUTIVE SUMMARY",
        "-" * 70,
        f"• Overall Text Similarity Score : {match_pct}% ({match_level})",
        f"• Required Skills Coverage     : {coverage_pct}%",
        f"• Total Skills in Job Posting  : {skill_data.get('jd_total_skills_count', 0)}",
        f"• Skills Matched in Resume     : {skill_data.get('matched_skills_count', 0)}",
        f"• Required Skills Missing      : {skill_data.get('missing_skills_count', 0)}",
        f"• Additional Resume Skills     : {len(extra_skills)}",
        "",
        "2. SKILL GAP ANALYSIS",
        "-" * 70,
        "MATCHED SKILLS (Detected in both Resume & Job Description):",
        f"  {', '.join(matched_skills) if matched_skills else 'None'}",
        "",
        "MISSING SKILLS (Present in Job Description, Missing in Resume):",
        f"  {', '.join(missing_skills) if missing_skills else 'None - Full skill match!'}",
        "",
        "ADDITIONAL CANDIDATE SKILLS (Bonus skills found in Resume):",
        f"  {', '.join(extra_skills[:15]) if extra_skills else 'None'}",
        "",
        "3. NLP KEYWORD RELEVANCE & TF-IDF CONTRIBUTION",
        "-" * 70,
        "Top Shared Terms (Resume & JD):"
    ]

    for item in top_shared[:8]:
        report_lines.append(f"  • {item['keyword'].ljust(20)} | Relevance: {item['relevance']}")

    report_lines.append("")
    report_lines.append("Top Missing Keywords from Job Description:")
    for item in missing_kws[:8]:
        report_lines.append(f"  • {item['keyword'].ljust(20)} | JD Importance: {item['jd_importance']}")

    report_lines.extend([
        "",
        "4. ACTIONABLE RECOMMENDATIONS FOR RESUME IMPROVEMENT",
        "-" * 70
    ])

    for i, rec in enumerate(recommendations, 1):
        report_lines.append(f"[{i}] {rec['title']}")
        report_lines.append(f"    {rec['text']}")
        report_lines.append("")

    report_lines.extend([
        "5. METHODOLOGY & ALGORITHMS USED",
        "-" * 70,
        "• Text Preprocessing: spaCy (Lowercasing, Cleaning, Tokenization, Stopwords, Lemmatization)",
        "• Feature Extraction : Scikit-learn TfidfVectorizer (Unigrams & Bigrams, N-gram range 1-2)",
        "• Similarity Metric  : Cosine Similarity between TF-IDF document vectors",
        "• Skill Extraction   : Domain-specific taxonomy with boundary regex and synonym mapping",
        "",
        "=" * 70,
        " END OF AUDIT REPORT — AI Job Matching Analyzer",
        "=" * 70
    ])

    return "\n".join(report_lines)
