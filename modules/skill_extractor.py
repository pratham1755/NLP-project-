"""
Module 4: Skill Extraction
Categorized skills taxonomy, regex phrase matching, synonym resolution,
and matched/missing skills calculation.
"""

import re
from typing import Dict, List, Set, Tuple, Any

# Comprehensive taxonomy of industry skills organized by category
SKILL_TAXONOMY = {
    "Programming Languages": [
        "python", "java", "c++", "c#", "c", "javascript", "typescript", "r", "scala",
        "go", "golang", "ruby", "php", "swift", "kotlin", "rust", "dart", "perl", "bash", "shell"
    ],
    "Data Science & AI/ML": [
        "machine learning", "deep learning", "natural language processing", "nlp",
        "computer vision", "tensorflow", "pytorch", "keras", "scikit-learn", "sklearn",
        "pandas", "numpy", "scipy", "matplotlib", "seaborn", "data analysis", "data analytics",
        "data science", "data cleaning", "data preprocessing", "feature engineering",
        "predictive modeling", "statistical analysis", "statistics", "hypothesis testing",
        "time series", "xgboost", "lightgbm", "random forest", "neural networks", "llm",
        "generative ai", "transformers", "hugging face", "opencv", "spacy", "nltk"
    ],
    "Databases & Big Data": [
        "sql", "mysql", "postgresql", "postgres", "sqlite", "oracle", "sql server",
        "mongodb", "nosql", "redis", "cassandra", "dynamodb", "elasticsearch",
        "spark", "apache spark", "pyspark", "hadoop", "hive", "kafka", "snowflake", "bigquery"
    ],
    "Business Intelligence & Visualization": [
        "power bi", "powerbi", "tableau", "excel", "advanced excel", "looker",
        "data visualization", "google data studio", "dax", "power query", "dashboard",
        "reporting", "business intelligence", "kpi tracking"
    ],
    "Cloud & DevOps": [
        "aws", "amazon web services", "azure", "microsoft azure", "gcp", "google cloud",
        "docker", "kubernetes", "k8s", "ci/cd", "jenkins", "git", "github", "gitlab",
        "terraform", "ansible", "linux", "cloud computing", "serverless", "lambda"
    ],
    "Web & Software Development": [
        "html", "html5", "css", "css3", "react", "react.js", "reactjs", "angular", "vue",
        "node.js", "nodejs", "express", "django", "flask", "fastapi", "spring boot",
        "rest api", "restful api", "graphql", "microservices", "api development", "tailwind"
    ],
    "Soft Skills & Management": [
        "problem solving", "critical thinking", "communication", "teamwork", "collaboration",
        "leadership", "project management", "agile", "scrum", "time management",
        "analytical skills", "presentation skills", "stakeholder management"
    ]
}

# Canonical name aliases / synonyms
SYNONYMS = {
    "powerbi": "Power BI",
    "power bi": "Power BI",
    "nlp": "Natural Language Processing",
    "natural language processing": "Natural Language Processing",
    "ml": "Machine Learning",
    "machine learning": "Machine Learning",
    "dl": "Deep Learning",
    "deep learning": "Deep Learning",
    "ai": "Artificial Intelligence",
    "postgres": "PostgreSQL",
    "postgresql": "PostgreSQL",
    "k8s": "Kubernetes",
    "kubernetes": "Kubernetes",
    "reactjs": "React",
    "react.js": "React",
    "react": "React",
    "nodejs": "Node.js",
    "node.js": "Node.js",
    "scikit-learn": "Scikit-Learn",
    "sklearn": "Scikit-Learn",
    "aws": "AWS",
    "amazon web services": "AWS",
    "gcp": "GCP",
    "google cloud": "GCP",
    "azure": "Azure",
    "microsoft azure": "Azure",
    "excel": "Excel",
    "advanced excel": "Excel",
    "sql": "SQL",
    "mysql": "MySQL",
    "html5": "HTML",
    "html": "HTML",
    "css3": "CSS",
    "css": "CSS",
    "golang": "Go",
    "ci/cd": "CI/CD"
}


def _canonicalize_skill(skill_key: str) -> str:
    """Returns nicely formatted canonical display name for a skill."""
    skill_lower = skill_key.lower().strip()
    if skill_lower in SYNONYMS:
        return SYNONYMS[skill_lower]
    # Default capitalization
    words = skill_lower.split()
    return " ".join(w.capitalize() for w in words)


def extract_skills_from_text(text: str) -> Dict[str, List[str]]:
    """
    Extracts skills from text, categorized according to SKILL_TAXONOMY.
    Returns: { category_name: [list of matched canonical skills] }
    """
    if not text:
        return {}

    # Lowercase text and normalize boundary punctuation for regex
    normalized_text = " " + re.sub(r"[^\w\s\+\#\./-]", " ", text.lower()) + " "

    found_by_category = {}
    found_canonical_all = set()

    for category, skill_list in SKILL_TAXONOMY.items():
        matched_in_category = set()
        for skill in skill_list:
            skill_lower = skill.lower()

            # Escape special regex characters in skill name (e.g. c++, c#, .net, ci/cd)
            escaped = re.escape(skill_lower)

            # Word boundary regex: allow spaces, punctuation around word
            pattern = rf"(?:\b|(?<=\s)){escaped}(?:\b|(?=\s))"

            if re.search(pattern, normalized_text):
                canonical = _canonicalize_skill(skill_lower)
                # Avoid duplicate synonyms (e.g., both powerbi and power bi)
                if canonical not in found_canonical_all:
                    matched_in_category.add(canonical)
                    found_canonical_all.add(canonical)

        if matched_in_category:
            found_by_category[category] = sorted(list(matched_in_category))

    return found_by_category


def compare_skills(resume_text: str, jd_text: str) -> Dict[str, Any]:
    """
    Compares skills between Resume and Job Description.
    Calculates:
    - Matched skills (found in both)
    - Missing skills (found in JD, missing in Resume)
    - Additional resume skills (found in Resume, not required by JD)
    - Category-by-category breakdown
    - Skill match percentage
    """
    resume_skills_by_cat = extract_skills_from_text(resume_text)
    jd_skills_by_cat = extract_skills_from_text(jd_text)

    # Flatten skills into set
    resume_skills_set = set()
    for cat, skills in resume_skills_by_cat.items():
        resume_skills_set.update(skills)

    jd_skills_set = set()
    for cat, skills in jd_skills_by_cat.items():
        jd_skills_set.update(skills)

    matched_skills = sorted(list(resume_skills_set.intersection(jd_skills_set)))
    missing_skills = sorted(list(jd_skills_set.difference(resume_skills_set)))
    extra_resume_skills = sorted(list(resume_skills_set.difference(jd_skills_set)))

    total_jd = len(jd_skills_set)
    total_matched = len(matched_skills)

    skill_coverage_pct = round((total_matched / total_jd * 100), 1) if total_jd > 0 else 0.0

    # Category breakdown comparison for visual charts
    all_categories = sorted(list(set(list(resume_skills_by_cat.keys()) + list(jd_skills_by_cat.keys()))))
    category_breakdown = []

    for cat in all_categories:
        res_cat = set(resume_skills_by_cat.get(cat, []))
        jd_cat = set(jd_skills_by_cat.get(cat, []))
        matched_cat = res_cat.intersection(jd_cat)
        missing_cat = jd_cat.difference(res_cat)

        category_breakdown.append({
            "category": cat,
            "jd_count": len(jd_cat),
            "resume_count": len(res_cat),
            "matched_count": len(matched_cat),
            "missing_count": len(missing_cat),
            "matched_skills": sorted(list(matched_cat)),
            "missing_skills": sorted(list(missing_cat))
        })

    return {
        "matched_skills": matched_skills,
        "missing_skills": missing_skills,
        "extra_resume_skills": extra_resume_skills,
        "resume_total_skills_count": len(resume_skills_set),
        "jd_total_skills_count": total_jd,
        "matched_skills_count": total_matched,
        "missing_skills_count": len(missing_skills),
        "skill_coverage_pct": skill_coverage_pct,
        "category_breakdown": category_breakdown,
        "resume_skills_by_category": resume_skills_by_cat,
        "jd_skills_by_category": jd_skills_by_cat
    }
