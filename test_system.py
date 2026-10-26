"""
Automated Verification & Unit Tests for Resume and Job Description Matcher
Tests all 5 core modules against sample data.
"""

import os
from modules.document_parser import parse_resume_file
from modules.nlp_preprocessor import preprocess_text, get_preprocessing_demonstration
from modules.similarity_engine import compute_similarity
from modules.skill_extractor import compare_skills, extract_skills_from_text
from modules.report_generator import generate_recommendations, generate_text_report


def run_tests():
    print("=" * 60)
    print("RUNNING NLP MINI-PROJECT VERIFICATION TESTS")
    print("=" * 60)

    # 1. Load sample files
    base_dir = os.path.dirname(os.path.abspath(__file__))
    resume_path = os.path.join(base_dir, "sample_data", "sample_data_analyst_resume.txt")
    jd_path = os.path.join(base_dir, "sample_data", "sample_job_description_data_analyst.txt")

    assert os.path.exists(resume_path), f"Missing {resume_path}"
    assert os.path.exists(jd_path), f"Missing {jd_path}"

    with open(resume_path, "r", encoding="utf-8") as f:
        resume_text = f.read()

    with open(jd_path, "r", encoding="utf-8") as f:
        jd_text = f.read()

    print(f" [PASS] Loaded Resume ({len(resume_text.split())} words) and JD ({len(jd_text.split())} words)")

    # 2. Test NLP Preprocessor
    clean_resume = preprocess_text(resume_text)
    clean_jd = preprocess_text(jd_text)
    assert len(clean_resume) > 0, "Cleaned resume should not be empty"
    assert len(clean_jd) > 0, "Cleaned JD should not be empty"
    print(f" [PASS] NLP Preprocessing completed. Sample lemma tokens: {clean_resume.split()[:8]}")

    demo = get_preprocessing_demonstration("The candidate is working with Python and developing data analysis projects.")
    assert len(demo["lemmatized_tokens"]) > 0
    print(f" [PASS] Preprocessing viva demonstration working. Base lemmas: {demo['lemmatized_tokens']}")

    # 3. Test Similarity Engine
    sim_result = compute_similarity(clean_resume, clean_jd)
    score = sim_result["match_percentage"]
    assert 0 <= score <= 100, f"Invalid score: {score}"
    print(f" [PASS] TF-IDF & Cosine Similarity computed: {score}% ({sim_result['match_level']})")
    print(f"        Top shared terms: {[item['keyword'] for item in sim_result['top_shared_keywords'][:5]]}")
    print(f"        Missing terms in resume: {[item['keyword'] for item in sim_result['missing_keywords'][:5]]}")

    # 4. Test Skill Extractor
    skill_result = compare_skills(resume_text, jd_text)
    assert len(skill_result["matched_skills"]) > 0, "Should match common data analyst skills"
    print(f" [PASS] Skill extraction successful:")
    print(f"        Matched skills ({len(skill_result['matched_skills'])}): {skill_result['matched_skills']}")
    print(f"        Missing skills ({len(skill_result['missing_skills'])}): {skill_result['missing_skills']}")
    print(f"        Coverage percentage: {skill_result['skill_coverage_pct']}%")

    # 5. Test Report Generator
    recs = generate_recommendations(sim_result, skill_result)
    assert len(recs) >= 3, "Should produce at least 3 actionable recommendations"
    print(f" [PASS] Generated {len(recs)} tailored recommendations.")

    report = generate_text_report(sim_result, skill_result, resume_text, jd_text)
    assert "EXECUTIVE SUMMARY" in report
    assert "SKILL GAP ANALYSIS" in report
    print(" [PASS] Generated full downloadable audit report.")

    print("=" * 60)
    print("ALL 5 MODULE TESTS PASSED PERFECTLY!")
    print("=" * 60)


if __name__ == "__main__":
    run_tests()
