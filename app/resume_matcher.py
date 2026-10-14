#!/usr/bin/env python3
"""
ResumeMatch AI – Career Guidance Agent
SDG 8 | IBM SkillsBuild AI/ML Internship
Author: Mehakpreet Singh Sandhu

Core features (offline, no API keys):
  • Resume ↔ Job Description match score (TF-IDF + cosine similarity)
  • Missing-skills list drawn from an Indian job-market skill bank
  • Three concrete improvement tips
  • Career-path matching + personal learning plan
"""

from __future__ import annotations

import argparse
import json
import re
import textwrap
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------------------------
# Skill bank tuned for the Indian tech / service job market (Tier-2/3 friendly)
# ---------------------------------------------------------------------------
SKILL_BANK: Dict[str, List[str]] = {
    "programming": [
        "python", "java", "javascript", "typescript", "c++", "c#", "go", "rust",
        "kotlin", "swift", "php", "ruby", "scala", "r", "matlab", "sql",
    ],
    "web": [
        "html", "css", "react", "angular", "vue", "node.js", "express", "django",
        "flask", "fastapi", "next.js", "bootstrap", "tailwind", "rest api", "graphql",
    ],
    "data": [
        "machine learning", "deep learning", "data analysis", "pandas", "numpy",
        "scikit-learn", "tensorflow", "pytorch", "power bi", "tableau", "excel",
        "data visualization", "statistics", "nlp", "computer vision", "etl",
    ],
    "cloud_devops": [
        "aws", "azure", "gcp", "docker", "kubernetes", "jenkins", "ci/cd",
        "terraform", "ansible", "linux", "git", "github", "gitlab",
    ],
    "soft": [
        "communication", "leadership", "teamwork", "problem solving",
        "project management", "agile", "scrum", "presentation", "time management",
    ],
    "domain": [
        "fintech", "healthcare", "ecommerce", "edtech", "cybersecurity",
        "iot", "blockchain", "mobile development", "ui/ux", "testing", "selenium",
    ],
}

# Flatten for fast lookup
ALL_SKILLS = sorted({s.lower() for cat in SKILL_BANK.values() for s in cat})

# Simple career path map (skill → suggested roles + next skills to learn)
CAREER_PATHS: Dict[str, Dict] = {
    "software_engineer": {
        "roles": ["Software Developer", "Backend Engineer", "Full-Stack Developer"],
        "core_skills": ["python", "java", "javascript", "sql", "git", "data structures"],
        "next_skills": ["system design", "docker", "aws", "react", "microservices"],
        "learning_plan": [
            "Master one backend language (Python/Java) + SQL",
            "Build 2-3 full-stack projects and push to GitHub",
            "Learn Docker + basic AWS (EC2, S3)",
            "Practice system-design fundamentals and DSA",
        ],
    },
    "data_analyst": {
        "roles": ["Data Analyst", "Business Analyst", "BI Developer"],
        "core_skills": ["excel", "sql", "python", "power bi", "tableau", "statistics"],
        "next_skills": ["pandas", "data visualization", "etl", "machine learning basics"],
        "learning_plan": [
            "Become expert in Excel + SQL",
            "Learn Python (pandas, matplotlib/seaborn)",
            "Master Power BI or Tableau and publish dashboards",
            "Complete 1-2 real-world analysis case studies",
        ],
    },
    "ml_engineer": {
        "roles": ["ML Engineer", "Data Scientist", "AI Engineer"],
        "core_skills": ["python", "machine learning", "scikit-learn", "pandas", "numpy", "statistics"],
        "next_skills": ["deep learning", "tensorflow", "pytorch", "nlp", "mlops"],
        "learning_plan": [
            "Solidify Python + core ML algorithms",
            "Complete end-to-end ML projects (classification, regression)",
            "Learn one deep-learning framework (TensorFlow/PyTorch)",
            "Explore MLOps basics (model deployment, monitoring)",
        ],
    },
    "frontend_dev": {
        "roles": ["Frontend Developer", "UI Engineer", "React Developer"],
        "core_skills": ["html", "css", "javascript", "react", "git"],
        "next_skills": ["typescript", "next.js", "tailwind", "testing", "performance"],
        "learning_plan": [
            "Deepen JavaScript + React fundamentals",
            "Learn TypeScript and modern CSS (Tailwind)",
            "Build responsive, accessible UI projects",
            "Add testing (Jest/Cypress) and performance optimization",
        ],
    },
    "devops": {
        "roles": ["DevOps Engineer", "SRE", "Cloud Engineer"],
        "core_skills": ["linux", "git", "docker", "ci/cd", "aws"],
        "next_skills": ["kubernetes", "terraform", "jenkins", "monitoring"],
        "learning_plan": [
            "Master Linux command line + Git",
            "Learn Docker and write Dockerfiles for real apps",
            "Set up a simple CI/CD pipeline (GitHub Actions/Jenkins)",
            "Explore one cloud (AWS preferred) and IaC with Terraform",
        ],
    },
}


def normalize_text(text: str) -> str:
    """Lower-case, remove extra whitespace and common punctuation noise."""
    text = text.lower()
    text = re.sub(r"[^\w\s+#.\-/]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_skills(text: str) -> List[str]:
    """Return list of skills found in the text (order-preserving, unique)."""
    text_norm = normalize_text(text)
    found = []
    for skill in ALL_SKILLS:
        # word-boundary friendly match
        pattern = r"(?<!\w)" + re.escape(skill) + r"(?!\w)"
        if re.search(pattern, text_norm):
            found.append(skill)
    return found


def compute_match_score(resume: str, jd: str) -> Tuple[float, float, float]:
    """
    Hybrid score = 0.55 * TF-IDF cosine + 0.45 * skill-overlap Jaccard.
    Returns (final_score, tfidf_score, skill_score).
    """
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=5000,
        sublinear_tf=True,
    )
    tfidf = vectorizer.fit_transform([normalize_text(resume), normalize_text(jd)])
    tfidf_score = float(cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0])

    r_skills = set(extract_skills(resume))
    j_skills = set(extract_skills(jd))
    if not j_skills:
        skill_score = 0.0
    else:
        skill_score = len(r_skills & j_skills) / len(j_skills)

    final = 0.55 * tfidf_score + 0.45 * skill_score
    return final, tfidf_score, skill_score


def missing_skills(resume: str, jd: str) -> List[str]:
    """Skills present in JD but missing (or weakly present) in resume."""
    resume_skills = set(extract_skills(resume))
    jd_skills = set(extract_skills(jd))
    missing = sorted(jd_skills - resume_skills)
    return missing


def generate_tips(score: float, missing: List[str], resume_skills: List[str]) -> List[str]:
    """Produce three actionable improvement tips."""
    tips = []

    if score < 0.40:
        tips.append(
            "Your overall match is low. Rewrite the Professional Summary and Skills "
            "section so they mirror the exact keywords used in the job description."
        )
    elif score < 0.60:
        tips.append(
            "You are in the medium-match range. Add a short 'Relevant Experience' "
            "bullet list that uses the same action verbs and tools mentioned in the JD."
        )
    else:
        tips.append(
            "Strong match! Still, quantify impact (e.g. 'reduced latency by 30%') "
            "and ensure every key requirement from the JD appears at least once."
        )

    if missing:
        top_missing = ", ".join(missing[:5])
        tips.append(
            f"Explicitly add these missing skills (if you have them) or learn the top ones: {top_missing}."
        )
    else:
        tips.append(
            "All major JD skills appear in your resume. Focus on stronger project descriptions "
            "and measurable outcomes."
        )

    if len(resume_skills) < 8:
        tips.append(
            "Your skill list is short. Expand it with tools, frameworks and soft skills "
            "that are relevant to the role (see the suggested learning plan)."
        )
    else:
        tips.append(
            "Keep a clean, prioritised skill section (group by category) and place "
            "the most relevant skills for this JD at the top."
        )

    return tips[:3]


def suggest_career_path(resume_skills: List[str]) -> Tuple[str, Dict]:
    """Pick the best-matching career path based on weighted skill overlap."""
    best_path = "software_engineer"
    best_score = -1.0
    resume_set = set(s.lower() for s in resume_skills)

    for path_id, info in CAREER_PATHS.items():
        core = set(s.lower() for s in info["core_skills"])
        overlap = len(resume_set & core) / max(len(core), 1)
        # Prefer software_engineer when strong backend languages + SQL present
        if path_id == "software_engineer" and len({"python", "java", "sql"} & resume_set) >= 2:
            overlap += 0.25
        if overlap > best_score:
            best_score = overlap
            best_path = path_id

    return best_path, CAREER_PATHS[best_path]


@dataclass
class MatchResult:
    score: float
    score_pct: int
    resume_skills: List[str]
    jd_skills: List[str]
    missing: List[str]
    tips: List[str]
    career_path_id: str
    career_info: Dict
    learning_plan: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "match_score": round(self.score, 4),
            "match_percentage": self.score_pct,
            "resume_skills": self.resume_skills,
            "jd_skills": self.jd_skills,
            "missing_skills": self.missing,
            "improvement_tips": self.tips,
            "suggested_career_path": self.career_path_id,
            "suggested_roles": self.career_info.get("roles", []),
            "next_skills_to_learn": self.career_info.get("next_skills", []),
            "personal_learning_plan": self.learning_plan,
        }


def analyse(resume: str, jd: str) -> MatchResult:
    """Full analysis pipeline."""
    score, _, _ = compute_match_score(resume, jd)
    score_pct = int(round(min(score, 1.0) * 100))
    resume_skills = extract_skills(resume)
    jd_skills = extract_skills(jd)
    missing = missing_skills(resume, jd)
    tips = generate_tips(score, missing, resume_skills)
    path_id, path_info = suggest_career_path(resume_skills)

    return MatchResult(
        score=score,
        score_pct=score_pct,
        resume_skills=resume_skills,
        jd_skills=jd_skills,
        missing=missing,
        tips=tips,
        career_path_id=path_id,
        career_info=path_info,
        learning_plan=path_info.get("learning_plan", []),
    )


def generate_html_report(result: MatchResult, resume_name: str = "Candidate", jd_title: str = "Target Role") -> str:
    """Create a self-contained HTML report."""
    score_color = "#22c55e" if result.score_pct >= 60 else "#f59e0b" if result.score_pct >= 40 else "#ef4444"
    missing_html = "".join(f"<li>{s}</li>" for s in result.missing[:12]) or "<li>None – great coverage!</li>"
    tips_html = "".join(f"<li>{t}</li>" for t in result.tips)
    skills_html = ", ".join(result.resume_skills[:20]) or "None detected"
    plan_html = "".join(f"<li>{step}</li>" for step in result.learning_plan)
    roles = ", ".join(result.career_info.get("roles", []))

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ResumeMatch AI Report – {resume_name}</title>
<style>
  :root {{ --primary: #0f766e; --bg: #f8fafc; --card: #ffffff; --text: #1e293b; }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: 'Segoe UI', system-ui, sans-serif; background: var(--bg); color: var(--text); line-height: 1.6; padding: 2rem; }}
  .container {{ max-width: 800px; margin: 0 auto; }}
  header {{ text-align: center; margin-bottom: 2rem; }}
  h1 {{ color: var(--primary); font-size: 1.8rem; }}
  .subtitle {{ color: #64748b; font-size: 0.95rem; }}
  .score-card {{ background: var(--card); border-radius: 16px; padding: 2rem; text-align: center;
                 box-shadow: 0 4px 20px rgba(0,0,0,0.06); margin-bottom: 1.5rem; }}
  .score-value {{ font-size: 3.5rem; font-weight: 700; color: {score_color}; }}
  .score-label {{ font-size: 1.1rem; color: #64748b; }}
  .card {{ background: var(--card); border-radius: 12px; padding: 1.25rem 1.5rem;
           box-shadow: 0 2px 12px rgba(0,0,0,0.04); margin-bottom: 1rem; }}
  .card h2 {{ font-size: 1.15rem; color: var(--primary); margin-bottom: 0.6rem; }}
  ul {{ padding-left: 1.2rem; }}
  li {{ margin-bottom: 0.35rem; }}
  .tag {{ display: inline-block; background: #ccfbf1; color: #0f766e; padding: 0.2rem 0.55rem;
          border-radius: 999px; font-size: 0.8rem; margin: 0.15rem; }}
  footer {{ text-align: center; margin-top: 2rem; font-size: 0.8rem; color: #94a3b8; }}
</style>
</head>
<body>
<div class="container">
  <header>
    <h1>ResumeMatch AI</h1>
    <p class="subtitle">Career Guidance Agent · SDG 8 · IBM SkillsBuild</p>
    <p style="margin-top:0.5rem"><strong>{resume_name}</strong> → <em>{jd_title}</em></p>
  </header>

  <div class="score-card">
    <div class="score-value">{result.score_pct}%</div>
    <div class="score-label">Resume–Job Match Score (TF-IDF Cosine Similarity)</div>
  </div>

  <div class="card">
    <h2>Detected Skills in Resume</h2>
    <p>{skills_html}</p>
  </div>

  <div class="card">
    <h2>Missing Skills (present in JD, absent in resume)</h2>
    <ul>{missing_html}</ul>
  </div>

  <div class="card">
    <h2>Three Improvement Tips</h2>
    <ul>{tips_html}</ul>
  </div>

  <div class="card">
    <h2>Suggested Career Path</h2>
    <p><strong>Best fit:</strong> {result.career_path_id.replace('_', ' ').title()}</p>
    <p><strong>Possible roles:</strong> {roles}</p>
    <p style="margin-top:0.6rem"><strong>Next skills to learn:</strong></p>
    <p>{' '.join(f'<span class="tag">{s}</span>' for s in result.career_info.get('next_skills', []))}</p>
  </div>

  <div class="card">
    <h2>Personal Learning Plan</h2>
    <ol style="padding-left:1.2rem">{plan_html}</ol>
  </div>

  <footer>
    Generated offline by ResumeMatch AI · No data leaves your machine · Free for students
  </footer>
</div>
</body>
</html>"""
    return html


def print_console_report(result: MatchResult) -> None:
    """Pretty-print results to the terminal."""
    bar = "█" * (result.score_pct // 5) + "░" * (20 - result.score_pct // 5)
    print("\n" + "=" * 60)
    print("  RESUMEMATCH AI – ANALYSIS REPORT")
    print("=" * 60)
    print(f"\n  Match Score : {result.score_pct}%  [{bar}]")
    print(f"\n  Resume skills detected ({len(result.resume_skills)}):")
    print("   ", ", ".join(result.resume_skills[:15]) or "(none)")
    print(f"\n  Missing skills ({len(result.missing)}):")
    print("   ", ", ".join(result.missing[:10]) or "(none – excellent)")
    print("\n  Improvement Tips:")
    for i, tip in enumerate(result.tips, 1):
        wrapped = textwrap.fill(tip, width=56, initial_indent="   ", subsequent_indent="     ")
        print(f"  {i}.{wrapped[3:]}")
    print(f"\n  Suggested path : {result.career_path_id.replace('_', ' ').title()}")
    print(f"  Possible roles : {', '.join(result.career_info.get('roles', []))}")
    print("\n  Personal Learning Plan:")
    for step in result.learning_plan:
        print(f"   • {step}")
    print("\n" + "=" * 60 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ResumeMatch AI – offline resume vs JD matcher & career coach"
    )
    parser.add_argument("--resume", "-r", type=Path, help="Path to resume text file")
    parser.add_argument("--jd", "-j", type=Path, help="Path to job-description text file")
    parser.add_argument("--html", type=Path, help="Write HTML report to this path")
    parser.add_argument("--json", type=Path, help="Write JSON result to this path")
    parser.add_argument("--demo", action="store_true", help="Run built-in demo")
    args = parser.parse_args()

    if args.demo or (not args.resume and not args.jd):
        # Built-in demo
        resume = Path(__file__).parent.parent / "data" / "sample_resume.txt"
        jd = Path(__file__).parent.parent / "data" / "sample_jd.txt"
        if not resume.exists():
            print("Demo files missing – create data/sample_resume.txt and sample_jd.txt")
            return
        resume_text = resume.read_text(encoding="utf-8")
        jd_text = jd.read_text(encoding="utf-8")
        name, title = "Demo Candidate", "Software Engineer (Sample JD)"
    else:
        if not args.resume or not args.jd:
            parser.error("Provide both --resume and --jd, or use --demo")
        resume_text = args.resume.read_text(encoding="utf-8")
        jd_text = args.jd.read_text(encoding="utf-8")
        name = args.resume.stem
        title = args.jd.stem

    result = analyse(resume_text, jd_text)
    print_console_report(result)

    if args.html:
        html = generate_html_report(result, name, title)
        args.html.write_text(html, encoding="utf-8")
        print(f"HTML report saved → {args.html}")

    if args.json:
        args.json.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
        print(f"JSON result saved → {args.json}")

    # Always write a default HTML next to the script when running demo
    if args.demo:
        out = Path(__file__).parent.parent / "docs" / "demo_report.html"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(generate_html_report(result, name, title), encoding="utf-8")
        print(f"Demo HTML report → {out}")


if __name__ == "__main__":
    main()
