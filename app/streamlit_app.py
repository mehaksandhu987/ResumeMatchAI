#!/usr/bin/env python3
"""
ResumeMatch AI – Web GUI (Streamlit)
SDG 8 | IBM SkillsBuild AI/ML Internship
Author: Mehakpreet Singh Sandhu

Run locally:  streamlit run app/streamlit_app.py
Deploy free:  Streamlit Community Cloud or Hugging Face Spaces
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow importing resume_matcher from same folder
APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import streamlit as st
from resume_matcher import analyse, generate_html_report

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="ResumeMatch AI",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
  .main-header {
    font-size: 2.2rem; font-weight: 700; color: #0f766e; margin-bottom: 0.2rem;
  }
  .sub-header { color: #64748b; font-size: 1rem; margin-bottom: 1.5rem; }
  .score-big {
    font-size: 3.5rem; font-weight: 800; text-align: center;
  }
  .tip-box {
    background: #f0fdfa; border-left: 4px solid #0f766e;
    padding: 0.75rem 1rem; margin: 0.4rem 0; border-radius: 0 8px 8px 0;
  }
  .stButton>button {
    background-color: #0f766e; color: white; font-weight: 600;
    border-radius: 8px; border: none; padding: 0.5rem 1.5rem;
  }
  .stButton>button:hover { background-color: #0d9488; color: white; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 📄 ResumeMatch AI")
    st.caption("Career Guidance Agent")
    st.markdown("---")
    st.markdown("**SDG 8** · Decent Work & Economic Growth")
    st.markdown("**IBM SkillsBuild** AI/ML Internship")
    st.markdown("**Author:** Mehakpreet Singh Sandhu")
    st.markdown("---")
    st.markdown("#### How to use")
    st.markdown(
        "1. Paste your **resume** (plain text)\n"
        "2. Paste the **job description**\n"
        "3. Click **Analyse**\n"
        "4. Read score, missing skills, tips & learning plan"
    )
    st.markdown("---")
    use_sample = st.button("Load sample data", use_container_width=True)

# ---------------------------------------------------------------------------
# Sample data paths
# ---------------------------------------------------------------------------
DATA = APP_DIR.parent / "data"
SAMPLE_RESUME = (DATA / "sample_resume.txt").read_text(encoding="utf-8") if (DATA / "sample_resume.txt").exists() else ""
SAMPLE_JD = (DATA / "sample_jd.txt").read_text(encoding="utf-8") if (DATA / "sample_jd.txt").exists() else ""

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown('<p class="main-header">ResumeMatch AI</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">Free offline-style resume ↔ job match score, missing skills, tips & career path</p>',
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Input area
# ---------------------------------------------------------------------------
col_r, col_j = st.columns(2)

if "resume_text" not in st.session_state:
    st.session_state.resume_text = ""
if "jd_text" not in st.session_state:
    st.session_state.jd_text = ""

if use_sample:
    st.session_state.resume_text = SAMPLE_RESUME
    st.session_state.jd_text = SAMPLE_JD

with col_r:
    st.subheader("Your Resume")
    resume_text = st.text_area(
        "Paste resume text",
        value=st.session_state.resume_text,
        height=320,
        placeholder="Paste your full resume here (plain text)...",
        label_visibility="collapsed",
        key="resume_area",
    )

with col_j:
    st.subheader("Job Description")
    jd_text = st.text_area(
        "Paste job description",
        value=st.session_state.jd_text,
        height=320,
        placeholder="Paste the full job description here...",
        label_visibility="collapsed",
        key="jd_area",
    )

st.session_state.resume_text = resume_text
st.session_state.jd_text = jd_text

analyse_clicked = st.button("🔍  Analyse Match", type="primary", use_container_width=False)

# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------
if analyse_clicked:
    if not resume_text.strip() or not jd_text.strip():
        st.warning("Please paste both a resume and a job description.")
    else:
        with st.spinner("Analysing resume vs job description..."):
            result = analyse(resume_text, jd_text)

        # Score colour
        pct = result.score_pct
        if pct >= 60:
            color = "#22c55e"
            band = "Strong match"
        elif pct >= 40:
            color = "#f59e0b"
            band = "Medium match"
        else:
            color = "#ef4444"
            band = "Low match – improve keywords & skills"

        st.markdown("---")
        st.subheader("Results")

        # Score row
        c1, c2, c3 = st.columns([1, 1, 2])
        with c1:
            st.markdown(
                f'<div class="score-big" style="color:{color}">{pct}%</div>',
                unsafe_allow_html=True,
            )
            st.caption("Match score (TF-IDF + skill overlap)")
        with c2:
            st.metric("Band", band)
            st.metric("Skills found in resume", len(result.resume_skills))
        with c3:
            st.markdown("**Suggested career path**")
            st.success(result.career_path_id.replace("_", " ").title())
            st.markdown("**Possible roles:** " + ", ".join(result.career_info.get("roles", [])))

        # Skills
        st.markdown("#### Skills detected in resume")
        if result.resume_skills:
            st.write(", ".join(f"`{s}`" for s in result.resume_skills))
        else:
            st.write("_None detected from skill bank_")

        st.markdown("#### Missing skills (in JD, not in resume)")
        if result.missing:
            st.write(", ".join(f"`{s}`" for s in result.missing))
        else:
            st.success("None – excellent coverage of JD skills!")

        # Tips
        st.markdown("#### Three improvement tips")
        for i, tip in enumerate(result.tips, 1):
            st.markdown(f'<div class="tip-box"><b>{i}.</b> {tip}</div>', unsafe_allow_html=True)

        # Learning plan
        st.markdown("#### Personal learning plan")
        for step in result.learning_plan:
            st.markdown(f"- {step}")

        if result.career_info.get("next_skills"):
            st.markdown("**Next skills to learn:** " + ", ".join(
                f"`{s}`" for s in result.career_info["next_skills"]
            ))

        # Download HTML report
        html = generate_html_report(result, "Candidate", "Target Role")
        st.download_button(
            label="⬇️ Download HTML report",
            data=html,
            file_name="ResumeMatch_Report.html",
            mime="text/html",
        )

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown("---")
st.caption(
    "ResumeMatch AI · SDG 8 · IBM SkillsBuild AI/ML Internship · Mehakpreet Singh Sandhu · "
    "Runs analysis in your session · Free for students"
)
