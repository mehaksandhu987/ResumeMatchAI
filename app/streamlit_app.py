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

APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import streamlit as st
from resume_matcher import analyse, generate_html_report

st.set_page_config(
    page_title="ResumeMatch AI",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# CSS – explicit dark text so tips stay readable in light & dark themes
# ---------------------------------------------------------------------------
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
    background-color: #f0fdfa !important;
    border-left: 4px solid #0f766e;
    padding: 0.75rem 1rem;
    margin: 0.5rem 0;
    border-radius: 0 8px 8px 0;
    color: #134e4a !important;
  }
  .tip-box b, .tip-box span, .tip-box p {
    color: #134e4a !important;
  }
  .stButton>button {
    background-color: #0f766e; color: white; font-weight: 600;
    border-radius: 8px; border: none; padding: 0.5rem 1.5rem;
  }
  .stButton>button:hover { background-color: #0d9488; color: white; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sample data
# ---------------------------------------------------------------------------
DATA = APP_DIR.parent / "data"
SAMPLE_RESUME = ""
SAMPLE_JD = ""
if (DATA / "sample_resume.txt").exists():
    SAMPLE_RESUME = (DATA / "sample_resume.txt").read_text(encoding="utf-8")
if (DATA / "sample_jd.txt").exists():
    SAMPLE_JD = (DATA / "sample_jd.txt").read_text(encoding="utf-8")

# Widget keys – must set session_state BEFORE creating the widgets
RESUME_KEY = "resume_area"
JD_KEY = "jd_area"

def load_sample():
    """Callback: fill text areas with sample resume + JD."""
    st.session_state[RESUME_KEY] = SAMPLE_RESUME
    st.session_state[JD_KEY] = SAMPLE_JD

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
    st.button(
        "Load sample data",
        use_container_width=True,
        on_click=load_sample,
        key="load_sample_btn",
    )

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown('<p class="main-header">ResumeMatch AI</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">Free resume ↔ job match score, missing skills, tips & career path</p>',
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Input area
# ---------------------------------------------------------------------------
col_r, col_j = st.columns(2)

with col_r:
    st.subheader("Your Resume")
    resume_text = st.text_area(
        "Paste resume text",
        height=320,
        placeholder="Paste your full resume here (plain text)...",
        label_visibility="collapsed",
        key=RESUME_KEY,
    )

with col_j:
    st.subheader("Job Description")
    jd_text = st.text_area(
        "Paste job description",
        height=320,
        placeholder="Paste the full job description here...",
        label_visibility="collapsed",
        key=JD_KEY,
    )

analyse_clicked = st.button("🔍  Analyse Match", type="primary")

# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------
if analyse_clicked:
    if not (resume_text or "").strip() or not (jd_text or "").strip():
        st.warning("Please paste both a resume and a job description (or click Load sample data).")
    else:
        with st.spinner("Analysing resume vs job description..."):
            result = analyse(resume_text, jd_text)

        pct = result.score_pct
        if pct >= 60:
            color = "#16a34a"
            band = "Strong match"
        elif pct >= 40:
            color = "#d97706"
            band = "Medium match"
        else:
            color = "#dc2626"
            band = "Low match – improve keywords & skills"

        st.markdown("---")
        st.subheader("Results")

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
            st.markdown(
                "**Possible roles:** " + ", ".join(result.career_info.get("roles", []))
            )

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

        # Tips – use st.info for reliable contrast (no white-on-white)
        st.markdown("#### Three improvement tips")
        for i, tip in enumerate(result.tips, 1):
            st.info(f"**{i}.** {tip}")

        st.markdown("#### Personal learning plan")
        for step in result.learning_plan:
            st.markdown(f"- {step}")

        if result.career_info.get("next_skills"):
            st.markdown(
                "**Next skills to learn:** "
                + ", ".join(f"`{s}`" for s in result.career_info["next_skills"])
            )

        html = generate_html_report(result, "Candidate", "Target Role")
        st.download_button(
            label="⬇️ Download HTML report",
            data=html,
            file_name="ResumeMatch_Report.html",
            mime="text/html",
        )

st.markdown("---")
st.caption(
    "ResumeMatch AI · SDG 8 · IBM SkillsBuild AI/ML Internship · Mehakpreet Singh Sandhu · "
    "Free for students"
)
