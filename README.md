# ResumeMatch AI – Career Guidance Agent

**SDG 8: Decent Work and Economic Growth**  
**IBM SkillsBuild AI/ML Internship**  
**Author: Mehakpreet Singh Sandhu**

> A free tool that tells you in seconds how well your resume fits a job, exactly what to fix, and what to learn next.

## Web GUI (recommended)

### Run locally

```bash
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

Open the URL shown in the terminal (usually http://localhost:8501).

### Deploy free (no credit card)

#### Option A – Streamlit Community Cloud (easiest)

1. Push this folder to a **public GitHub repository**
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub
3. Click **New app** → select your repo
4. Main file path: `app/streamlit_app.py` (or `app.py`)
5. Click **Deploy**

Your app will be live at `https://<you>-<repo>-<app>.streamlit.app`

#### Option B – Hugging Face Spaces (also free)

1. Create a Space at [huggingface.co/new-space](https://huggingface.co/new-space)
2. Choose **Streamlit** as the SDK
3. Upload all project files (or connect a GitHub repo)
4. Ensure `app.py` is the entry point and `requirements.txt` is present
5. Space builds and goes live at `https://huggingface.co/spaces/<you>/<space-name>`

## CLI (optional)

```bash
python app/resume_matcher.py --demo
python app/resume_matcher.py -r resume.txt -j jd.txt --html report.html
```

## Project structure

```
ResumeMatchAI/
├── app.py                 # Entry for cloud deploy
├── app/
│   ├── streamlit_app.py   # Web GUI
│   └── resume_matcher.py  # Core engine
├── data/
│   ├── sample_resume.txt
│   └── sample_jd.txt
├── docs/
│   ├── Concept_Note_ResumeMatchAI.pdf
│   ├── LeanCanvas_ResumeMatchAI.pdf
│   └── demo_report.html
├── ppt/
│   └── ResumeMatchAI_Presentation.pptx
├── requirements.txt
└── README.md
```

## How scoring works

- **55%** TF-IDF cosine similarity (language / keyword overlap)
- **45%** skill-overlap Jaccard (JD skills present in resume)
- Skill bank tuned for the Indian job market

## Licence

Open for educational and non-commercial use.
