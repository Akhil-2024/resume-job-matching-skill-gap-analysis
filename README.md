# Resume–Job Matching & Skill Gap Analysis

An NLP-based application that matches resumes with job descriptions using **Sentence-BERT, TF-IDF, and skill-gap analysis**.

The system analyzes semantic similarity, keyword relevance, required skills, preferred skills, and provides personalized recommendations through an interactive Streamlit interface.

## Features

* Upload resume in PDF format
* Extract resume text automatically
* Sentence-BERT semantic similarity
* TF-IDF keyword similarity
* Resume chunk-based matching
* Required and preferred skill detection
* Matched and missing skill identification
* Weighted resume–job compatibility score
* Personalized improvement recommendations
* Interactive Streamlit dashboard

## Tech Stack

* Python
* NLP
* Sentence-BERT
* TF-IDF
* Scikit-learn
* Streamlit
* PyPDF

## Architecture

```text
Resume PDF
    |
    v
PDF Text Extraction
    |
    v
Resume Chunking
    |
    +-------------------+
    |                   |
    v                   v
Sentence-BERT         TF-IDF
Semantic Score     Keyword Score
    |                   |
    +---------+---------+
              |
              v
       Skill Extraction
              |
        +-----+-----+
        |           |
        v           v
   Required      Preferred
    Skills         Skills
        |           |
        +-----+-----+
              |
              v
        Skill Gap Analysis
              |
              v
        Weighted Match Score
              |
              v
       Recommendations
              |
              v
      Streamlit Dashboard
```

## Scoring Method

The current baseline score is calculated as:

```text
Final Score =
0.45 × Semantic Similarity
+ 0.20 × TF-IDF Similarity
+ 0.35 × Skill Match Score
```

The skill score gives greater importance to required skills:

```text
Skill Match =
0.80 × Required Skill Score
+ 0.20 × Preferred Skill Score
```

These weights are currently heuristic and can later be tuned using labeled resume–job relevance data.

## Project Structure

```text
resume-job-matching-skill-gap-analysis/
│
├── app.py
├── matcher.py
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore
```

## Installation

Clone the repository:

```bash
git clone https://github.com/Akhil-2024/resume-job-matching-skill-gap-analysis.git
```

Move into the project folder:

```bash
cd resume-job-matching-skill-gap-analysis
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the Application

Start the Streamlit app:

```bash
streamlit run app.py
```

Then open:

```text
http://localhost:8501
```

## Example Output

The application provides:

```text
Overall Match Score

Semantic Similarity
TF-IDF Similarity
Skill Match Score

Required Skills
✓ Python
✓ SQL
✓ Machine Learning
✓ XGBoost

Preferred Skills
✓ Git
✗ AWS
✗ Docker

Recommendations
- Improve job-specific project descriptions
- Add relevant missing preferred skills
```

## NLP Pipeline

### Sentence-BERT

Sentence-BERT is used to generate semantic embeddings for resume chunks and the job description.

The system compares these embeddings using cosine similarity to determine how closely the meaning of the resume aligns with the job description.

### TF-IDF

TF-IDF captures exact keyword and phrase overlap between resume sections and the job description.

### Skill Gap Analysis

Skills are extracted from both the resume and job description and divided into:

* Required skills
* Preferred skills
* Matched skills
* Missing skills

This provides explainable feedback in addition to the overall similarity score.

## Future Improvements

* Section-aware resume scoring
* Larger skill ontology
* Named Entity Recognition for automatic skill extraction
* Multiple candidate ranking
* Cross-Encoder reranking
* Learning-to-rank models
* SHAP-based explanation
* Resume recommendation generation
* Labeled resume–job dataset for weight optimization
* Docker deployment
* Cloud deployment

## Author

Akhilesh Kumar Patel

M.Tech, Control & Automation
Indian Institute of Technology Delhi
