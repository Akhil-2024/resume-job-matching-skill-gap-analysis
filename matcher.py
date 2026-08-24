import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader


# ============================================================
# 1. SKILL DATABASE
# ============================================================

skills = [
    "python",
    "sql",
    "c++",
    "machine learning",
    "deep learning",
    "pandas",
    "numpy",
    "scikit learn",
    "xgboost",
    "tensorflow",
    "pytorch",
    "nlp",
    "natural language processing",
    "aws",
    "docker",
    "streamlit",
    "fastapi",
    "postgresql",
    "pyspark",
    "spark",
    "git",
    "linux",
    "matplotlib",
    "seaborn",
    "statistics",
    "bert",
    "sentence bert",
    "transformers"
]


# ============================================================
# 2. PDF TEXT EXTRACTION
# ============================================================

def extract_text_from_pdf(pdf_file):

    reader = PdfReader(
        pdf_file
    )

    text = ""

    for page in reader.pages:

        page_text = page.extract_text()

        if page_text:

            text += (
                page_text
                + "\n"
            )

    return text


# ============================================================
# 3. TEXT CLEANING
# ============================================================

def clean_text(text):

    text = text.lower()

    # Normalize technology names
    text = text.replace(
        "scikit-learn",
        "scikit learn"
    )

    text = text.replace(
        "sentence-bert",
        "sentence bert"
    )

    text = text.replace(
        "sentencebert",
        "sentence bert"
    )

    # Preserve C++
    text = text.replace(
        "c++",
        "cplusplus"
    )

    # Remove special characters
    text = re.sub(
        r"[^a-zA-Z0-9\s]",
        " ",
        text
    )

    # Restore C++
    text = text.replace(
        "cplusplus",
        "c++"
    )

    # Remove extra spaces
    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# 4. RESUME CHUNKING
# ============================================================

def split_resume_into_chunks(
    text,
    chunk_size=80
):

    # Try to preserve bullet boundaries
    text = text.replace(
        "•",
        "\n•"
    )

    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    chunks = []

    current_chunk = []

    current_word_count = 0

    for line in lines:

        words = line.split()

        # If adding this line makes
        # the chunk too large
        if (
            current_word_count
            + len(words)
            > chunk_size
            and current_chunk
        ):

            chunks.append(
                " ".join(
                    current_chunk
                )
            )

            current_chunk = []

            current_word_count = 0

        current_chunk.append(
            line
        )

        current_word_count += len(
            words
        )

    # Add final chunk
    if current_chunk:

        chunks.append(
            " ".join(
                current_chunk
            )
        )

    return chunks


# ============================================================
# 5. SKILL EXTRACTION
# ============================================================

def extract_skills(text):

    found_skills = set()

    cleaned_text = clean_text(
        text
    )

    for skill in skills:

        pattern = (
            r"(?<!\w)"
            + re.escape(skill)
            + r"(?!\w)"
        )

        if re.search(
            pattern,
            cleaned_text,
            re.IGNORECASE
        ):

            found_skills.add(
                skill
            )

    return found_skills


# ============================================================
# 6. REQUIRED VS PREFERRED JOB SKILLS
# ============================================================

def extract_job_skills(
    job_text
):

    required_skills = set()

    preferred_skills = set()

    # Replace line breaks with spaces
    # so complete sentences remain together
    normalized_job = re.sub(
        r"\s+",
        " ",
        job_text
    ).strip()

    # Split into sentences
    sentences = re.split(
        r"[.!?]",
        normalized_job
    )

    preferred_keywords = [
        "preferred",
        "plus",
        "nice to have",
        "nice-to-have",
        "beneficial",
        "desirable",
        "good to have",
        "advantage",
        "bonus"
    ]

    for sentence in sentences:

        sentence = sentence.strip()

        if not sentence:

            continue

        sentence_skills = extract_skills(
            sentence
        )

        if not sentence_skills:

            continue

        sentence_lower = (
            sentence.lower()
        )

        is_preferred = any(
            keyword in sentence_lower
            for keyword
            in preferred_keywords
        )

        if is_preferred:

            preferred_skills.update(
                sentence_skills
            )

        else:

            required_skills.update(
                sentence_skills
            )

    # Required status has priority
    preferred_skills -= (
        required_skills
    )

    return (
        required_skills,
        preferred_skills
    )


# ============================================================
# 7. TF-IDF CHUNK-BASED SCORE
# ============================================================

def calculate_tfidf_score(
    resume_text,
    job_text,
    top_k=3
):

    resume_chunks = (
        split_resume_into_chunks(
            resume_text
        )
    )

    if not resume_chunks:

        return 0.0

    # Clean each resume chunk
    clean_chunks = [
        clean_text(chunk)
        for chunk
        in resume_chunks
    ]

    clean_job = clean_text(
        job_text
    )

    # Create ONE TF-IDF vocabulary
    # using all resume chunks + JD
    documents = (
        clean_chunks
        + [clean_job]
    )

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        sublinear_tf=True
    )

    tfidf_matrix = (
        vectorizer.fit_transform(
            documents
        )
    )

    # Last vector = job description
    job_vector = (
        tfidf_matrix[-1]
    )

    # Previous vectors = resume chunks
    resume_vectors = (
        tfidf_matrix[:-1]
    )

    similarities = (
        cosine_similarity(
            resume_vectors,
            job_vector
        ).flatten()
    )

    # Take best matching chunks
    top_scores = sorted(
        similarities,
        reverse=True
    )[:top_k]

    if not top_scores:

        return 0.0

    # Weighted top-k average
    weights = [
        1 / (i + 1)
        for i in range(
            len(top_scores)
        )
    ]

    weighted_score = (
        sum(
            score * weight
            for score, weight
            in zip(
                top_scores,
                weights
            )
        )
        / sum(weights)
    )

    return (
        weighted_score
        * 100
    )


# ============================================================
# 8. SENTENCE-BERT SEMANTIC SCORE
# ============================================================

def calculate_semantic_score(
    resume_text,
    job_text,
    model,
    top_k=3
):

    resume_chunks = (
        split_resume_into_chunks(
            resume_text
        )
    )

    if not resume_chunks:

        return 0.0

    # Use original natural-language
    # resume chunks for SBERT
    chunk_embeddings = (
        model.encode(
            resume_chunks,
            normalize_embeddings=True
        )
    )

    job_embedding = (
        model.encode(
            [job_text],
            normalize_embeddings=True
        )
    )

    similarities = (
        cosine_similarity(
            chunk_embeddings,
            job_embedding
        ).flatten()
    )

    top_scores = sorted(
        similarities,
        reverse=True
    )[:top_k]

    if not top_scores:

        return 0.0

    weights = [
        1 / (i + 1)
        for i in range(
            len(top_scores)
        )
    ]

    weighted_score = (
        sum(
            score * weight
            for score, weight
            in zip(
                top_scores,
                weights
            )
        )
        / sum(weights)
    )

    return (
        weighted_score
        * 100
    )


# ============================================================
# 9. MATCH CATEGORY
# ============================================================

def get_match_category(
    score
):

    if score >= 80:

        return "Excellent Match"

    elif score >= 65:

        return "Good Match"

    elif score >= 50:

        return "Moderate Match"

    else:

        return "Low Match"


# ============================================================
# 10. RECOMMENDATION SYSTEM
# ============================================================

def generate_recommendations(
    missing_required,
    missing_preferred,
    semantic_score,
    tfidf_score,
    required_score
):

    recommendations = []

    # Required skill gaps
    if missing_required:

        recommendations.append(
            "Priority skill gaps: "
            + ", ".join(
                sorted(
                    missing_required
                )
            )
        )

    # Preferred skill gaps
    if missing_preferred:

        recommendations.append(
            "Preferred skills that could strengthen "
            "your profile: "
            + ", ".join(
                sorted(
                    missing_preferred
                )
            )
        )

    # Semantic alignment
    if semantic_score < 60:

        recommendations.append(
            "Improve project and experience descriptions "
            "so they better reflect the responsibilities "
            "mentioned in the job description."
        )

    # Keyword overlap
    if tfidf_score < 30:

        recommendations.append(
            "Use more relevant job-specific keywords "
            "naturally in your Projects, Experience "
            "and Skills sections."
        )

    # Required skill coverage
    if required_score < 70:

        recommendations.append(
            "Your required-skill coverage is relatively low. "
            "Prioritize the most important missing "
            "required skills."
        )

    if not recommendations:

        recommendations.append(
            "Your resume is strongly aligned "
            "with this job description."
        )

    return recommendations


# ============================================================
# 11. MAIN ANALYSIS FUNCTION
# ============================================================

def analyze_resume(
    resume,
    job_description,
    model
):

    # --------------------------------------------------------
    # TF-IDF SCORE
    # --------------------------------------------------------

    tfidf_score = (
        calculate_tfidf_score(
            resume,
            job_description
        )
    )


    # --------------------------------------------------------
    # SEMANTIC SCORE
    # --------------------------------------------------------

    semantic_score = (
        calculate_semantic_score(
            resume,
            job_description,
            model
        )
    )


    # --------------------------------------------------------
    # RESUME SKILLS
    # --------------------------------------------------------

    resume_skills = (
        extract_skills(
            resume
        )
    )


    # --------------------------------------------------------
    # JOB SKILLS
    # --------------------------------------------------------

    (
        required_job_skills,
        preferred_job_skills

    ) = extract_job_skills(
        job_description
    )


    # --------------------------------------------------------
    # REQUIRED SKILL MATCHING
    # --------------------------------------------------------

    matched_required = (
        resume_skills
        & required_job_skills
    )

    missing_required = (
        required_job_skills
        - resume_skills
    )


    # --------------------------------------------------------
    # PREFERRED SKILL MATCHING
    # --------------------------------------------------------

    matched_preferred = (
        resume_skills
        & preferred_job_skills
    )

    missing_preferred = (
        preferred_job_skills
        - resume_skills
    )


    # --------------------------------------------------------
    # REQUIRED SKILL SCORE
    # --------------------------------------------------------

    if required_job_skills:

        required_score = (
            len(
                matched_required
            )
            /
            len(
                required_job_skills
            )
        ) * 100

    else:

        required_score = 0.0


    # --------------------------------------------------------
    # PREFERRED SKILL SCORE
    # --------------------------------------------------------

    if preferred_job_skills:

        preferred_score = (
            len(
                matched_preferred
            )
            /
            len(
                preferred_job_skills
            )
        ) * 100

    else:

        preferred_score = None


    # --------------------------------------------------------
    # COMBINED SKILL SCORE
    # --------------------------------------------------------

    if preferred_score is not None:

        skill_match_score = (
            0.80
            * required_score

            + 0.20
            * preferred_score
        )

    else:

        # If JD has no preferred skills,
        # only required skills matter
        skill_match_score = (
            required_score
        )


    # --------------------------------------------------------
    # FINAL MATCH SCORE
    # --------------------------------------------------------

    final_score = (
        0.45
        * semantic_score

        + 0.20
        * tfidf_score

        + 0.35
        * skill_match_score
    )


    # --------------------------------------------------------
    # CATEGORY
    # --------------------------------------------------------

    match_category = (
        get_match_category(
            final_score
        )
    )


    # --------------------------------------------------------
    # RECOMMENDATIONS
    # --------------------------------------------------------

    recommendations = (
        generate_recommendations(
            missing_required,
            missing_preferred,
            semantic_score,
            tfidf_score,
            required_score
        )
    )


    # --------------------------------------------------------
    # RETURN EVERYTHING TO APP.PY
    # --------------------------------------------------------

    return {

        "tfidf_score":
            tfidf_score,

        "semantic_score":
            semantic_score,

        "required_score":
            required_score,

        "preferred_score":
            preferred_score,

        "skill_match_score":
            skill_match_score,

        "final_score":
            final_score,

        "match_category":
            match_category,

        "resume_skills":
            resume_skills,

        "required_job_skills":
            required_job_skills,

        "preferred_job_skills":
            preferred_job_skills,

        "matched_required":
            matched_required,

        "missing_required":
            missing_required,

        "matched_preferred":
            matched_preferred,

        "missing_preferred":
            missing_preferred,

        "recommendations":
            recommendations
    }


# ============================================================
# 12. TERMINAL TEST
# ============================================================

# This block runs ONLY when you execute:
#
# python matcher.py
#
# It will NOT run when app.py imports matcher.

if __name__ == "__main__":

    print(
        "Loading Sentence-BERT model..."
    )

    model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )


    # --------------------------------------------------------
    # LOAD LOCAL RESUME
    # --------------------------------------------------------

    resume = extract_text_from_pdf(
        "resume.pdf"
    )


    # --------------------------------------------------------
    # SAMPLE JOB DESCRIPTION
    # --------------------------------------------------------

    job_description = """
    We are looking for a Data Scientist to build and deploy
    machine learning solutions using large-scale datasets.

    The candidate should have experience in Python, SQL,
    Machine Learning, Scikit-learn and XGBoost.

    The role involves exploratory data analysis, feature
    engineering, model development, model evaluation and
    communicating insights from data.

    Experience with AWS, Docker, Git and deployment of
    machine learning applications is preferred.
    """


    # --------------------------------------------------------
    # RUN ANALYSIS
    # --------------------------------------------------------

    results = analyze_resume(
        resume,
        job_description,
        model
    )


    # --------------------------------------------------------
    # PRINT RESULT
    # --------------------------------------------------------

    print(
        "\n===================================="
    )

    print(
        " RESUME-JOB MATCH ANALYSIS"
    )

    print(
        "===================================="
    )


    print(
        "\nTF-IDF Score:",
        round(
            results[
                "tfidf_score"
            ],
            2
        ),
        "%"
    )


    print(
        "Semantic Score:",
        round(
            results[
                "semantic_score"
            ],
            2
        ),
        "%"
    )


    print(
        "Required Skill Score:",
        round(
            results[
                "required_score"
            ],
            2
        ),
        "%"
    )


    if (
        results[
            "preferred_score"
        ]
        is not None
    ):

        print(
            "Preferred Skill Score:",
            round(
                results[
                    "preferred_score"
                ],
                2
            ),
            "%"
        )

    else:

        print(
            "Preferred Skill Score: N/A"
        )


    print(
        "Combined Skill Match Score:",
        round(
            results[
                "skill_match_score"
            ],
            2
        ),
        "%"
    )


    print(
        "\nFinal Match Score:",
        round(
            results[
                "final_score"
            ],
            2
        ),
        "%"
    )


    print(
        "Match Category:",
        results[
            "match_category"
        ]
    )


    # --------------------------------------------------------
    # REQUIRED SKILLS
    # --------------------------------------------------------

    print(
        "\nRequired Job Skills:"
    )

    for skill in sorted(
        results[
            "required_job_skills"
        ]
    ):

        print(
            "-",
            skill
        )


    print(
        "\nMatched Required Skills:"
    )

    if results[
        "matched_required"
    ]:

        for skill in sorted(
            results[
                "matched_required"
            ]
        ):

            print(
                "✓",
                skill
            )

    else:

        print(
            "None"
        )


    print(
        "\nMissing Required Skills:"
    )

    if results[
        "missing_required"
    ]:

        for skill in sorted(
            results[
                "missing_required"
            ]
        ):

            print(
                "✗",
                skill
            )

    else:

        print(
            "None"
        )


    # --------------------------------------------------------
    # PREFERRED SKILLS
    # --------------------------------------------------------

    print(
        "\nPreferred Job Skills:"
    )

    if results[
        "preferred_job_skills"
    ]:

        for skill in sorted(
            results[
                "preferred_job_skills"
            ]
        ):

            print(
                "-",
                skill
            )

    else:

        print(
            "None"
        )


    print(
        "\nMatched Preferred Skills:"
    )

    if results[
        "matched_preferred"
    ]:

        for skill in sorted(
            results[
                "matched_preferred"
            ]
        ):

            print(
                "✓",
                skill
            )

    else:

        print(
            "None"
        )


    print(
        "\nMissing Preferred Skills:"
    )

    if results[
        "missing_preferred"
    ]:

        for skill in sorted(
            results[
                "missing_preferred"
            ]
        ):

            print(
                "✗",
                skill
            )

    else:

        print(
            "None"
        )


    # --------------------------------------------------------
    # RECOMMENDATIONS
    # --------------------------------------------------------

    print(
        "\nRecommendations:"
    )

    for recommendation in results[
        "recommendations"
    ]:

        print(
            "-",
            recommendation
        )