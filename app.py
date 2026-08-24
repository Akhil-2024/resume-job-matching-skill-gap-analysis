import streamlit as st

from sentence_transformers import SentenceTransformer

from matcher import (
    extract_text_from_pdf,
    analyze_resume
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Resume Job Matcher",
    page_icon="🎯",
    layout="wide"
)


# ============================================================
# LOAD SENTENCE-BERT MODEL
# ============================================================

@st.cache_resource
def load_model():

    return SentenceTransformer(
        "all-MiniLM-L6-v2"
    )


model = load_model()


# ============================================================
# TITLE
# ============================================================

st.title(
    "🎯 Resume–Job Matching & Skill Gap Analysis"
)

st.write(
    """
    Upload your resume and paste a job description
    to analyze semantic similarity, keyword relevance,
    required skills, preferred skills, and skill gaps.
    """
)

st.divider()


# ============================================================
# INPUT SECTION
# ============================================================

left, right = st.columns(
    2
)


# ------------------------------------------------------------
# Resume Upload
# ------------------------------------------------------------

with left:

    st.subheader(
        "📄 Upload Resume"
    )

    uploaded_resume = st.file_uploader(
        "Choose your resume",
        type=["pdf"]
    )


# ------------------------------------------------------------
# Job Description
# ------------------------------------------------------------

with right:

    st.subheader(
        "💼 Job Description"
    )

    job_description = st.text_area(
        "Paste the complete job description",
        height=300,
        placeholder="""
Example:

We are looking for a Data Scientist
with experience in Python, SQL,
Machine Learning, Scikit-learn
and XGBoost.

Experience with AWS, Docker,
and Git is preferred.
"""
    )


# ============================================================
# ANALYZE BUTTON
# ============================================================

st.divider()

analyze_button = st.button(
    "🔍 Analyze Resume",
    type="primary",
    use_container_width=True
)


# ============================================================
# RUN ANALYSIS
# ============================================================

if analyze_button:

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    if uploaded_resume is None:

        st.error(
            "Please upload a resume PDF."
        )

    elif not job_description.strip():

        st.error(
            "Please paste a job description."
        )

    else:

        # ----------------------------------------------------
        # Resume Text Extraction
        # ----------------------------------------------------

        with st.spinner(
            "Analyzing resume..."
        ):

            resume_text = (
                extract_text_from_pdf(
                    uploaded_resume
                )
            )

            results = analyze_resume(
                resume_text,
                job_description,
                model
            )


        # ====================================================
        # OVERALL RESULT
        # ====================================================

        st.divider()

        st.header(
            "📊 Match Analysis"
        )


        # ----------------------------------------------------
        # Final score
        # ----------------------------------------------------

        final_score = results[
            "final_score"
        ]

        category = results[
            "match_category"
        ]


        st.metric(
            "Overall Match Score",
            f"{final_score:.2f}%"
        )

        st.progress(
            min(
                int(final_score),
                100
            )
        )


        st.write(
            f"### Match Category: **{category}**"
        )


        # ====================================================
        # SCORE BREAKDOWN
        # ====================================================

        st.subheader(
            "Score Breakdown"
        )


        col1, col2, col3 = (
            st.columns(3)
        )


        # ----------------------------------------------------
        # Semantic
        # ----------------------------------------------------

        with col1:

            st.metric(
                "Semantic Similarity",
                f'{results["semantic_score"]:.2f}%'
            )


        # ----------------------------------------------------
        # TF-IDF
        # ----------------------------------------------------

        with col2:

            st.metric(
                "TF-IDF Similarity",
                f'{results["tfidf_score"]:.2f}%'
            )


        # ----------------------------------------------------
        # Skill score
        # ----------------------------------------------------

        with col3:

            st.metric(
                "Skill Match",
                f'{results["skill_match_score"]:.2f}%'
            )


        # ====================================================
        # REQUIRED / PREFERRED SCORE
        # ====================================================

        col4, col5 = (
            st.columns(2)
        )


        with col4:

            st.metric(
                "Required Skill Score",
                f'{results["required_score"]:.2f}%'
            )


        with col5:

            preferred_score = (
                results[
                    "preferred_score"
                ]
            )

            if preferred_score is not None:

                st.metric(
                    "Preferred Skill Score",
                    f"{preferred_score:.2f}%"
                )

            else:

                st.metric(
                    "Preferred Skill Score",
                    "N/A"
                )


        # ====================================================
        # REQUIRED SKILLS
        # ====================================================

        st.divider()

        st.header(
            "Required Skills"
        )


        required_col1, required_col2 = (
            st.columns(2)
        )


        # ----------------------------------------------------
        # Matched required
        # ----------------------------------------------------

        with required_col1:

            st.subheader(
                "✅ Matched"
            )

            matched_required = (
                results[
                    "matched_required"
                ]
            )

            if matched_required:

                for skill in sorted(
                    matched_required
                ):

                    st.success(
                        skill
                    )

            else:

                st.info(
                    "No required skills matched."
                )


        # ----------------------------------------------------
        # Missing required
        # ----------------------------------------------------

        with required_col2:

            st.subheader(
                "❌ Missing"
            )

            missing_required = (
                results[
                    "missing_required"
                ]
            )

            if missing_required:

                for skill in sorted(
                    missing_required
                ):

                    st.error(
                        skill
                    )

            else:

                st.success(
                    "No required skills missing."
                )


        # ====================================================
        # PREFERRED SKILLS
        # ====================================================

        st.divider()

        st.header(
            "Preferred Skills"
        )


        preferred_col1, preferred_col2 = (
            st.columns(2)
        )


        # ----------------------------------------------------
        # Matched preferred
        # ----------------------------------------------------

        with preferred_col1:

            st.subheader(
                "✅ Matched"
            )

            matched_preferred = (
                results[
                    "matched_preferred"
                ]
            )

            if matched_preferred:

                for skill in sorted(
                    matched_preferred
                ):

                    st.success(
                        skill
                    )

            else:

                st.info(
                    "No preferred skills matched."
                )


        # ----------------------------------------------------
        # Missing preferred
        # ----------------------------------------------------

        with preferred_col2:

            st.subheader(
                "⚠️ Missing"
            )

            missing_preferred = (
                results[
                    "missing_preferred"
                ]
            )

            if missing_preferred:

                for skill in sorted(
                    missing_preferred
                ):

                    st.warning(
                        skill
                    )

            else:

                st.success(
                    "No preferred skills missing."
                )


        # ====================================================
        # DETECTED RESUME SKILLS
        # ====================================================

        st.divider()

        st.header(
            "🧠 Detected Resume Skills"
        )


        resume_skills = (
            results[
                "resume_skills"
            ]
        )


        if resume_skills:

            st.write(
                ", ".join(
                    sorted(
                        resume_skills
                    )
                )
            )

        else:

            st.info(
                "No skills detected."
            )


        # ====================================================
        # JOB SKILLS
        # ====================================================

        st.divider()

        st.header(
            "💼 Skills Detected From Job Description"
        )


        job_col1, job_col2 = (
            st.columns(2)
        )


        with job_col1:

            st.subheader(
                "Required"
            )

            required_job_skills = (
                results[
                    "required_job_skills"
                ]
            )

            if required_job_skills:

                for skill in sorted(
                    required_job_skills
                ):

                    st.write(
                        "•",
                        skill
                    )

            else:

                st.write(
                    "None detected"
                )


        with job_col2:

            st.subheader(
                "Preferred"
            )

            preferred_job_skills = (
                results[
                    "preferred_job_skills"
                ]
            )

            if preferred_job_skills:

                for skill in sorted(
                    preferred_job_skills
                ):

                    st.write(
                        "•",
                        skill
                    )

            else:

                st.write(
                    "None detected"
                )


        # ====================================================
        # RECOMMENDATIONS
        # ====================================================

        st.divider()

        st.header(
            "💡 Personalized Recommendations"
        )


        for recommendation in (
            results[
                "recommendations"
            ]
        ):

            st.write(
                "•",
                recommendation
            )


        # ====================================================
        # EXTRACTED TEXT
        # ====================================================

        st.divider()

        with st.expander(
            "📄 View Extracted Resume Text"
        ):

            st.text(
                resume_text
            )