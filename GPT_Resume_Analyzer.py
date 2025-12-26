import streamlit as st
from pypdf import PdfReader
import json
from datetime import datetime
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import FreqDist
import google.generativeai as genai
import ssl

# Fix SSL certificate issues for NLTK data downloads on Streamlit Cloud
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Download NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

# Page Configuration
st.set_page_config(
    page_title="Resume Analyzer with Gemini",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .score-excellent { color: #28a745; font-weight: bold; font-size: 24px; }
    .score-good { color: #ffc107; font-weight: bold; font-size: 24px; }
    .score-fair { color: #fd7e14; font-weight: bold; font-size: 24px; }
    .score-poor { color: #dc3545; font-weight: bold; font-size: 24px; }
    .strength-box { background-color: #d4edda; padding: 15px; border-radius: 5px; margin: 10px 0; }
    .weakness-box { background-color: #f8d7da; padding: 15px; border-radius: 5px; margin: 10px 0; }
    .suggestion-box { background-color: #d1ecf1; padding: 15px; border-radius: 5px; margin: 10px 0; }
    .skill-badge { display: inline-block; background-color: #e7f3ff; color: #004085;
                   padding: 5px 10px; border-radius: 15px; margin: 5px; font-size: 12px; }
    </style>
""", unsafe_allow_html=True)

def extract_text_from_pdf(pdf_file):
    """Extract text from PDF file using pypdf for improved accuracy"""
    try:
        pdf_reader = PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
        return None

def extract_keywords(text, num_keywords=15):
    """Extract important keywords from text using NLTK"""
    if not text:
        return []

    # Tokenize and convert to lowercase
    tokens = word_tokenize(text.lower())

    # Remove stopwords and non-alphabetic tokens
    stop_words = set(stopwords.words('english'))
    keywords = [
        token for token in tokens
        if token.isalpha() and token not in stop_words and len(token) > 3
    ]

    # Get frequency distribution
    freq_dist = FreqDist(keywords)

    # Return top keywords
    return [word for word, _ in freq_dist.most_common(num_keywords)]

def get_score_color(score):
    """Return color based on score"""
    if score >= 80:
        return "🟢", "#28a745"
    elif score >= 60:
        return "🟡", "#ffc107"
    elif score >= 40:
        return "🟠", "#fd7e14"
    else:
        return "🔴", "#dc3545"

def create_score_bar(score):
    """Create a visual score bar"""
    filled = int(score / 5)
    empty = 20 - filled
    bar = "█" * filled + "░" * empty
    return f"{bar} {score}/100"

def analyze_resume_with_gpt(resume_text, job_description, resume_keywords, jd_keywords, api_key):
    """Use Google Gemini 2.5 Pro to analyze resume match with reliable JSON output"""

    try:
        genai.configure(api_key=api_key)

        # Model fallback chain for 2025
        models_to_try = [
            'gemini-2.0-pro-exp',
            'gemini-2.0-flash-exp',
            'gemini-2.0-pro',
            'gemini-2.5-pro',
            'gemini-1.5-flash',
            'gemini-pro'
        ]

        model = None
        used_model = None

        for model_name in models_to_try:
            try:
                model = genai.GenerativeModel(model_name)
                used_model = model_name
                st.session_state.used_model = used_model
                break
            except Exception:
                continue

        if not model:
            st.error("Error: No available Gemini models. Please check your API key.")
            return None

        prompt = f"""You are an expert resume analyst and recruiter. Analyze the following resume against the job description and provide a detailed analysis.

RESUME:
{resume_text[:2000]}...

JOB DESCRIPTION:
{job_description[:1500]}...

RESUME KEYWORDS EXTRACTED: {', '.join(resume_keywords)}
JOB DESCRIPTION KEYWORDS: {', '.join(jd_keywords)}

Please provide your analysis in the following JSON format:
{{
    "match_score": <0-100>,
    "skills_match": <0-100>,
    "experience_match": <0-100>,
    "strengths": ["strength 1", "strength 2", "strength 3"],
    "weaknesses": ["weakness 1", "weakness 2", "weakness 3"],
    "suggestions": ["suggestion 1", "suggestion 2", "suggestion 3"],
    "missing_skills": ["skill 1", "skill 2"],
    "summary": "Brief 2-3 sentence summary of the match"
}}

IMPORTANT: Return ONLY valid JSON, no markdown or extra text. Be critical but fair. The match score should reflect how well the resume matches the job description."""

        response = model.generate_content(prompt)
        response_text = response.text

        if not response_text:
            st.error("Error: No content in API response")
            return None

        response_text = response_text.strip()

        # Remove markdown code blocks if present
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.startswith("```"):
            response_text = response_text[3:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
        response_text = response_text.strip()

        # Parse JSON response
        analysis = json.loads(response_text)
        return analysis

    except json.JSONDecodeError as e:
        st.error(f"Error: Invalid JSON response from API. Please try again. (JSON Error: {str(e)})")
        return None
    except Exception as e:
        st.error(f"Error calling Gemini API: {str(e)}")
        return None

def display_analysis_report(analysis):
    """Display the analysis report with formatted output"""
    if not analysis:
        return

    col1, col2, col3 = st.columns(3)

    # Overall Match Score
    with col1:
        score = analysis.get("match_score", 0)
        emoji, color = get_score_color(score)
        st.markdown(f"<h3>Overall Match {emoji}</h3>", unsafe_allow_html=True)
        st.markdown(f"<div style='color: {color}; font-size: 36px; font-weight: bold;'>{score}</div>", unsafe_allow_html=True)
        st.text(create_score_bar(score))

    # Skills Match
    with col2:
        skills_score = analysis.get("skills_match", 0)
        emoji, color = get_score_color(skills_score)
        st.markdown(f"<h3>Skills Match {emoji}</h3>", unsafe_allow_html=True)
        st.markdown(f"<div style='color: {color}; font-size: 36px; font-weight: bold;'>{skills_score}</div>", unsafe_allow_html=True)
        st.text(create_score_bar(skills_score))

    # Experience Match
    with col3:
        exp_score = analysis.get("experience_match", 0)
        emoji, color = get_score_color(exp_score)
        st.markdown(f"<h3>Experience Match {emoji}</h3>", unsafe_allow_html=True)
        st.markdown(f"<div style='color: {color}; font-size: 36px; font-weight: bold;'>{exp_score}</div>", unsafe_allow_html=True)
        st.text(create_score_bar(exp_score))

    st.divider()

    # Summary
    st.subheader("📋 Summary")
    st.info(analysis.get("summary", "No summary available"))

    st.divider()

    # Strengths
    st.subheader("✅ Strengths")
    for strength in analysis.get("strengths", []):
        st.markdown(f"<div class='strength-box'>• {strength}</div>", unsafe_allow_html=True)

    # Weaknesses
    st.subheader("⚠️ Weaknesses")
    for weakness in analysis.get("weaknesses", []):
        st.markdown(f"<div class='weakness-box'>• {weakness}</div>", unsafe_allow_html=True)

    # Missing Skills
    st.subheader("🎯 Missing Skills")
    missing_skills = analysis.get("missing_skills", [])
    if missing_skills:
        cols = st.columns(len(missing_skills))
        for idx, skill in enumerate(missing_skills):
            with cols[idx]:
                st.markdown(f"<span class='skill-badge'>{skill}</span>", unsafe_allow_html=True)
    else:
        st.success("No major missing skills identified!")

    # Suggestions
    st.subheader("💡 Recommendations")
    for i, suggestion in enumerate(analysis.get("suggestions", []), 1):
        st.markdown(f"<div class='suggestion-box'>{i}. {suggestion}</div>", unsafe_allow_html=True)

def main():
    st.title("📄 Resume Analyzer with Google Gemini")
    st.markdown("Analyze your resume against job descriptions using AI-powered insights (FREE with Gemini API)")

    # Sidebar for API Key - Production Ready
    with st.sidebar:
        st.header("⚙️ Configuration")

        # Verify Gemini is loaded
        st.write("✅ Gemini API loaded")

        # Show which model is being used
        if "used_model" in st.session_state:
            st.info(f"📊 Using model: **{st.session_state.used_model}**")

        # Try to get API key from st.secrets (production)
        api_key = None

        try:
            api_key = st.secrets.get("GEMINI_API_KEY", None)
            if api_key:
                st.success("✅ Using production Gemini API key from secrets")
        except (AttributeError, KeyError):
            # st.secrets not available or key doesn't exist
            pass

        # Fallback to manual input (local testing)
        if not api_key:
            st.warning("⚠️ PRODUCTION MODE: Using secrets")
            st.info("""
            **For Local Testing:**
            1. Create `.streamlit/secrets.toml` in your project:
            ```
            GEMINI_API_KEY = "your-api-key-here"
            ```
            2. Or paste key below (not recommended for production)
            """)
            manual_key = st.text_input(
                "🔑 Google Gemini API Key (for local testing)",
                type="password",
                help="Get your FREE API key from ai.google.dev/tutorials/python_quickstart"
            )
            if manual_key:
                api_key = manual_key
                st.warning("⚠️ LOCAL TESTING MODE: Using manual input (not secure for production)")

        # Validate API key exists
        if not api_key:
            st.error("""
            ❌ **ERROR: Google Gemini API Key Not Found**

            **Get FREE Access:**
            1. Visit: https://ai.google.dev/tutorials/python_quickstart
            2. Click "Get API Key"
            3. Create new API key (FREE, 60 requests/minute)

            **Production Deployment:**
            - Add GEMINI_API_KEY to Streamlit Secrets (https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app#secrets-management)

            **Local Testing:**
            - Create `.streamlit/secrets.toml`:
              ```
              GEMINI_API_KEY = "your-key-here"
              ```
            - Or paste your API key above

            **⚠️ NEVER hardcode keys in code!**
            """)
            st.stop()

        st.markdown("---")
        st.markdown("""
        ### How to use:
        1. ✅ API Key configured (from secrets or input)
        2. 📄 Upload or paste your resume
        3. 📋 Enter or paste the job description
        4. 🚀 Click 'Analyze Resume'
        5. 📊 Review the detailed analysis report

        ### Features:
        - ✅ AI-powered matching score (0-100)
        - 📊 Skills & experience analysis
        - 🎯 Skill gap identification
        - 💡 Personalized recommendations
        - 🎨 Color-coded visualization
        - 🆓 FREE with Google Gemini API (60 req/min)
        """)

    # Main Content
    st.subheader("📥 Resume Upload")
    col1, col2 = st.columns(2)

    resume_text = ""
    with col1:
        st.markdown("**Option 1: Upload PDF or Text File**")
        st.info("📤 Upload your resume (PDF or TXT format)")
        resume_file = st.file_uploader(
            "Choose your resume (PDF or TXT)",
            type=["pdf", "txt"],
            key="resume_upload"
        )

        if resume_file:
            if resume_file.type == "application/pdf":
                resume_text = extract_text_from_pdf(resume_file)
            else:
                resume_text = resume_file.read().decode("utf-8")

            if resume_text:
                st.success(f"✓ Loaded {len(resume_text)} characters")

    with col2:
        st.markdown("**Option 2: Paste Resume Text**")
        st.info("📝 Or copy-paste your resume text here")
        resume_text_pasted = st.text_area(
            "Or paste your resume here",
            height=150,
            key="resume_text"
        )
        if resume_text_pasted:
            resume_text = resume_text_pasted

    st.divider()

    st.subheader("📋 Job Description")
    col1, col2 = st.columns(2)

    job_description = ""
    with col1:
        st.markdown("**Option 1: Upload Text File**")
        st.info("📤 Upload the job description (TXT format)")
        jd_file = st.file_uploader(
            "Choose job description (TXT)",
            type=["txt"],
            key="jd_upload"
        )

        if jd_file:
            job_description = jd_file.read().decode("utf-8")
            st.success(f"✓ Loaded {len(job_description)} characters")

    with col2:
        st.markdown("**Option 2: Paste Description**")
        st.info("📝 Or copy-paste the job description here")
        jd_pasted = st.text_area(
            "Or paste the job description here",
            height=150,
            key="jd_text"
        )
        if jd_pasted:
            job_description = jd_pasted

    st.divider()

    # Analysis Button
    if st.button("🚀 Analyze Resume", type="primary", use_container_width=True):
        # Input validation
        validation_errors = []

        if not api_key:
            validation_errors.append("API Key Error: Please enter your Gemini API key in the sidebar (left side)")

        if not resume_text or not resume_text.strip():
            validation_errors.append("Please provide a resume (paste text or upload a file)")
        elif len(resume_text.strip()) < 50:
            validation_errors.append("Resume is too short (at least 50 characters required)")

        if not job_description or not job_description.strip():
            validation_errors.append("Please provide a job description (paste text or upload a file)")
        elif len(job_description.strip()) < 50:
            validation_errors.append("Job description is too short (at least 50 characters required)")

        if validation_errors:
            for error in validation_errors:
                st.error(f"❌ {error}")
        else:
            with st.spinner("🔄 Analyzing resume... This may take a few seconds"):
                # Extract keywords
                resume_keywords = extract_keywords(resume_text, num_keywords=15)
                jd_keywords = extract_keywords(job_description, num_keywords=15)

                # Get GPT analysis
                analysis = analyze_resume_with_gpt(
                    resume_text,
                    job_description,
                    resume_keywords,
                    jd_keywords,
                    api_key
                )

                if analysis:
                    st.session_state.analysis = analysis
                    st.session_state.resume_keywords = resume_keywords
                    st.session_state.jd_keywords = jd_keywords

    # Display results if available
    if "analysis" in st.session_state:
        st.divider()
        st.header("📊 Analysis Report")
        display_analysis_report(st.session_state.analysis)

        st.divider()
        st.subheader("🔑 Keywords Detected")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Resume Keywords**")
            for keyword in st.session_state.resume_keywords:
                st.markdown(f"<span class='skill-badge'>{keyword}</span>", unsafe_allow_html=True)

        with col2:
            st.markdown("**Job Description Keywords**")
            for keyword in st.session_state.jd_keywords:
                st.markdown(f"<span class='skill-badge'>{keyword}</span>", unsafe_allow_html=True)

        # Export results
        st.divider()
        if st.button("📥 Export Report as Text", use_container_width=True):
            report = f"""
RESUME ANALYSIS REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

SCORES:
- Overall Match: {st.session_state.analysis.get('match_score', 0)}/100
- Skills Match: {st.session_state.analysis.get('skills_match', 0)}/100
- Experience Match: {st.session_state.analysis.get('experience_match', 0)}/100

SUMMARY:
{st.session_state.analysis.get('summary', 'N/A')}

STRENGTHS:
{chr(10).join(f"• {s}" for s in st.session_state.analysis.get('strengths', []))}

WEAKNESSES:
{chr(10).join(f"• {w}" for w in st.session_state.analysis.get('weaknesses', []))}

MISSING SKILLS:
{', '.join(st.session_state.analysis.get('missing_skills', []))}

RECOMMENDATIONS:
{chr(10).join(f"{i}. {s}" for i, s in enumerate(st.session_state.analysis.get('suggestions', []), 1))}

RESUME KEYWORDS: {', '.join(st.session_state.resume_keywords)}
JOB KEYWORDS: {', '.join(st.session_state.jd_keywords)}
"""
            st.download_button(
                label="Download Report",
                data=report,
                file_name=f"resume_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                use_container_width=True
            )

if __name__ == "__main__":
    main()
