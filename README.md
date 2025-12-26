# 📄 Resume Analyzer with GPT-4o-mini

A Streamlit app that analyzes resumes against job descriptions using OpenAI's GPT-4o-mini model with NLTK keyword extraction.

## Features

✅ **AI-Powered Analysis**

- Match score (0-100) between resume and job description
- Skills match & experience match scoring
- Identified strengths and weaknesses
- Missing skills detection
- Personalized recommendations

📊 **Keyword Extraction**

- NLTK-based keyword extraction from resume and job description
- Frequency-based ranking of skills
- Visual skill badges

🎨 **Beautiful UI**

- Color-coded score bars (🟢 Excellent, 🟡 Good, 🟠 Fair, 🔴 Poor)
- Formatted analysis report with expandable sections
- Export analysis as text file

## Installation

### 1. Clone the repository

```bash
git clone <your-repo>
cd "Python Projects"
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure API Key (Choose One)

#### Option A: Production (Recommended)

Create `.streamlit/secrets.toml` in your project root:

```toml
OPENAI_API_KEY = "sk-proj-YOUR_API_KEY_HERE"
```

**⚠️ IMPORTANT:** Add `.streamlit/secrets.toml` to `.gitignore` (already included)

#### Option B: Local Testing

Paste your API key directly in the sidebar when running the app locally.

## Running Locally

```bash
streamlit run GPT_Resume_Analyzer.py
```

The app will:

1. Check for API key in `.streamlit/secrets.toml` (production)
2. Fall back to manual input field (local testing)
3. Stop execution if no API key is found

## Deployment to Streamlit Cloud

### Step 1: Push to GitHub

```bash
git add .
git commit -m "Add resume analyzer app"
git push origin main
```

### Step 2: Deploy on Streamlit Cloud

1. Go to [streamlit.io/cloud](https://streamlit.io/cloud)
2. Click "New app"
3. Select your GitHub repository and branch
4. Click "Deploy"

### Step 3: Add API Key to Secrets

1. In Streamlit Cloud dashboard, go to your app
2. Click "Settings" → "Secrets"
3. Add your OpenAI API key:
   ```
   OPENAI_API_KEY = "sk-proj-..."
   ```
4. Click "Save"
5. Streamlit will automatically redeploy

## API Cost

- Uses **free tier OpenAI API** (gpt-4o-mini)
- Typical cost: ~$0.01-0.02 per resume analysis
- [OpenAI Pricing](https://openai.com/pricing)

## Project Structure

```
Python Projects/
├── GPT_Resume_Analyzer.py      # Main application
├── requirements.txt             # Python dependencies
├── .gitignore                   # Git ignore rules
├── .streamlit/
│   └── secrets.toml.example    # Secrets template (copy this)
└── README.md                    # This file
```

## How It Works

1. **Resume Input** → PDF/TXT upload or text paste
2. **Job Description Input** → Text upload or paste
3. **Keyword Extraction** → NLTK extracts top 15 keywords from each
4. **GPT Analysis** → GPT-4o-mini scores the match
5. **Report Generation** → Formatted analysis with recommendations

## Security Best Practices

✅ **DO:**

- Use `.streamlit/secrets.toml` for production
- Add `.streamlit/secrets.toml` to `.gitignore`
- Use environment variables for secrets
- Regenerate API keys if exposed

❌ **DON'T:**

- Hardcode API keys in code
- Share API keys in public repos
- Use API key for multiple apps
- Commit secrets to git

## Troubleshooting

### "API Key Not Found"

- Make sure `.streamlit/secrets.toml` exists with correct key
- Or paste your key in the sidebar for local testing
- Check API key is valid at [platform.openai.com](https://platform.openai.com/account/api-keys)

### "PDF extraction failed"

- Ensure PDF is text-based (not scanned image)
- Try converting to TXT first

### "JSON parsing error from GPT"

- API key might be invalid
- Check OpenAI account status
- Verify `gpt-4o-mini` model is available in your account

## Requirements

- Python 3.8+
- OpenAI API key
- See `requirements.txt` for dependencies

## License

MIT License - Feel free to use and modify!

---

**Created with ❤️ using Streamlit & OpenAI**
