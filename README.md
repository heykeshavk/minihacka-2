# ProofHire AI — Resume/JD Matcher Backend

A FastAPI backend for the provided hackathon problem statement:

> Upload a resume + paste a job description → match %, missing keywords, ATS-friendliness checks and improvement tips.

## Important honesty / hallucination controls

- No generative AI is used in this MVP.
- Skill matching uses a curated, explicit alias dictionary.
- Resume evidence is quoted from extracted resume text; the backend does not invent supporting evidence.
- “Missing keyword evidence” means the detector did not find a supported term. It does **not** mean the candidate lacks that skill.
- The skill match score is the proportion of detected job-description skills also detected in the resume.
- Text similarity uses TF-IDF cosine similarity (scikit-learn).
- ATS-friendliness is a transparent format/readability checklist, **not** an official ATS vendor score.
- The included demo example is synthetic and clearly labelled. It is not scraped from Indeed and does not claim to represent real listings.
- Scanned PDFs without selectable text are rejected with a clear message; OCR is not falsely claimed.

## Requirements

Python 3.10+ recommended.

## Windows PowerShell setup

```powershell
cd proofhire_resume_matcher
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

Open:
- API docs: http://127.0.0.1:8000/docs
- Health: http://127.0.0.1:8000/health
- Synthetic demo payload: http://127.0.0.1:8000/api/demo/sample

## Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/` | API info |
| GET | `/health` | Health check |
| GET | `/api/skills` | Supported skill dictionary |
| POST | `/api/analyze` | Upload PDF/DOCX/TXT + form field `job_description` |
| POST | `/api/analyze-text` | Analyze JSON `resume_text` and `job_description` |
| POST | `/api/report` | Download an HTML report from analysis JSON |
| GET | `/api/demo/sample` | Clearly labelled synthetic sample input |

## Test

```powershell
pytest -q
```

## Frontend request example (browser)

```javascript
const form = new FormData();
form.append("resume", fileInput.files[0]);
form.append("job_description", jobDescription);

const response = await fetch("http://127.0.0.1:8000/api/analyze", {
  method: "POST",
  body: form
});
const result = await response.json();
```

Do not manually set `Content-Type` when using `FormData`; the browser adds the multipart boundary.

## API example for `/api/analyze-text`

```json
{
  "resume_text": "Python developer. Built a data analysis project with Python, Pandas, SQL and FastAPI.",
  "job_description": "We need Python, SQL, FastAPI, Docker and data analysis skills."
}
```

## Production notes

- Set `allow_origins` to your actual frontend origin(s) in `main.py`.
- Add authentication, rate limits, malware scanning, logging policy and secure storage before public deployment.
- This MVP does not persist uploaded resumes. Avoid logging or retaining personal data unnecessarily.
- Do not use this heuristic score as the sole basis for hiring or rejecting people.
