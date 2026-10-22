from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_text_analysis_has_no_fake_ai_claim():
    response = client.post("/api/analyze-text", json={
        "resume_text": (
            "Aarav Sharma\nEmail: aarav@example.com\nPhone: +91 98765 43210\n"
            "Skills\nPython, SQL, Pandas, FastAPI, Git\n"
            "Projects\nBuilt a Python data analysis project using Pandas and SQL. "
            "Created REST API endpoints with FastAPI and wrote basic tests.\nEducation\nBSc Computer Science"
        ),
        "job_description": (
            "Junior developer role requiring Python, SQL, Pandas, data analysis, "
            "FastAPI, Docker, Git and data visualization. Build APIs and reports."
        )
    })
    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is True
    assert "python" in body["skills"]["matched"]
    assert "docker" in body["skills"]["missing_keyword_evidence"]
    assert body["analysis_method"]["generative_ai_used"] is False
    assert "hiring probabilities" in body["summary"]["disclaimer"]

def test_too_short_text_rejected():
    response = client.post("/api/analyze-text", json={
        "resume_text": "too short",
        "job_description": "This job description is also intentionally too short"
    })
    assert response.status_code == 422

def test_demo_is_labeled_synthetic():
    response = client.get("/api/demo/sample")
    assert response.status_code == 200
    assert response.json()["label"] == "SYNTHETIC_DEMO_DATA_NOT_REAL_JOB_LISTINGS"
