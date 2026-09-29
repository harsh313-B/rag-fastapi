from fastapi.testclient import TestClient

from main import app


client = TestClient(app)


def test_ask_endpoint():

    response = client.post(
        "/ask",
        json={
            "question": "What does FastAPI use to validate data?"
        }
    )

    assert response.status_code == 200

    data = response.json()

    assert data["question"] == "What does FastAPI use to validate data?"

    assert "answer" in data

    assert "sources" in data

    assert len(data["sources"]) > 0

def test_missing_question():

    response = client.post(
        "/ask",
        json={}
    )

    assert response.status_code == 422