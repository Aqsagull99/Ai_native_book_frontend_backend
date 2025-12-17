from fastapi.testclient import TestClient
from backend.main import app

# Create test client
client = TestClient(app)

def test_simple():
    response = client.get("/")
    assert response.status_code == 200