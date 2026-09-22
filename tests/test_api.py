from src.api.app import create_app

def test_health():
    client=create_app().test_client(); r=client.get("/health"); assert r.status_code==200 and r.json["status"]=="ok"
