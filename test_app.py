import pytest
import json
from app import app 

@pytest.fixture
def client():
    with app.test_client() as client:
        yield client

# Teste para verificar a rota principal
def test_home(client):
    response = client.get('/')
    assert response.status_code == 200
    assert "API de Predição de Cogumelos com Machine Learning" in response.data.decode('utf-8')

# Teste para verificar predição válida
def test_predict_valid(client):
    # JSON de exemplo com os dados necessários para a predição
    test_data = {
        "data": {
            "cap-shape": "1",
            "cap-surface": "2",
            "cap-color": "1",
            "bruises": "0",
            "odor": "1",
            "gill-attachment": "1",
            "gill-spacing": "1",
            "gill-size": "1",
            "gill-color": "1",
            "stalk-shape": "4",
            "stalk-root": "2",
            "stalk-surface-above-ring": "1",
            "stalk-surface-below-ring": "1",
            "stalk-color-above-ring": "2",
            "stalk-color-below-ring": "0",
            "veil-type": "1",
            "veil-color": "1",
            "ring-number": "1",
            "ring-type": "1",
            "spore-print-color": "0",
            "population": "5",
            "habitat": "4"
        }
    }
    
    # Faz a solicitação POST com os dados de teste
    response = client.post(
        "/predict",
        data=json.dumps(test_data),
        content_type="application/json"
    )
    
    # Verifica se a resposta está correta e o status é 200
    assert response.status_code == 200
    data = json.loads(response.data)
    assert "prediction" in data  # Verifica se a resposta contém o campo 'prediction'

# Teste para verificar erro quando os dados estão faltando
def test_predict_invalid(client):
    # Dados de entrada incompletos
    test_data = {
        "data": {
            "cap-shape": "x",
            "cap-surface": "s"
        }
    }

    # Faz a solicitação POST com dados incompletos
    response = client.post(
        "/predict",
        data=json.dumps(test_data),
        content_type="application/json"
    )

    # Verifica se a resposta contém erro
    assert response.status_code == 400
    data = json.loads(response.data)
    assert "error" in data 

