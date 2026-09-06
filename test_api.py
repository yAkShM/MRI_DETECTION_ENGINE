import requests
import torch
import io

print("Testing /health")
try:
    response = requests.get("http://127.0.0.1:8000/health", timeout=5)
    print("Health Status:", response.status_code)
    print("Health JSON:", response.json())
except Exception as e:
    print("Health failed:", e)

print("\nCreating dummy tensor for /predict/pt")
# Dummy tensor to match [4, 128, 128, 128]
dummy = torch.randn(4, 128, 128, 128)
buffer = io.BytesIO()
torch.save(dummy, buffer)
buffer.seek(0)

print("\nTesting /predict/pt")
try:
    files = {'file': ('dummy.pt', buffer, 'application/octet-stream')}
    response = requests.post("http://127.0.0.1:8000/predict/pt", files=files)
    print("Predict Status:", response.status_code)
    try:
        print("Predict JSON:", response.json())
    except:
        print("Predict content:", response.text)
except Exception as e:
    print("Predict failed:", e)

buffer.seek(0)
print("\nTesting /predict/pt/mask")
try:
    files = {'file': ('dummy.pt', buffer, 'application/octet-stream')}
    response = requests.post("http://127.0.0.1:8000/predict/pt/mask", files=files)
    print("Predict Mask Status:", response.status_code)
    print("Predict Mask Content Length:", len(response.content))
except Exception as e:
    print("Predict Mask failed:", e)
