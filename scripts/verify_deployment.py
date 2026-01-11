import requests
import time
import sys

BASE_URL = "http://localhost:8000/api/v1"

def wait_for_server(timeout=30):
    start = time.time()
    while time.time() - start < timeout:
        try:
            response = requests.get(f"{BASE_URL}/health")
            if response.status_code == 200:
                print("Server is healthy!")
                return True
        except requests.RequestException:
            pass
        time.sleep(1)
    print("Server failed to allow connection within timeout.")
    return False

def test_ask():
    print("Testing /ask endpoint...")
    payload = {
        "query": "What is yoga?"
    }
    try:
        response = requests.post(f"{BASE_URL}/ask", json=payload)
        if response.status_code == 200:
            data = response.json()
            print("Response received:")
            print(f"Query: {data.get('query')}")
            print(f"Answer: {data.get('response', {}).get('content')[:100]}...")
            return True
        else:
            print(f"Ask failed with status {response.status_code}: {response.text}")
            return False
    except Exception as e:
        print(f"Ask request failed: {e}")
        return False

if __name__ == "__main__":
    if not wait_for_server():
        sys.exit(1)
    
    if not test_ask():
        sys.exit(1)
    
    print("Verification successful!")
    sys.exit(0)
