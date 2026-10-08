import urllib.request
import urllib.parse
import json
import ssl
import sys
import os

ctx = ssl.create_default_context()
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

token_path = r'C:\Users\Lohan\.gemini\antigravity\plugin_data\gdrive\token.json'
with open(token_path, 'r', encoding='utf-8') as f:
    d = json.load(f)

req_data = urllib.parse.urlencode({
    'client_id': d['client_id'],
    'client_secret': d['client_secret'],
    'refresh_token': d['token']['refresh_token'],
    'grant_type': 'refresh_token'
}).encode('utf-8')

req = urllib.request.Request('https://oauth2.googleapis.com/token', data=req_data, method='POST')
with opener.open(req) as resp:
    res = json.loads(resp.read().decode('utf-8'))

token = res['access_token']

def download_file(file_id, out_name):
    print(f"Telechargement de {out_name} (ID: {file_id})...")
    url = f"https://www.googleapis.com/drive/v3/files/{file_id}?alt=media"
    req = urllib.request.Request(url, headers={'Authorization': 'Bearer ' + token})
    with opener.open(req) as resp:
        with open(out_name, 'wb') as f:
            chunk_size = 1024 * 1024 * 4
            downloaded = 0
            while True:
                chunk = resp.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                print(f"  {downloaded / (1024*1024):.1f} MB telecharges...", end='\r')
    print(f"\nTermine: {out_name} ({os.path.getsize(out_name)} octets)")

# Telecharger la nouvelle version de best_mobilenetv5_rocks_w4a8.onnx (ID: 1vRZ6qT9Lutz_2fSqbJVtZM0bFBeqxLHu)
download_file('1vRZ6qT9Lutz_2fSqbJVtZM0bFBeqxLHu', 'best_mobilenetv5_rocks_w4a8.onnx')

# Telecharger aussi best_mobilenetv5_rocks_w4a8.tflite (ID: 1gEjpgUwmtZuS7Y8LZXh3qbhtQ9XrknQu)
download_file('1gEjpgUwmtZuS7Y8LZXh3qbhtQ9XrknQu', 'best_mobilenetv5_rocks_w4a8.tflite')

# Telecharger best_mobilenetv5_rocks_int8.onnx (ID: 1iNrWJC_kZHoKmZxPXDOb0QXom51pUHdi)
download_file('1iNrWJC_kZHoKmZxPXDOb0QXom51pUHdi', 'best_mobilenetv5_rocks_int8.onnx')

# Telecharger best_mobilenetv5_rocks_w4a8_npu.onnx (ID: 1FzDde92TVHXeCZ9IxoyCa3DBlQXUaxU_)
download_file('1FzDde92TVHXeCZ9IxoyCa3DBlQXUaxU_', 'best_mobilenetv5_rocks_w4a8_npu.onnx')
