import urllib.request
import urllib.parse
import json
import ssl

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
print("OAuth access token ready.")

# 1. Update README_lithotheque_edge_models.md on Google Drive (ID: 15tL4VUmOpQL5Z4FCgwxUEb_9OXHJlzhj)
with open('README_lithotheque_edge_models.md', 'rb') as f:
    content = f.read()

req_readme = urllib.request.Request(
    'https://www.googleapis.com/upload/drive/v3/files/15tL4VUmOpQL5Z4FCgwxUEb_9OXHJlzhj?uploadType=media',
    data=content,
    headers={'Authorization': f'Bearer {token}', 'Content-Type': 'text/markdown'},
    method='PATCH'
)
with opener.open(req_readme) as resp:
    print('Update README on Drive:', resp.status)

# 2. Upload / Update benchmark_snapdragon_100_runs.csv in folder 11D28L2Ax68L6lWQkDJcmxJVwEJkv30KK
with open('benchmark_snapdragon_100_runs.csv', 'rb') as f:
    csv_bytes = f.read()

q_str = "'11D28L2Ax68L6lWQkDJcmxJVwEJkv30KK' in parents and name = 'benchmark_snapdragon_100_runs.csv' and trashed = false"
search_url = 'https://www.googleapis.com/drive/v3/files?q=' + urllib.parse.quote(q_str)
req_search = urllib.request.Request(search_url, headers={'Authorization': f'Bearer {token}'})
with opener.open(req_search) as resp:
    search_res = json.loads(resp.read().decode('utf-8'))
existing = search_res.get('files', [])

if existing:
    f_id = existing[0]['id']
    req_csv = urllib.request.Request(
        f'https://www.googleapis.com/upload/drive/v3/files/{f_id}?uploadType=media',
        data=csv_bytes,
        headers={'Authorization': f'Bearer {token}', 'Content-Type': 'text/csv'},
        method='PATCH'
    )
    with opener.open(req_csv) as resp:
        print('Update existing CSV on Drive:', resp.status, 'ID:', f_id)
else:
    boundary = '-------314159265358979323846'
    meta = json.dumps({
        'name': 'benchmark_snapdragon_100_runs.csv',
        'parents': ['11D28L2Ax68L6lWQkDJcmxJVwEJkv30KK'],
        'mimeType': 'text/csv'
    })
    body = (
        f'--{boundary}\r\n'
        f'Content-Type: application/json; charset=UTF-8\r\n\r\n'
        f'{meta}\r\n'
        f'--{boundary}\r\n'
        f'Content-Type: text/csv\r\n\r\n'
    ).encode('utf-8') + csv_bytes + f'\r\n--{boundary}--\r\n'.encode('utf-8')

    req_csv = urllib.request.Request(
        'https://www.googleapis.com/upload/drive/v3/files?uploadType=multipart',
        data=body,
        headers={
            'Authorization': f'Bearer {token}',
            'Content-Type': f'multipart/related; boundary={boundary}'
        },
        method='POST'
    )
    with opener.open(req_csv) as resp:
        csv_res = json.loads(resp.read().decode('utf-8'))
        print('Created CSV on Drive:', resp.status, 'ID:', csv_res.get('id'))
