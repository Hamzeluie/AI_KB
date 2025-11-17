import requests
import json

url = 'https://api-dev.vexu.ai/api/v1/server/user-profile'
params = {
  'user_id': '5',
}
headers = {
  'Authorization': 'Bearer vexu_6W3Qr84dHNJRDHQIYdC3VLLb4eWsdko4MGGNTD99ttV6jvNN1K0PwZcXNGSc8dsO'
}
response = requests.request('GET', url, params=params, headers=headers)
response.json()
print(json.dumps(response.json(), indent=2))