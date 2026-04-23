import requests

# url = "http://YOUR_SERVER_IP:8081/predict"
url="http://127.0.0.1:8081/predict"

# Your text prompt (must be lowercase + end with dot)
text_prompt = "peoples ."

files = {
    "image": ("test.jpg", open("/home/rahim/Downloads/frame_1.jpg", "rb"), "image/jpeg")
}

data = {"text": text_prompt,"image_url":"https://assets.bizclikmedia.net/1200/2dbcd30a46fdcee5316274280a35ae9a:df75ad027f562a79efc47c4cec74b4e4/gettyimages-1406550413.jpg.jpg"}

response = requests.post(url, files=None, data=data)

print("Status:", response.status_code)
print("Response:", response.json())
