import cv2
import requests

img1 = cv2.imread("/home/junior/Documents/quickium/api_gym/face_recognition/joao.jpg")
img2 = cv2.imread("/home/junior/Documents/quickium/api_gym/face_recognition/joao2.jpg")
img1 = cv2.resize(img1, (224, 224))
img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2BGR)
img2 = cv2.resize(img2, (224, 224))
img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)
img1 = img1.tolist()
img2 = img2.tolist()

r = requests.post("https://apigym.pages.quickium.com/compare_faces",
                  json={"photo1": img1, "photo2": img2})

score = r.json()

print(f"Score: {score}/100")
