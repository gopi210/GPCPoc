import io
from google.cloud import vision
vision_client = vision.Client("MyProject47443-3090c70d9f52.json")
file_name = 'C:\\Users\\398714\\Desktop\\Demo\\NapaParts\\BK-7651085\\BottomAngle.jpg'
with io.open(file_name,'rb') as image_file:
    content = image_file.read()
    image = vision_client.image(content=content)
labels = image.detect_labels()
for label in labels:
    print(label.description)
    print(str(label.score))
