import cv2
import torch
from PIL import Image
from facenet_pytorch import MTCNN
import torchvision.transforms as transforms

# 예측 함수
def predict_ethnicity(image):
    image = transform(image)
    image = image.unsqueeze(0)

    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)

    return predicted.item()

# 입력값 받기
model_path = input("Path to model: ")
img_path = input("Path to image: ")

model = torch.load(model_path)
model.eval()

# 입력을 위해 전처리
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# MTCNN
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(keep_all=True, device=device)

# 이미지 로딩
img_array = cv2.imread(img_path)
img = Image.fromarray(cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB))

# 얼굴 검출
boxes, probs = mtcnn.detect(img)

if boxes is not None:
    asians = 0
    for box in boxes:
        # 얼굴 크롭
        face_region = img.crop((box[0], box[1], box[2], box[3]))
        if predict_ethnicity(face_region) == 0:
            asians += 1
            flag = True
        else:
            flag = False

        # 바운딩 박스 그리기
        if flag: color = (0, 0, 255)
        else: color = (0, 255, 0)
        cv2.rectangle(img_array, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)

    cv2.imshow(img_array)
    print(f"Asians: {asians} ({asians / len(boxes) * 100:.2f}%)")
