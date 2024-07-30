import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cv2
import torch
from models.model import get_object_detection_model
import torchvision.transforms as T

def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = get_object_detection_model(num_classes=2)

    # Change this to an existing checkpoint file created during training
    model_checkpoint = 'checkpoints/model_0.pth'
    model.load_state_dict(torch.load(model_checkpoint))
    model.to(device)
    model.eval()

    cap = cv2.VideoCapture(0)
    transform = T.Compose([T.ToTensor()])

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            prediction = model(img)

        for element in prediction[0]['boxes']:
            box = element.int().cpu().numpy()
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)

        cv2.imshow('Object Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
