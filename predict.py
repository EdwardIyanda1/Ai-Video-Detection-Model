import torch
from torchvision import transforms
from PIL import Image
from main import AiDetectorCNN  # Import the model architecture

def predict(image_path, model_path='models/ai_detector.pth'):
    # 1. Load the model
    model = AiDetectorCNN()
    model.load_state_dict(torch.load(model_path))
    model.eval() # Set model to evaluation mode

    # 2. Prepare the image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    img = Image.open(image_path).convert('RGB')
    img = transform(img).unsqueeze(0) # Add batch dimension

    # 3. Predict
    with torch.no_grad():
        output = model(img)
        probability = output.item()

    # 4. Interpret Result
    # 0 = Authentic, 1 = Manipulated
    if probability > 0.5:
        print(f"Prediction: Manipulated (Score: {probability:.4f})")
    else:
        print(f"Prediction: Authentic (Score: {probability:.4f})")

# Example usage:
predict('dataset/train/manipulated/face_0000.jpg')