from torchvision import transforms
from PIL import Image

def transform_image(path):

    image_transform = transforms.Compose([transforms.Resize((224,224)),
                                          transforms.ToTensor()])
    
    image = Image.open(path)
    image = image_transform(image)
    image = image.unsqueeze(0)

    return image.cpu()
