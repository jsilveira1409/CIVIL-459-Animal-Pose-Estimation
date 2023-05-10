import torch
import torchvision
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms


model = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=True)
model.eval()

# get image from the data-animalpose/images/train/ directory
image = Image.open('../output/cropped_3_0.jpg')

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])



# Convert cropped_image to a PyTorch tensor and normalize
input_tensor = transform(image)
input_tensor = input_tensor.unsqueeze(0)

output = model(input_tensor)['out']

# Convert output to a binary mask (assuming the animal class is 1)
mask = torch.argmax(output, dim=1) == 1
foreground = input_tensor * mask.numpy()

# Save the cropped image
plt.imsave('test.jpg', foreground)
# save the output
plt.imsave('output.jpg', output[0, 1, :, :].detach().numpy())