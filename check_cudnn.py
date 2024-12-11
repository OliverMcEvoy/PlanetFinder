import torch
import torch.backends.cudnn as cudnn

print("CUDA available:", torch.cuda.is_available())
print("cuDNN enabled:", cudnn.enabled)
print("cuDNN version:", cudnn.version())

# Check if a simple operation uses cuDNN
x = torch.randn(1, 3, 224, 224).cuda()
model = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1).cuda()
output = model(x)
print("Operation completed successfully.")