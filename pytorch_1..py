#Tensors
import torch
import torchvision

print("Pytorch:",torch.__version__)
print("Torchvision:",torchvision.__version__)
print("CUDA available:",torch.cuda.is_available())

x=torch.tensor([1,2,3,4,5])
print(x)

#Tensors
x=torch.tensor([[1,2],[3,4]])
print(x.shape)
print(x * 2)

#Datasets and Dataloaders
from torch.utils.data import TensorDataset, DataLoader

x=torch.randn(2,3,4)
y=torch.randint(0,2,(100,))
dataset=TensorDataset(x,y)
loader=DataLoader(dataset,batch_size=16)
for features,labels in loader:
    print(features,labels)

#Transforms
from torchvision import transforms
transform=transforms.Compose([transforms.ToTensor()])

#Build a model
from torch import nn

model = nn.Sequential(
    nn.Linear(5, 16),
    nn.ReLU(),
    nn.Linear(16, 2))
print(model)

#Autograd
x=torch.tensor(3.0,require_grad=True)
y=x**2
y.backward()
print(x.grad)

#Optimization
optimizer=torch.optim.SGD(model.parameters(),lr=0.01)
optimizer.zero_grad()

#Save and load a model
torch.save(model.state_dict(), "model.pth")

model.load_state_dict(
    torch.load(
        "model.pth",
        map_location="cpu",
        weights_only=True
    )
)
model.eval()