import torch


x = torch.tensor([1,2,3,4,5])
y = torch.tensor([10,20,30,40,50])

result = torch.where(x > 3, x, y)
print(result)