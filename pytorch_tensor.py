import torch

# x = torch.empty(2,2,2,3)
# print(x)


# y = torch.rand(2,2, dtype=torch.double)
# print(y.dtype)

# z = torch.tensor([2.5,0.1])
# print(z.size())

a = torch.rand(2,2)
b = torch.rand(2,2)
# b.add_(a)

# print(b)

# print(a)
# print(b)

# c = a + b
# c = torch.add(a,b)
# print(c)


c = torch.mul(a,b)
print(c)
