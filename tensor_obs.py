import torch

my_torch = torch.arange(10)
# tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

my_torch = my_torch.reshape(2,5)
# print(my_torch)
# tensor([[0, 1, 2, 3, 4],
#         [5, 6, 7, 8, 9]])

my_torch2 = torch.arange(15)
my_torch2 = my_torch2.reshape(3,-1)
# print(my_torch2)
# output
# tensor([[ 0,  1,  2,  3,  4],
#         [ 5,  6,  7,  8,  9],
#         [10, 11, 12, 13, 14]])


my_torch3 =torch.arange(10)
# print(my_torch3)
# tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])


my_torch4 = my_torch3.view(2,5)
# print(my_torch4)

# tensor([[0, 1, 2, 3, 4],
#         [5, 6, 7, 8, 9]])


#reshape and view
my_torch5 = torch.arange(10)
print(my_torch5)
# tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

my_torch5[1] = 101
print(my_torch5)   
# tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]) 


my_torch6 = my_torch5.reshape(2,5)
print(my_torch6)
# tensor([[  0, 101,   2,   3,   4],
#         [  5,   6,   7,   8,   9]])

# slice
my_torch7 = torch.arange(10)
print(my_torch7)

#grab a specific items
my_torch7[7]

#tenso