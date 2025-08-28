import torch

attn_weight = torch.tensor([
    [[1.0, 2.0, 3.0, 4.0],
     [2.0, 2.0, 2.0, 2.0],
     [4.0, 3.0, 2.0, 1.0]],
    
    [[1.0, 0.0, 0.0, 0.0],
     [0.0, 1.0, 0.0, 0.0],
     [0.0, 0.0, 1.0, 0.0]]
])

result0 = torch.softmax(attn_weight, dim=-1)
print(result0)

for i in range(attn_weight.shape[0]):      # 遍历样本数
    for j in range(attn_weight.shape[1]):  # 遍历注意力头数
        slice_ij = attn_weight[i, j]       # 取出 [*, *, 4] 的向量
        res_ij = torch.softmax(slice_ij, 0)
        print(f"Sample {i}, Head {j}: {slice_ij} | softmax : {res_ij}")

