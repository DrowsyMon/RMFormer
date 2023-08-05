import torch
import torch.nn as nn

# https://blog.csdn.net/qq_19841133/article/details/126245602
# 1d absolute con-sin postion enmbedding
def create_1d_absolute_sin_cos_embedding(pos_len, dim):
    assert dim % 2 == 0, "wrong dimension!"
    position_emb = torch.zeros(pos_len, dim, dtype=torch.float)
    # i矩阵
    i_matrix = torch.arange(dim//2, dtype=torch.float)
    i_matrix /= dim / 2
    i_matrix = torch.pow(10000, i_matrix)
    i_matrix = 1 / i_matrix
    i_matrix = i_matrix.to(torch.long)
    # pos矩阵
    pos_vec = torch.arange(pos_len).to(torch.long)
    # 矩阵相乘，pos变成列向量，i_matrix变成行向量
    out = pos_vec[:, None] @ i_matrix[None, :]
    # 奇/偶数列
    emb_cos = torch.cos(out)
    emb_sin = torch.sin(out)
    # 赋值
    position_emb[:, 0::2] = emb_sin
    position_emb[:, 1::2] = emb_cos
    return position_emb

def create_1d_learnable_embedding(pos_len, dim):
    pos_emb = nn.Embedding(pos_len, dim)
    # 初始化成全0
    nn.init.constant_(pos_emb.weight, 0)
    return pos_emb


def create_2d_relative_bias_trainable_embedding(n_head, h, w, dim):
    pos_emb = nn.Embedding((2*w-1)*(2*h-1), n_head)
    nn.init.constant_(pos_emb.weight, 0.)

    def get_2d_relative_position_index(height, width):
        # m1/m2.shape = [h, w]，m1所有行值相同，m2所有列数相同
        m1, m2 = torch.meshgrid(torch.arange(height), torch.arange(width))
        # [2, h, 2]
        coords = torch.stack([m1, m2], dim=0)
        # 将h和w维度拉直,[2, h*w]
        coords_flatten = torch.flatten(coords, start_dim=1)
        # 变成3维列向量[2, h*w, 1] 减去 3维行向量，得到坐标差值
        # relative_coords_bias.shape = [2, h*w, h*w],反应两个方向任何两个点之间的差值
        relative_coords_bias = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        # 方向差距变为正数，bias ∈ [0, 2(h - 1)]/[0, 2(w - 1)]
        relative_coords_bias[0, :, :] += height - 1
        relative_coords_bias[1, :, :] += width - 1
        # 将两个方向转换一个方向坐标， [i, j] -> [i*cols + j]
        relative_coords_bias[0, :, :] *= relative_coords_bias[1, :, :].max()+1
        return relative_coords_bias.sum(0)  # [h*w, h*w]
    relative_pos_bias = get_2d_relative_position_index(h, w)
    # 基于相对bias去Embedding中去查
    bias_emb = pos_emb(relative_pos_bias.flatten()).reshape([h*w, h*w, n_head])
    # 转置一下n_head，放到第0维
    bias_emb = bias_emb.permute(2, 0, 1).unsqueeze(0)  # [1, n_head, h*w, h*w]
    return bias_emb

def create_2d_absolute_sin_cos_embedding(h, w, dim):
    # 奇数列和偶数列sin_cos，还有h和w方向，因此维度是4的倍数
    assert dim % 4 == 0, "wrong dimension"

    pos_emb = torch.zeros([h*w, dim])
    m1, m2 = torch.meshgrid(torch.arange(h), torch.arange(w))
    # [2, h, 2]
    coords = torch.stack([m1, m2], dim=0)
    # 高度方向的emb
    h_emb = create_1d_absolute_sin_cos_embedding(torch.flatten(coords[0]).numel(), dim // 2)
    # 宽度方向的emb
    w_emb = create_1d_absolute_sin_cos_embedding(torch.flatten(coords[1]).numel(), dim // 2)
    # 拼接起来
    pos_emb[:, :dim//2] = h_emb
    pos_emb[:, dim//2:] = w_emb
    return pos_emb

def create_1d_learnable_embedding(pos_len, dim):
    pos_emb = nn.Embedding(pos_len, dim)
    # 初始化成全0
    nn.init.constant_(pos_emb.weight, 0)
    return pos_emb




if __name__ == '__main__':
    # print(create_1d_absolute_sin_cos_embedding(4, 4))

    # emb = create_2d_relative_bias_trainable_embedding(1, 1024, 1024, 32)
    # print(emb.shape)

    emb = create_2d_absolute_sin_cos_embedding(1024, 1024, 32)
    print(emb.shape)