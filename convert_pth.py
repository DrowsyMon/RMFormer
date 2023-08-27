import torch
ckpt_path1 = 'save_models/Atemp/NN_DH_1.pth'
ckpt_out = 'save_models/model_DH_final.pth'


state1 = torch.load(ckpt_path1)

tmp1 = state1['net']

ori_dict = dict()

state1['net'] = ori_dict


checkpoint = {
                "net": tmp1
            }

torch.save(checkpoint, ckpt_out)

print('ok.........')

