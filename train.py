import torch
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, utils
import torch.optim as optim
from tensorboardX import SummaryWriter
import glob
import tqdm
import os
from dataloader_collect import RandomFlip,RandomCrop,ToTensorLab,SalObjDataset,RescaleT
from model import myNet1, LR_Scheduler
from losses.lossfunc import loss_vit_simple_edgelist
from net_test import test_Net
from eval import qual_eval
from model.helper.helper_blocks import setup_seed 
from myconfig import myParser
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
setup_seed(1234)

args = myParser()


### --------------------loading data---------------------------------
train_img_dir = args.data_dir + args.tra_img_dir
train_label_dir = args.data_dir + args.tra_label_dir
train_edge_dir = args.data_dir + args.tra_edge_dir
test_img_dir = args.data_dir+args.test_img_dir
test_label_dir = args.data_dir+args.test_label_dir
pred_dir = args.data_dir+args.pred_results_dir


image_ext = '.jpg'
label_ext = '.png'
tra_img_name_list = glob.glob(args.data_dir + args.tra_img_dir + '*' + image_ext)
tra_lbl_name_list = []
tra_edg_name_list = []
for img_path in tra_img_name_list:
    img_name = img_path.split("/")[-1]
    aaa = img_name.split(".")
    bbb = aaa[0:-1]
    imidx = bbb[0]
    for i in range(1,len(bbb)):
        imidx = imidx + "." + bbb[i]
    tra_lbl_name_list.append(args.data_dir + args.tra_label_dir + imidx + label_ext)
    tra_edg_name_list.append(args.data_dir + args.tra_edge_dir + imidx + label_ext)

print("---")
print("train images: ", len(tra_img_name_list))
print("train labels: ", len(tra_lbl_name_list))
print("train edge: ", len(tra_edg_name_list))
print("---")
salobj_dataset = SalObjDataset(
    img_name_list=tra_img_name_list,
    lbl_name_list=tra_lbl_name_list,
    edge_name_list=tra_edg_name_list,
    transform=transforms.Compose([
        RescaleT(1600),
        RandomCrop(1536),
        RandomFlip(0.5),
        ToTensorLab(flag=0)]))
salobj_dataloader = DataLoader( salobj_dataset, 
                                batch_size=args.batchsize, \
                                shuffle=True, 
                                num_workers=16,
                                pin_memory=True,
                                drop_last=True)

### --------------------def model---------------------------------

net = myNet1(args)


if torch.cuda.is_available():
    net.cuda()


print("---define optimizer...")
optimizer = optim.SGD(
                # [
                #     {'params':net.lr_branch.parameters(), 'lr':args.init_lr*0.1},
                #     {'params':net.refine.parameters()},
                #     {'params':net.predictor.parameters()},  
                #     {'params':net.hr_branch.parameters()},
                # ],
                [
                    {'params':net.lr_branch.parameters(), 'lr':args.init_lr*0.1},
                    {'params':net.predictor.parameters()},  
                    {'params':net.rrs1.patch_embed1.parameters()},
                    {'params':net.rrs1.in_conv.parameters()},
                    {'params':net.rrs1.stageconv.parameters()},
                    {'params':net.rrs1.deconv.parameters()},
                    {'params':net.rrs1.mid_conv.parameters()},
                    {'params':net.rrs1.sideout.parameters()},
                    {'params':net.rrs1.sideout_e.parameters()},
                    {'params':net.rrs1.refiner.parameters()},
                    {'params':net.rrs2.patch_embed1.parameters()},
                    {'params':net.rrs2.in_conv.parameters()},
                    {'params':net.rrs2.stageconv.parameters()},
                    {'params':net.rrs2.deconv.parameters()},
                    {'params':net.rrs2.mid_conv.parameters()},
                    {'params':net.rrs2.sideout.parameters()},
                    {'params':net.rrs2.sideout_e.parameters()},
                    {'params':net.rrs2.refiner.parameters()},
                ],
                lr=args.init_lr,
                momentum=0.9,
                dampening=0,
                weight_decay=0.0005,
                nesterov=True,
                )

scheduler = LR_Scheduler('cos',args.init_lr,args.train_scheduler_num,\
                        args.itr_epoch,warmup_epochs=args.warmup_epoch_num)





# ------- 5. training process --------
print("---start training...")
# save checkpoint for resume 
start_epoch = 0
if args.resume == True:
    checkpoint = torch.load(args.resume_path)
    net.load_state_dict(checkpoint['net'],strict=False)
    optimizer.load_state_dict(checkpoint['optimizer'])
    start_epoch = checkpoint['epoch'] +1
    del checkpoint
    print('resume training..................')

ite_num = 0
running_loss = 0.0
running_tar_loss = 0.0
running_edge_loss = 0.0
current_lr = 0.0
ite_num4val = 0
trloss = 0
trloss2 = 0
best,best2 = 0,0

if not os.path.exists(args.writer_path):
    os.makedirs(args.writer_path, exist_ok=True)

writer = SummaryWriter(args.writer_path)
GG = start_epoch
b_epoch,b_MAE,b_maxF,b_meanF,b_mba =0,0,0,0,0


for epoch in range(start_epoch, args.epoch_num):
    net.train()
    show_dict = {'epoch':epoch}
    for i, data in enumerate(tqdm.tqdm(salobj_dataloader, ncols=60,postfix=show_dict)):
        ite_num = ite_num + 1
        ite_num4val = ite_num4val + 1

        inputs, labels, edges = data['image'], data['label'],data['edge']
        showimg1 = inputs[:,0:3,:,:]
        showimg2 = labels
        showimg3 = F.interpolate(edges,(384,384), mode='bilinear', align_corners=True)

        inputs = inputs.type(torch.FloatTensor)
        labels = labels.type(torch.FloatTensor)
        edges = edges.type(torch.FloatTensor)

        # wrap them in Variable
        if torch.cuda.is_available():
            inputs_v, labels_v = Variable(inputs.cuda(), requires_grad=False), Variable(labels.cuda(),
                                                                                        requires_grad=False)
            edge_v  = Variable(edges.cuda(), requires_grad=False)
        else:
            inputs_v, labels_v = Variable(inputs, requires_grad=False), Variable(labels, requires_grad=False)
            edge_v  = Variable(edges, requires_grad=False)

        # y zero the parameter gradients
        optimizer.zero_grad()


        scheduler(optimizer, i, epoch)
        # forward + backward + optimize
        #########################################
        #########################################
        dout,dout_c, dmid_list,de_list, attn_out = net(inputs_v)
        lossp, loss_all, bceloss, ssimloss, iouloss,loss_e = \
            loss_vit_simple_edgelist(dout,dout_c, dmid_list,de_list,attn_out,\
            edge_v, labels_v)

        if i == 0 : 
            torch.cuda.empty_cache()

        (loss_all).backward()

        optimizer.step()

        current_lr = optimizer.state_dict()['param_groups'][0]['lr']

        writer.add_scalar('Train param/lr',current_lr,((i+1)+epoch*args.itr_epoch))

        current_lr_other = optimizer.state_dict()['param_groups'][2]['lr']
        writer.add_scalar('Train param/lr_other',current_lr_other,((i+1)+epoch*args.itr_epoch))
        current_lr_hr = optimizer.state_dict()['param_groups'][1]['lr']
        writer.add_scalar('Train param/lr_hr',current_lr_hr,((i+1)+epoch*args.itr_epoch))
        #########################################
        #########################################
        # print statistics
        running_loss += loss_all.item()
        running_tar_loss += lossp.item()
        running_edge_loss += loss_e.item()

# ------- tensorboard --------
        # tensorboard
        trloss = (running_loss / ite_num4val)
        trloss2 = (running_tar_loss / ite_num4val)
        trlosse = (running_edge_loss / ite_num4val)

        writer.add_scalar('loss/total',trloss, ((i+1)+epoch*args.itr_epoch) )
        writer.add_scalar('loss/P_loss',trloss2, ((i+1)+epoch*args.itr_epoch))
        writer.add_scalar('loss/E_loss',trlosse, ((i+1)+epoch*args.itr_epoch))
        
        writer.add_scalar('3_type/BEC',bceloss.item(), ((i+1)+epoch*args.itr_epoch))
        writer.add_scalar('3_type/IOUloss',iouloss.item(), ((i+1)+epoch*args.itr_epoch))

        # if ((ite_num % (args.itr_epoch*args.save_interval))==0)|(ite_num == args.itr_epoch):
        if (ite_num == 10):  
            checkpoint = {
                "net": net.state_dict(),
                'optimizer':optimizer.state_dict(),
                "epoch": epoch
            }
            save_dir = args.model_dir + "epoch_%d.pth" % (epoch)
            torch.save(checkpoint,save_dir)
        
            # eval at 1st epoch and >no_eval_interval
            # if args.eval == True & ((epoch == start_epoch) | (epoch > args.no_eval_interval)) :
            if args.eval == True :
                test_Net(save_dir,test_img_dir,pred_dir,args)

                MAE, maxF, meanF, mba = qual_eval(test_label_dir,pred_dir)
                # MAE, maxF, meanF = 0,0,0
                writer.add_scalar('Eval/MAE',MAE, epoch)
                writer.add_scalar('Eval/maxF',maxF, epoch)
                writer.add_scalar('Eval/meanF',meanF, epoch)
                writer.add_scalar('Eval/mba',meanF, mba)

                if epoch == start_epoch :
                    best = maxF + meanF - MAE
                    filename = (args.model_dir + "bestone.pth")
                    torch.save(checkpoint, filename)
                    b_epoch = epoch
                    b_MAE = MAE
                    b_maxF = maxF
                    b_meanF = meanF
                    b_mba = mba
                elif (epoch > start_epoch) & (epoch > args.no_eval_interval):  # save model with best performance
                    if maxF + meanF - MAE > best:
                        best = maxF + meanF - MAE
                        filename = (args.model_dir + "bestone.pth")
                        torch.save(checkpoint, filename)
                        b_epoch = epoch
                        b_MAE = MAE
                        b_maxF = maxF
                        b_meanF = meanF
                        b_mba = mba

                print('b_epoch_%d_MAE_%3f_maxF_%3f_meanF_%mba%3f'%(b_epoch,b_MAE,b_maxF,b_mba))
           
            running_loss = 0.0
            running_tar_loss = 0.0
            running_edge_loss = 0.0
            net.train()  # resume train
            ite_num4val = 0
            del checkpoint

# ------- pic shoot in tensorboard --------
        if (ite_num % int(args.itr_epoch/4) -1)==0:
            showimg1[:,0,:,:] = showimg1[:,0,:,:]*0.229+0.485
            showimg1[:,1,:,:] = showimg1[:,1,:,:]*0.224+0.456
            showimg1[:,2,:,:] = showimg1[:,2,:,:]*0.225+0.406

            writer.add_image('Input/input_image',utils.make_grid(showimg1,nrow=2),global_step = GG)
            writer.add_image('Input/gt',utils.make_grid(showimg2,nrow=2),global_step = GG)
            writer.add_image('Input/error_map_gt',utils.make_grid(showimg3,nrow=2),global_step = GG)
            writer.add_image('Output/d0_predict', utils.make_grid(torch.sigmoid(dout),nrow=2),global_step = GG)
            writer.add_image('Output/d2', utils.make_grid(torch.sigmoid(dout_c),nrow=2), global_step = GG)

            GG += 1
        del dout, loss_all,\
            bceloss, ssimloss, iouloss
