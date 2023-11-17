import torch
import torch.nn as nn
import torch.nn.functional as F
from .pointnet2_utils import PointNetSetAbstractionMsg, PointNetSetAbstraction, PointNetFeaturePropagation
import sys
sys.path.append('llama/')
'''
customized version for object articulation project

num_class: # of class of an object
output: class prob, part_embedding
'''
class get_model(nn.Module):
    def __init__(self, num_points, num_class, llama_model, ckpt_dir, normal_channel=True, seq_len=10, tok_dim=32000):
        super(get_model, self).__init__()
        if '7b' in ckpt_dir:
            self.emb_dim = 4096
        elif '13b' in ckpt_dir:
            self.emb_dim = 5120
        else:
            raise NotImplementedError
        self.num_points = num_points
        in_channel = 3 if normal_channel else 0
        self.normal_channel = normal_channel
        self.sa1 = PointNetSetAbstractionMsg(512, [0.1, 0.2, 0.4], [16, 32, 128], in_channel,[[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        self.sa2 = PointNetSetAbstractionMsg(128, [0.2, 0.4, 0.8], [32, 64, 128], 320,[[64, 64, 128], [128, 128, 256], [128, 128, 256]])
        self.sa3 = PointNetSetAbstraction(None, None, None, 640 + 3, [256, 512, 1024], True)
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(256, num_class) # fc layer for object class


        self.fp3 = PointNetFeaturePropagation(in_channel=1024+640, mlp=[256,256])
        self.fp2 = PointNetFeaturePropagation(in_channel=256+320, mlp=[256, 128])
        self.fp1 = PointNetFeaturePropagation(in_channel=128+in_channel, mlp=[128, 256, 512])
        self.conv_part1 = nn.Conv1d(512, 1024, 1)
        self.bn_part1 = nn.BatchNorm1d(1024)
        self.drop_part1 = nn.Dropout(0.4)
        self.conv_part2 = nn.Conv1d(1024, 2048, 1)
        self.bn_part2 = nn.BatchNorm1d(2048)
        self.drop_part2 = nn.Dropout(0.5)
        self.conv_part3 = nn.Conv1d(2048, self.emb_dim, 1)
        # get pseudo embedding size: B seq_len 4096 -> output would be softmaxed B seq_len 32000

        self.llama_model = llama_model

    def forward(self, xyz):
        B, _, N = xyz.shape
        if self.normal_channel:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None
        l1_xyz, l1_points = self.sa1(xyz, norm)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        x = l3_points.view(B, 1024)
        if B > 1:
            x = self.drop1(F.relu(self.bn1(self.fc1(x))))
            x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        else:
            x = self.drop1(F.relu(self.fc1(x)))
            x = self.drop2(F.relu(self.fc2(x)))
        x = self.fc3(x)
        x = F.log_softmax(x, -1) # object prob

        '''
        part embedding
        '''
        #Feature Propagation layers
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(xyz, l1_xyz, norm, l1_points)
        x_part = self.drop_part1(F.relu(self.bn_part1(self.conv_part1(l0_points))))
        x_part = self.drop_part2(F.relu(self.bn_part2(self.conv_part2(x_part))))

        embed = self.conv_part3(x_part).transpose(-2, -1) # create embeddings for language model...
        logits = F.softmax(self.llama_model.model.forward_for_embeddings(embed, 0), dim=-1) # get logits from llama
        assert logits.shape[-1] == 32000
        
        # get tokens from logits
        # let's not use sample top p...
        tokens = torch.argmax(logits, dim=-1) 


        return x, logits, tokens


class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target, trans_feat):
        total_loss = F.nll_loss(pred, target)

        return total_loss


'''
original version of pointnet2_cls
'''

'''

class get_model(nn.Module):
    def __init__(self,num_class,normal_channel=True):
        super(get_model, self).__init__()
        in_channel = 3 if normal_channel else 0
        self.normal_channel = normal_channel
        self.sa1 = PointNetSetAbstractionMsg(512, [0.1, 0.2, 0.4], [16, 32, 128], in_channel,[[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        self.sa2 = PointNetSetAbstractionMsg(128, [0.2, 0.4, 0.8], [32, 64, 128], 320,[[64, 64, 128], [128, 128, 256], [128, 128, 256]])
        self.sa3 = PointNetSetAbstraction(None, None, None, 640 + 3, [256, 512, 1024], True)
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(256, num_class)

    def forward(self, xyz):
        B, _, N = xyz.shape
        if self.normal_channel:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None
        l1_xyz, l1_points = self.sa1(xyz, norm)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        x = l3_points.view(B, 1024)
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        x = F.log_softmax(x, -1)


        return x,l3_points


'''