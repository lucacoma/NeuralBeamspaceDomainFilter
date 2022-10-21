# Lint as python3
"""Contains network modules implementing A Neural Beamspace-Domain
Filter for Real-Time Multi-Channel Speech Enhancement"""
import network_modules
import torch
torch.cuda.empty_cache()
#os.environ['CUDA_VISIBLE_DEVICES'] = '1'
"""
Beam Filter Module
"""
# Get cpu or gpu for training
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class BFM(torch.nn.Module):
    def __init__(self,
                 k1: tuple = (2, 3),  # GLU layers kernel
                 k2: tuple = (1, 3),  # Kernel size U-Net block
                 M:  int = 5,  # Number of beams
                 intra_connect: str = "cat",  # intra connection type, "cat" by default
                 norm_type: str = "BN",
                 c:  int = 64,  # number of channels
                 q:  int = 3,  # Number of S-TCN groups
                 p:  int = 6,  # the number of Squeezed-TCMs within a group, 6 by default
                 kd1:  int = 5,  # kernel size in the Squeezed-TCM (dilation-part), 5 by default
                 cd1:  int = 64,  # channel number in the Squeezed-TCM (dilation-part), 64 by default
                 is_causal: bool = True,  # is_causal: bool = True,
                 embed_dim: int = 64,  # embedded dimension, 64 by default
                 ):

        super(BFM, self).__init__()
        self.k1 = k1
        self.k2 = k2
        self.M = M
        self.intra_connect = intra_connect
        self.c = c
        self.q = q
        self.p = p
        self.is_causal = is_causal

        # Encoder/Decoder
        self.M = self.M + 1 # Since we are concatenating the spectrum at the input (--> 2 more channels)
        self.en = modules_beamspace.U2Net_Encoder(self.M * 2, k1, k2, c, intra_connect, norm_type)
        self.de = modules_beamspace.U2Net_Decoder(embed_dim, c, k1, k2, intra_connect, norm_type)

        # squeezed temporal convolutional networks (S-TCN) bottleneck
        stcn_list = []
        """
        Sono giusti i prossimi due parametri?
        """
        norm_type_stcm = "IN"
        d_feat = 192  # channel number in the Squeezed-TCM(pointwise-part), 256 by default

        for _ in range(q):
            stcn_list.append(network_modules.SqueezedTCNGroup(kd1, cd1, d_feat, p, is_causal, norm_type=norm_type_stcm ))
        self.stcns = torch.nn.ModuleList(stcn_list)

        # Weight estimator module
        self.w_e_m = network_modules.LSTM_BF(embed_dim, M)

    def forward(self, inpt_B: torch.Tensor, inpt_Y: torch.Tensor):
        """
        :param inpt: (B, T, F, D, 2) -> (batchsize, 2 * beamformer directions, seqlen, freqsize)
        :param inpt: (B, T, F, 2) -> (batchsize, 2 * beamformer directions, seqlen, freqsize)

        :return: output:
        """
        #inpt = torch.zeros(B, T, F, D, 2)

        b_size, seq_len, freq_len, D, _ = inpt_B.shape

        # Incorporate STFT + beamspace input??? (paper pag 5)
        # either sum (but probably wrong)
        # inpt = torch.zeros(b_size, seq_len, freq_len, D, 2)
        # inpt = inpt.to(device)
        # for i in range(D):
        #     inpt[:, :, :, i, :] = inpt_B[:, :, :, i, :] + inpt_Y #torch.cat((inpt_B[:, :, :, 0, :], inpt_Y), dim=1)
        # or concatenate (probably right??)
        inpt_Y = inpt_Y.unsqueeze(dim=3)  # Add channel dimension
        inpt = torch.cat((inpt_B, inpt_Y), dim=3) # concatenate beamspace and ref mic spectrum around channel dimension

        x = inpt.transpose(-2, -1).contiguous()
        x = x.view(b_size, seq_len, freq_len, -1).permute(0,3,1,2)

        x, en_list = self.en(x)
        c = x.shape[1]
        x = x.transpose(-2, -1).contiguous().view(b_size, -1, seq_len) # boh
        x_acc = torch.autograd.Variable(torch.zeros(x.size()), requires_grad=True).to(x.device) # boh
        for i in range(len(self.stcns)):
            x = self.stcns[i](x)
            x_acc = x_acc + x # boh
        x = x_acc  # boh

        x = x.view(b_size, c, -1, seq_len).transpose(-2, -1).contiguous()
        x = self.de(x, en_list)
        # Weight estimator
        we = self.w_e_m(x)
        X_bfm = torch.sum(we*inpt_B,dim=3)
        return X_bfm, x_acc, en_list


"""
Residual refinement module
"""

class RRM(torch.nn.Module):
    def __init__(self,
                 k1: tuple = (2, 3),  # GLU layers kernel
                 k2: tuple = (1, 3),  # Kernel size U-Net block
                 M: int = 5,  # Number of beams
                 intra_connect: str = "cat",  # intra connection type, "cat" by default
                 norm_type: str = "BN",
                 c: int = 64,  # number of channels
                 q: int = 3,  # Number of S-TCN groups
                 p: int = 6,  # the number of Squeezed-TCMs within a group, 6 by default
                 kd1: int = 5,  # kernel size in the Squeezed-TCM (dilation-part), 5 by default
                 cd1: int = 64,  # channel number in the Squeezed-TCM (dilation-part), 64 by default
                 c_res: int = 16,  # Channels in res block
                 k_res: tuple = (2, 3),  # kernel size resnet
                 s_res: int = 1,  # stride size resnet
                 is_causal: bool = True,  # is_causal: bool = True,
                 embed_dim: int = 64,  # embedded dimension, 64 by default
                 b: int = 3,  # number of ResConv in ResBlock
    ):
        super(RRM, self).__init__()
        self.k1 = k1
        self.k2 = k2
        self.M = M
        self.intra_connect = intra_connect
        self.c = c
        self.q = q
        self.p = p
        self.is_causal = is_causal
        self.b = b
        self.k_res = k_res
        self.s_res = s_res
        self.c_res = c_res
        self.embed_dim = embed_dim


        # Decoder
        self.de = network_modules.U2Net_Decoder(embed_dim, c, k1, k2, intra_connect, norm_type)

        # Point conv????
        self.point_conv = torch.nn.Conv2d(self.embed_dim+2, self.c_res, (1, 1))

        # Res block
        res_block_list = []
        for _ in range(b):
            res_block_list.append(network_modules.ResConv2D(channels=self.c_res,kernel_size=k_res, stride=s_res))
        self.res_block = torch.nn.ModuleList(res_block_list)

        # Output convolution
        self.out_conv = torch.nn.Conv2d(in_channels=c_res, out_channels=2, kernel_size=(1, 1))

    def forward(self, inpt_B: torch.Tensor, inpt_Y: torch.Tensor, en_list: torch.Tensor):
        """
        :param inpt: (B, 64, T, F) -> (batchsize, embed_dim, seqlen, freqsize) Embedding output
        :param inpt: (B, 1, T, F) -> (batchsize, 1, seqlen, freqsize)  Reference Mic spectrum

        :return: output:
        """
        b_size, _, seq_len = inpt_B.shape
        x = inpt_B.view(b_size, self.c, -1, seq_len).transpose(-2, -1).contiguous()
        x = self.de(x, en_list)
        Y = inpt_Y.permute(0, 3, 1, 2).contiguous()
        x = torch.cat((x, Y), dim=1)
        x = self.point_conv(x)  # Point convolution???
        for i in range(len(self.res_block)):
            x = self.res_block[i](x)
        X = self.out_conv(x)
        X = X.permute((0, 2, 3, 1))
        return X
