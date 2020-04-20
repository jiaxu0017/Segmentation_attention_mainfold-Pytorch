import torch
from torch.nn import Module,Conv2d,Parameter,Softmax

torch_ver = torch.__version__[:3]

__all__ = ['PAM_Module', 'CAM_Module','Attention_Module']

class PAM_Module(Module):
    def __init__(self,in_dim):
        super(PAM_Module, self).__init__()

        self.chanel_in = in_dim

        self.query_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = Parameter(torch.zeros(1))
        self.softmax = Softmax(dim=-1)

    def forward(self,x):
        '''
        PAM_Module function forward
        Args:
            x: input feature maps(B * C * H * W)

        Returns:
            out: attention value + input feature(B * C/8 * H * W)
            attention: B * (H * W) * ( H* W)

        '''
        # get x shape
        m_batchsize, C, height, width = x.size()

        # preprocess three vectors
        # proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0,2,1)
        # proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)
        # proj_value = self.value_conv(x).view(m_batchsize,-1, width*height)

        proj_query = kronecker(self.query_conv(x)).permute(0,2,1)
        proj_key = kronecker(self.key_conv(x))
        proj_value = kronecker(self.value_conv(x))

        # get feature map of attetion
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)

        # get out feature
        out = torch.bmm(proj_value, attention.permute(0,2,1))
        # out = out.view(m_batchsize, C, height, width)

        out = Inverse_kronecker(out,m_batchsize, C,height,width)
        # get out add x(origin feature tensor)
        try:
            out = self.gamma*out+x
        except RuntimeError:
            print('self.gamma',self.gamma)
            print('out.size:',out.shape)
            print('x.size:',x.shape)
        return out


def kronecker(input):
    # print(input.shape)
    # H = torch.sum(input, dim=3)
    # W = torch.sum(input, dim=2)
    H = torch.mean(input, dim=3)
    W = torch.mean(input, dim=2)
    out = torch.cat((H, W), dim = 2)

    # W_,H_ = out.chunk(2,2)
    # print('H_ & W_',H_, W_)

    # return H, W, out
    return out

def Inverse_kronecker(input,m_batchsize, C,height,width):
    # W_,H_ = input.chunk(2,2)

    # H_ = torch.zeros((m_batchsize,C,height))
    # W_ = torch.zeros((m_batchsize,C,width))

    H_ = input[:,:,0:height]
    W_ = input[:,:,height:width+height]

    # print(H_.shape)
    # print(W_.shape)

    # print(W_.shape, H_.shape)
    # print(W_[0,1,:], H_[0,1,:])
    w = H_.shape[2]
    h = W_.shape[2]
    W_min = torch.min(W_,dim=2)
    H_min = torch.min(H_,dim=2)

    W_ = W_.reshape(W_.shape[0],W_.shape[1],1, W_.shape[2])
    H_ = H_.reshape(H_.shape[0],H_.shape[1],H_.shape[2],1)

    W_min = W_min.values
    W_min = W_min.reshape(W_min.shape[0],W_min.shape[1],1,1)
    H_min = H_min.values
    H_min = H_min.reshape(H_min.shape[0],H_min.shape[1],1,1)
    # print(W_.shape)
    # print('H_ & W_',H_, W_)

    W_mar = W_.expand(W_.shape[0],W_.shape[1],w, W_.shape[3])
    H_mar = H_.expand(H_.shape[0],H_.shape[1],H_.shape[2],h)

    W_min = W_min.expand(W_min.shape[0],W_min.shape[1],w,h)
    H_min = H_min.expand(H_min.shape[0],H_min.shape[1],w,h)
    # print(W_mar.shape)
    # print(H_mar.shape)
    # print(W_mar[0, 1, :])
    # print(H_mar[0, 1, :])

    x = W_min.values
    # print('123',W_min)
    # print('123',x)

    return W_mar+H_mar-W_min-H_min

class CAM_Module(Module):
    """ Channel attention module"""

    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.gamma = Parameter(torch.zeros(1))
        self.softmax = Softmax(dim=-1)

    def forward(self,x):
        '''
        Calcuate attetion between channels
        Args:
            x: input feature maps (B * C * H * W)

        Returns:
            out: attention value + input feature (B * C * H * W)
            attention: B * C * C

        '''

        m_batchsize, C, height, wight = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize,C, -1).permute(0,2,1)
        proj_value = x.view(m_batchsize,C, -1)

        energy = torch.bmm(proj_query,proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = self.softmax(energy_new)

        out = torch.bmm(attention,proj_value)
        out = out.view(m_batchsize,C, height, wight)
        mean = torch.mean(out)
        out = out/mean

        out = self.gamma*out + x
        return out

class Attention_Module(Module):
    """ Position attention module"""
    #Ref from SAGAN
    def __init__(self, in_dim):
        super(Attention_Module, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = Parameter(torch.zeros(1))

        self.softmax = Softmax(dim=-1)
    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out




if __name__ == '__main__':
    input = torch.ones([1,16,9,5])
    for i  in range(9):
        for j in range(5):
            input[:,:,i,j] = i * 5 + j
    # print(input.size())
    print(input[0,1,])

    # test kronecker
    output_H, output_W , output= kronecker(input)
    # print('H & W:',output_H.size(), output_W.size())
    # print('out',output.size())
    print('H & W:',output_H.shape, output_W.shape)
    # print(output)

    # test Inverse_kronecker
    # out = kronecker(input)
    # print(H[0,1,],W[0,1,])
    out = Inverse_kronecker(output, input.shape[0],input.shape[1],input.shape[2],input.shape[3])
    print(out.shape)
    # # print(out[0,1,])
    # out = out/5


    # test PAM_Module
    # model = PAM_Module(16)
    # out = model(input)
    # print(out.shape)

