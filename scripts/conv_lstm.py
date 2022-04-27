import torch
import torch.nn as nn

class ConvLSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, bias, mode="zeros"):
        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias
        self.mode = mode

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)
        
    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_state(self, batch_size, image_size):
        height, width = image_size
        """ Initializing hidden and cell state """
        if(self.mode == "zeros"):
            h = torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device)
            c = torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device)
        elif(self.mode == "random"):
            h = torch.randn(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device)
            c = torch.randn(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device)
        elif(self.mode == "learned"):
            h = self.learned_h.repeat(batch_size, 1, height, width, device=self.conv.weight.device)
            c = self.learned_c.repeat(batch_size, 1, height, width, device=self.conv.weight.device)
        
        return h, c

        
class ConvLSTM(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers, batch_first=False, bias=True, return_all_layers=False):
        super(ConvLSTM, self).__init__()
        
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
       
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers
        
        conv_lstms  = []
        # iterating over no of layers
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            conv_lstms.append(ConvLSTMCell(input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))

        self.conv_lstms = nn.ModuleList(conv_lstms)

    def forward(self, x, hidden_state=None):

        b, in_frames, c, h, w = x.size()

        if hidden_state is not None:
            raise NotImplementedError()
        else:
            hidden_state = self._init_hidden(batch_size=b,
                                             image_size=(h, w))

        # iterating over no of layers
        for i in range(self.num_layers):

            h, c = hidden_state[i]
            each_layer_output = []
            # iterating over sequence length

            for t in range(in_frames * 2):
                if t < in_frames:
                    h, c = self.conv_lstms[i](x[:, t, :, :, :], [h, c])
                    each_layer_output.append(x[:, t, :, :, :])

                else:
                    h_prev = each_layer_output[-1]
                    h, c = self.conv_lstms[i](h_prev, [h, c])
                    each_layer_output.append(h)

            stacked_layer_output = torch.stack(each_layer_output, dim=1)
        return stacked_layer_output

    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.conv_lstms[i].init_state(batch_size, image_size))
        return init_states

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param