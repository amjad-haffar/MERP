import os
import sys
# sys.path.append(os.path.abspath('../..'))
sys.path.append(os.path.abspath(''))
print(sys.path)
import util

import pandas as pd

import torch
from torch import nn
from torch.nn import functional as F

class lstm_double(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(lstm_double, self).__init__()

        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        # self.attn = MutliHeadAttention1D(in_features, out_features, kernel_size)

        self.lstm2 = nn.LSTM(hidden_dim*2, hidden_dim, batch_first=True, bidirectional=True)

        self.fc2 = nn.Linear(hidden_dim*2, 1)

        # self.dropout2 = nn.Dropout(0.2)
        self.actout = nn.Tanh()
        
        # smoothing gaussian kernel
        # kernel = torch.FloatTensor([[[0.006, 0.061, 0.242, 0.383, 0.242, 0.061, 0.006]]]) # sigma = 1
        # kernel = torch.FloatTensor([[[0.0099, 0.0301, 0.0587, 0.0733, 0.0587, 0.0301, 0.0099]]]) # sigma = 1.5, kernel size = 7

        # self.register_buffer('kernel', kernel) # to device workaround
        

    def forward(self, x):
        # print('forward')

        lstm_out, lstm_c = self.lstm(x)
        # print(lstm_out.narrow(0,1,0))
        # print(f'lstm weight max = {self.lstm.weight_ih_l0.max()}\tweight min = {self.lstm.weight_ih_l0.min()}\tmean = {self.lstm.weight_ih_l0.mean()}')

        lstm_out, _ = self.lstm2(lstm_out, lstm_c)
        
        # out = self.dropout2(lstm_out)

        out = self.fc2(lstm_out)
        
        out = self.actout(out)
        # print('3 ', out.shape)
        out = out.flatten(1) # remove last dimension [8,10,1]
        out = out.unsqueeze(1) # create channel dim for conv [8,1,10]

        # Apply smoothing
        # out = F.conv1d(out, self.kernel, padding=3)

        # print(out)
        # print('out shape: ', out.shape)
        return out

    def load_my_state_dict(self, state_dict):
 
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                 continue
            if isinstance(param, nn.Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            own_state[name].copy_(param)

class lstm_single(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(lstm_single, self).__init__()

        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        # self.attn = MutliHeadAttention1D(in_features, out_features, kernel_size)

        self.dropout1 = nn.Dropout(0.5)
        self.act1 = nn.LeakyReLU(0.1)

        self.fc3 = nn.Linear(hidden_dim, 1)
        

    def forward(self, x):
        # print('forward')

        lstm_out, _ = self.lstm(x)
        # print(lstm_out.narrow(0,1,0))
        # print(f'lstm weight max = {self.lstm.weight_ih_l0.max()}\tweight min = {self.lstm.weight_ih_l0.min()}\tmean = {self.lstm.weight_ih_l0.mean()}')
        
        lstm_out = self.dropout1(lstm_out)
        lstm_out = self.act1(lstm_out)
        # print(lstm_out)

        out = self.fc3(lstm_out)
        # print(out)
        # print('out shape: ', out.shape)
        return out

class lstm_single_2fc(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(lstm_single_2fc, self).__init__()

        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        # self.attn = MutliHeadAttention1D(in_features, out_features, kernel_size)

        self.dropout1 = nn.Dropout(0.5)
        self.act1 = nn.LeakyReLU(0.1)

        self.fc1 = nn.Linear(hidden_dim, hidden_dim//2)

        self.dropout2 = nn.Dropout(0.5)
        self.act2 = nn.LeakyReLU(0.1)

        self.fc2 = nn.Linear(hidden_dim//2, 1)
        

    def forward(self, x):
        # print('forward')

        lstm_out, _ = self.lstm(x)
        # print(lstm_out.narrow(0,1,0))
        # print(f'lstm weight max = {self.lstm.weight_ih_l0.max()}\tweight min = {self.lstm.weight_ih_l0.min()}\tmean = {self.lstm.weight_ih_l0.mean()}')
        
        lstm_out = self.dropout1(lstm_out)
        lstm_out = self.act1(lstm_out)
        # print(lstm_out)

        out = self.fc1(lstm_out)

        out = self.dropout2(out)
        out = self.act2(out)

        out = self.fc2(out)

        # print(out)
        # print('out shape: ', out.shape)
        return out

class Three_FC_layer(torch.nn.Module):
    def __init__(self, input_dim = 724, hidden_dim=128):
    # def __init__(self, input_dim = 1582, reduced_dim=128, fc_dim = 64):
        super(Three_FC_layer, self).__init__()
        # self.reduce_dim = nn.Linear(input_dim, reduced_dim, bias=False)

        self.fc1 = nn.Linear(input_dim, hidden_dim)#, bias=False)
        self.dropout1 = nn.Dropout(0.5)
        self.lr1 = nn.LeakyReLU(0.1)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim//2)#, bias=False)
        self.dropout2 = nn.Dropout(0.5)
        self.lr2 = nn.LeakyReLU(0.1)
        self.fc_out = nn.Linear(hidden_dim//2, out_features=1)#, bias=False)  # output
        self.actout = nn.Tanh()

        # kernel = torch.FloatTensor([[[0.006, 0.061, 0.242, 0.383, 0.242, 0.061, 0.006]]]) # sigma = 1
        kernel = torch.FloatTensor([[[0.0099, 0.0301, 0.0587, 0.0733, 0.0587, 0.0301, 0.0099]]]) # sigma = 1.5, kernel size = 7

        self.register_buffer('kernel', kernel)

    def forward(self, x):
        # out = self.class_dim(self.lr2(self.dropout2(self.fc2(self.lr1(self.dropout1(self.fc1(self.reduced_rgb(x))))))))
        # out = self.reduce_dim(x)
        out = self.fc1(x)
        # print(out)
        # print(self.fc1.weight.grad)
        # print(self.fc1.weight)
        # print(f'weight max = {self.fc1.weight.max()}\tweight min = {self.fc1.weight.min()}\tmean = {self.fc1.weight.mean()}')
        
        # print('1 ', x.shape)
        out = self.dropout1(out)
        out = self.lr1(out)
        out = self.fc2(out)
        # print('2 ', out.shape)
        out = self.dropout2(out)
        out = self.lr2(out)
        out = self.fc_out(out)
        out = self.actout(out)
        # print('3 ', out.shape)
        out = out.flatten(1)
        out = out.unsqueeze(1)
        
        # Apply smoothing
        out = F.conv1d(out, self.kernel, padding=3)
        return out
    
    def load_my_state_dict(self, state_dict):
 
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                 continue
            if isinstance(param, nn.Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            own_state[name].copy_(param)

class lstm_double_audio_matched(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            batch_first=True,
            bidirectional=True
        )

        self.lstm2 = nn.LSTM(
            hidden_dim * 2,
            hidden_dim,
            batch_first=True,
            bidirectional=True
        )

        # Match the representation size used in the personalized model
        self.audio_fc = nn.Linear(
            hidden_dim * 2,
            128
        )

        self.audio_act = nn.LeakyReLU(0.1)

        # Same regularisation strength
        self.dropout = nn.Dropout(0.4)

        self.fc_out = nn.Linear(
            128,
            1
        )

        self.actout = nn.Tanh()


    def forward(self, x):

        lstm_out, lstm_state = self.lstm(x)

        lstm_out, _ = self.lstm2(
            lstm_out,
            lstm_state
        )

        out = self.audio_fc(lstm_out)
        out = self.audio_act(out)

        out = self.dropout(out)

        out = self.fc_out(out)
        out = self.actout(out)

        out = out.flatten(1)
        out = out.unsqueeze(1)

        return out


    def load_my_state_dict(self, state_dict):

        own_state = self.state_dict()

        for name, param in state_dict.items():

            if name not in own_state:
                continue

            if isinstance(param, nn.Parameter):
                param = param.data

            own_state[name].copy_(param)
class lstm_double_late_simple(torch.nn.Module):
    def __init__(self, audio_dim, profile_dim, hidden_dim):
        super().__init__()

        self.lstm = nn.LSTM(
            audio_dim,
            hidden_dim,
            batch_first=True,
            bidirectional=True
        )

        self.lstm2 = nn.LSTM(
            hidden_dim * 2,
            hidden_dim,
            batch_first=True,
            bidirectional=True
        )

        self.fc2 = nn.Linear(
            hidden_dim * 2 + profile_dim,
            1
        )

        self.actout = nn.Tanh()

    def forward(self, audio, profile):

        lstm_out, lstm_state = self.lstm(audio)

        lstm_out, _ = self.lstm2(
            lstm_out,
            lstm_state
        )

        # profile: [batch, profile_dim]
        # repeat it across the dynamic timesteps
        profile = profile.unsqueeze(1).expand(
            -1,
            lstm_out.size(1),
            -1
        )

        combined = torch.cat(
            [lstm_out, profile],
            dim=2
        )

        out = self.fc2(combined)
        out = self.actout(out)

        out = out.flatten(1)
        out = out.unsqueeze(1)

        return out
    def load_my_state_dict(self, state_dict):

        own_state = self.state_dict()

        for name, param in state_dict.items():
            if name not in own_state:
                continue

            if isinstance(param, nn.Parameter):
                param = param.data

            own_state[name].copy_(param)

class lstm_double_late_profile_branch(torch.nn.Module):
    def __init__(self, audio_dim, profile_dim, hidden_dim):
        super().__init__()

        # Audio branch: unchanged
        self.lstm = nn.LSTM(
            audio_dim,
            hidden_dim,
            batch_first=True,
            bidirectional=True
        )

        self.lstm2 = nn.LSTM(
            hidden_dim * 2,
            hidden_dim,
            batch_first=True,
            bidirectional=True
        )

        # Small profile branch
        self.profile_fc = nn.Linear(
            profile_dim,
            32
        )
        self.profile_act = nn.LeakyReLU(0.1)

        # Smaller fusion layer
        self.fusion_fc = nn.Linear(
            hidden_dim * 2 + 32,
            128
        )
        self.fusion_act = nn.LeakyReLU(0.1)

        self.fc_out = nn.Linear(
            128,
            1
        )
        self.profile_dropout = nn.Dropout(0.4)
        self.fusion_dropout = nn.Dropout(0.4)

        self.actout = nn.Tanh()

    def forward(self, audio, profile):

        lstm_out, lstm_state = self.lstm(audio)
        lstm_out, _ = self.lstm2(lstm_out, lstm_state)

        profile_out = self.profile_fc(profile)
        profile_out = self.profile_act(profile_out)
        profile_out = self.profile_dropout(profile_out)

        profile_out = profile_out.unsqueeze(1).expand(
            -1,
            lstm_out.size(1),
            -1
        )

        combined = torch.cat(
            [lstm_out, profile_out],
            dim=2
        )

        out = self.fusion_fc(combined)
        out = self.fusion_act(out)
        out = self.fusion_dropout(out)

        out = self.fc_out(out)
        out = self.actout(out)

        out = out.flatten(1)
        out = out.unsqueeze(1)

        return out


    def load_my_state_dict(self, state_dict):

        own_state = self.state_dict()

        for name, param in state_dict.items():

            if name not in own_state:
                continue

            if isinstance(param, nn.Parameter):
                param = param.data

            own_state[name].copy_(param)
    
class lstm_double_late_middle(torch.nn.Module):
    def __init__(self, audio_dim, profile_dim, hidden_dim):
        super().__init__()

        # =====================================================
        # AUDIO BRANCH
        # =====================================================

        self.lstm = nn.LSTM(
            audio_dim,
            hidden_dim,
            batch_first=True,
            bidirectional=True
        )

        self.lstm2 = nn.LSTM(
            hidden_dim * 2,
            hidden_dim,
            batch_first=True,
            bidirectional=True
        )


        # =====================================================
        # PROFILE BRANCH
        # 9 profile features -> compact learned representation
        # =====================================================

        self.profile_fc = nn.Linear(
            profile_dim,
            32
        )

        self.profile_act = nn.LeakyReLU(0.1)


        # =====================================================
        # DIRECT FUSION -> OUTPUT
        #
        # NO extra fusion hidden layer here.
        # =====================================================

        self.fc_out = nn.Linear(
            hidden_dim * 2 + 32,
            1
        )

        self.actout = nn.Tanh()


    def forward(self, audio, profile):

        # =====================================================
        # AUDIO
        # =====================================================

        lstm_out, lstm_state = self.lstm(audio)

        lstm_out, _ = self.lstm2(
            lstm_out,
            lstm_state
        )


        # =====================================================
        # PROFILE
        # profile: [batch, 9]
        # -> [batch, 32]
        # =====================================================

        profile_out = self.profile_fc(profile)
        profile_out = self.profile_act(profile_out)


        # Repeat static listener representation
        # across the 30 emotion timesteps
        #
        # [batch, 32]
        # ->
        # [batch, 30, 32]

        profile_out = profile_out.unsqueeze(1).expand(
            -1,
            lstm_out.size(1),
            -1
        )


        # =====================================================
        # FUSION
        # =====================================================

        combined = torch.cat(
            [lstm_out, profile_out],
            dim=2
        )


        # DIRECTLY predict from combined representation
        out = self.fc_out(combined)

        out = self.actout(out)


        # Match existing MERP output shape
        # [batch, 30, 1]
        # ->
        # [batch, 1, 30]

        out = out.flatten(1)
        out = out.unsqueeze(1)

        return out


    def load_my_state_dict(self, state_dict):

        own_state = self.state_dict()

        for name, param in state_dict.items():

            if name not in own_state:
                continue

            if isinstance(param, nn.Parameter):
                param = param.data

            own_state[name].copy_(param)

class lstm_double_adaptive_gating(torch.nn.Module):
    def __init__(self, audio_dim, profile_dim, hidden_dim):
        super().__init__()

        # =====================================================
        # AUDIO BRANCH
        # =====================================================

        self.lstm = nn.LSTM(
            audio_dim,
            hidden_dim,
            batch_first=True,
            bidirectional=True
        )

        self.lstm2 = nn.LSTM(
            hidden_dim * 2,
            hidden_dim,
            batch_first=True,
            bidirectional=True
        )

        # Project audio representation into common fusion space
        self.audio_fc = nn.Linear(
            hidden_dim * 2,
            128
        )

        self.audio_act = nn.LeakyReLU(0.1)


        # =====================================================
        # PROFILE BRANCH
        # =====================================================

        self.profile_fc1 = nn.Linear(
            profile_dim,
            32
        )

        self.profile_act1 = nn.LeakyReLU(0.1)

        # Bring profile into SAME dimension as audio
        self.profile_fc2 = nn.Linear(
            32,
            128
        )

        self.profile_act2 = nn.LeakyReLU(0.1)


        # =====================================================
        # ADAPTIVE GATE
        #
        # Input:
        # audio 128 + profile 128 = 256
        #
        # Output:
        # 2 weights:
        # [audio weight, profile weight]
        # =====================================================

        self.gate_fc = nn.Linear(
            128 * 2,
            2
        )


        # =====================================================
        # REGULARISATION + OUTPUT
        # =====================================================

        self.dropout = nn.Dropout(0.4)

        self.fc_out = nn.Linear(
            128,
            1
        )

        self.actout = nn.Tanh()


    def forward(self, audio, profile):

        # =====================================================
        # AUDIO
        # =====================================================

        lstm_out, lstm_state = self.lstm(audio)

        lstm_out, _ = self.lstm2(
            lstm_out,
            lstm_state
        )

        audio_out = self.audio_fc(lstm_out)
        audio_out = self.audio_act(audio_out)

        # [batch, timesteps, 128]


        # =====================================================
        # PROFILE
        # =====================================================

        profile_out = self.profile_fc1(profile)
        profile_out = self.profile_act1(profile_out)

        profile_out = self.profile_fc2(profile_out)
        profile_out = self.profile_act2(profile_out)

        # [batch, 128]

        profile_out = profile_out.unsqueeze(1).expand(
            -1,
            audio_out.size(1),
            -1
        )

        # [batch, timesteps, 128]


        # =====================================================
        # ADAPTIVE GATING
        # =====================================================

        gate_input = torch.cat(
            [audio_out, profile_out],
            dim=2
        )

        gate_logits = self.gate_fc(gate_input)

        # Force weights to sum to 1
        gate_weights = torch.softmax(
            gate_logits,
            dim=2
        )

        audio_weight = gate_weights[:, :, 0:1]
        profile_weight = gate_weights[:, :, 1:2]


        # =====================================================
        # WEIGHTED FUSION
        # =====================================================

        fused = (
            audio_weight * audio_out
            +
            profile_weight * profile_out
        )

        fused = self.dropout(fused)


        # =====================================================
        # PREDICTION
        # =====================================================

        out = self.fc_out(fused)
        out = self.actout(out)

        out = out.flatten(1)
        out = out.unsqueeze(1)

        return out


    def load_my_state_dict(self, state_dict):

        own_state = self.state_dict()

        for name, param in state_dict.items():

            if name not in own_state:
                continue

            if isinstance(param, nn.Parameter):
                param = param.data

            own_state[name].copy_(param)

class lstm_double_profile_attention(torch.nn.Module):
    def __init__(self, audio_dim, profile_dim, hidden_dim):
        super().__init__()

        # =====================================================
        # AUDIO BRANCH
        # =====================================================

        self.lstm = nn.LSTM(
            audio_dim,
            hidden_dim,
            batch_first=True,
            bidirectional=True
        )

        self.lstm2 = nn.LSTM(
            hidden_dim * 2,
            hidden_dim,
            batch_first=True,
            bidirectional=True
        )


        # =====================================================
        # PROFILE BRANCH
        # raw profile -> compact learned representation
        # =====================================================

        self.profile_fc = nn.Linear(
            profile_dim,
            32
        )

        self.profile_act = nn.LeakyReLU(0.1)


        # =====================================================
        # PROFILE ATTENTION
        #
        # Learns a weight for each of the 32 profile dimensions
        # =====================================================

        self.profile_attention = nn.Linear(
            32,
            32
        )


        # =====================================================
        # FUSION
        # =====================================================

        self.fusion_fc = nn.Linear(
            hidden_dim * 2 + 32,
            128
        )

        self.fusion_act = nn.LeakyReLU(0.1)

        self.dropout = nn.Dropout(0.4)

        self.fc_out = nn.Linear(
            128,
            1
        )

        self.actout = nn.Tanh()


    def forward(self, audio, profile):

        # =====================================================
        # AUDIO
        # =====================================================

        lstm_out, lstm_state = self.lstm(audio)

        lstm_out, _ = self.lstm2(
            lstm_out,
            lstm_state
        )


        # =====================================================
        # PROFILE REPRESENTATION
        # =====================================================

        profile_out = self.profile_fc(profile)
        profile_out = self.profile_act(profile_out)


        # =====================================================
        # PROFILE ATTENTION
        #
        # sigmoid gives each profile dimension a weight 0..1
        # =====================================================

        attention_weights = torch.sigmoid(
            self.profile_attention(profile_out)
        )

        attended_profile = (
            profile_out * attention_weights
        )


        # Repeat static listener representation
        # across dynamic timesteps
        attended_profile = attended_profile.unsqueeze(1).expand(
            -1,
            lstm_out.size(1),
            -1
        )


        # =====================================================
        # FUSION
        # =====================================================

        combined = torch.cat(
            [lstm_out, attended_profile],
            dim=2
        )

        out = self.fusion_fc(combined)
        out = self.fusion_act(out)
        out = self.dropout(out)

        out = self.fc_out(out)
        out = self.actout(out)

        out = out.flatten(1)
        out = out.unsqueeze(1)

        return out


    def load_my_state_dict(self, state_dict):

        own_state = self.state_dict()

        for name, param in state_dict.items():

            if name not in own_state:
                continue

            if isinstance(param, nn.Parameter):
                param = param.data

            own_state[name].copy_(param)

class Three_FC_profile_branch(torch.nn.Module):
    def __init__(
        self,
        audio_dim,
        profile_dim,
        hidden_dim=128
    ):
        super().__init__()

        # -----------------------------
        # Audio branch
        # -----------------------------
        self.audio_fc1 = nn.Linear(
            audio_dim,
            hidden_dim
        )
        self.audio_dropout1 = nn.Dropout(0.5)
        self.audio_act1 = nn.LeakyReLU(0.1)

        self.audio_fc2 = nn.Linear(
            hidden_dim,
            hidden_dim // 2
        )
        self.audio_dropout2 = nn.Dropout(0.5)
        self.audio_act2 = nn.LeakyReLU(0.1)

        # -----------------------------
        # Profile branch
        # -----------------------------
        self.profile_fc = nn.Linear(
            profile_dim,
            32
        )
        self.profile_act = nn.LeakyReLU(0.1)

        # -----------------------------
        # Fusion
        # audio 64 + profile 32 = 96
        # -----------------------------
        self.fusion_fc = nn.Linear(
            hidden_dim // 2 + 32,
            hidden_dim // 2
        )
        # self.fusion_dropout = nn.Dropout(0.1)
        self.fusion_act = nn.LeakyReLU(0.1)

        self.fc_out = nn.Linear(
            hidden_dim // 2,
            1
        )

        self.actout = nn.Tanh()

        kernel = torch.FloatTensor([[
            [0.0099, 0.0301, 0.0587,
             0.0733,
             0.0587, 0.0301, 0.0099]
        ]])

        self.register_buffer(
            'kernel',
            kernel
        )

    def forward(self, audio, profile):

        # Audio representation
        audio_out = self.audio_fc1(audio)
        audio_out = self.audio_dropout1(audio_out)
        audio_out = self.audio_act1(audio_out)

        audio_out = self.audio_fc2(audio_out)
        audio_out = self.audio_dropout2(audio_out)
        audio_out = self.audio_act2(audio_out)

        # Profile representation
        profile_out = self.profile_fc(profile)
        profile_out = self.profile_act(profile_out)

        # If profile is [B, profile_dim]
        # and audio is [B, T, audio_dim],
        # repeat profile across timesteps
        profile_out = profile_out.unsqueeze(1).expand(
            -1,
            audio_out.size(1),
            -1
        )

        # Fuse
        combined = torch.cat(
            [audio_out, profile_out],
            dim=2
        )

        out = self.fusion_fc(combined)
        # out = self.fusion_dropout(out)
        out = self.fusion_act(out)

        out = self.fc_out(out)
        out = self.actout(out)

        out = out.flatten(1)
        out = out.unsqueeze(1)

        out = F.conv1d(
            out,
            self.kernel,
            padding=3
        )

        return out