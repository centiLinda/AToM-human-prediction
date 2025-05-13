import torch
import torch.nn as nn
import sys

class model_encdec(nn.Module):
    
    def __init__(self, settings, pretrained_model):
        super(model_encdec, self).__init__()

        self.device = torch.device('cpu')

        self.name_model = 'autoencoder'
        self.use_cuda = settings["use_cuda"]
        self.dim_embedding_key = 64 
        self.past_len = settings["past_len"]
        self.future_len = settings["future_len"]
        
        # LAYERS
        self.abs_past_encoder = pretrained_model.abs_past_encoder
        self.norm_past_encoder = pretrained_model.norm_past_encoder
        self.norm_fut_encoder = pretrained_model.norm_fut_encoder
        self.res_past_encoder = pretrained_model.res_past_encoder
        self.social_pooling_X = pretrained_model.social_pooling_X
        self.decoder = pretrained_model.decoder
        self.decoder_x = pretrained_model.decoder_x
        # self.decoder_x_abs = pretrained_model.decoder_x_abs
        self.decoder_2 = pretrained_model.decoder_2
        self.decoder_2_x = pretrained_model.decoder_2_x
        # self.decoder_2_x_abs = pretrained_model.decoder_2_x_abs
        self.input_query_w = pretrained_model.input_query_w
        self.past_memory_w = pretrained_model.past_memory_w

        self.encoder_dest = pretrained_model.encoder_dest
        # self.encoder_dest = MLP(input_dim = 2, output_dim = 64, hidden_size=(64, 128))
        self.traj_abs_past_encoder = pretrained_model.traj_abs_past_encoder
        self.interaction = pretrained_model.interaction
        self.num_decompose = 2
        self.decompose = pretrained_model.decompose

    def forward(self, past, abs_past, seq_start_end, end_pose, goal):
        abs_past_state = self.traj_abs_past_encoder(abs_past)
        abs_past_state_social = self.interaction(abs_past_state, seq_start_end, end_pose)
        destination_feat = self.encoder_dest(goal)
        state_conc = torch.cat((abs_past_state_social, destination_feat), dim=1)

        x_true = past.clone()
        x_hat = torch.zeros_like(x_true)
        batch_size = past.size(0)
        prediction_single = torch.zeros((batch_size, self.future_len-1, 2))
        reconstruction = torch.zeros((batch_size, self.past_len, 2))

        for decompose_i in range(self.num_decompose):
            x_hat, y_hat = self.decompose[decompose_i](x_true, x_hat, state_conc)
            # x_hat = x_hat[:, :-5, :] #NOTE mofify length, or retrain
            # y_hat = y_hat[:, :-2, :]
            prediction_single += y_hat
            reconstruction += x_hat
        
        for i_frame in range(1, self.future_len):
            prediction_single[:, i_frame-1] += goal * i_frame / self.future_len
        prediction_single = torch.cat((prediction_single, goal.unsqueeze(1)), dim=1)
        
        return prediction_single