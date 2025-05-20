import torch.nn as nn
import torch
import torch.nn.functional as F


class Decoder(nn.Module):
    def __init__(
        self,
        dims,
        dropout=None,
        dropout_prob=0.1,
        norm_layers=(),
        latent_in=(),
        weight_norm=True,
        use_tanh=True
    ):
        super(Decoder, self).__init__()
        #self.512 = 512
        ##########################################################
        # <================START MODIFYING CODE<================>
        ##########################################################
        # **** YOU SHOULD IMPLEMENT THE MODEL ARCHITECTURE HERE ****
        # Define the network architecture based on the figure shown in the assignment page.
        # Read the instruction carefully for layer details.
        # Pay attention that your implementation should include FC layers, weight_norm layers,
        # Leaky ReLU layers, Dropout layers and a tanh layer.
        self.fc = nn.Linear(3, 1)
        self.dropout_prob = nn.Dropout(dropout_prob)
        self.th = nn.Tanh()

        self.fc1 = nn.utils.weight_norm(nn.Linear(3, 512))
        self.fc2 = nn.utils.weight_norm(nn.Linear(512, 512))
        self.fc3 = nn.utils.weight_norm(nn.Linear(512, 512))
        self.fc4 = nn.utils.weight_norm(nn.Linear(512, 509))

        self.fc5 = nn.utils.weight_norm(nn.Linear(512, 512))
        self.fc6 = nn.utils.weight_norm(nn.Linear(512, 512))
        self.fc7 = nn.utils.weight_norm(nn.Linear(512, 512))
        
        self.fc8 = nn.Linear(512, 1)

        self.leaky_relu = nn.LeakyReLU(0.1, inplace=True)
        # ***********************************************************************
        ##########################################################
        # <================END MODIFYING CODE<================>
        ##########################################################
    
    # input: N x 3
    def forward(self, input):

        ##########################################################
        # <================START MODIFYING CODE<================>
        ##########################################################
        # **** YOU SHOULD IMPLEMENT THE FORWARD PASS HERE ****
        # Based on the architecture defined above, implement the feed forward procedure
        
        x = self.fc1(input)
        x = self.leaky_relu(x)
        x = self.dropout_prob(x)

        x = self.fc2(x)
        x = self.leaky_relu(x)
        x = self.dropout_prob(x)

        x = self.fc3(x)
        x = self.leaky_relu(x)
        x = self.dropout_prob(x)

        x = self.fc4(x)
        x = self.leaky_relu(x)
        x = self.dropout_prob(x)

        x = torch.cat([x, input], dim=1) #concatenate the input with the output of the 4th layer like the figure
        x = self.fc5(x)
        x = self.leaky_relu(x)
        x = self.dropout_prob(x)

        x = self.fc6(x)
        x = self.leaky_relu(x)
        x = self.dropout_prob(x)

        x = self.fc7(x)
        x = self.leaky_relu(x)
        x = self.dropout_prob(x)

        x = self.fc8(x)
        x = self.th(x) # final layer tanh activation
        # ***********************************************************************
        ##########################################################  
        # <================END MODIFYING CODE<================>
        ##########################################################

        return x
