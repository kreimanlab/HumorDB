from transformers import Blip2VisionModel
import torch
from eva_vit import *

class ClassificationOutput():
    def __init__(self, logits=None, loss=None):
        self.logits = logits
        self.loss = loss

class ViTgForClassification(torch.nn.Module):
    def __init__(self, num_classes = 2):
        super(ViTgForClassification, self).__init__()
        self.embed_dim = 1408
        self.init_scale = 0.001
        #self.vitg = Blip2VisionModel.from_pretrained('Salesforce/blip2-opt-2.7b')
        self.vitg = create_eva_vit_g()
        self.fc_norm = nn.LayerNorm(self.embed_dim)
        self.head = nn.Linear(self.embed_dim, num_classes)
        if isinstance(self.head, nn.Linear):
            trunc_normal_(self.head.weight, std=.02)
            self.head.weight.data.mul_(self.init_scale)
            self.head.bias.data.mul_(self.init_scale)
    def forward(self, pixel_values= None, labels= None):
        x = pixel_values
        #features = self.vitg(x).pooler_output
        features = self.vitg(x)
        if self.fc_norm is not None:
            features = self.fc_norm(features.mean(1))
        else:
            features = features[:, 0]
        logits = self.head(features)
        loss = None
        if labels is not None:
            loss_func = torch.nn.CrossEntropyLoss()
            loss = loss_func(logits, labels)
        return ClassificationOutput(logits=logits, loss=loss)

model = ViTgForClassification()
print(model)