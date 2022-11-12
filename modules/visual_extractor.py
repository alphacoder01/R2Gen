import torch
import torch.nn as nn
import torchvision.models as models
import torchxrayvision as xrv


class VisualExtractor(nn.Module):
    def __init__(self, args):
        super(VisualExtractor, self).__init__()
        if args.use_xrayvision:
            self.visual_extractor = "densenet121-res224-mimic_nb"
            model = xrv.models.DenseNet(weights=self.visual_extractor)
        elif args.use_pretrained:
            print("using inhouse pretrained model")
            self.pretrained = args.visual_extractor_pretrained
            model = getattr(models, 'resnet101')(pretrained=self.pretrained)
            model.fc = nn.Linear(2048,15)
            model.load_state_dict(torch.load('/home/sweta/scratch/828-Project/Pre-training/2nd_pass_epoch_10_weights.pth'))
        else:
            self.visual_extractor = args.visual_extractor
            self.pretrained = args.visual_extractor_pretrained
            print(f"Using {self.visual_extractor} model")
            model = getattr(models, self.visual_extractor)(pretrained=self.pretrained)
        modules = list(model.children())[:-2]
        self.model = nn.Sequential(*modules)
        self.avg_fnt = torch.nn.AvgPool2d(kernel_size=7, stride=1, padding=0)

    def forward(self, images):
        patch_feats = self.model(images)
        avg_feats = self.avg_fnt(patch_feats).squeeze().reshape(-1, patch_feats.size(1))
        batch_size, feat_size, _, _ = patch_feats.shape
        patch_feats = patch_feats.reshape(batch_size, feat_size, -1).permute(0, 2, 1)
        return patch_feats, avg_feats
