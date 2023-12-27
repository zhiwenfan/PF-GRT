from .viewAtten.view_atten import NoPEGNT
import torch

if __name__ == "__main__":
    
    config = {
        "backbone_type":"ResNetFPN",
        "resolution": (8, 2),
        "netwidth":64, 
        "trans_depth":4,
        "resnetfpn": {
            "initial_dim":128,
            "block_dims": [128, 196, 256]  # s1, s2, s3
        }
    }
    
    NopeGntModel = NoPEGNT(config)
    pts = torch.ones((4048, 64, 3))
    ray_d = torch.ones((4048, 3)) 
    src_images = torch.ones((1,10,3,512,512)) 
    dists = torch.ones((4048, 64)) 
    z_vals = torch.ones((4048, 64)) 

    device = "cuda:0"
    # NopeGntModel = NopeGntModel.to(device)
    outputs = NopeGntModel(src_images , pts, ray_d, dists=dists, z_vals=z_vals)

    for out in outputs:
        print(f"{out.shape}")
    