#%%

from dataclasses import dataclass
from typing import Literal, Optional
from torch import Tensor, nn, relu, tanh, tensor, uint8
from typing import List, Optional, Sequence, Union

# @dataclass
# class Z_scoring:
#     transformation_type: Optional[Literal["affine-", "logit"]] = "affine" # affine: standard z_scoring, logit: for numerical stability if rejection-sampling, identity: no z-scoring, typically lowers performance
#     structure: Optional[str] = "independent" # independent or structured

ZScoreType = Literal["affine-independent", "affine-structured", "logit-independent", "logit-structured", "structured", "independent", "none"]

def build_zuko_flow(
    which_nf: str,
    batch_x: Tensor,
    batch_y: Tensor,
    z_score_x: Optional[ZScoreType] = "affine-independent",
    z_score_y: Optional[ZScoreType] = "affine-independent",
    hidden_features: Union[Sequence[int], int] = 50,
    num_transforms: int = 5,
    embedding_net: nn.Module = nn.Identity(),
    **kwargs,
):
    
    if (z_score_x == "affine-independent"):
        print("affine-independent")
        
    
    
    # if (z_score_x[0].transformation_type == "logit"):
    #     print("logit transformation style")
    # else:
    #     print("not logit transformation style")
    
    
build_zuko_flow(
    which_nf="nsf",
    batch_x=tensor([[1, 2, 3], [4, 5, 6]]),
    batch_y=tensor([[1, 2, 3], [4, 5, 6]]),
    z_score_x="affine-independent",
    z_score_y="affine-independent",
    hidden_features=50,
    num_transforms=5,
    embedding_net=nn.Identity(),
)


# %%
import torch
batch_x=tensor([[1, 2, 3], [4, 5, 6]])

min_x = batch_x.min(dim=1)[0]
max_x = batch_x.max(dim=1)[0]


min = torch.min(batch_x, dim=0).values


print(min)

# %%

raise ValueError(
            "Invalid z-scoring option. Use 'affine-independent', 'affine-structured'", 
            "'logit-independent', 'logit-structured', 'identity'."
        )
# %%
