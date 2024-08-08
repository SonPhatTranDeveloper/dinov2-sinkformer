import torch
import torch.nn as nn

from models.attention import WeightedCombinationAttention
from dinov2.layers import NestedTensorBlock as Block
from functools import partial
from dinov2.vision_transformer import DinoVisionTransformer, vit_small

from copy import deepcopy


def vit_small_sinkhorn(patch_size=16, num_register_tokens=0, **kwargs):
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        block_fn=partial(Block, attn_class=WeightedCombinationAttention),
        num_register_tokens=num_register_tokens,
        **kwargs,
    )
    return model


def create_vit_and_copy_weight():
    # Load pretrained ViT model
    vit_transformers = vit_small(patch_size=14,
                                 img_size=526,
                                 init_values=1.0,
                                 num_register_tokens=4,
                                 block_chunks=0)
    vit_transformers.load_state_dict(
        torch.load("pretrained/dinov2_vits14_reg4_pretrain.pth")
    )

    # Create ViT Sinkformer
    vit_sinkformers = vit_small_sinkhorn(patch_size=14,
                                         img_size=526,
                                         init_values=1.0,
                                         num_register_tokens=4,
                                         block_chunks=0)

    # Copy the weight
    vit_sinkformers.load_state_dict(vit_transformers.state_dict(), strict=False)

    # Load the weight of softmax attention layer to sinkhorn attention layer
    for block_sinkformer, block_transformer in zip(vit_sinkformers.blocks, vit_transformers.blocks):
        # Load all other blocks
        block_sinkformer.load_state_dict(block_transformer.state_dict(), strict=False)

        # Load the softmax weight
        attn_sinkformer = block_sinkformer.attn
        attn_transformer = block_transformer.attn

        # Load the weights for softmax
        attn_sinkformer.softmax_attn.load_state_dict(
            attn_transformer.state_dict(),
            strict=False
        )
        attn_sinkformer.sinkhorn_attn.load_state_dict(
            attn_transformer.state_dict(),
            strict=False
        )

    # Copy back the last block of the sinkformer to the transformer
    vit_transformers.blocks[11] = vit_sinkformers.blocks[11]

    # Create a deep copy of the model
    return deepcopy(vit_transformers)


class DINOClassificationModel(nn.Module):
    def __init__(self, hidden_size, num_classes):
        """
        Load the pretrained DINOv2 Classification Model
        """
        # Initialize module
        super(DINOClassificationModel, self).__init__()

        # Load result with register
        self.embedding_size = 384
        self.number_of_heads = 6
        self.num_classes = num_classes
        self.hidden_size = hidden_size

        # Copy the result
        self.transformers = create_vit_and_copy_weight()

        # Add the classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.embedding_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.num_classes)
        )

    def forward(self, inputs):
        """
        Forward the inputs
        inputs: tensor of size (batch_size, image_height, image_width, channels)
        """
        # Pass through the transformers and normalization
        outputs = self.transformers(inputs)
        outputs = self.transformers.norm(outputs)
        outputs = self.classifier(outputs)
        return outputs


if __name__ == "__main__":
    # Create model
    create_vit_and_copy_weight()

