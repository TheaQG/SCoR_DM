'''
    Script containing neural network modules for DDPM_DANRA_Downscaling.
    The encoder and decoder modules are used in a UNET for downscaling in the DDPM.
    The following modules are defined:
        - SinusoidalEmbedding: sinusoidal embedding module
        - ImageSelfAttention: image self-attention module
        - Encoder: encoder module
        - DecoderBlock: decoder block module
        - Decoder: decoder module
        
'''
import torch
import logging
import torch.nn as nn
from torchvision.models.resnet import ResNet, BasicBlock
from typing import Optional, Iterable
import functools

# Set up logging
logger = logging.getLogger(__name__)


class SigmaEmbed(nn.Module):
    """
        sigma-embedding -> R^time_dim for FiLM like conditioning.
    """
    def __init__(self, time_dim: int):
        super().__init__()
        self.time_dim = time_dim
        self.net = nn.Sequential(
            nn.Linear(1, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim)
        )
    def forward(self, c_noise: torch.Tensor) -> torch.Tensor:
        # must be [B, 1]
        if not (c_noise.ndim == 2 and c_noise.shape[1] == 1):
            raise ValueError(
                f"[SigmaEmbed] c_noise must be [B, 1], got {tuple(c_noise.shape)}."
            )
        # c_noise: [B, 1]
        return self.net(c_noise)

class EDMPrecondUNet(nn.Module):
    """
        Wraps Encoder/Decoder with EDM preconditioning.
        Predicts x0 directly (no division by sigma) and combines skip/out as in Karras et al. (2022).
    """
    def __init__(self,
                 encoder: nn.Module,
                 decoder: nn.Module,
                 sigma_data: float = 1.0,
                 predict_residual: bool = True,
                 ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.sigma_data = sigma_data
        self.predict_residual = predict_residual

        # time_embedding size is already defined in encoder
        time_dim = getattr(encoder, "time_embedding", 128)
        self.sigma_emb = SigmaEmbed(time_dim)

    def _combine_with_labels(self, t_emb: torch.Tensor, y: torch.Tensor | None) -> torch.Tensor:
        """
        Build a season-aware embedding t_comb by adding either a categorical label embedding
        or a continuous DOY sin/cos projection to the sigma embedding t_emb.

        Uses the encoder's embedding modules (label_emb / temporal_emb) to guarantee consistency.
        """
        if y is None:
            return t_emb

        enc = self.encoder
        longs = (torch.long, torch.int64, torch.int32)

        # Categorical labels
        if getattr(enc, "num_classes", None) is not None and y.dtype in longs:
            y_in = y
            if y_in.ndim == 2 and y_in.shape[1] == 1:
                y_in = y_in[:, 0]
            elif y_in.ndim != 1:
                raise ValueError(f"[EDMPrecondUNet] categorical y must be [B] or [B,1], got {tuple(y.shape)}")
            y_emb = enc.label_emb(y_in.to(t_emb.device))
            return t_emb + y_emb

        # Continuous DOY sin/cos
        if torch.is_floating_point(y):
            y2 = y.view(y.shape[0], -1)
            if y2.shape[1] != 2:
                raise ValueError(f"[EDMPrecondUNet] expected DOY sin/cos with last dim==2, got {tuple(y2.shape)}")
            return t_emb + enc.temporal_emb(y2.to(t_emb.device))

        # Otherwise leave unchanged
        return t_emb

    def _precond(self, sigma: torch.Tensor):
        """ Compute preconditioning coefficients. (as in Karras et al. 2022) with strict shape control for """
        # Accept sigma of shape [B] or [B, 1], [B, 1, 1, 1] - coerce to [B]
        if sigma.ndim == 0:
            sigma = sigma.unsqueeze(0)
        if sigma.ndim > 1:
            sigma = sigma.reshape(sigma.shape[0]) # -> [B]

        # sigma: [B]
        s2 = sigma**2
        sd = self.sigma_data
        sd2 = sd**2

        c_in    = 1.0 / torch.sqrt(s2 + sd2)            # [B]
        c_skip  = sd2 / (s2 + sd2)                      # [B]
        c_out   = sigma * sd / torch.sqrt(s2 + sd2)     # [B] corrected formula
        # c_noise = 0.25 * log(sigma^2) = 0.25 * 2 * log(sigma) 
        # c_noise = 0.5 * log(sigma) as a scalar feature -> [B,1]
        c_noise = (sigma.log() * 0.5).unsqueeze(-1)  # [B, 1]   
        return c_in, c_skip, c_out, c_noise
    
    def forward(self,
                x_t: torch.Tensor,
                sigma: torch.Tensor,
                *,
                cond_img: torch.Tensor | None = None,
                lsm_cond: torch.Tensor | None = None,
                topo_cond: torch.Tensor | None = None,
                y: torch.Tensor | None = None,
                lr_ups: torch.Tensor | None = None # <- needed if predict_residual = True
                ) -> torch.Tensor:
        B = x_t.shape[0]
        c_in, c_skip, c_out, c_noise = self._precond(sigma) # c_noise is [B, 1]

        # Sigmaa embedding [B, time_dim]
        t_emb = self.sigma_emb(c_noise)  # [B, time_dim]
        assert t_emb.ndim == 2 and t_emb.shape[1] == self.encoder.time_embedding, \
            f"[EDMPrecondUnet] Expected t_emb to be [B, {self.encoder.time_embedding}], got {tuple(t_emb.shape)}."

        # === Residual-aware preconditioning ===
        # NOTE: LR upsampled must be scaled in HR space!

        # Build combined time for decoder FiLM (season-aware)
        t_comb = self._combine_with_labels(t_emb, y)  # [B, time_dim]

        if self.predict_residual:
            if lr_ups is None:
                raise ValueError("lr_ups must be provided when predict_residual is True.")
            
            # Shift the noisy input by the baseline so the network denoises the residual r := x0 - lr_ups
            x_shift = x_t - lr_ups
            x_in = c_in.view(B, 1, 1, 1) * x_shift  # [B, C, H, W] Scale residual input

            # Encoder gets (t_emb,y) so it will add label/DOY internally; decoder gets season-aware t_comb
            enc_fmaps = self.encoder(x_in, t_emb, y=y, cond_img=cond_img, lsm_cond=lsm_cond, topo_cond=topo_cond)
            out = self.decoder(*enc_fmaps, t=t_comb) # [B, 1, H, W] (treat as direct x0 head OR residual head)

            # Predict x0 as: baseline + preconditioned skip residual + network residual head
            x0_hat = lr_ups + c_skip.view(B, 1, 1, 1) * x_shift + c_out.view(B, 1, 1, 1) * out # As in Karras et al. (2022)

        else:
            # Standard EDM preconditioning (no residual awareness)
            x_in = c_in.view(B, 1, 1, 1) * x_t  # [B, C, H, W] Scale input
            enc_fmaps = self.encoder(x_in, t_emb, y=y, cond_img=cond_img, lsm_cond=lsm_cond, topo_cond=topo_cond)
            out = self.decoder(*enc_fmaps, t=t_comb) # [B, 1, H, W] (treat as direct x0 head OR residual head)
            x0_hat = c_skip.view(B, 1, 1, 1) * x_t + c_out.view(B, 1, 1, 1) * out

        return x0_hat
    


class SinusoidalEmbedding(nn.Module):
    '''
        Gaussian random features for encoding the time steps.
        (Named SinusoidalEmbedding to match the original DDPM implementation.)
        Randomly samples weights during initialization. These weights
        are fixed during optimization - non-trainable.
    '''
    def __init__(self, embed_dim: int, scale: float = 30.0, device=None, dtype=torch.float32):
        super().__init__()
        if embed_dim % 2 != 0:
            raise ValueError(f"Embedding dimension must be even, got {embed_dim}.")
        
        # Fixed random frequencies; saved in state_dict but no gradients
        # Initialize the weights as random Gaussian weights multiplied by the scale
        W = torch.randn(embed_dim // 2, dtype=dtype, device=device) * scale
        self.register_buffer('W', W, persistent=True)  # persistent=True to save in state_dict

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B] or [B, 1] (time steps); project to [B, embed_dim//2], then concat sin/cos -> [B, embed_dim]
        x = x.view(-1).to(device=self.W.device, dtype=self.W.dtype)  # [B]
        x_proj = x[:, None] * self.W[None, :] * (2.0 * torch.pi)  # [B, embed_dim//2]
        return torch.cat([x_proj.sin(), x_proj.cos()], dim=-1)  # [B, embed_dim]

        


# class ImageSelfAttention(nn.Module):
#     ''' 
#         MAYBRITT SCHILLINGER
#         Class for image self-attention. Self-attention is a mechanism that allows the model to focus on more important features.
#         Focus on one thing and ignore other things that seem irrelevant at the moment.
#         The attention value is of size (N, C, H, W), where N is the number of samples, C is the number of channels, and H and W are the height and width of the input.
#     '''
#     def __init__(self, input_channels:int, n_heads:int, device = None):
#         '''
#             Initialize the class.
#             Input:
#                 - input_channels: number of input channels
#                 - n_heads: number of heads (how many different parts of the input are attended to)
#         '''
#         # Initialize the class
#         super(ImageSelfAttention, self).__init__()
#         # Set the device
#         if device is None:
#             self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         else:
#             self.device = device
#         # Set the device for the class
#         self.to(self.device)
        
#         # Set the class variables
#         self.input_channels = input_channels
#         self.n_heads = n_heads
#         # Multi-head attention layer, for calculating the attention
#         self.mha = nn.MultiheadAttention(self.input_channels, self.n_heads, batch_first=True)
#         # Layer normalization layer, for normalizing the input
#         self.layernorm = nn.LayerNorm([self.input_channels])
#         # FF layer, for calculating the attention value
#         self.ff = nn.Sequential(
#             nn.LayerNorm([self.input_channels]), # Layer normalization
#             nn.Linear(self.input_channels, self.input_channels), # Linear layer
#             nn.GELU(), # GELU activation function
#             nn.Linear(self.input_channels, self.input_channels) # Linear layer
#         )
        
#     def forward(self, x:torch.Tensor):
#         '''
#             Forward function for the class. The self-attention is applied to the input x.
#             Self-attention is calculated by calculating the dot product of the input with itself.
#         '''

#         # shape of x: (N, C, H, W), (N samples, C channels, height, width)
#         _, C, H, W = x.shape

#         # Reshape the input to (N, C, H*W) and permute to (N, H*W, C)
#         x = x.reshape(-1, C, H*W).permute(0, 2, 1)
#         # Normalize the input
#         x_ln = self.layernorm(x)
#         # Calculate the attention value and attention weights 
#         attn_val, _ = self.mha(x_ln, x_ln, x_ln)
#         # Add the attention value to the input
#         attn_val = attn_val + x
#         # Apply the FF layer to the attention value
#         attn_val = attn_val + self.ff(attn_val)
#         # Reshape the attention value to (N, C, H, W)
#         attn_val = attn_val.permute(0, 2, 1).reshape(-1, C, H, W)
#         return attn_val

class ImageSelfAttention(nn.Module):
    """
        Channel-first image self-attention over flattened spatial tokens.
        Returns y = x + MHA(LayerNorm(x)) + FF(LayerNorm(x + MHA(...))) (pre-LayerNorm residual block).
    """
    def __init__(self,
                 input_channels: int,
                 n_heads: int,
                 dropout: float = 0.0,):
        super().__init__()
        if input_channels % n_heads != 0:
            raise ValueError(f"Number of input channels ({input_channels}) must be divisible by number of heads ({n_heads}).")
        self.input_channels = input_channels
        self.n_heads = n_heads

        self.mha = nn.MultiheadAttention(embed_dim=input_channels, num_heads=n_heads, dropout=dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(input_channels)
        self.ln2 = nn.LayerNorm(input_channels)
        self.ff = nn.Sequential(
            nn.Linear(input_channels, input_channels),
            nn.GELU(),
            nn.Linear(input_channels, input_channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, C, H, W)
        N, C, H, W = x.shape
        x_tokens = x.reshape(N, C, H * W).permute(0, 2, 1)  # (N, S, C), S=H*W

        h = self.ln1(x_tokens)
        attn_out, _ = self.mha(h, h, h)  # (N, S, C)
        h = x_tokens + attn_out

        y_tokens = h + self.ff(self.ln2(h))  # (N, S, C)
        y = y_tokens.permute(0, 2, 1).reshape(N, C, H, W)  # (N, C, H, W)

        return y


class Encoder(ResNet):
    '''
        Class for the encoder. The encoder is used to encode the input data.
        The encoder is a ResNet with self-attention layers, and will be part of a UNET used for downscaling in the DDPM.
        The encoder consists of five feature maps, one for each layer of the ResNet.
        The encoder works as a downsample block, and will be used to downsample the input.
    '''
    def __init__(self,
                 input_channels:int,
                 time_embedding:int, 
                 block=BasicBlock,
                 block_layers:list=[2, 2, 2, 2],
                 n_heads:int=4,
                 num_classes:Optional[int]=None,
                 cond_on_img=False,
                 cond_img_dim = None,
                 device = None
                 ):
        '''
            Initialize the class. 
            Input:
                - input_channels: number of input channels
                - time_embedding: size of the time embedding
                - block: block to use for the ResNet
                - block_layers: list containing the number of blocks for each layer (default: [2, 2, 2, 2], 4 layers with 2 blocks each)
                - n_heads: number of heads for the self-attention layers (default: 4, meaning 4 heads for each self-attention layer)
        '''
        # Initialize the class
        self.block = block
        self.block_layers = block_layers
        self.time_embedding = time_embedding
        self.input_channels = input_channels + 1 # +1 for the HR image (noised)
        self.n_heads = n_heads
        self.num_classes = num_classes

        
        # Initialize the ResNet with the given block and block_layers
        super(Encoder, self).__init__(self.block, self.block_layers)

        # Device attribute (do not move submodules here, call self.to(device) at the end of init)
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                
        # Initialize the sinusoidal time embedding layer with the given time_embedding
        self.sinusoidal_embedding = SinusoidalEmbedding(self.time_embedding)

        
        # Set the channels for the feature maps (five feature maps, one for each layer, with 64, 64, 128, 256, 512 channels)
        fmap_channels = [64, 64, 128, 256, 512]

        # Set the time projection layers, for projecting the time embedding onto the feature maps
        self.time_projection_layers = self.make_time_projections(fmap_channels)
        # Set the attention layers, for calculating the attention for each feature map
        self.attention_layers = self.make_attention_layers(fmap_channels)
        
        # === FiLM (scale & shift by DOY feature) layers per feature map ===
        self.film_layers = nn.ModuleList([
            nn.Sequential(
                nn.SiLU(),
                nn.Linear(self.time_embedding, 2 * ch)  # -> [B, 2*C]
            ) for ch in fmap_channels
        ])

        # Set the first convolutional layer, with N input channels(=input_channels) and 64 output channels
        self.conv1 = nn.Conv2d(
            self.input_channels, 64, 
            kernel_size=(8, 8), # Previous kernelsize (7,7)
            stride=(2, 2), 
            padding=(3, 3), 
            bias=False)
        
        # Set the second convolutional layer, with 64 input channels and 64 output channels
        self.conv2 = nn.Conv2d(
            64, 64, 
            kernel_size=(8, 8), # Previous kernelsize (7,7)
            stride=(2, 2),  
            padding=(3, 3),
            bias=False)

        # If conditional, set the label embedding layer from the number of classes to the time embedding size
        # A tiny MLP from R^2 -> R^time_embedding
        temporal_dim = int(self.time_embedding)
        self.temporal_emb = nn.Sequential(
            nn.Linear(2, temporal_dim),
            nn.SiLU(),
            nn.Linear(temporal_dim, temporal_dim)
        )
        
        if num_classes is not None:
            self.num_classes = num_classes
            self.label_emb = nn.Embedding(num_classes + 1, self.time_embedding)
            with torch.no_grad():
                self.label_emb.weight[0].fill_(0.0) # 0 is the null class (CFG), so ensure no conditioning influence

        #delete unwanted layers, i.e. maxpool(=self.maxpool), fully connected layer(=self.fc) and average pooling(=self.avgpool
        del self.maxpool, self.fc, self.avgpool

    def _add_time_bias(self, fmap: torch.Tensor, proj: nn.Module, t_vec: torch.Tensor, where: str) -> torch.Tensor:
        """
        Project t_vec -> [B, C], reshape to [B, C, 1, 1] and add to fmap.
        Enforces a strict 2-D [B, time_embedding] input to the projection.
        """
        assert fmap.ndim == 4, f"{where}: expected fmap to be 4D NCHW, got {tuple(fmap.shape)}"

        # --- Hard enforcement: make t_vec strictly [B, time_embedding] ---
        if t_vec.ndim != 2:
            t_vec = t_vec.reshape(t_vec.shape[0], -1)

        if t_vec.shape[1] != self.time_embedding:
            raise ValueError(
                f"{where}: time vector must be [B,{self.time_embedding}] but got {tuple(t_vec.shape)}. "
                f"Check where you build the time/class/season embeddings."
            )

        bias = proj(t_vec)  # [B, C]
        if bias.ndim != 2:
            raise ValueError(f"{where}: projection produced {tuple(bias.shape)}; expected [B, C].")

        bias = bias.unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]
        return fmap + bias

    def _apply_film(self, fmap: torch.Tensor, film_mlp: nn.Module, t_vec: torch.Tensor, where: str) -> torch.Tensor:
        """
        FiLM modulation: produce per-channel gamma,beta from t_vec and apply:
            y = (1 + gamma) * x + beta
        Expects t_vec to be strictly [B, time_embedding].
        """
        assert fmap.ndim == 4, f"{where}: expected fmap to be 4D NCHW, got {tuple(fmap.shape)}"

        if t_vec.ndim != 2:
            t_vec = t_vec.reshape(t_vec.shape[0], -1)
        if t_vec.shape[1] != self.time_embedding:
            raise ValueError(f"{where}: t_vec must be [B,{self.time_embedding}] but got {tuple(t_vec.shape)}.")

        gb = film_mlp(t_vec)  # [B, 2*C]
        C = fmap.shape[1]
        if gb.ndim != 2 or gb.shape[1] != 2 * C:
            raise ValueError(f"{where}: FiLM MLP output must be [B,{2*C}], got {tuple(gb.shape)}.")

        gamma, beta = torch.chunk(gb, 2, dim=1)      # each [B, C]
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)    # [B, C, 1, 1]
        beta  = beta.unsqueeze(-1).unsqueeze(-1)     # [B, C, 1, 1]
        return fmap * (1.0 + gamma) + beta

    def forward(self,  # type: ignore
                x:torch.Tensor, 
                t:torch.Tensor, 
                y:Optional[torch.Tensor]=None, 
                cond_img:Optional[torch.Tensor]=None, 
                lsm_cond:Optional[torch.Tensor]=None, 
                topo_cond:Optional[torch.Tensor]=None
                ):
        '''
            Forward function for the class. The input x and time embedding t are used to calculate the output.
            The output is the encoded input x.
            Input:
                x:          [B, C_hr, H, W] input tensor, noised image (+ already concatenated channels if any)
                t:          [B] | [B, 1] scalar times, or pre-embedded [B, time_embedding]
                y:          label tensor, optional:
                                - Long [B] for categorical labels (dtype long/int) (season/month/day id) -> label_emb
                                - Float [B, 2] for continuous scalar labels (e.g. sin/cos DOY) -> temporal_emb
                cond_img:   [B, C_lr, H, W] conditional image tensor, optional (must be concatenated correctly, if multiple channels)
                lsm_cond:   [B, 1, H, W] conditional tensor for land-sea mask, optional
                topo_cond:  [B, 1, H, W] conditional tensor for topography, optional

            Output:
                - fmap1, fmap2, fmap3, fmap4, fmap5: feature maps

        '''
        dev = x.device


        # === 1) Canonicalize time to [B, time_embedding] ===
        t = t.to(dev)
        if t.ndim == 2 and t.shape[1] == self.time_embedding:
            t_base = t  # Already embedded
        elif (t.ndim == 1) or (t.ndim == 2 and t.shape[1] == 1):
            t_base = self.sinusoidal_embedding(t.view(-1))  # [B, time_embedding]
        else:
            raise ValueError(f"[Encoder.forward] Unexpected t shape {tuple(t.shape)}. Must be [B], [B,1], or [B,{self.time_embedding}].")

        # === 2) Add optional label (categorical) or sin/cos (continuous) to time ===
        if (self.num_classes is not None) and (y is not None) and y.dtype in (torch.long, torch.int64, torch.int32):
            y_in = y.to(dev)
            # Accept [B] or [B,1]; reject anything else
            if y_in.ndim == 2 and y_in.shape[1] == 1:
                y_in = y_in[:, 0]          # -> [B]
            elif y_in.ndim != 1:
                raise ValueError(f"[Encoder.forward] categorical y must be [B] or [B,1], got {tuple(y.shape)}")
            y_emb = self.label_emb(y_in)    # [B, time_embedding]
            t_comb = t_base + y_emb         # stays [B, time_embedding]
        elif (y is not None) and torch.is_floating_point(y):
            # sin/cos DOY: y shape [B, 2]
            ycont = y.to(dev)
            if ycont.dim() > 2:
                ycont = ycont.view(ycont.shape[0], -1)
            assert ycont.shape[-1] == 2, f"Expected sin/cos with last dim==2, got shape {tuple(ycont.shape)}."
            t_comb = t_base + self.temporal_emb(ycont)  # [B, time_embedding]
        else:
            t_comb = t_base  # [B, time_embedding]
        # Final safety: must be strictly 2-D [B, time_embedding]
        if not (t_comb.ndim == 2 and t_comb.shape[1] == self.time_embedding):
            raise ValueError(
                f"[Encoder.forward] t_comb must be [B,{self.time_embedding}], got {tuple(t_comb.shape)}. "
                f"Check sigma_emb/cfg and label embedding inputs."
            )

        # === 3) Concatenate optional conditional inputs to x ===
        if lsm_cond is not None:
            if lsm_cond.shape[0] != x.shape[0]:
                raise ValueError(f"Batch mismatch: x= {x.shape[0]}, lsm_cond={lsm_cond.shape[0]}.")
            x = torch.cat([x, lsm_cond.to(dev)], dim=1)
        if topo_cond is not None:
            if topo_cond.shape[0] != x.shape[0]:
                raise ValueError(f"Batch mismatch: x= {x.shape[0]}, topo_cond={topo_cond.shape[0]}.")
            x = torch.cat([x, topo_cond.to(dev)], dim=1)
        if cond_img is not None:
            if cond_img.shape[0] != x.shape[0]:
                raise ValueError(f"Batch mismatch: x= {x.shape[0]}, cond_img={cond_img.shape[0]}.")
            x = torch.cat([x, cond_img.to(dev)], dim=1)



        # === 4) Validate shape/channels ===
        if x.ndim != 4:
            raise ValueError(f"Encoder expects NCHW input; got {x.shape}")
        if x.shape[1] != self.input_channels:
            raise ValueError(f"In-channel mismatch: got {x.shape[1]}, expected {self.input_channels}. "
                            "Check dual_lr / geo with_mask and dataset stacking.")

        assert t_comb.shape == (x.shape[0], self.time_embedding), \
            f"[Encoder] bad t_comb: {tuple(t_comb.shape)} vs (B,{self.time_embedding}) with B={x.shape[0]}"

        # === 5) Encoder path with time-bias at each stage ===
        # fmap1
        fmap1 = self.conv1(x)
        fmap1 = self._apply_film(fmap1, self.film_layers[0], t_comb, "Encoder FiLM after conv1")
        fmap1 = self.attention_layers[0](fmap1)

        # conv2 -> bn1 -> relu
        x2 = self.conv2(fmap1)
        x2 = self.bn1(x2)
        x2 = self.relu(x2)

        # fmap2
        fmap2 = self.layer1(x2)
        fmap2 = self._apply_film(fmap2, self.film_layers[1], t_comb, "Encoder FiLM after fmap2")
        fmap2 = self.attention_layers[1](fmap2)
        
        # fmap3
        fmap3 = self.layer2(fmap2)
        fmap3 = self._apply_film(fmap3, self.film_layers[2], t_comb, "Encoder FiLM after fmap3")
        fmap3 = self.attention_layers[2](fmap3)
        
        # fmap4
        fmap4 = self.layer3(fmap3)
        fmap4 = self._apply_film(fmap4, self.film_layers[3], t_comb, "Encoder FiLM after fmap4")
        fmap4 = self.attention_layers[3](fmap4)

        # fmap5
        fmap5 = self.layer4(fmap4)
        fmap5 = self._apply_film(fmap5, self.film_layers[4], t_comb, "Encoder FiLM after fmap5")
        fmap5 = self.attention_layers[4](fmap5)
        
        # Return the feature maps
        return fmap1, fmap2, fmap3, fmap4, fmap5
        # # fmap1
        # fmap1 = self.conv1(x)
        # fmap1 = self._add_time_bias(fmap1, self.time_projection_layers[0], t_comb, "after conv1")
        # fmap1 = self.attention_layers[0](fmap1)

        # # conv2 -> bn1 -> relu
        # x2 = self.conv2(fmap1)
        # x2 = self.bn1(x2)
        # x2 = self.relu(x2)

        # # fmap2
        # fmap2 = self.layer1(x2)
        # fmap2 = self._add_time_bias(fmap2, self.time_projection_layers[1], t_comb, "at fmap2")
        # fmap2 = self.attention_layers[1](fmap2)
        
        # # fmap3
        # fmap3 = self.layer2(fmap2)
        # fmap3 = self._add_time_bias(fmap3, self.time_projection_layers[2], t_comb, "at fmap3")
        # fmap3 = self.attention_layers[2](fmap3)
        
        # # fmap4
        # fmap4 = self.layer3(fmap3)
        # fmap4 = self._add_time_bias(fmap4, self.time_projection_layers[3], t_comb, "at fmap4")
        # fmap4 = self.attention_layers[3](fmap4)

        # # fmap5
        # fmap5 = self.layer4(fmap4)
        # fmap5 = self._add_time_bias(fmap5, self.time_projection_layers[4], t_comb, "at fmap5")
        # fmap5 = self.attention_layers[4](fmap5)
        
        # # Return the feature maps
        # return fmap1, fmap2, fmap3, fmap4, fmap5
    
    
    def make_time_projections(self, fmap_channels:Iterable[int]):
        '''
            Function for making the time projection layers. The time projection layers are used to project the time embedding onto the feature maps.
            Input:
                - fmap_channels: list containing the number of channels for each feature map
        '''
        # Initialize the time projection layers consisting of a SiLU activation function and a linear layer. 
        # The SiLU activation function is used to introduce non-linearity. One time projection layer is used for each feature map.
        # The number of input channels for each time projection layer is the size of the time embedding, and the number of output channels is the number of channels for the corresponding feature map.
        # Only the first time projection layer has a different number of input channels, namely the number of input channels for the first convolutional layer.
        layers = nn.ModuleList([
            nn.Sequential(
                nn.SiLU(),
                nn.Linear(self.time_embedding, ch)
            ) for ch in fmap_channels ])
        
        return layers
    
    def make_attention_layers(self, fmap_channels:Iterable[int]):
        '''
            Function for making the attention layers. The attention layers are used to calculate the attention for each feature map.
            Input:
                - fmap_channels: list containing the number of channels for each feature map
        '''
        # Initialize the attention layers. One attention layer is used for each feature map.
        
        # NEW: attention only at layers >= len(fmap_channels) - 2
        fmap_channels = list(fmap_channels)
        layers = nn.ModuleList(
            [ImageSelfAttention(ch, self.n_heads) if i >= len(fmap_channels) - 2 else nn.Identity() for i, ch in enumerate(fmap_channels)]
        )

        # OLD: Attention at all layers
        # layers = nn.ModuleList([
        #     ImageSelfAttention(ch, self.n_heads) for ch in fmap_channels
        # ])
        
        return layers
    



class DecoderBlock(nn.Module):
    '''
        Class for the decoder block. The decoder block is used to decode the encoded input.
        Part of a UNET used for downscaling in the DDPM. The decoder block consists of a transposed convolutional layer, a convolutional layer, and a self-attention layer.
        The decoder block works as an upsample block, and will be used to upsample the input.
    '''
    def __init__(
            self,
            input_channels:int,
            output_channels:int,
            time_embedding:int,
            upsample_scale:int=2,
            activation: type[nn.Module] = nn.ReLU,
            compute_attn:bool=True,
            n_heads:int=4,
            device = None,
            *,
            use_resize_conv: bool = True,
            norm: str = "instance", # "instance" | "group"
            gn_groups: int = 8
            ):
        '''
            Initialize the class.
            Input:
                - input_channels: number of input channels
                - output_channels: number of output channels
                - time_embedding: size of the time embedding
                - upsample_scale: scale factor for the transposed convolutional layer (default: 2, meaning the output will be twice the size of the input)
                - activation: activation function to use (default: ReLU)
                - compute_attn: boolean indicating whether to compute the attention (default: True)
                - n_heads: number of heads for the self-attention layer (default: 4, meaning 4 heads for the self-attention layer)
                - use_resize_conv: whether to use resize convolution instead of transposed convolution for upsampling (default: True)
                - norm: normalization layer to use (default: instance normalization)
                - gn_groups: number of groups for group normalization (default: 8)
        '''

        # Initialize the class
        super().__init__()

        # Keep device attribute for reference (only move submodules at the end of init with self.to(device))
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Initialize the class variables
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.upsample_scale = upsample_scale
        self.time_embedding = time_embedding
        self.compute_attn = compute_attn
        self.n_heads = n_heads
        self.use_resize_conv = use_resize_conv
        self.norm_kind = norm
        self.gn_groups = gn_groups
        
        # -------------------------------
        # (A) Upsampling path
        # -------------------------------
        if self.use_resize_conv:
            # resize -> conv_up keeps channels the same
            self.upsample = nn.Upsample(scale_factor = self.upsample_scale, mode="bilinear", align_corners=False)
            self.conv_up = nn.Conv2d(self.input_channels, self.input_channels, kernel_size=3, padding=1, bias=True)
            mid_ch = self.input_channels
        else:
            # deconv path (kept for ablation/toggling purposes)
            self.transpose = nn.ConvTranspose2d(
                self.input_channels, self.input_channels, 
                kernel_size=self.upsample_scale, stride=self.upsample_scale)
            mid_ch = self.transpose.out_channels # (= input_channels with current setup)

        # -------------------------------
        # (B) Normalization helpers
        # -------------------------------
        def make_norm(c: int):
            if self.norm_kind == "group":
                return nn.GroupNorm(num_groups=max(1, min(self.gn_groups, c)), num_channels=c)
            return nn.InstanceNorm2d(c)
        
        # Norm before main 3x3 (acts on mid_ch)
        self.norm1 = make_norm(mid_ch)

        # Main 3x3 conv to reach the block's output width
        self.conv = nn.Conv2d(mid_ch, self.output_channels, kernel_size=3, padding=1)

        # Norm after main conv
        self.norm2 = make_norm(self.conv.out_channels)

        # Activation 
        self.activation = activation()

        # -------------------------------
        # (C) Time embedding path (FiLM)
        # -------------------------------
        self.sinusoidal_embedding = SinusoidalEmbedding(self.time_embedding)
        # Produce gamma || beta for FiLM (2 * output_channels)
        self.time_projection_layer = nn.Sequential(
                nn.SiLU(),
                nn.Linear(self.time_embedding, 2 * self.output_channels)
            )
        
        # -------------------------------
        # (D) Attention layer
        # -------------------------------
        if self.compute_attn:
            self.attention = ImageSelfAttention(self.output_channels, self.n_heads)
        else:
            self.attention = nn.Identity()

    
    def forward(self,
                fmap:torch.Tensor,
                prev_fmap:Optional[torch.Tensor]=None,
                t:Optional[torch.Tensor]=None
                ):
        '''
            Args:
                fmap:       [B, C_in, H, W] incoming feature map
                prev_fmap:  optional skip/residual to add after the main conv (must match output shape)
                t:          timestep; either scalar indices [B] to be sinusoidally embedded here,
                            or a precomputed embedding [B, time_dim]

            Returns:
                [B, C_out, H*ups, W*ups]
        '''

        # === Pick the right norm attributes regardless of init version ===
        n1 = getattr(self, "norm1", getattr(self, "instance_norm1", None))
        n2 = getattr(self, "norm2", getattr(self, "instance_norm2", None))
        if n1 is None or n2 is None:
            raise ValueError("Norm layers not found; possible init error.")
        
        # === Upsample path (resize-congv or deconv) ===
        if getattr(self, "use_resize_conv", True):
            x = self.upsample(fmap) # bilinear upsample
            x = self.conv_up(x)     # 3x3 conv
            x = n1(x)               # norm on mid_ch
        else:
            x = self.transpose(fmap) # ConvTranspose2d upsample (OLD, kept for ablation/toggling purposes)
            x = n1(x)

        # === Main conv to output width + norm ===
        x = self.conv(x)        # [B, C_out, H*ups, W*ups]
        x = n2(x)               # norm on output

        # === Add previous feature map (residual/skip) if given ===
        if prev_fmap is not None and torch.is_tensor(prev_fmap):
            if prev_fmap.shape != x.shape:
                raise AssertionError(f"prev_fmap shape {prev_fmap.shape} must match output shape {tuple(x.shape)}")
            if prev_fmap.device != x.device:
                prev_fmap = prev_fmap.to(x.device)
            x = x + prev_fmap

        # === FiLM modulation from time embedding ===
        if t is not None:
            # Accept either raw timesteps [B] or pre-embedded [B, time_dim]
            if t.dim() == 1 or (t.dim() == 2 and t.shape[-1] != getattr(self, "time_embedding", self.time_embedding)):
                t_emb = self.sinusoidal_embedding(t.view(-1))  # [B, time_dim]
            else:
                t_emb = t
            gb = self.time_projection_layer(t_emb)  # [B, 2*C_out]
            C = x.shape[1]
            if gb.shape[1] != 2 * C:
                raise ValueError(f"Time projection output {tuple(gb.shape)} must be [B,{2*C}] to match x with {C} channels.")
            gamma, beta = torch.chunk(gb, 2, dim=1)      # each [B, C_out]
            gamma = gamma.unsqueeze(-1).unsqueeze(-1)    # [B, C_out, 1, 1]
            beta  = beta.unsqueeze(-1).unsqueeze(-1)     # [B, C_out, 1, 1]
            x = x * (1.0 + gamma) + beta
            # t_proj = self.time_projection_layer(t_emb).unsqueeze(-1).unsqueeze(-1)  # [B, C_out, 1, 1]
            # if t_proj.device != x.device:
            #     t_proj = t_proj.to(x.device)
            # x = x + t_proj  # broadcast add

        # === Non-linearity ===
        x = self.activation(x)

        # === Optional spatial self-attention (residual) ===
        attn = getattr(self, "attention", None)
        if attn is not None:
            # NEW: single-added attention (pre-norm)
            x = attn(x)
            
            # OLD: double-added attention (pre-norm + post-norm)
            # attn_out = attn(x)
            # x = x + attn_out.to(self.device) if attn_out is not None else x # 

        return x



class Decoder(nn.Module):
    '''
        Class for the decoder. The decoder is used to decode the encoded input.
        The decoder is a UNET with self-attention layers, and will be part of a UNET used for downscaling in the DDPM.
        The decoder consists of five feature maps, one for each layer of the UNET.
        The decoder works as an upsample block, and will be used to upsample the input.
    '''
    def __init__(self,
                 last_fmap_channels:int,
                 output_channels:int,
                 time_embedding:int,
                 first_fmap_channels:int=64,
                 n_heads:int=4,
                 device = None,
                 *,
                 use_resize_conv: bool = True,
                 norm: str = "instance", # "instance" | "group"
                 gn_groups: int = 8,
                 activation: type[nn.Module] = nn.ReLU
                 ):
        '''
            Initialize the class. 
            Input:
                - last_fmap_channels: number of channels for the last feature map
                - output_channels: number of output channels
                - time_embedding: size of the time embedding
                - first_fmap_channels: number of channels for the first feature map (default: 64)
                - n_heads: number of heads for the self-attention layers (default: 4, meaning 4 heads for each self-attention layer)
        '''

        # Initialize the class
        super().__init__()

        # Set the device
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Initialize the class variables
        self.last_fmap_channels = last_fmap_channels
        self.output_channels = output_channels
        self.time_embedding = time_embedding
        self.first_fmap_channels = first_fmap_channels
        self.n_heads = n_heads
        self.use_resize_conv = use_resize_conv
        self.norm = norm
        self.gn_groups = gn_groups
        self.activation = activation

        # Initialize the residual layers (four residual layers)
        self.residual_layers = self.make_layers()

        # Initialize the final layer: no attention, identity activation (if no extra non-linearity is wanted at the end)
        self.final_layer = DecoderBlock(
            self.residual_layers[-1].input_channels,
            self.output_channels,
            time_embedding=self.time_embedding,
            activation=nn.Identity,
            compute_attn=False, # No computation of attention on final layer
            n_heads=self.n_heads,
            device=self.device,
            use_resize_conv=self.use_resize_conv,
            norm=self.norm,
            gn_groups=self.gn_groups,
            )
        # After creating final layer, make sure no activation or normalization (otherwise mode ljust learns zero mean/unit variance)
        if hasattr(self.final_layer, "norm1"): 
            self.final_layer.norm1 = nn.Identity() # type: ignore
        if hasattr(self.final_layer, "norm2"):
            self.final_layer.norm2 = nn.Identity() # type: ignore
        self.final_layer.activation = nn.Identity() 


    def forward(self, *fmaps, t:Optional[torch.Tensor]=None):
        '''
            Forward function for the class.
            Input:
                - fmaps: feature maps
                - t: time embedding tensor
        '''
        # Expect n+1 feature maps if you built n residual layers
        assert len(fmaps) == len(self.residual_layers) + 1, f"Decoder expected {len(self.residual_layers)+1} feature maps, got {len(fmaps)}"

        # Reverse the feature maps in a list, fmaps(reversed): fmap5, fmap4, fmap3, fmap2, fmap1
        fmaps = list(reversed(fmaps))
        output = None

        # Loop over the residual layers
        for idx, block in enumerate(self.residual_layers):
            if idx == 0:
                # If idx is 0, the first residual layer is used.
                output = block(fmaps[idx], fmaps[idx+1], t)
            else:
                # If idx is not 0, the other residual layers are used.
                output = block(output, fmaps[idx+1], t)
        # No previous fmap is passed to the final decoder block
        # and no attention is computed
        output = self.final_layer(output, t=t)
        return output

      
    def make_layers(self, n:int=4):
        '''
            Function for making the residual layers. 
            Input:
                - n: number of residual layers (default: 4)
        '''
        # Initialize the residual layers
        layers = []

        # Loop over the number of residual layers
        for i in range(n):
            in_ch = self.last_fmap_channels if i == 0 else layers[i-1].output_channels
            out_ch = in_ch // 2 if i != (n-1) else self.first_fmap_channels

            # Initialize the residual layer as a decoder block
            layers.append(DecoderBlock(
                in_ch,
                out_ch, 
                time_embedding=self.time_embedding,
                compute_attn=(i < 2), # Attention only on first two decoders (i.e. lowest spatial resolutions, closest to bottleneck). Adding attention to larger maps is very expensive and cause instability in training.
                n_heads=self.n_heads,
                device=self.device,
                use_resize_conv=self.use_resize_conv,
                norm=self.norm,
                gn_groups=self.gn_groups,
                activation=self.activation
            ))

        return nn.ModuleList(layers)


class ScoreNet(nn.Module):
    '''
        Class for the diffusion net. The diffusion net is used to encode and decode the input.
        UNet-based score model: encoder-decoder with time conditioning
        Assumes VE-SDE (Variance Exploding SDE) training target; for EDM preconditioning see note below.
    '''
    def __init__(self,
                 marginal_prob_std, # callable: t->[B], matches VE schedule
                 encoder: nn.Module,
                 decoder: nn.Module,
                 device = None,
                 debug_pre_sigma_div: bool = True
                 ):
        '''
            Initialize the class.
            Input:
                - marginal_prob_std: marginal probability standard deviation (for Score-Based Generative Modeling)
                - encoder: encoder module
                - decoder: decoder module
        '''
        # Initialize the class
        super().__init__()

        # Set the device
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                
        # Set the marginal probability standard deviation
        self.marginal_prob_std = marginal_prob_std
        # Set the encoder and decoder modules
        self.encoder = encoder
        self.decoder = decoder
        self.debug_pre_sigma_div = debug_pre_sigma_div

        # Move everything to the device, single move for whole model tree
        self.to(self.device)

    
    def forward(self,
                x:torch.Tensor,
                t:torch.Tensor,
                y:Optional[torch.Tensor]=None,
                cond_img:Optional[torch.Tensor]=None,
                lsm_cond:Optional[torch.Tensor]=None,
                topo_cond:Optional[torch.Tensor]=None
                ):
        '''
            Forward function for the class.
            Input:
                - x: input tensor
                - t: time embedding tensor 
                - y: label tensor
        '''
        # === Device/dtype alignment ===
        dev = x.device
        t = t.to(dev).float()
        if y is not None:
            # either categorical (long) or continuous (float)
            if y.dtype in (torch.long, torch.int64, torch.int32):
                y = y.to(dev).long()
            else:
                y = y.to(dev)
        if cond_img is not None:
            cond_img = cond_img.to(dev)
        if lsm_cond is not None:
            lsm_cond = lsm_cond.to(dev)
        if topo_cond is not None:
            topo_cond = topo_cond.to(dev)

        # === Encode ===
        # Expect encoder to return a list/tuple of feature maps for skip connections
        enc_fmaps = self.encoder(x, t, y=y, cond_img=cond_img, lsm_cond=lsm_cond, topo_cond=topo_cond)

        # Optional sanity check (helps catch shape/order mismatches early)
        # assert isinstance(enc_fmaps, (list, tuple)) and len(enc_fmaps) >= 2, "Encoder must return a list/tuple of >=2 feature maps"

        # === Decode with season-aware FiLM ===
        time_dim = getattr(self.encoder, "time_embedding", 128)
        # Build t_emb (if not already embedded)
        if t.ndim == 2 and t.shape[1] == time_dim:
            t_emb = t
        else:
            t_emb = self.encoder.sinusoidal_embedding(t.view(-1))

        # Reuse the encoder's embedding heads for consistency
        def _combine_with_labels(t_in: torch.Tensor, y_in: torch.Tensor | None) -> torch.Tensor:
            if y_in is None:
                return t_in
            if getattr(self.encoder, "num_classes", None) is not None and y_in.dtype in (torch.long, torch.int64, torch.int32):
                yi = y_in
                if yi.ndim == 2 and yi.shape[1] == 1:
                    yi = yi[:, 0]
                elif yi.ndim != 1:
                    raise ValueError(f"[ScoreNet] categorical y must be [B] or [B,1], got {tuple(y_in.shape)}")
                return t_in + self.encoder.label_emb(yi.to(t_in.device))
            if torch.is_floating_point(y_in):
                y2 = y_in.view(y_in.shape[0], -1)
                if y2.shape[1] != 2:
                    raise ValueError(f"[ScoreNet] expected DOY sin/cos with last dim==2, got {tuple(y2.shape)}")
                return t_in + self.encoder.temporal_emb(y2.to(t_in.device))
            return t_in

        t_comb = _combine_with_labels(t_emb, y)
        score = self.decoder(*enc_fmaps, t=t_comb)

        # === DEBUG: Distribution before sigma-division ===
        if getattr(self, "debug_pre_sigma_div", True):
            with torch.no_grad():
                pre_m = float(score.mean())
                pre_s = float(score.std())
                svals =self.marginal_prob_std(t)
                logger.info(f"[pre-σ-div] mean = {pre_m:.4g}, std = {pre_s:.4g}, σ ∈ [{svals.min():.4g}, {svals.max():.4g}]")

        # === VE-SDE normalization ===
        # NOTE: for EDM-style preconditioning, the score output would need to be scaled by a (t-dependent) factor here
        std = self.marginal_prob_std(t) # [B]
        score = score / std.view(-1, 1, 1, 1) # [B, C, H, W] broadcast div

        return score


def marginal_prob_std(t: torch.Tensor,
                      sigma: float,
                      eps: float = 1e-5,
                      ) -> torch.Tensor:
    """
        Safer marginal probability standard deviation function that ensures numerical stability.
        Computes the standard deviation of p_(0t)(x(t)|x(0)) for VE-SDE with lognormal variance schedule sigma^t.
        Sigma: base (>1), e.g. 25.0
        returns shape [B], dtype/device same as t
    """
    t = t.to(dtype=torch.float32, device=t.device)
    s = torch.tensor(sigma, dtype=t.dtype, device=t.device)
    # sigma^(2t) = exp(2t log(sigma))
    sigma_t_sq = torch.exp((2. * t) * torch.log(s))
    std = torch.sqrt((sigma_t_sq - 1.) / (2. * torch.log(s)))
    # small floor to avoid dicision blow ups when t ~ 0
    return torch.clamp(std, min=eps)


def diffusion_coeff(t, sigma, device = None):
    '''
        Function to compute the diffusion coefficient
        of our SDE.
        Input:
            - t: A vector of time steps
            - sigma: the sigma parameter in our SDE

        Returns:
            - The vector of diffusion coefficients
    '''

    diff_coeff = sigma**t
    diff_coeff = diff_coeff.to(t.device)
    return diff_coeff

sigma_marg =  25.0#@param {'type':'number'}
marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=sigma_marg)
diffusion_coeff_fn = functools.partial(diffusion_coeff, sigma=sigma_marg)

# def loss_fn(model,
#             x,
#             marginal_prob_std,
#             t_eps=1e-3, # to avoid dead gradients near t=0
#             y = None,
#             cond_img = None,
#             lsm_cond = None,
#             topo_cond = None,
#             sdf_cond = None):
#     '''
#         The loss function for training SBGM.

#         Input:
#             - model: A PyTorch model that represents a time-dependent Score Based model
#             - x: The input tensor (mini-batch of training data)
#             - marginal_prob_std: A function that gives the std of 
#                 the perturbation kernel
#             - eps: A small constant to avoid division by zero
#     '''
#     # Sample a random time step for each sample in the mini-batch
#     random_t = torch.rand(x.shape[0], device=x.device) * (1. - t_eps) + t_eps
#     # Sample a random noise vector for each sample in the mini-batch
#     z = torch.randn_like(x)
#     # Compute the std of the perturbation kernel at the random time step
#     std = marginal_prob_std(random_t)
#     # Perturb the input x with the random noise vector z
#     perturbed_x = x + std[:, None, None, None] * z

#     for name, arr in [('cond_img', cond_img), ('lsm_cond', lsm_cond), ('topo_cond', topo_cond), ('y', y)]:
#         if arr is not None and arr.shape[0] != x.shape[0]:
#             raise ValueError(f'Batch size mismatch: x={x.shape[0]}, {name}={arr.shape[0]}')
    
#     # Estimate the score at the perturbed input x and the random time step t
#     score = model(perturbed_x, random_t, y=y, cond_img=cond_img, lsm_cond=lsm_cond, topo_cond=topo_cond)
    
#     # logger.info('[Loss fn] Score computed.')

#     max_land_weight=1.0
#     min_sea_weight=0.5
#     if sdf_cond is not None:
#         sdf_weights = torch.sigmoid(sdf_cond) * (max_land_weight - min_sea_weight) + min_sea_weight
#         sdf_weights = sdf_weights.to(x.device)
#     else:
#         sdf_weights = torch.ones_like(x).to(x.device)
    
    
#     # Compute the loss
#     loss = torch.mean(torch.sum(sdf_weights * (score * std[:, None, None, None] + z)**2, dim=(1, 2, 3)))
#     return loss


