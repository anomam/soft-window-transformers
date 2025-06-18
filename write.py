# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "matplotlib>=3.10.3",
#     "numpy",
#     "pydantic>=2.11.7",
#     "pyyaml>=6.0.2",
#     "torch>=2.3",
#     "tqdm>=4.67.1",
#     "typer",
# ]
# ///

import datetime as dt
import math
from abc import ABC, abstractmethod
from contextlib import contextmanager
from enum import Enum
from pathlib import Path
from typing import Callable, Final, Iterator, NamedTuple, Optional, Self

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from pydantic import BaseModel
from torch import Tensor, nn
from torch.nn import functional as F
from tqdm import tqdm
from typer import Typer

# ---------------------------------------------------------------------------------------------------------------------------
# DATA & PROCESSING
# ---------------------------------------------------------------------------------------------------------------------------

DIR_ROOT = Path(__file__).parent
DIR_ASSETS = DIR_ROOT / "assets"
DIR_SOFT_WINDOW = DIR_ASSETS / "soft_window_ckpt/"

FP_VOCAB = DIR_ROOT / "vocab.txt"

N_MAX_LABEL_SIZE: Final[int] = 64


class GravesTokenizer:
    """One-hot encoder for text characters"""

    def __init__(self) -> None:
        with FP_VOCAB.open("r") as f:
            chars = f.read().split("\n")
            chars = [char for char in chars if char]  # order matters
        pad_char = "\x00"
        self.chars = [pad_char] + chars
        self.pad_id = 0
        self.char_to_id = {char: idx for idx, char in enumerate(self.chars)}
        self.vocab_size = len(self.chars)

    def tokenize(self, text: str) -> list[int]:
        return [self.char_to_id[char] for char in text]  # + [self.pad_id]

    def tokenize_batch(self, strings: list[str], seq_len: int) -> np.ndarray:
        """Will use padding"""
        if not strings:
            return np.array([])
        batch = self.pad_id * np.ones((len(strings), seq_len), dtype=np.int8)
        for i, string in enumerate(strings):
            enc = self.tokenize(string)
            enc = enc[:seq_len]
            batch[i, : len(enc)] = np.array(enc, dtype=np.int8)
        return batch


class SequenceData(NamedTuple):  # (B, T, ...)
    vals: Tensor
    lengths: Tensor


def offsets_to_coords(offsets: np.ndarray) -> np.ndarray:
    """
    convert from offsets to coordinates
    """
    return np.concatenate([np.cumsum(offsets[:, :2], axis=0), offsets[:, 2:3]], axis=1)


def _split_coords_to_strokes(coords: np.ndarray) -> list[np.ndarray]:
    # Find indices where the split should occur
    split_points = np.where(coords[:, :, -1] == 1)[1]

    # Add start and end indices
    all_points = np.concatenate(([0], split_points + 1, [len(coords)]))

    # Create splits using the indices
    splits = [coords[0, all_points[i] : all_points[i + 1], :] for i in range(len(all_points) - 1)]
    return splits


# ---------------------------------------------------------------------------------------------------------------------------
# PLOTTING
# ---------------------------------------------------------------------------------------------------------------------------


class Style(BaseModel):
    rainbow: bool = True
    lw: int = 2


def plot_coords(
    coords: np.ndarray,
    label: Optional[str] = None,
    style: Style = Style(),
    ax: Optional[Axes] = None,
    highlight_mask: Optional[np.ndarray] = None,
) -> Axes:
    """
    Expected dimensions for coords: (1, time, strokes)
    """
    if not ax:
        _, ax = plt.subplots()
    splits = _split_coords_to_strokes(coords)
    for split in splits:
        color = np.random.rand(3).tolist() if style.rainbow else "k"
        ax.plot(split[:, 0], -split[:, 1], color=color, linewidth=style.lw)
    if label is not None:
        ax.set_title(label)

    # plot highlight: will highlight strokes that have been flagged in strokeset
    # eg: can find filtered out strokes using train set limit filter
    # see test named: `test_train_ds_limit_filter`
    if highlight_mask is not None:
        bool_mask = highlight_mask.squeeze().astype(bool)
        for idx in np.where(bool_mask)[0]:
            current = coords[0, idx, :2]
            prev = coords[0, idx - 1, :2]
            x, y = zip(prev, current)
            ax.plot(x, -np.array(y), color="r", linewidth=3)

    return ax


# ---------------------------------------------------------------------------------------------------------------------------
# Mixture Density Network Layer
# ---------------------------------------------------------------------------------------------------------------------------


class MDNInter(NamedTuple):
    log_pi: Tensor  # (B, T, G), from log_softmax
    log_sigma: Tensor  # (B, T, G, Dout)
    mu: Tensor  # (B, T, G, Dout)
    rho: Optional[Tensor] = None  # (B, T, G), in [-1, 1]

    def get_last_step(self) -> "MDNInter":
        return MDNInter(*[val[:, -1:] if val is not None else None for val in self])  # pyright: ignore [reportArgumentType]


class MDNLayer(nn.Module):
    """Mixture Density Network layer

    Attributes:
        in_dim (int): the number of dimensions in the input, eg: out of the RNN, hidden size
        out_dim (int): the number of dimensions in the output, 2 if handwriting
        num_gaussians (int): the number of mixture component
        dim_wise (bool): whether to model data for each dimension separately
    """

    def __init__(
        self, din: int, dout: int, num_gaussians: int = 30, dim_wise: bool = False, with_rho: bool = False
    ) -> None:
        super(MDNLayer, self).__init__()
        self.in_dim = din
        self.out_dim = dout
        self.num_gaussians = num_gaussians
        self.dim_wise = dim_wise

        odim_log_pi = dout * num_gaussians if dim_wise else num_gaussians
        self.log_pi = nn.Linear(din, odim_log_pi)

        self.log_sigma = nn.Linear(din, dout * num_gaussians)
        self.mu = nn.Linear(din, dout * num_gaussians)
        self.with_rho = with_rho
        if with_rho:
            self.rho = nn.Linear(din, num_gaussians)

    def forward(self, minibatch: Tensor, bias: float = 0.0) -> MDNInter:
        """Forward for MDN

        Args:
            minibatch (torch.Tensor): tensor of shape (B, T, D_in)
                B is the batch size and T is data lengths of this batch,
                and D_in is in_dim.
            bias (float): used for bias sampling

        Returns:
            torch.Tensor: Tensor of shape (B, T, G) or (B, T, G, D_out)
                Log of mixture weights. G is num_gaussians and D_out is out_dim.
            torch.Tensor: Tensor of shape (B, T, G, D_out)
                the log of standard deviation of each Gaussians.
            torch.Tensor: Tensor of shape (B, T, G, D_out)
                mean of each Gaussians
        """
        B = len(minibatch)
        if self.dim_wise:
            # (B, T, G, D_out)
            log_pi = self.log_pi(minibatch).view(B, -1, self.num_gaussians, self.out_dim)
            log_pi = F.log_softmax(log_pi * (1 + bias), dim=2)
        else:
            # (B, T, G)
            log_pi = F.log_softmax(self.log_pi(minibatch) * (1 + bias), dim=2)
        log_sigma = self.log_sigma(minibatch) - bias
        log_sigma = log_sigma.view(B, -1, self.num_gaussians, self.out_dim)
        mu = self.mu(minibatch)
        mu = mu.view(B, -1, self.num_gaussians, self.out_dim)
        rho = None
        if self.with_rho:
            rho = torch.tanh(self.rho(minibatch))  # (B, T, G)
        return MDNInter(log_pi, log_sigma, mu, rho)


def _to_one_hot(tensor: Tensor, n: int, fill_with: float = 1.0) -> Tensor:
    # we perform one hot encode with respect to the last axis
    one_hot = torch.FloatTensor(tensor.size() + (n,)).zero_()
    one_hot = one_hot.to(tensor.device)
    one_hot.scatter_(len(tensor.size()), tensor.unsqueeze(-1), fill_with)
    return one_hot


def _mdn_get_most_probable_sigma_and_mu_and_rho(
    mdn_inter: MDNInter, use_max: bool = False
) -> tuple[Tensor, Tensor, Optional[Tensor]]:
    """Return the mean and standard deviation of the Gaussian component
    whose weight coefficient is the largest as the most probable predictions.

    Args:
        mdn_inter:
            log_pi (torch.Tensor): Tensor of shape (B, T, G) or (B, T, G, D_out)
                The log of multinomial distribution of the Gaussians.
                B is the batch size, T is data length of this batch,
                G is num_gaussians of class MDNLayer.
            log_sigma (torch.Tensor): Tensor of shape (B, T, G, D_out)
                The standard deviation of the Gaussians. D_out is out_dim of class
                MDNLayer.
            mu (torch.Tensor): Tensor of shape (B, T, G, D_out)
                The means of the Gaussians. D_out is out_dim of class MDNLayer.

    Returns:
        tuple: tuple of torch.Tensor
            torch.Tensor of shape (B, T, D_out). The standardd deviations
            of the most probable Gaussian component.
            torch.Tensor of shape (B, T, D_out). Means of the Gaussians.
    """
    log_pi, log_sigma, mu, rho = mdn_inter
    max_rho = None

    dim_wise = len(log_pi.shape) == 4
    B, T, num_gaussians, _ = mu.shape
    # Get the indexes of the largest log_pi
    if use_max:
        _, component = torch.max(log_pi, dim=2)  # (B, T) or (B, T, C_out)
    else:
        pi = torch.exp(log_pi).reshape(-1, num_gaussians)  # (B * T, G)
        component = torch.multinomial(pi, 1)
        component = component.reshape(B, T)

    # Convert max_component to one_hot manner
    # if dim_wise: (B, T, D_out) -> (B, T, D_out, G)
    # else: (B, T) -> (B, T, G)
    one_hot = _to_one_hot(component, num_gaussians)

    if rho is not None:
        # rho (B, T, G)
        max_rho = torch.sum(rho * one_hot, dim=2)  # (B, T)

    if dim_wise:
        # (B, T, G, D_out)
        one_hot = one_hot.transpose(2, 3)
        assert one_hot.shape == mu.shape
    else:
        # Expand the dim of one_hot as  (B, T, G) -> (B, T, G, d_out)
        one_hot = one_hot.unsqueeze(3).expand_as(mu)

    # Multiply one_hot and sum to get mean(mu) and standard deviation(sigma)
    # of the Gaussians whose weight coefficient(log_pi) is the largest.
    #  (B, T, G, d_out) -> (B, T, d_out)
    max_mu = torch.sum(mu * one_hot, dim=2)
    max_sigma = torch.exp(torch.sum(log_sigma * one_hot, dim=2))

    return (
        max_sigma,  # (B, T, Dout)
        max_mu,  # (B, T, Dout)
        max_rho,  # (B, T) or None
    )


def mdn_sample(outputs: MDNInter, use_max: bool = False) -> torch.Tensor:
    (
        sigmas,  # (B, T, 2)
        mus,  # (B, T, 2)
        rho,  # (B, T)
    ) = _mdn_get_most_probable_sigma_and_mu_and_rho(outputs, use_max=use_max)
    assert rho is not None

    B, T = rho.shape
    # Sample from the selected normal distributions
    # build covariance matrix
    cov = sigmas.new_zeros((B, T, 2, 2))
    cov[..., 0, 0], cov[..., 1, 1] = sigmas[..., 0].pow(2), sigmas[..., 1].pow(2)
    cov[..., 0, 1] = cov[..., 1, 0] = rho * sigmas[..., 0] * sigmas[..., 1]
    multiv_normal = torch.distributions.MultivariateNormal(mus, cov)
    # sample only once for each parameter set
    samples = multiv_normal.sample()
    return samples


# ---------------------------------------------------------------------------------------------------------------------------
# Torch modules
# ---------------------------------------------------------------------------------------------------------------------------


class LabelEncoder(nn.Module):
    """Following Graves implementation of soft window, the encoder will be just do 1 hot encoding"""

    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size

    def forward(self, tokens: Tensor) -> Tensor:
        return F.one_hot(tokens.long(), num_classes=self.vocab_size).float()  # (1, U, Dc)


class MultiHeadScaledDotProdAttention(nn.Module):
    """General multihead scaled dot product attention:
    can be used for cross attention, self attention, causal, ..."""

    def __init__(self, Dx: int, Dc: int, size_hidden: int, n_head: int) -> None:
        super().__init__()
        assert size_hidden % n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_kv = nn.Linear(Dc, 2 * size_hidden)
        self.c_q = nn.Linear(Dx, size_hidden)
        # output projection
        self.c_proj = nn.Linear(size_hidden, Dx)  # needed for residual connection
        self.n_head = n_head
        self.size_hidden = size_hidden

    def forward(self, x: Tensor, context: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """
        Assumes mask (1, 1, Tx, Tc)
        """
        B, Tx, _ = x.size()  # batch size, sequence length
        Tc = context.size(1)
        Dh = self.size_hidden  # embedding dim

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q = self.c_q(x)
        k, v = self.c_kv(context).split(self.size_hidden, dim=2)
        q = q.view(B, Tx, self.n_head, Dh // self.n_head).transpose(1, 2)  # (B, nh, Tx, hs)
        k = k.view(B, Tc, self.n_head, Dh // self.n_head).transpose(1, 2)  # (B, nh, Tc, hs)
        v = v.view(B, Tc, self.n_head, Dh // self.n_head).transpose(1, 2)  # (B, nh, Tc, hs)

        if mask is not None and mask.dtype != torch.bool:
            mask = mask.to(torch.bool)
        # attention layer
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        # re-assemble all head outputs side by side
        y = y.transpose(1, 2).contiguous().view(B, Tx, Dh)
        y = self.c_proj(y)  # output projection
        return y


def sequence_mask_tensor(lengths: Tensor, maxlen: Optional[int] = None) -> Tensor:
    """Create a mask from length values"""
    if maxlen is None:
        maxlen = int(lengths.max().item())
    return torch.arange(maxlen, device=lengths.device)[None, :] < lengths[:, None]


class SoftWindowAttention(nn.Module):
    def __init__(
        self,
        Dx: int,
        Datt: int,
        n_gaussians: int,
        n_head: int = 1,
        max_context_len: int = N_MAX_LABEL_SIZE,
    ) -> None:
        super().__init__()
        self.fc = nn.Linear(Dx, 3 * n_gaussians)  # for alpha, beta, kappa
        self.self_attn = MultiHeadScaledDotProdAttention(n_gaussians, n_gaussians, Datt, n_head)

        self.n_gaussians = n_gaussians
        self.max_context_len = max_context_len

    def forward(
        self,
        x: Tensor,  # (B, Tx, Dx)
        context: Tensor,  # (B, Tc, Dc)
        context_len: Tensor,  # (B, 1) - Context len within the batch
    ) -> tuple[Tensor, Tensor]:
        Tx = x.size(1)
        Tc = context.size(1)
        params = self.fc(x)  # (B, Tx, 3 * G)
        alpha_h, beta_h, kappa_h = torch.split(params, self.n_gaussians, dim=2)  # (B, Tx, G)
        alpha, beta = F.softplus(alpha_h), F.softplus(beta_h)  # (B, Tx, G)
        # from: https://github.com/sjvasquez/handwriting-synthesis/blob/master/rnn_cell.py#L88
        beta = torch.clamp(beta, min=0.01, max=torch.inf)  # (B, Tx, G)

        causal_mask = torch.tril(torch.ones(Tx, Tx, device=x.device)).view(1, 1, Tx, Tx)
        kappa_h = kappa_h + self.self_attn(kappa_h, kappa_h, mask=causal_mask)
        kappa = F.softplus(kappa_h) / 25.0  # (B, Tx, G), can use softplus instead of exp()
        kappa = torch.cumsum(kappa, dim=1)  # (B, Tx, G)

        phi = self.compute_attention_weights(alpha, beta, kappa, Tc)

        # missing attention sequence mask
        # see: https://github.com/sjvasquez/handwriting-synthesis/blob/master/rnn_cell.py#L98
        mask = sequence_mask_tensor(context_len.squeeze(1), maxlen=self.max_context_len)  # (B, C)
        w = torch.bmm(
            phi * mask.unsqueeze(1),  # (B, T, C)
            context,  # (B, C, Dc)
        )  # (B, T, Dc)
        return w, phi

    def compute_attention_weights(self, alpha: Tensor, beta: Tensor, k: Tensor, char_seq_size: int):
        alpha = alpha.unsqueeze(3).repeat(1, 1, 1, char_seq_size)  # (B, T, G) -> (B, T, G, C)
        beta = beta.unsqueeze(3).repeat(1, 1, 1, char_seq_size)  # (B, T, G) -> (B, T, G, C)
        k = k.unsqueeze(3).repeat(1, 1, 1, char_seq_size)  # (B, T, G) -> (B, T, G, C)
        u = torch.arange(char_seq_size, device=alpha.device)  # (C,)

        densities = alpha * torch.exp(-beta * (k - u).pow(2))  # (B, T, G, C)
        phi = densities.sum(dim=2)  # (B, T, C)
        return phi  # (B, T, C)


class NewGELU(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """

    def forward(self, x: Tensor) -> Tensor:
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


class CausalBlock(nn.Module):
    """an unassuming Transformer block"""

    def __init__(self, Dx: int, size_hidden: int, n_head: int, window_size: int) -> None:
        super().__init__()
        # --- causal self attention
        self.ln_1 = nn.LayerNorm(size_hidden)
        self.self_attn = MultiHeadScaledDotProdAttention(Dx, Dx, size_hidden, n_head)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(window_size, window_size, dtype=torch.bool)).view(1, 1, window_size, window_size),
        )
        # --- MLP
        self.ln_3 = nn.LayerNorm(size_hidden)
        self.mlp = nn.ModuleDict(
            dict(
                c_fc=nn.Linear(size_hidden, 4 * size_hidden),
                c_proj=nn.Linear(4 * size_hidden, size_hidden),
                act=NewGELU(),
            )
        )
        m = self.mlp
        self.mlpf = lambda x: m.c_proj(m.act(m.c_fc(x)))  # MLP forward

    def forward(
        self,
        x: Tensor,
        x_mask: Optional[Tensor],
        fn_xattn: Optional[Callable[[Tensor], Tensor]] = None,
    ) -> Tensor:
        x = self.ln_1(x)
        causal_mask: Tensor = self.causal_mask  # pyright: ignore [reportAssignmentType]
        if x_mask is not None:  # (B, 1, 1, Tx)
            Tx = x.size(1)
            causal_mask = causal_mask[:, :, :Tx, :Tx] & x_mask
        x = x + self.self_attn(x, x, mask=causal_mask)
        if fn_xattn:
            x = x + fn_xattn(x)
        x = x + self.mlpf(self.ln_3(x))
        return x


class HwDecoderTransformer(nn.Module):
    def __init__(
        self,
        Din: int,
        size_hidden: int,
        n_head: int,
        window_size: int,
        n_layers: int,
        Dout: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.window_size = window_size
        self.size_hidden = size_hidden
        self.Dout = Dout or size_hidden

        Dx = size_hidden  # block will take Dh as input dim
        self.transformer = nn.ModuleDict(
            dict(
                xte=nn.Linear(Din, size_hidden),
                xpe=nn.Embedding(window_size, size_hidden),
                h=nn.ModuleList([CausalBlock(Dx, size_hidden, n_head, window_size) for _ in range(n_layers)]),
                ln_f=nn.LayerNorm(size_hidden),
            )
        )
        self.fc = nn.Linear(size_hidden, self.Dout, bias=False)

    def forward(self, x: SequenceData) -> Tensor:
        x_vals = x.vals
        # --- causal masking - only valid sequence lengths
        Tx = x_vals.size(1)
        x_mask = sequence_mask_tensor(x.lengths.squeeze(1), maxlen=Tx)
        x_mask = x_mask.unsqueeze(1)  # (B, 1, Tx), n heads
        causal_mask = x_mask.unsqueeze(3) & x_mask.unsqueeze(2)  # (B, 1, Tx, 1) * (B, 1, 1, Tx) = (B, 1, Tx, Tx)
        # --- decoder
        device = x_vals.device
        t = x_vals.size(1)
        assert t <= self.window_size, f"Cannot forward sequence of length {t}, block size is only {self.window_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)  # shape (1, t)

        # forward the GPT model itself
        tok_emb = self.transformer.xte(x_vals)  # token embeddings of shape (b, t, h)
        pos_emb = self.transformer.xpe(pos)  # position embeddings of shape (1, t, h)
        x_vals = tok_emb + pos_emb

        for block in self.transformer.h:
            vals = (x_vals, causal_mask)
            x_vals = block(*vals)
        x_vals = self.transformer.ln_f(x_vals)
        logits = self.fc(x_vals)

        return logits

    def get_window_size(self) -> int:
        return self.window_size


class ParallelSWXfmrXAttn(nn.Module):
    """Parallel Soft Window Transformer cross attention."""

    def __init__(
        self,
        xfm_decoder: HwDecoderTransformer,
        Dc: int,
        Datt: int,
        n_gaussians: int = 10,
    ) -> None:
        super().__init__()
        self.Dw = Dc
        self.Dout = self.Dw + xfm_decoder.Dout
        self.xfm_decoder = xfm_decoder
        self.xattn = SoftWindowAttention(xfm_decoder.size_hidden, Datt, n_gaussians)

    @classmethod
    def from_params(
        cls,
        Din: int,
        Dc: int,
        Datt: int,
        size_hidden: int,
        n_head: int,
        window_size: int,
        n_layers: int = 1,
        n_gaussians: int = 10,
    ) -> Self:
        xfm_decoder = HwDecoderTransformer(Din, size_hidden, n_head, window_size, n_layers, Dout=size_hidden)
        return cls(xfm_decoder, Dc, Datt, n_gaussians)

    def forward(self, x: SequenceData, context: SequenceData) -> tuple[Tensor, Tensor]:
        """
        x: Tensor,  # (B, Tx, Dx) - Input stroke features
        context: Tensor,  # (B, Tc, Dc) - Encoded context

        and lengths of dim (B, 1)
        """
        out = self.xfm_decoder(x)  # out <B, Tx, H>
        w, phi = self.xattn(out, context.vals, context.lengths)  # w <B, Tx, Dw>, phi <B, Tx, Tc>
        out = torch.cat([out, w], dim=2)
        return out, phi

    def get_window_size(self) -> int:
        return self.xfm_decoder.get_window_size()

    @property
    def d_out(self) -> int:
        return self.Dout


class EncoderXfmDecoder(nn.Module):
    def __init__(self, encoder: LabelEncoder, xattn: ParallelSWXfmrXAttn) -> None:
        super().__init__()
        self.encoder = encoder
        self.xattn = xattn
        self.d_out = self.xattn.d_out

    def forward(self, x: SequenceData, label: SequenceData) -> tuple[Tensor, Tensor | None]:
        context = self.encoder(label.vals)
        out, attn_weights = self.xattn(x, SequenceData(context, label.lengths))
        return out, attn_weights

    def get_window_size(self) -> int:
        return self.xattn.get_window_size()


class HwOutputs(NamedTuple):
    mdn_outputs: MDNInter
    bce_outputs: Tensor


class HwXfmModule(nn.Module):
    def __init__(self, mdn_layer: MDNLayer, pred_xfm: HwDecoderTransformer, encdec: EncoderXfmDecoder) -> None:
        super().__init__()
        self.pred_xfm = pred_xfm
        self.encdec = encdec
        self.mdn_layer = mdn_layer

        self.window_size = -1
        assert (mod := self.pred_xfm) or (mod := self.encdec), "should provide predictor and/or encdec"
        if self.pred_xfm and self.encdec:
            # check that same window for pred & enc dec
            assert self.pred_xfm.get_window_size() == self.encdec.get_window_size()
        self.window_size = mod.get_window_size()

        # count params
        print("number of parameters: %.2fM" % (sum(p.numel() for p in self.parameters()) / 1e6,))

    def forward(self, x: SequenceData, label: SequenceData, bias: float = 0.0) -> tuple[HwOutputs, None]:
        # go through transformer(s)
        out, phi = self.encdec(x, label)
        x = SequenceData(out, x.lengths)
        out = self.pred_xfm(x)

        # split mdn & bce outputs
        mdn_params = out[..., :-1]
        mdn_inter = self.mdn_layer(mdn_params, bias=bias)
        bce_outputs = out[..., -1:]

        return HwOutputs(mdn_inter, bce_outputs), phi


class Config(BaseModel):
    window_size: int  # max time length of decoder inputs
    d_soft_window_attention: int  # dimension of attention space for kappa
    n_soft_window_gaussians: int
    decoder_hidden_size: int
    n_heads_first_layer: int
    n_layers_first_layer: int
    n_heads_after: int
    n_layers_after: int
    n_gaussians_mdn: int


def load_module(dir_dump: Path, vocab_size: int, use_fast: bool = False) -> HwXfmModule:
    def get_fastest() -> torch.device:
        if torch.cuda.is_available():
            # nvidia gpu
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            # apple silicon
            return torch.device("mps")
        else:
            return torch.device("cpu")

    fp_ckpt = dir_dump / "module.pt"
    fp_config = dir_dump / "config.yml"

    with fp_config.open("r") as f:
        cfg = Config(**yaml.safe_load(f))

    encoder = LabelEncoder(vocab_size)
    d_strokes = 3  # (x1, x2) position, (x3) pen raise
    d_mdn, d_bce = 2, 1  # 2 for positions, 1 for pen raise
    n_chars = encoder.vocab_size  # size of 1 hot vectors
    xfm_layer_1 = HwDecoderTransformer(
        d_strokes, cfg.decoder_hidden_size, cfg.n_heads_first_layer, cfg.window_size, cfg.n_layers_first_layer
    )
    cross_attention = ParallelSWXfmrXAttn(
        xfm_layer_1, n_chars, cfg.d_soft_window_attention, cfg.n_soft_window_gaussians
    )
    softwindow_xfm = EncoderXfmDecoder(encoder, cross_attention)
    xfm_layers_after = HwDecoderTransformer(
        softwindow_xfm.d_out, cfg.decoder_hidden_size, cfg.n_heads_after, cfg.window_size, cfg.n_layers_after
    )
    mdn_layer = MDNLayer(xfm_layers_after.Dout - d_bce, d_mdn, cfg.n_gaussians_mdn, with_rho=True)
    module = HwXfmModule(mdn_layer, xfm_layers_after, softwindow_xfm)

    ckpt = torch.load(fp_ckpt, map_location=torch.device("cpu"))
    module.load_state_dict(ckpt["state_dict"], strict=True)

    if use_fast:
        device = get_fastest()
        module.to(device)

    return module


# ---------------------------------------------------------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------------------------------------------------------


@contextmanager
def eval_mode(est: Optional[torch.nn.Module] = None) -> Iterator[None]:
    """Not doing anything by default"""
    with torch.no_grad():
        prev_training = None
        if est:
            prev_training = est.training
            est.eval()
        yield
        if est and (prev_training is True):
            est.train()


def get_device(est: torch.nn.Module) -> torch.device:
    return next(est.parameters()).device


class BaseHandWriter(ABC):
    def __init__(self, tokenizer: GravesTokenizer) -> None:
        self._tokenizer = tokenizer

    @abstractmethod
    def write_offsets(self, seq_len: int, text: str, bias: float = 0.0) -> np.ndarray:
        raise NotImplementedError

    def write_and_plot(self, seq_len: int, text: str, bias: float = 0.0) -> tuple[Figure, Axes]:
        x = self.write_offsets(seq_len=seq_len, text=text, bias=bias)

        # plot
        f, ax = plt.subplots()
        coords = offsets_to_coords(x[0])
        plot_coords(coords[None], ax=ax)  # unsqueeze
        return f, ax

    def _init_x(self, batch_size: int, device: torch.device) -> SequenceData[Tensor]:
        x_len = torch.ones((batch_size, 1), device=device)
        return SequenceData(
            torch.cat([torch.zeros((batch_size, 1, 2)), torch.ones((batch_size, 1, 1))], dim=2).to(device), x_len
        )

    def _init_label(self, text: str | None, device: torch.device) -> SequenceData[Tensor]:
        if not text:
            return SequenceData(torch.tensor([], device=device), torch.zeros((1, 1), device=device))
        enc_text = torch.tensor(self._tokenizer.tokenize_batch([text], N_MAX_LABEL_SIZE), device=device)
        enc_len = torch.tensor([len(text)], device=device).unsqueeze(0)
        return SequenceData(enc_text, enc_len)

    def _predict_next_offset(self, params: HwOutputs) -> torch.Tensor:
        # sample pen lift
        bern_params = torch.sigmoid(params.bce_outputs[:, -1:, :])
        pen_raise = torch.bernoulli(bern_params)  # (B, 1, 1)
        # sample stroke
        x1x2 = mdn_sample(params.mdn_outputs.get_last_step())  # (B, 1, 2)
        out = torch.concatenate([x1x2, pen_raise], dim=-1)
        return out


class XfmHandwriter(BaseHandWriter):
    """
    Will build sequence by adding last prediction to original input 1 by 1
    """

    def __init__(
        self,
        est: HwXfmModule,
        tokenizer: GravesTokenizer,
        overlap: float = 0.5,
    ) -> None:
        self._tokenizer = tokenizer
        self._est = est
        self._overlap = overlap
        self._window_size = est.window_size

    def write_offsets(self, seq_len: int, text: str, bias: float = 0.0) -> np.ndarray:
        batch_size = 1
        with eval_mode(self._est):
            # init
            device = get_device(self._est)
            label = self._init_label(text, device)
            x_init = self._init_x(batch_size, device)
            length_one = x_init.lengths

            # loop
            inputs = (x_init, label)
            all_xs = torch.zeros(batch_size, seq_len, 3, dtype=torch.float32, device=device)
            all_xs[:, :1, :] = x_init.vals
            for t in tqdm(range(1, seq_len), desc="generate", leave=False):
                # calculate offset for time t
                next_params, attn_weights = self._est(*inputs, bias=bias)
                x_vals = self._predict_next_offset(next_params)
                all_xs[:, t : t + 1, :] = x_vals

                if self._terminate(attn_weights, label):
                    break
                window_len = min(t + 1, self._window_size)
                next_x = SequenceData(all_xs[:, t + 1 - window_len : t + 1, :], window_len * length_one)
                inputs = (next_x, label)

        return all_xs.detach().cpu().numpy()

    def _predict_next_offset(self, params: HwOutputs) -> torch.Tensor:
        # sample pen lift
        bern_params = torch.sigmoid(params.bce_outputs[:, -1:, :])
        pen_raise = torch.bernoulli(bern_params)  # (B, 1, 1)
        # sample stroke
        x1x2 = mdn_sample(params.mdn_outputs.get_last_step())  # (B, 1, 2)
        out = torch.concatenate([x1x2, pen_raise], dim=-1)
        return out

    def _terminate(self, attn_weights: Tensor, label: SequenceData) -> bool:
        threshold = 0.95
        str_len = int(label.lengths[0].item())
        w = attn_weights[0, -1, :]  # (0, -1, U)
        max_w, _ = w.max(0)
        last_w = w[str_len + 1]  # otherwise use string length
        ratio = last_w / max_w
        is_done = ratio > threshold
        return bool(is_done.item())


# ---------------------------------------------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------------------------------------------

APP = Typer(name="swt")


class TypeModel(str, Enum):
    soft_window = "soft-window"


@APP.command()
def handwrite(
    text: str,
    bias: float = 15.0,
    seq_len: int = 800,
    seed: Optional[int] = None,
    model_type: TypeModel = TypeModel.soft_window,
    show: bool = False,
) -> None:
    # seeding
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
    # load module
    match model_type:
        case TypeModel.soft_window:
            dir_dump = DIR_SOFT_WINDOW
    tokenizer = GravesTokenizer()
    module = load_module(dir_dump, tokenizer.vocab_size)
    # create writer
    writer = XfmHandwriter(module, tokenizer)
    f, ax = writer.write_and_plot(seq_len=seq_len, text=text, bias=bias)
    ax.axis('off')
    plt.show()
    f.savefig(f"{text}_{dt.datetime.now().isoformat()}.png")


if __name__ == "__main__":
    APP()
