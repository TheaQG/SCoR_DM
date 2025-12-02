import torch
import tqdm
import logging
import math

import numpy as np

from monitoring import tensor_stats
# Set up logging
logger = logging.getLogger(__name__)


# === EDM sampler (Karras et al., 2022) ===
@torch.no_grad()
def edm_sampler(score_model,
                batch_size: int,
                num_steps: int,
                device: torch.device | str,
                img_size: int,
                y=None,
                cond_img=None,
                lsm_cond=None,
                topo_cond=None,
                # EDM schedule defaults (can be overridden via cfg)
                sigma_min: float = 0.002,
                sigma_max: float = 80,
                rho: float = 7.0,
                S_churn: float = 0.0,
                S_min: float = 0.0,
                S_max: float = float('inf'),
                S_noise: float = 1.0,
                lr_ups: torch.Tensor | None = None,
                cfg_guidance: dict | None = None,
                cfg_diagnostics: dict | None = None,
                *,
                sigma_star: float = 1.0,
                # --- scale-aware control (late-step ramp) ---
                sigma_star_mode: str = "global",   # "global" or "late_ramp"
                ramp_start_frac: float = 0.60,     # start of ramp as fraction of steps (0..1)
                ramp_end_frac: float = 0.85,       # end of ramp as fraction of steps (0..1)
                ramp_start_sigma: float | None = None,  # optional: start ramp when sigma <= this
                ramp_end_sigma: float | None = None,    # optional: end ramp when sigma <= this
                ):
  """
      Karras EDM sampler with Heun updates.
      Expects score_model(x_t, sigma, cond_img=..., lsm_cond=..., topo_cond=..., y=..., lr_ups=...) -> x0_hat.
      Returns a tensor shaped like the model outputs (i.e. a sample batch, shape (B, C, H, W)).
  """
  # Move all conditional tensors to the correct device
  def to_dev(t): return None if t is None else t.to(device)
  cond_img, lsm_cond, topo_cond, y, lr_ups = map(to_dev, (cond_img, lsm_cond, topo_cond, y, lr_ups))
  
  if isinstance(cfg_guidance, dict) and cfg_guidance.get('enabled', False):
    cfg_enabled = True
    base_scale = float(cfg_guidance.get('guidance_scale', 0.0))
    null_label_id = int(cfg_guidance.get('null_label_id', 0))
    null_scalar = float(cfg_guidance.get('null_scalar_value', 0.0))
    null_geo_value = float(cfg_guidance.get('null_geo_value', -5.0))
    lr_null_strategy = str(cfg_guidance.get('null_lr_strategy', 'zero')).lower()
    lr_null_scalar = float(cfg_guidance.get('null_lr_scalar', 0.0))
  else:
    cfg_enabled = False
    base_scale = 0.0
    null_label_id = 0
    null_scalar = 0.0
    null_geo_value = -5.0
    lr_null_strategy = 'zero'
    lr_null_scalar = 0.0

  # Sampler logging
  if cfg_enabled and cfg_guidance is not None:
    logger.info(f"[sampler] CFG enabled: base_scale={base_scale}, sigma_weighted={bool(cfg_guidance.get('sigma_weighted', True))}, "
                f"drop_lr_ups_in_uncond={bool(cfg_guidance.get('drop_lr_ups_in_uncond', False))}")
  
  # New grid-sampler logging
  logger.info(f"[sampler] EDM sampler settings: sigma_min={sigma_min}, sigma_max={sigma_max}, rho={rho}, "
              f"S_churn={S_churn}, S_min={S_min}, S_max={S_max}, S_noise={S_noise}")

  logger.info(f"[sampler] sigma*: {sigma_star:.3f}, mode={sigma_star_mode}, "
              f"ramp_frac=({ramp_start_frac:.2f},{ramp_end_frac:.2f}), "
              f"ramp_sigma=({ramp_start_sigma},{ramp_end_sigma})")
  
  # Set the max guidance scale and make sure that base_scale <= gmax
  gmax = float(cfg_guidance.get('guidance_scale_max', base_scale)) if (cfg_enabled and cfg_guidance is not None) else base_scale
  base_scale = min(base_scale, gmax)


  def _make_null_y(y_in):
    if y_in is None:
      return None
    # float -> sin/cos DOY
    if y_in.dtype in (torch.float16, torch.float32, torch.float64):
      return torch.zeros_like(y_in).fill_(null_scalar)
    # int/long -> categorical (4, 12, 365 classes)
    return torch.full_like(y_in, null_label_id)
  def _null_lr_like(t):
      if t is None: return None
      if lr_null_strategy == 'noise':
          return torch.randn_like(t)
      if lr_null_strategy == 'scalar':
          return t.new_full(t.shape, lr_null_scalar)
      return torch.zeros_like(t)  # 'zero'

  def _null_geo_like(t):
      if t is None: return None
      if t.ndim >= 2 and t.shape[1] >= 2:
          out = t.clone()
          out[:, 0, ...] = null_geo_value
          out[:, 1, ...] = 0.0
          return out
      return t.new_full(t.shape, null_geo_value)
  
  device = torch.device(device)


  # Build nulls for unconditional branch
  null_img = _null_lr_like(cond_img) if (cfg_enabled and cond_img is not None) else None
  null_lsm = _null_geo_like(lsm_cond) if (cfg_enabled and lsm_cond is not None) else None
  null_topo = _null_geo_like(topo_cond) if (cfg_enabled and topo_cond is not None) else None
  null_y = _make_null_y(y) if (cfg_enabled and y is not None) else None

  drop_lr_ups_in_uncond = bool(cfg_guidance and cfg_guidance.get('drop_lr_ups_in_uncond', False))
  null_lr_ups = _null_lr_like(lr_ups) if (cfg_enabled and lr_ups is not None and drop_lr_ups_in_uncond) else lr_ups

  # Build Karras sigma (noise) schedule (decreasing)
  def get_sigmas_K(n_steps, s_min, s_max, rho_):
    i = torch.arange(n_steps, device=device, dtype=torch.float32) # 0, ..., n_steps-1
    ramp = i / max(n_steps - 1, 1) # in [0, 1]
    min_inv = s_min ** (1 / rho_) # Karras min
    max_inv = s_max ** (1 / rho_) # Karras max
    sig = (max_inv + ramp * (min_inv - max_inv)) ** rho_ # Karras sigma
    return sig
  
  def _smoothstep01(x: torch.Tensor) -> torch.Tensor:
      # clamp to [0,1] then apply smoothstep 3x^2 - 2x^3
      x = torch.clamp(x, 0.0, 1.0)
      return x * x * (3.0 - 2.0 * x)
    
  # Build base sigma schedule
  sigmas = get_sigmas_K(num_steps, float(sigma_min), float(sigma_max), float(rho))

  # === Scale-aware sigma_star inference knob ===
  if not torch.is_tensor(sigmas):
      sigmas = torch.tensor(sigmas, dtype=torch.float32, device=device)

  # Build per-step scale factors f_i
  if sigma_star_mode.lower() == "global":
      f = torch.full_like(sigmas, float(sigma_star))
  else:
      # Late-step ramp: start~end window in which the scale increases from 1 -> sigma_star
      N = int(sigmas.shape[0])

      if (ramp_start_sigma is not None) and (ramp_end_sigma is not None):
          # Use sigma thresholds (later steps have smaller sigma)
          if torch.any(sigmas <= float(ramp_start_sigma)):
              i0 = int(torch.nonzero(sigmas <= float(ramp_start_sigma), as_tuple=False).min().item())
          else:
              i0 = int(0.6 * (N - 1))
          if torch.any(sigmas <= float(ramp_end_sigma)):
              i1 = int(torch.nonzero(sigmas <= float(ramp_end_sigma), as_tuple=False).min().item())
          else:
              i1 = int(0.85 * (N - 1))
      else:
          # Fractional indices
          i0 = max(0, min(int(round(ramp_start_frac * (N - 1))), N - 1))
          i1 = max(i0, min(int(round(ramp_end_frac   * (N - 1))), N - 1))

      idx = torch.arange(N, device=device, dtype=torch.float32)
      denom = max(float(i1 - i0), 1.0)
      t = (idx - float(i0)) / denom
      w = _smoothstep01(t)
      w = torch.where(idx < i0, torch.zeros_like(w), w)
      w = torch.where(idx > i1, torch.ones_like(w), w)
      f = 1.0 + (float(sigma_star) - 1.0) * w

  # Apply per-step scaling
  sigmas = f * sigmas

  # Effective churn window:
  # - global: keep scaling bounds
  # - late_ramp: leave bounds in original units; scaled 'sigma' already controls entry.
  if sigma_star_mode.lower() == "global":
      S_min_eff = float(S_min) * float(sigma_star)
      S_max_eff = float(S_max) * float(sigma_star)
  else:
      S_min_eff = float(S_min)
      S_max_eff = float(S_max)

  # Append terminal sigma=0 step
  sigmas = torch.cat([sigmas, sigmas.new_zeros(1)]) # add sigma=0 for final step

  # Infer spatial shape
  shape_hint = None
  for tns in (cond_img, lsm_cond, topo_cond):
      if tns is not None:
          shape_hint = tns
          break
  if shape_hint is not None:
      _, _, H, W = shape_hint.shape
  else:
      H = W = int(img_size)

  # Infer channels from model if possible
  C_out = getattr(getattr(score_model, 'decoder', None), 'output_channels', None)
  if C_out is None:
    # Fallback: assume single-channel output
    C_out = 1

  B = int(batch_size)
  # Initial sample: Gaussian noise with sigma_max stddev
  x = torch.randn(B, C_out, H, W, device=device) * float(sigma_max)

  if lr_ups is not None and (lr_ups.shape[0] != x.shape[0] or lr_ups.shape[2:] != x.shape[2:]):
      raise ValueError(f"lr_ups shape {lr_ups.shape} does not match the expected batch size {x.shape[0]} and spatial shape {x.shape[2:]}")

  def _denoise_with_cfg(x_in, sigma_vec):
    if cfg_enabled and base_scale > 0.0:
      # Optional sigma weighting (weaker guidance a large noise)
      s = base_scale

      if isinstance(cfg_guidance, dict) and bool(cfg_guidance.get('sigma_weighted', True)):
        # Use inverse-sigma weighting; clamp to [0, base_scale]
        sig = float(sigma_vec[0]) if sigma_vec.ndim == 1 else float(sigma_vec)
        s = base_scale * min(1.0, float(sigma_min) / max(sig, 1e-5))
      # Unconditional 
      x0_uc = score_model(x_in,
                          sigma_vec,
                          cond_img=null_img,
                          lsm_cond=null_lsm,
                          topo_cond=null_topo,
                          y=null_y,
                          lr_ups=null_lr_ups)
      # Conditional
      x0_c = score_model(x_in,
                        sigma_vec,
                        cond_img=cond_img,
                        lsm_cond=lsm_cond,
                        topo_cond=topo_cond,
                        y=y,
                        lr_ups=lr_ups)
      # Linear combination
      denoised = x0_uc + s * (x0_c - x0_uc) # Use s = basescale or s = sigma-weighted scale 
    else:
      # No classifier-free guidance
      denoised = score_model(x_in,
                            sigma_vec,
                            cond_img=cond_img,
                            lsm_cond=lsm_cond,
                            topo_cond=topo_cond,
                            y=y,
                            lr_ups=lr_ups)
    return denoised

  diag = isinstance(cfg_diagnostics, dict) and cfg_diagnostics.get("per_batch_stats", False)
  diag_every = int(cfg_diagnostics.get("log_every", max(1, num_steps // 4))) if (diag and isinstance(cfg_diagnostics, dict)) else 0
  
  for i in range(num_steps):
    sigma = sigmas[i]
    sigma_next = sigmas[i + 1]

    x_in = x
    # Churn (stochasticity injection at high sigmas)
    if (S_min_eff <= float(sigma) <= S_max_eff) and (S_churn > 0):
      gamma = min(S_churn / num_steps, math.sqrt(2.0) - 1.0) # Stochasticity factor
      eps = torch.randn_like(x) * S_noise # Noise scaled by S_noise
      sigma_hat = sigma * (1 + gamma) # Increased sigma with stochasticity injection
      x_in = x + eps * torch.sqrt(sigma_hat**2 - sigma**2) # Perturb x_in to x_hat
    else:
      sigma_hat = sigma # No stochasticity injection

    sigma_hat_vec = torch.full((B,), float(sigma_hat), device=device, dtype=x.dtype) 
    # (Optional safety) Check lr_ups shape
    if lr_ups is not None:
      assert lr_ups.shape[0] == B and lr_ups.shape[2:] == (H, W), f"lr_ups shape {lr_ups.shape} does not match the expected batch size {B} and spatial shape {(H, W)}"

    denoised = _denoise_with_cfg(x_in, sigma_hat_vec)

    d = (x_in - denoised) / sigma_hat # Score-based derivative

    # Euler step 
    x_euler = x_in + (sigma_next - sigma_hat) * d
    if i == num_steps - 1:
      x = x_euler
      break

    # Heun correction (2nd order Runge-Kutta)
    sigma_next_vec = torch.full((B,), float(sigma_next), device=device, dtype=x.dtype)
    denoised_next = _denoise_with_cfg(x_euler, sigma_next_vec)
    d_next = (x_euler - denoised_next) / sigma_next

    x = x_in + (sigma_next - sigma_hat) * 0.5 * (d + d_next)


    # Diagnostics logging
    if diag and (i % diag_every == 0 or i in (0, num_steps - 1)):
      tensor_stats(x_in, f"sampling/iter{i:03d}/x_in")
      if lr_ups is not None:
        tensor_stats(lr_ups, f"sampling/iter{i:03d}/lr_hr_norm")
      tensor_stats(denoised, f"sampling/iter{i:03d}/x0_hat")
  
  if diag:
    tensor_stats(x, "sampling/final_x")

  return x































def guided_score_fn(score_model,
                    x,                        # (B, C, H, W)    noisy sample
                    t,                        # (B,)            time/sigma
                    y=None,                   # (B,)            season/class index
                    cond_img=None,            # (B, C_lr, H, W) conditionals
                    lsm_cond=None,            # (B, 2, H,W)     value||mask, 2-channels
                    topo_cond=None,           # (B, 2, H,W)     value||mask, 2-channels
                    null_token: int = 0,      # NULL index used at train-time
                    scale: float = 2.0,):     # guidance weight
  '''
    Classifier-free guidance wrapper that:
    - Keeps geo *values* channel unchanged
    - Zeroes only the mask channel for the unconditional branch
    - Uses the correct NULL class token
  '''
  # --------------------------------------------------------------------------------------
  # 1) Build unconditional (dropped) inputs
  # --------------------------------------------------------------------------------------
  null_cond_img = torch.zeros_like(cond_img) if cond_img is not None else None

  def strip_mask(tensor):
    """Set mask channel (idx=1) to zero, leaving the value channel (idx=0) unchanged."""
    if tensor is None or tensor.shape[1] != 2:
      return tensor
    
    t_null = tensor.clone()
    t_null[:, 1, :, :] = 0.0    # mask -> 0, value unchanged
    return t_null

  null_lsm = strip_mask(lsm_cond)
  null_topo = strip_mask(topo_cond)

  null_y = torch.full_like(y, null_token) if y is not None else None

  # --------------------------------------------------------------------------------------
  # 2) Forward passes
  # --------------------------------------------------------------------------------------

  # Compute the score for the conditional and unconditional cases.
  score_cond = score_model(x, t, y, cond_img, lsm_cond, topo_cond)
  score_uncond = score_model(x, t, null_y, null_cond_img, null_lsm, null_topo)

  # --------------------------------------------------------------------------------------
  # 3) Linear combination (1+w)s_c - 2*s_u
  # --------------------------------------------------------------------------------------
  guided_score = (1.0 + scale) * score_cond - scale * score_uncond
  return guided_score


#@title Define the Euler-Maruyama sampler (double click to expand or collapse)

## The number of sampling steps.
num_steps =  500#@param {'type':'integer'}
def Euler_Maruyama_sampler(score_model, 
                           marginal_prob_std,
                           diffusion_coeff, 
                           batch_size=64, 
                           num_steps=num_steps, 
                           device='cuda', 
                           eps=1e-3,
                           img_size=64,
                           y=None,
                           cond_img=None,
                           lsm_cond=None,
                           topo_cond=None,
                           cfg=None
                           ):
  """Generate samples from score-based models with the Euler-Maruyama solver.

  Args:
    score_model: A PyTorch model that represents the time-dependent score-based model.
    marginal_prob_std: A function that gives the standard deviation of
      the perturbation kernel.
    diffusion_coeff: A function that gives the diffusion coefficient of the SDE.
    batch_size: The number of samplers to generate by calling this function once.
    num_steps: The number of sampling steps. 
      Equivalent to the number of discretized time steps.
    device: 'cuda' for running on GPUs, and 'cpu' for running on CPUs.
    eps: The smallest time step for numerical stability.
  
  Returns:
    Samples.    
  """
  t = torch.ones(batch_size, device=device)
  init_x = torch.randn(batch_size, 1, img_size, img_size, device=device) * marginal_prob_std(t)[:, None, None, None]
  time_steps = torch.linspace(1., eps, num_steps, device=device)
  step_size = time_steps[0] - time_steps[1]
  x = init_x
  mean_x = x  # Initialize mean_x to ensure it is always defined
  with torch.no_grad():
    for time_step in tqdm.tqdm(time_steps):
      batch_time_step = torch.ones(batch_size, device=device) * time_step
      g = diffusion_coeff(batch_time_step)

      if cfg is None:
        cfg = {}
      if cfg.get('classifier_free_guidance', {}).get('enabled', False):
        # If classifier-free guidance is enabled, use the guided score function.
        scale = cfg['classifier_free_guidance'].get('guidance_scale', 2.0)
        score = guided_score_fn(score_model,
                                x,
                                batch_time_step,
                                y,
                                cond_img,
                                lsm_cond,
                                topo_cond,
                                scale=scale,
                                null_token = 0)
      else:
        # Else, use the standard score model (cheaper computation).
        score = score_model(x, batch_time_step, y, cond_img, lsm_cond, topo_cond)

      # Update the mean_x with the score model output.
      mean_x = x + (g**2)[:, None, None, None] * score * step_size
      x = mean_x + torch.sqrt(step_size) * g[:, None, None, None] * torch.randn_like(x)      
  # Do not include any noise in the last sampling step.
  return mean_x


#@title Define the Predictor-Corrector sampler (double click to expand or collapse)

signal_to_noise_ratio = 0.16 #@param {'type':'number'}

## The number of sampling steps.
num_steps =  800#@param {'type':'integer'}
def pc_sampler(score_model, 
               marginal_prob_std,
               diffusion_coeff,
               batch_size=64, 
               num_steps=num_steps, 
               snr=signal_to_noise_ratio,                
               device='cuda',
               eps=1e-3,
               img_size=64,
               y=None,
               cond_img=None,
               lsm_cond=None,
               topo_cond=None,
               cfg=None):
  """Generate samples from score-based models with Predictor-Corrector method.

  Args:
    score_model: A PyTorch model that represents the time-dependent score-based model.
    marginal_prob_std: A function that gives the standard deviation
      of the perturbation kernel.
    diffusion_coeff: A function that gives the diffusion coefficient 
      of the SDE.
    batch_size: The number of samplers to generate by calling this function once.
    num_steps: The number of sampling steps. 
      Equivalent to the number of discretized time steps.    
    device: 'cuda' for running on GPUs, and 'cpu' for running on CPUs.
    eps: The smallest time step for numerical stability.
  
  Returns: 
    Samples.
  """
  t = torch.ones(batch_size, device=device)
  init_x = torch.randn(batch_size, 1, img_size, img_size, device=device) * marginal_prob_std(t)[:, None, None, None]
  time_steps = np.linspace(1., eps, num_steps)
  step_size = time_steps[0] - time_steps[1]
  x = init_x
  x_mean = x  # Initialize x_mean to ensure it is always defined

  with torch.no_grad():
    for time_step in tqdm.tqdm(time_steps):
      batch_time_step = torch.ones(batch_size, device=device) * time_step
      # Corrector step (Langevin MCMC)
      if cfg is None:
        cfg = {}
      if cfg.get('classifier_free_guidance', {}).get('enabled', False):
        # If classifier-free guidance is enabled, use the guided score function.
        scale = cfg['classifier_free_guidance'].get('guidance_scale', 2.0)
        # Clamp to max guidance scale if specified
        max_scale = cfg['classifier_free_guidance'].get('guidance_scale_max', None)
        if max_scale is not None and scale > max_scale:
          scale = max_scale

        score = guided_score_fn(score_model,
                                x,
                                batch_time_step,
                                y,
                                cond_img,
                                lsm_cond,
                                topo_cond,
                                scale=scale)
      else:
        # Else, use the standard score model (cheaper computation).
        score = score_model(x, batch_time_step, y, cond_img, lsm_cond, topo_cond)
      
      grad = score                                                                                       
      grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
      noise_norm = np.sqrt(np.prod(x.shape[1:]))
      eps_norm = 1e-12
      langevin_step_size = 2 * (snr * noise_norm / (grad_norm + eps_norm))**2
      x = x + langevin_step_size * grad + torch.sqrt(2 * langevin_step_size) * torch.randn_like(x)      

      # Predictor step (Euler-Maruyama)
      g = diffusion_coeff(batch_time_step)

      if cfg.get('classifier_free_guidance', {}).get('enabled', False):
        # If classifier-free guidance is enabled, use the guided score function.
        scale = cfg['classifier_free_guidance'].get('guidance_scale', 2.0)
        score = guided_score_fn(score_model,
                                x,
                                batch_time_step,
                                y,
                                cond_img,
                                lsm_cond,
                                topo_cond,
                                scale=scale)
      else:
        # Else, use the standard score model (cheaper computation).
        score = score_model(x, batch_time_step, y, cond_img, lsm_cond, topo_cond)

      x_mean = x + (g**2)[:, None, None, None] * score * step_size

      # x_mean = x + (g**2)[:, None, None, None] * score_model(x, batch_time_step, y, cond_img, lsm_cond, topo_cond) * step_size
      x = x_mean + torch.sqrt(g**2 * step_size)[:, None, None, None] * torch.randn_like(x)      
    
    # The last step does not include any noise
    return x_mean


#@title Define the ODE sampler (double click to expand or collapse)

from scipy import integrate

## The error tolerance for the black-box ODE solver
error_tolerance = 1e-5 #@param {'type': 'number'}
def ode_sampler(score_model,
                marginal_prob_std,
                diffusion_coeff,
                num_steps=100,
                batch_size=64, 
                atol=error_tolerance, 
                rtol=error_tolerance, 
                device='cuda', 
                z=None,
                eps=1e-3,
                img_size=64,
                y=None,
                cond_img=None,
                lsm_cond=None,
                topo_cond=None,
                cfg=None
                ):
  """Generate samples from score-based models with black-box ODE solvers.

  Args:
    score_model: A PyTorch model that represents the time-dependent score-based model.
    marginal_prob_std: A function that returns the standard deviation 
      of the perturbation kernel.
    diffusion_coeff: A function that returns the diffusion coefficient of the SDE.
    batch_size: The number of samplers to generate by calling this function once.
    atol: Tolerance of absolute errors.
    rtol: Tolerance of relative errors.
    device: 'cuda' for running on GPUs, and 'cpu' for running on CPUs.
    z: The latent code that governs the final sample. If None, we start from p_1;
      otherwise, we start from the given z.
    eps: The smallest time step for numerical stability.
  """
  t = torch.ones(batch_size, device=device)
  # Create the latent code
  if z is None:
    init_x = torch.randn(batch_size, 1, 32, 32, device=device) \
      * marginal_prob_std(t)[:, None, None, None]
  else:
    init_x = z
    
  shape = init_x.shape

  def score_eval_wrapper(sample, time_steps):
    """A wrapper of the score-based model for use by the ODE solver."""
    sample = torch.tensor(sample, device=device, dtype=torch.float32).reshape(shape)
    time_steps = torch.tensor(time_steps, device=device, dtype=torch.float32).reshape((sample.shape[0], ))    
    with torch.no_grad():    
      score = score_model(sample, time_steps)
    return score.cpu().numpy().reshape((-1,)).astype(np.float64)
  
  def ode_func(t, x):        
    """The ODE function for use by the ODE solver."""
    time_steps = np.ones((shape[0],)) * t    
    g = diffusion_coeff(torch.tensor(t)).cpu().numpy()
    return  -0.5 * (g**2) * score_eval_wrapper(x, time_steps)
  
  # Run the black-box ODE solver.
  res = integrate.solve_ivp(ode_func, (1., eps), init_x.reshape(-1).cpu().numpy(), rtol=rtol, atol=atol, method='RK45')  
  logger.info(f"Number of function evaluations: {res.nfev}")
  x = torch.tensor(res.y[:, -1], device=device).reshape(shape)

  return x
