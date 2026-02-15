---
title: "PDE-Refiner: Achieving Accurate Long Rollouts with Neural PDE Solvers"
date: 2026-02-13
permalink: /posts/pde-refiner/
layout: single
author_profile: true
toc: false
toc_sticky: false
toc_label: "On this page"
classes: wide
pdf: /files/posts/pde-refiner/PDE-Refiner-Blog.pdf
---

{% include toc %}
<div style="clear: both;"></div>

> **Summary.** PDE-Refiner improves long-horizon rollouts of neural PDE solvers by refining each predicted step through an iterative denoising process. The key benefit is better correction of frequency-domain errors that otherwise accumulate and destabilize autoregressive rollouts.
{: .notice--info}

## Overview

Neural surrogate models for time-dependent PDEs can be dramatically faster than classical solvers—especially when high-resolution simulations must be repeated many times (design loops, optimization, uncertainty quantification). The practical bottleneck is not one-step prediction, but **stable long rollouts**: when a model predicts a trajectory step-by-step, small errors compound until the rollout becomes unreliable.

This post summarizes the main ideas and results of *PDE-Refiner: Achieving Accurate Long Rollouts with Neural PDE Solvers* (NeurIPS 2023) [[paper]](https://arxiv.org/abs/2308.05732).

---

## 1. Problem setting: learned rollouts for time-dependent PDEs

We consider PDEs of the form

$$
u_t = F(t, x, u, u_x, u_{xx}, \ldots),
$$

where $$u(t,x)$$ is the state evolving over time. A neural PDE solver often learns a one-step evolution map:

$$
\hat{u}(t+\Delta t) = \mathcal{M}(\hat{u}(t)).
$$

A rollout is autoregressive:

$$
\hat{u}(t+2\Delta t)=\mathcal{M}(\hat{u}(t+\Delta t)),\;\;\ldots
$$

Even if one-step error is small, **errors accumulate** over steps. In chaotic or turbulent systems, small mismatches can get amplified and lead to rapid divergence from the ground truth.

---

## 2. Key diagnosis: frequency components are learned unevenly

A central empirical observation is that neural PDE solvers do not learn all spatial frequency components equally well.

Any spatial field can be decomposed into Fourier modes (frequencies):
- **Low frequencies** capture large-scale structure (smooth patterns).
- **High frequencies** capture fine details (small-scale structure).

Standard training objectives (like MSE) push the model to reduce error where the signal energy is largest. That typically means the model focuses on dominant, high-energy components and can under-correct low-energy bands. This is easy to miss in short-horizon plots because the large-scale dynamics look correct.

However, nonlinear PDEs couple modes. A small error in a “weak” frequency band can seed a growing mismatch that later contaminates larger scales, causing long-horizon instability.

### Spectrum intuition

<a href="/images/posts/pde-refiner/KM_FFT_Spectrum-1.png" class="image-popup">
  <img src="/images/posts/pde-refiner/KM_FFT_Spectrum-1.png" alt="Frequency spectrum mismatch">
</a>

**How to read this:** the rollout can look plausible at first, while frequency-domain mismatches quietly accumulate until the trajectory collapses.

---

## 3. Working example: Kuramoto–Sivashinsky (KS) dynamics

The paper uses the **Kuramoto–Sivashinsky (KS)** equation as a primary test case:

$$
u_t + u u_x + u_{xx} + \nu u_{xxxx} = 0.
$$

KS is a strong benchmark because it is chaotic and sensitive: tiny errors grow with time, so long-horizon stability is hard.

A common baseline trains with one-step mean squared error:

$$
\mathcal{L}_{\mathrm{MSE}} = \|u(t) - \mathcal{M}(u(t-\Delta t))\|^2.
$$

This can produce good one-step predictions, but long rollouts still fail—especially when the model does not reproduce the frequency content accurately.

---

## 4. Why “fixing rollout strategy” alone is not enough (expanded)

A natural idea is: “If rollouts fail due to compounding error, let’s change training/rollout strategies.”

There are several common strategies:

### 4.1 Train with longer context (history)
Instead of predicting from a single previous state, models use multiple previous states:

$$
\hat{u}(t+\Delta t)=\mathcal{M}(u(t),u(t-\Delta t),\ldots,u(t-H\Delta t)).
$$

This can stabilize rollouts slightly, because the model sees more temporal context. But it does not necessarily fix the fundamental issue: if the model still learns an inaccurate spectrum, the error can still accumulate—just slower.

<div style="display:flex; gap:20px; margin:20px 0; align-items:flex-start; flex-wrap:wrap;">

  <div style="flex:1; min-width:280px; text-align:center;">
    <a href="/images/posts/pde-refiner/KS_History_Rollout_Loss_Short-1.png" class="image-popup">
      <img src="/images/posts/pde-refiner/KS_History_Rollout_Loss_Short-1.png" alt="History Rollout Loss Short" style="width:100%; height:auto; border-radius:6px;">
    </a>
    <div style="font-size:0.9rem; opacity:0.75; margin-top:6px;">Short rollout horizon</div>
  </div>

  <div style="flex:1; min-width:280px; text-align:center;">
    <a href="/images/posts/pde-refiner/KS_History_Rollout_Loss_Long-1.png" class="image-popup">
      <img src="/images/posts/pde-refiner/KS_History_Rollout_Loss_Long-1.png" alt="History Rollout Loss Long" style="width:100%; height:auto; border-radius:6px;">
    </a>
    <div style="font-size:0.9rem; opacity:0.75; margin-top:6px;">Long rollout horizon</div>
  </div>

</div>


**History Bar Plot System:**
<a href="/images/posts/pde-refiner/KS_barplot_history-1.png" class="image-popup">
  <img src="/images/posts/pde-refiner/KS_barplot_history-1.png" alt="bar plot history system">
</a>


### 4.2 Multi-step training / pushforward training
Instead of training only one-step, the model is trained on multiple steps, hoping it learns to correct drift. This often improves stability, but the paper finds it is still limited when frequency-domain mismatches persist.

### 4.3 Change the loss (e.g., Sobolev / derivative losses)
A Sobolev loss emphasizes derivatives and can indirectly emphasize higher frequencies. These losses can help, but they still optimize a single-shot prediction. They do not explicitly give the model a mechanism to **iteratively correct** a predicted state.

### 4.4 The core point
Many rollout tricks improve stability *a bit*, but they do not reliably solve:
- **small-but-important frequency mismatches** that accumulate during autoregressive rollout.

So the paper proposes a different idea:
> Don’t only change training schedules—change the prediction process itself so the model can refine and correct a step multiple times.

---

## 5. PDE-Refiner: iterative denoising refinement (expanded, step-by-step)

PDE-Refiner predicts the next state and then **refines it** through a small number of correction steps.

The refinement is set up as a **denoising problem**:
- add noise to the predicted state,
- train the model to estimate/remove that noise,
- repeat with decreasing noise magnitude.

This design matters because noise injects energy across frequencies, forcing corrections across the spectrum rather than only the dominant modes.

### 5.1 Step-by-step algorithm (clear version)

At time $$t$$ we want $$u(t+\Delta t)$$.

**Step 1 — Initial prediction**
Compute a first estimate:

$$
\hat{u}_0(t+\Delta t)=\mathcal{M}_{\text{base}}(u(t), \text{context}).
$$

This estimate is usually good in large-scale structure, but may contain systematic errors in fine-scale content.

**Step 2 — Add noise**
For refinement step $$k$$:

$$
\tilde{u}_k = \hat{u}_k + \sigma_k \epsilon_k,\quad \epsilon_k \sim \mathcal{N}(0,1).
$$

**Step 3 — Predict the noise**
The model predicts the injected noise:

$$
\hat{\epsilon}_k = \mathcal{R}(\tilde{u}_k,\; \text{context},\; k).
$$

**Step 4 — Denoise / refine**
Update the refined estimate:

$$
\hat{u}_{k+1} = \tilde{u}_k - \sigma_k \hat{\epsilon}_k.
$$

**Step 5 — Repeat with smaller noise**
Use a decreasing schedule:

$$
\sigma_1 > \sigma_2 > \cdots > \sigma_K.
$$

This turns refinement into a coarse-to-fine correction process:
- early steps fix large-scale structure,
- later steps recover smaller-scale details.

### 5.2 Why adding noise helps (intuition)
If the model only sees clean predictions, it tends to correct what contributes most to MSE (dominant modes).
Adding noise forces the model to learn corrections robustly across amplitudes and frequencies.

### 5.3 Refinement illustration

<a href="/images/posts/pde-refiner/KS_PDERefiner_fft_intermediate-1.png" class="image-popup">
  <img src="/images/posts/pde-refiner/KS_PDERefiner_fft_intermediate-1.png" alt="Refinement improves spectrum">
</a>

<div style="display:flex; gap:20px; margin:20px 0; flex-wrap:wrap;">

  <div style="flex:1; min-width:280px; text-align:center;">
    <a href="/images/posts/pde-refiner/KS_fft_after_1_steps_Original-1.png" class="image-popup">
      <img src="/images/posts/pde-refiner/KS_fft_after_1_steps_Original-1.png" style="width:100%;">
    </a>
    <div style="font-size:0.9rem; opacity:0.75;">Initial prediction spectrum</div>
  </div>

  <div style="flex:1; min-width:280px; text-align:center;">
    <a href="/images/posts/pde-refiner/KS_fft_after_1_steps_Corrected-1.png" class="image-popup">
      <img src="/images/posts/pde-refiner/KS_fft_after_1_steps_Corrected-1.png" style="width:100%;">
    </a>
    <div style="font-size:0.9rem; opacity:0.75;">After refinement correction</div>
  </div>

</div>

---

## 6. Training objective (denoising loss)

Instead of predicting a direct residual correction, PDE-Refiner is trained to predict the injected noise. For a sampled refinement step $$k$$:

$$
\mathcal{L}_k
=
\mathbb{E}_{\epsilon_k}
\left[
\left\|
\epsilon_k -
\mathcal{R}(u(t) + \sigma_k \epsilon_k,\; \text{context},\; k)
\right\|^2
\right].
$$

This trains the refiner to operate at different noise levels (different scales), which supports coarse-to-fine correction.

---

## 7. Why it works: improved accuracy across frequency bands (expanded)

The paper’s key mechanism is that refinement reduces error **more uniformly across frequencies**.

### 7.1 What baseline training tends to do
With standard MSE training, the model reduces error where the energy is highest. That often means:
- dominant low-frequency modes are good,
- higher-frequency modes are under-corrected,
- long rollouts drift when those small errors get amplified.

### 7.2 What PDE-Refiner changes
PDE-Refiner introduces multiple correction opportunities per step. Because refinement is framed as denoising with varying noise magnitudes, the model learns to correct:
- large-scale structure (early steps),
- finer-scale details (later steps).

This matters because in chaotic systems, a small fine-scale mismatch can act like a perturbation that grows over time.

### 7.3 Frequency-domain evidence

<a href="/images/posts/pde-refiner/KS_frequency_spectrum_precision-1.png" class="image-popup">
  <img src="/images/posts/pde-refiner/KS_frequency_spectrum_precision-1.png" alt="Spectrum precision">
</a>

<div style="display:flex; gap:20px; margin:20px 0; flex-wrap:wrap;">

  <div style="flex:1; min-width:280px; text-align:center;">
    <a href="/images/posts/pde-refiner/KS_fft_after_1000_steps_Original-1.png" class="image-popup">
      <img src="/images/posts/pde-refiner/KS_fft_after_1000_steps_Original-1.png" style="width:100%;">
    </a>
    <div style="font-size:0.9rem; opacity:0.75;">
      Baseline spectrum after long rollout
    </div>
  </div>

  <div style="flex:1; min-width:280px; text-align:center;">
    <a href="/images/posts/pde-refiner/KS_fft_after_1000_steps_Corrected-1.png" class="image-popup">
      <img src="/images/posts/pde-refiner/KS_fft_after_1000_steps_Corrected-1.png" style="width:100%;">
    </a>
    <div style="font-size:0.9rem; opacity:0.75;">
      PDE-Refiner spectrum after long rollout
    </div>
  </div>

</div>
<a href="/images/posts/pde-refiner/KS_correlation_over_rollout_time-1.png" class="image-popup">
  <img src="/images/posts/pde-refiner/KS_correlation_over_rollout_time-1.png">
</a>


---

## 8. Experiments

### 8.1 Experiment 1 — 1D Kuramoto–Sivashinsky rollouts

KS is chaotic, so the correct question is not “does it look okay at one step,” but:
> **How long can the model roll out before it loses correlation with the ground truth?**

#### Setup (high-level)
The model is trained on KS trajectories and evaluated by rolling out many steps autoregressively. Evaluation focuses on long horizons where compounding error matters.

A common metric in the paper is correlation over time (or correlation time): how long predictions remain aligned with ground truth before dropping below a threshold.

#### Baselines vs PDE-Refiner
Baselines can show:
- good one-step error,
- plausible short-term patterns,
- but loss of correlation at longer horizons.

PDE-Refiner improves the long-horizon behavior by refining each step, reducing the drift that accumulates from subtle frequency errors.

<a href="/images/posts/pde-refiner/KS_barplot_correlation_time_reordered-1.png" class="image-popup">
  <img src="/images/posts/pde-refiner/KS_barplot_correlation_time_reordered-1.png" alt="KS correlation time">
</a>

#### Interpreting the result
This bar plot shows that PDE-Refiner maintains accurate rollouts significantly longer than:
- plain MSE training,
- and multiple alternative rollout strategies.

The important point is that refinement improves long-term stability without requiring a separate simulator or handcrafted correction.

<div style="display:flex; gap:20px; margin:25px 0; align-items:flex-start; flex-wrap:wrap;">

  <div style="flex:1; min-width:300px; text-align:center;">
    <a href="/images/posts/pde-refiner/KS_correlation_over_rollout_time-1.png" class="image-popup">
      <img src="/images/posts/pde-refiner/KS_correlation_over_rollout_time-1.png"
           alt="Correlation over rollout time"
           style="width:100%; height:auto; border-radius:6px;">
    </a>
    <div style="font-size:0.9rem; opacity:0.75; margin-top:6px;">
      Correlation between prediction and ground truth over rollout horizon
    </div>
  </div>

  <div style="flex:1; min-width:300px; text-align:center;">
    <a href="/images/posts/pde-refiner/KS_Methods_Rollout_Loss-1.png" class="image-popup">
      <img src="/images/posts/pde-refiner/KS_Methods_Rollout_Loss-1.png"
           alt="Rollout loss comparison"
           style="width:100%; height:auto; border-radius:6px;">
    </a>
    <div style="font-size:0.9rem; opacity:0.75; margin-top:6px;">
      Error accumulation across rollout steps for different methods
    </div>
  </div>

</div>


These show:
- qualitative rollouts,
- how methods compare over time,
- and what “failure” looks like.

---

### 8.2 Experiment 2 — Generalization across viscosity

The paper also varies viscosity $$\nu$$, which changes how strongly high frequencies are damped. This tests whether the method generalizes across regimes.

<a href="/images/posts/pde-refiner/KS_conditional_rollout_over_viscosity-1.png" class="image-popup">
  <img src="/images/posts/pde-refiner/KS_conditional_rollout_over_viscosity-1.png" alt="Viscosity generalization">
</a>

Interpretation: PDE-Refiner improves rollout behavior across viscosities, suggesting robustness rather than tuning to a single setting.

---

### 8.3 Experiment 3 — 2D Kolmogorov flow (turbulence benchmark)

Kolmogorov flow is a harder, more realistic setting because turbulence is strongly nonlinear and sensitive to fine-scale error.

A simplified form of the dynamics:

$$
\partial_t u + \nabla \cdot (u \otimes u)
=
\nu \nabla^2 u - \nabla p + f.
$$

The takeaway is consistent: when small-scale errors are not controlled, long rollouts drift. Refinement improves stability by correcting a broader range of frequency content.

<a href="/images/posts/pde-refiner/KS_fno_barplot_correlation_time-1.png" class="image-popup">
  <img src="/images/posts/pde-refiner/KS_fno_barplot_correlation_time-1.png">
</a>

<a href="/images/posts/pde-refiner/KS_performance_over_resolution-1.png" class="image-popup">
  <img src="/images/posts/pde-refiner/KS_performance_over_resolution-1.png">
</a>

---

## 9. Uncertainty estimation via sampling (expanded)

PDE-Refiner enables a practical uncertainty signal because refinement involves stochastic noise. We can run multiple refinement samples (different noise draws) and compare their rollouts.

- If multiple sampled rollouts stay similar for a long time, confidence is higher.
- If they diverge quickly, uncertainty is higher—and the true rollout typically fails sooner.

This is useful operationally: it provides a “trust horizon” estimate without training a full ensemble.

<div style="gap:20px; margin:25px 0; align-items:flex-start; flex-wrap:wrap;">

  <div style="flex:1; min-width:300px; text-align:center;">
    <a href="/images/posts/pde-refiner/KS_diffusion_uncertainty_estimation_scatter_plot-1.png" class="image-popup">
      <img src="/images/posts/pde-refiner/KS_diffusion_uncertainty_estimation_scatter_plot-1.png" 
           alt="Uncertainty estimation scatter"
           style="width:100%; height:auto; border-radius:6px;">
    </a>
    <div style="font-size:0.9rem; opacity:0.75; margin-top:6px;">
      Uncertainty estimation scatter
    </div>
  </div>
<br />
  <div style="flex:1; min-width:300px; text-align:center;">
    <a href="/images/posts/pde-refiner/KS_diffusion_sample_std-1.png" class="image-popup">
      <img src="/images/posts/pde-refiner/KS_diffusion_sample_std-1.png"
           alt="Diffusion samples standard deviation"
           style="width:100%; height:auto; border-radius:6px;">
    </a>
    <div style="font-size:0.9rem; opacity:0.75; margin-top:6px;">
      Diffusion samples standard deviation
    </div>
  </div>

</div>

---

## 10. Pros, cons, and trade-offs

**Pros**
- Improves long-horizon rollout stability in chaotic regimes.
- Corrects errors across a broader frequency range.
- Robust across viscosity (regime shifts).
- Provides a usable uncertainty signal from sampling.

**Cons / trade-offs**
- More compute: refinement requires multiple model evaluations per timestep.
- Requires choosing refinement steps and noise schedule (design choices).

---

## Future work

Possible directions include:
- faster refinement schedules (fewer steps with similar performance),
- distilling refinement into a cheaper predictor,
- extending to larger-scale 2D/3D PDE systems,
- combining refinement with physics-informed constraints or conservation priors,
- automatic stopping criteria based on uncertainty signals,
- exploring refinement with other neural operator families.

---

## Conclusion

Long-horizon rollout failure is strongly connected to frequency-domain error accumulation: standard training often under-corrects low-energy frequency components that later become dynamically important.

PDE-Refiner improves stability by refining each predicted step through iterative denoising, producing better frequency coverage, longer stable rollouts, and a natural uncertainty signal via sampling.

---

## Reference

Lippe, P., et al. (2023). *PDE-Refiner: Achieving Accurate Long Rollouts with Neural PDE Solvers.* NeurIPS 2023.

- [arXiv page](https://arxiv.org/abs/2308.05732)
- [PDF](https://arxiv.org/pdf/2308.05732.pdf)
