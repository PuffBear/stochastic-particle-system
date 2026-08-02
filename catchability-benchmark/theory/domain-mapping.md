# Domain Mapping: Where Real Systems Fall in ρ-κ Space

**Purpose:** Show that the SPS benchmark corresponds to physically realistic operating regimes, and that the benchmark's parameter grid covers systems that practitioners actually build.

---

## How to estimate ρ and κ for a real system

**ρ = (signal drift speed · √observation_interval) / observation_noise_std**

In practice: ρ is small when the signal you are trying to detect (the drifting target) is hard to distinguish from background noise in a single observation. It is large when the target's motion is easily detectable.

**κ = target_drift_speed / collector_max_speed**

κ < 1: collector is faster than the target. κ > 1: target outruns the collector.

For systems where the "target" is a diffusing contaminant or passive particle, the relevant drift speed is the mean advection velocity of the medium (current, wind, fluid flow), not the Brownian fluctuation.

---

## Domain estimates

### 1. Ocean microplastics monitoring (AUV swarms)

**System:** 4 autonomous underwater vehicles (AUVs) sampling for microplastic particles in a current.

**Parameter estimates:**
- Target drift: ocean surface current α ≈ 0.05–0.15 m/s
- AUV speed: v_max ≈ 0.5–1.5 m/s (Bluefin-9 class)
- Observation noise: turbulent velocity fluctuations σ ≈ 0.03–0.10 m/s
- Sampling interval: dt ≈ 5–30 seconds

**ρ estimate:** α·√dt / σ ≈ 0.10·√10 / 0.05 ≈ 6.3 (high — current is easily detectable)
**κ estimate:** α / v_max ≈ 0.10 / 1.0 = 0.10 (low — AUVs are much faster than current)

**SPS regime:** (low κ, high ρ) — bottom-left of the grid.

**Implication from predicted pattern:** In this regime, sensing is easy but catching is trivial (AUVs outrun the drift). Communication benefit is predicted to be low because each agent can independently detect the current direction. Coverage strategy dominates.

**Caveat:** If the target is sparse (few particles per m³), effective ρ may be much lower — sparse sampling makes the detection problem harder even if current speed is easily measured.

---

### 2. Agricultural drone swarms (pest or pathogen tracking)

**System:** 4 UAVs tracking a plume of airborne pest eggs or spores diffusing downwind.

**Parameter estimates:**
- Target drift: wind speed α ≈ 1–5 m/s
- UAV speed: v_max ≈ 10–20 m/s (agricultural multirotor)
- Observation noise: turbulent gusts σ ≈ 0.5–2.0 m/s
- Sampling interval: dt ≈ 1–5 seconds

**ρ estimate:** α·√dt / σ ≈ 2·√2 / 1.0 ≈ 2.8 (moderate-high — wind direction is detectable)
**κ estimate:** α / v_max ≈ 3 / 15 = 0.20 (moderate — UAVs are faster but not overwhelmingly so at high wind)

**SPS regime:** (mid κ, moderate ρ) — near cell (mid, mid), the confirmed C03 regime.

**Implication:** This is the regime where SPS-C03 shows +1.19 particle improvement. Agricultural drone swarms at moderate wind speeds are predicted to benefit from shared velocity summaries.

---

### 3. Search-and-rescue (moving person tracking)

**System:** 4 ground robots searching for a moving person in a disaster zone.

**Parameter estimates:**
- Target drift: average walking/crawling speed α ≈ 0.5–1.5 m/s
- Robot speed: v_max ≈ 0.3–0.8 m/s (typical ground robot in rubble)
- Observation noise: RF/acoustic signal noise σ ≈ 0.3–1.0 m/s equivalent
- Sampling interval: dt ≈ 1–5 seconds

**ρ estimate:** α·√dt / σ ≈ 1.0·√2 / 0.5 ≈ 2.8 (moderate)
**κ estimate:** α / v_max ≈ 1.0 / 0.5 = 2.0 (κ > 1 — target outruns collectors)

**SPS regime:** (high κ, moderate ρ) — right column of the grid.

**Implication from predicted pattern:** High κ regime — collectors cannot outrun the target. The theoretical prediction is near-zero coordination benefit because actuation cannot execute the implied interception. Coverage and waiting strategies dominate over directed pursuit. Communication about velocity direction is of limited value when you cannot act on it.

**This is an important calibration result for the paper:** it shows the benchmark correctly identifies a regime where coordination through velocity sharing does *not* help, which is as useful as knowing where it does.

---

### 4. Oil spill cleanup (autonomous surface vessels)

**System:** 4 autonomous surface vessels (ASVs) collecting oil slick in wind and current.

**Parameter estimates:**
- Slick drift: wind + current combined α ≈ 0.1–0.5 m/s
- ASV speed: v_max ≈ 1–3 m/s
- Observation noise: visual/radar slick boundary uncertainty σ ≈ 0.05–0.2 m/s
- Sampling interval: dt ≈ 10–60 seconds

**ρ estimate:** α·√dt / σ ≈ 0.3·√30 / 0.1 ≈ 16.4 (very high — slick drift is easily observable)
**κ estimate:** α / v_max ≈ 0.3 / 2.0 = 0.15 (low — ASVs much faster than drift)

**SPS regime:** (low κ, very high ρ) — outside the grid (ρ higher than our highest cell).

**Implication:** Sensing is easy and collectors are fast. This is the regime where simple greedy strategies (go to nearest slick) likely dominate coordinated communication. The SPS benchmark at current parameter ranges does not directly represent this system — but the domain mapping makes this explicit.

---

### 5. Wildfire ember tracking (UAV swarms)

**System:** UAV swarm tracking flying embers to predict spot fire ignition points.

**Parameter estimates:**
- Ember drift: wind speed α ≈ 5–15 m/s
- UAV speed: v_max ≈ 10–20 m/s
- Observation noise: thermal camera noise σ ≈ 1–5 m/s equivalent
- Sampling interval: dt ≈ 0.5–2 seconds

**ρ estimate:** α·√dt / σ ≈ 10·√1 / 2 = 5 (high)
**κ estimate:** α / v_max ≈ 10 / 15 ≈ 0.67 (moderate — UAVs have limited advantage over wind)

**SPS regime:** (mid-high κ, high ρ) — upper-right of the grid.

**Implication:** In this regime, sensing is easy but the wind is fast relative to UAV speed. The coordination benefit is predicted to be low to moderate — collectors can detect the drift direction easily but cannot outrun embers moving in strong wind. Positioning ahead of predicted trajectories matters more than velocity sharing.

---

## Summary table

| Domain | ρ estimate | κ estimate | SPS grid region | Predicted coordination gain |
|---|---|---|---|---|
| Ocean AUV (microplastics) | High (3–7) | Low (0.05–0.15) | Outside grid (high ρ, low κ) | Low — sensing easy, catching trivial |
| Agricultural UAV (pestspores) | Moderate (1–4) | Moderate (0.15–0.30) | Centre — near C03 cell | Moderate — confirmed +1.19 regime |
| Search-and-rescue (ground robot) | Moderate (1–3) | High (1–3) | Outside grid (high κ) | Low — target outruns collectors |
| Oil spill (ASV) | Very high (10+) | Low (0.10–0.20) | Outside grid (very high ρ) | Low — greedy sufficient |
| Wildfire ember (UAV) | High (3–6) | Moderate-high (0.5–1.0) | Upper-right cell | Low to moderate |

**Key observation:** The agricultural UAV regime is the closest match to SPS-C03's confirmed positive result. This is not a coincidence — the SPS parameters were chosen to model a physically realistic mid-difficulty collection problem. The domain mapping confirms that the confirmed regime corresponds to a real system class.

---

## What the domain mapping adds to the paper

Without the domain mapping, the paper presents a result at one confirmed operating point and a grid showing how gain varies. With the domain mapping, it answers: "yes, but does this correspond to any real system?" The answer is yes — specifically agricultural drone swarms at moderate wind speeds. And it answers "where does it *not* help?" — specifically when κ > 1 (target outruns collector), which covers search-and-rescue with slow ground robots. Both answers are useful to practitioners.
