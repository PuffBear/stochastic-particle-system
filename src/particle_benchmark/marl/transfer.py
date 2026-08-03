"""Immutable checkpoint bundles and deterministic FR-B3 transfer inference."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Literal, Mapping

import numpy as np
from numpy.typing import NDArray

from particle_benchmark.environment import ParticleEnvConfig
from particle_benchmark.io import canonical_json_bytes
from particle_benchmark.marl.representations import (
    ADAPTER_CONTRACT_VERSION,
    FrozenObservationStandardizer,
    Representation,
)


Architecture = Literal["ippo", "commnet"]
ARCHITECTURES: tuple[Architecture, ...] = ("ippo", "commnet")
BUNDLE_SCHEMA_VERSION = "fr_b3_policy_bundle_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def environment_sha256(config: ParticleEnvConfig) -> str:
    return hashlib.sha256(canonical_json_bytes(asdict(config))).hexdigest()


@dataclass(frozen=True)
class LoadedPolicyBundle:
    path: Path
    architecture: Architecture
    representation: Representation
    training_seed: int
    checkpoint_episode: int
    metadata: Mapping[str, object]
    standardizer: FrozenObservationStandardizer
    policy: Any


def write_policy_bundle(
    output_dir: Path,
    *,
    policy: Any,
    architecture: Architecture,
    representation: Representation,
    training_seed: int,
    checkpoint_episode: int,
    canonical_config: ParticleEnvConfig,
    standardizer: FrozenObservationStandardizer,
) -> dict[str, object]:
    """Write one immutable policy, standardizer, and provenance bundle."""

    if output_dir.exists():
        raise FileExistsError(f"immutable policy bundle already exists: {output_dir}")
    if architecture not in ARCHITECTURES:
        raise ValueError(f"unsupported architecture: {architecture!r}")
    if representation != standardizer.representation:
        raise ValueError("bundle and standardizer representations differ")
    if training_seed <= 0 or checkpoint_episode <= 0:
        raise ValueError("training_seed and checkpoint_episode must be positive")
    if int(policy.n_agents) != canonical_config.collector_count:
        raise ValueError("policy agent count differs from canonical environment")

    output_dir.mkdir(parents=True, exist_ok=False)
    model_path = output_dir / "model.pt"
    standardizer_path = output_dir / "standardizer.json"
    policy.save(model_path)
    standardizer.save(standardizer_path)

    architecture_parameters: dict[str, int] = {}
    if architecture == "commnet":
        architecture_parameters = {
            "h_dim": int(policy.h_dim),
            "n_comm_rounds": int(policy.net.n_comm_rounds),
        }
    metadata = {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "adapter_contract_version": ADAPTER_CONTRACT_VERSION,
        "architecture": architecture,
        "architecture_parameters": architecture_parameters,
        "representation": representation,
        "training_scale": "canonical",
        "training_seed": training_seed,
        "checkpoint_episode": checkpoint_episode,
        "obs_dim": int(policy.obs_dim),
        "n_agents": int(policy.n_agents),
        "canonical_environment_sha256": environment_sha256(canonical_config),
        "model": {"path": model_path.name, "sha256": _sha256(model_path)},
        "standardizer": {
            "path": standardizer_path.name,
            "sha256": _sha256(standardizer_path),
            "sample_count": standardizer.sample_count,
        },
    }
    metadata_path = output_dir / "metadata.json"
    with metadata_path.open("xb") as handle:
        handle.write(canonical_json_bytes(metadata))
    return metadata


def load_policy_bundle(
    bundle_dir: Path, *, canonical_config: ParticleEnvConfig
) -> LoadedPolicyBundle:
    """Load a bundle only after exact metadata, config, and hash validation."""

    metadata_path = bundle_dir / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"bundle metadata is missing: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("schema_version") != BUNDLE_SCHEMA_VERSION:
        raise ValueError("unsupported policy-bundle schema")
    if metadata.get("adapter_contract_version") != ADAPTER_CONTRACT_VERSION:
        raise ValueError("policy bundle uses a different observation contract")
    architecture = str(metadata.get("architecture"))
    representation = str(metadata.get("representation"))
    if architecture not in ARCHITECTURES:
        raise ValueError(f"unsupported bundled architecture: {architecture!r}")
    if metadata.get("training_scale") != "canonical":
        raise ValueError("FR-B3 transfer requires canonical-scale training")
    if metadata.get("canonical_environment_sha256") != environment_sha256(
        canonical_config
    ):
        raise ValueError("bundle canonical-environment hash mismatch")
    if int(metadata.get("n_agents", -1)) != canonical_config.collector_count:
        raise ValueError("bundle agent count mismatch")

    model_record = dict(metadata["model"])
    standardizer_record = dict(metadata["standardizer"])
    model_path = bundle_dir / str(model_record["path"])
    standardizer_path = bundle_dir / str(standardizer_record["path"])
    for path, expected_hash in (
        (model_path, str(model_record["sha256"])),
        (standardizer_path, str(standardizer_record["sha256"])),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"bundle artifact is missing: {path}")
        if _sha256(path) != expected_hash:
            raise ValueError(f"bundle artifact hash mismatch: {path.name}")

    standardizer = FrozenObservationStandardizer.load(standardizer_path)
    if representation != standardizer.representation:
        raise ValueError("bundle and standardizer representations differ")
    obs_dim = int(metadata["obs_dim"])
    n_agents = int(metadata["n_agents"])
    if architecture == "ippo":
        from particle_benchmark.marl.ippo import IPPO

        policy = IPPO(obs_dim=obs_dim, n_agents=n_agents)
    else:
        from particle_benchmark.marl.commnet import CommNet

        parameters = dict(metadata["architecture_parameters"])
        policy = CommNet(
            obs_dim=obs_dim,
            n_agents=n_agents,
            h_dim=int(parameters["h_dim"]),
            n_comm_rounds=int(parameters["n_comm_rounds"]),
        )
    policy.load(model_path)
    return LoadedPolicyBundle(
        path=bundle_dir,
        architecture=architecture,
        representation=representation,
        training_seed=int(metadata["training_seed"]),
        checkpoint_episode=int(metadata["checkpoint_episode"]),
        metadata=metadata,
        standardizer=standardizer,
        policy=policy,
    )


def _clip_unit_ball(actions: NDArray[np.float64]) -> NDArray[np.float64]:
    norms = np.linalg.norm(actions, axis=1, keepdims=True)
    return actions / np.maximum(norms, 1.0)


def deterministic_actions(
    bundle: LoadedPolicyBundle,
    observations: tuple[Mapping[str, object], ...],
    *,
    communication_ablated: bool = False,
) -> NDArray[np.float64]:
    """Return action means with no policy sampling or normalization updates."""

    if communication_ablated and bundle.architecture != "commnet":
        raise ValueError("communication ablation is defined only for CommNet")
    import torch
    from particle_benchmark.marl.networks import flatten_all_observations

    flat = flatten_all_observations(observations)
    if flat.shape != (int(bundle.metadata["n_agents"]), int(bundle.metadata["obs_dim"])):
        raise ValueError("adapted observation layout differs from checkpoint metadata")
    with torch.no_grad():
        if bundle.architecture == "ippo":
            means = []
            for agent_id, network in enumerate(bundle.policy.networks):
                network.eval()
                action_mean, _ = network.forward(
                    torch.from_numpy(flat[agent_id]).unsqueeze(0)
                )
                means.append(action_mean.squeeze(0).cpu().numpy())
            action_means = np.asarray(means, dtype=np.float64)
        else:
            bundle.policy.net.eval()
            action_tensor, _ = bundle.policy.net.forward_step(
                torch.from_numpy(flat), comm_ablated=communication_ablated
            )
            action_means = action_tensor.cpu().numpy().astype(np.float64)
    if action_means.shape != (int(bundle.metadata["n_agents"]), 2):
        raise ValueError("policy returned an invalid action shape")
    if not np.all(np.isfinite(action_means)):
        raise ValueError("policy returned non-finite action means")
    return _clip_unit_ball(action_means)
