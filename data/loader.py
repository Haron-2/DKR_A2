import hashlib
import logging
import os
from pathlib import Path
import shutil
import tempfile
import time
from urllib.error import URLError
from urllib.request import Request, urlopen

import numpy as np
import torch
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.datasets import Actor, WikipediaNetwork
from torch_geometric.utils import remove_self_loops, to_undirected

from data.validation import validate_dataset


logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = PROJECT_ROOT / "datasets"
GEOM_GCN_REVISION = "f1fc0d14b3b019c562737240d06ec83b07d16a8f"
FILTERED_REVISION = "a431395582e929d88271309716bea4fe24ce6318"
GEOM_GCN_URL = (
    "https://raw.githubusercontent.com/graphdml-uiuc-jlu/geom-gcn/"
    + GEOM_GCN_REVISION
)
FILTERED_URL = (
    "https://raw.githubusercontent.com/yandex-research/heterophilous-graphs/"
    + FILTERED_REVISION
    + "/data"
)
SUPPORTED_DATASETS = (
    ("chameleon", "raw"),
    ("chameleon", "filtered"),
    ("squirrel", "raw"),
    ("squirrel", "filtered"),
    ("actor", "raw"),
)
CACHE_VERSION = 1
GRAPH_POLICY = "undirected_coalesced_without_self_loops"
TENSOR_FIELDS = (
    "x", "y", "edge_index", "source_edge_index", "node_id",
    "train_mask", "val_mask", "test_mask",
)


class _WikipediaNetwork(WikipediaNetwork):
    processed_url = GEOM_GCN_URL

    @property
    def raw_dir(self) -> str:
        return str(Path(self.root) / "raw" / self.name / "raw")

    @property
    def processed_dir(self) -> str:
        return str(Path(self.root) / "processed" / self.name / "pyg_raw")


class _Actor(Actor):
    url = GEOM_GCN_URL

    @property
    def raw_dir(self) -> str:
        return str(Path(self.root) / "raw" / "actor" / "raw")

    @property
    def processed_dir(self) -> str:
        return str(Path(self.root) / "processed" / "actor" / "pyg_raw")


def file_sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def dataset_fingerprint(data: Data) -> str:
    digest = hashlib.sha256()
    for name in TENSOR_FIELDS:
        tensor = getattr(data, name).detach().cpu().contiguous()
        digest.update(f"{name}:{tensor.dtype}:{list(tensor.shape)}:".encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _source_files(name: str, mode: str) -> list[dict]:
    directory = Path("raw") / name / mode
    if mode == "filtered":
        filename = f"{name}_filtered.npz"
        return [{"path": (directory / filename).as_posix(),
                 "url": f"{FILTERED_URL}/{filename}"}]

    upstream_name = "film" if name == "actor" else name
    sources = []
    for filename in ("out1_node_feature_label.txt", "out1_graph_edges.txt"):
        sources.append({
            "path": (directory / filename).as_posix(),
            "url": f"{GEOM_GCN_URL}/new_data/{upstream_name}/{filename}",
        })
    for split_id in range(10):
        filename = f"{upstream_name}_split_0.6_0.2_{split_id}.npz"
        sources.append({"path": (directory / filename).as_posix(),
                        "url": f"{GEOM_GCN_URL}/splits/{filename}"})
    return sources


def _download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = Request(url, headers={"User-Agent": "DKR-A2-data-preparation/1.0"})
    for attempt in range(3):
        temporary_path = None
        try:
            logger.info("Downloading %s", destination.name)
            with urlopen(request, timeout=60) as response:
                with tempfile.NamedTemporaryFile(
                    dir=destination.parent, suffix=".part", delete=False
                ) as target:
                    temporary_path = Path(target.name)
                    shutil.copyfileobj(response, target)
            if temporary_path.stat().st_size == 0:
                raise OSError(f"Empty download: {url}")
            temporary_path.replace(destination)
            return
        except (URLError, OSError) as exc:
            if attempt == 2:
                raise OSError(f"Failed to download {url}") from exc
            logger.warning("Download failed; retrying %s: %s", url, exc)
            time.sleep(attempt + 1)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)


def _load_filtered(path: Path) -> Data:
    with np.load(path, allow_pickle=False) as arrays:
        required = (
            "node_features", "node_labels", "edges",
            "train_masks", "val_masks", "test_masks",
        )
        missing = set(required) - set(arrays.files)
        if missing:
            raise ValueError(f"Missing arrays in {path}: {sorted(missing)}")
        # The authors' NPZ files store splits as rows: [10, N].
        return Data(
            x=torch.from_numpy(arrays["node_features"].copy()),
            y=torch.from_numpy(arrays["node_labels"].copy()),
            edge_index=torch.from_numpy(arrays["edges"].copy()).t().contiguous(),
            train_mask=torch.from_numpy(arrays["train_masks"].copy()).t().contiguous(),
            val_mask=torch.from_numpy(arrays["val_masks"].copy()).t().contiguous(),
            test_mask=torch.from_numpy(arrays["test_masks"].copy()).t().contiguous(),
        )


def _check_source_hashes(root: Path, sources: list[dict]) -> None:
    for source in sources:
        path = root / source["path"]
        # Processed caches are self-contained; missing raw files need no download.
        if path.exists() and file_sha256(path) != source["sha256"]:
            raise ValueError(f"Source checksum mismatch: {path}")


def _read_cache(path: Path, root: Path, name: str, mode: str) -> Data:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    metadata = payload["metadata"]
    revision = FILTERED_REVISION if mode == "filtered" else GEOM_GCN_REVISION
    if (
        metadata["cache_version"] != CACHE_VERSION
        or metadata["source_revision"] != revision
        or metadata["graph_policy"] != GRAPH_POLICY
        or metadata["dataset"] != name
        or metadata["mode"] != mode
    ):
        raise ValueError(f"Cache configuration changed; rebuild {name}/{mode}")
    data = Data(**payload["tensors"])
    validate_dataset(data)
    if dataset_fingerprint(data) != metadata["fingerprint"]:
        raise ValueError(f"Processed tensor checksum mismatch: {path}")
    _check_source_hashes(root, metadata["sources"])
    data.metadata = metadata
    data.num_classes = metadata["summary"]["num_classes"]
    return data


def get_dataset(
    name: str,
    mode: str = "raw",
    root: str | Path | None = None,
    *,
    offline: bool = False,
    rebuild: bool = False,
) -> Data:
    """Return CPU Data with [N, 10] masks and unnormalized features.

    The model graph is undirected, without self-loops or duplicate edges.
    source_edge_index preserves edges before this processing. node_id records
    vertex order within a version, not a mapping between raw and filtered nodes.
    Existing caches need no network. offline forbids downloading missing sources.
    """
    name, mode = name.lower(), mode.lower()
    if (name, mode) not in SUPPORTED_DATASETS:
        raise ValueError(
            f"Unsupported dataset condition: {name}/{mode}. "
            "Actor has only raw mode; no official filtered version is configured."
        )
    root = DEFAULT_ROOT if root is None else Path(root).resolve()
    cache_path = root / "processed" / name / mode / "data.pt"
    if cache_path.exists() and not rebuild:
        cached = _read_cache(cache_path, root, name, mode)
        logger.info("Loaded cache: %s/%s", name, mode)
        return cached

    sources = _source_files(name, mode)
    missing = [source for source in sources if not (root / source["path"]).is_file()]
    if offline and missing:
        raise FileNotFoundError(
            f"Offline preparation needs {root / missing[0]['path']}; "
            "run scripts.prepare_data once with network access."
        )
    for source in missing:
        _download_file(source["url"], root / source["path"])

    if mode == "filtered":
        data = _load_filtered(root / sources[0]["path"])
        split_source = "Platonov et al. official NPZ masks"
    elif name == "actor":
        data = _Actor(root=str(root), force_reload=True)[0]
        split_source = "Geom-GCN film_split_0.6_0.2_{0..9}"
    else:
        data = _WikipediaNetwork(
            root=str(root), name=name, geom_gcn_preprocess=True, force_reload=True
        )[0]
        split_source = f"Geom-GCN {name}_split_0.6_0.2_{{0..9}}"

    data.x = data.x.to(torch.float32).contiguous()
    data.y = data.y.to(torch.int64).contiguous()
    data.source_edge_index = data.edge_index.to(torch.int64).clone().contiguous()
    data.node_id = torch.arange(data.x.shape[0])
    edge_index, _ = remove_self_loops(data.source_edge_index)
    data.edge_index = to_undirected(edge_index, num_nodes=data.x.shape[0])
    for field in ("train_mask", "val_mask", "test_mask"):
        mask = getattr(data, field)
        if not torch.all((mask == 0) | (mask == 1)):
            raise ValueError(f"{name}/{mode}: {field} contains nonbinary values")
        setattr(data, field, mask.to(torch.bool).contiguous())

    summary = validate_dataset(data)
    for source in sources:
        path = root / source["path"]
        source["sha256"] = file_sha256(path)
        source["bytes"] = path.stat().st_size
    metadata = {
        "cache_version": CACHE_VERSION,
        "dataset": name,
        "mode": mode,
        "source_revision": FILTERED_REVISION if mode == "filtered" else GEOM_GCN_REVISION,
        "split_source": split_source,
        "split_ids": list(range(10)),
        "split_seed": None,
        "graph_policy": GRAPH_POLICY,
        "source_edge_policy": "upstream_PyG_edges" if mode == "raw" else "upstream_NPZ_edges",
        "feature_transform": "none",
        "node_id_scope": "version_local; no raw-filtered mapping supplied",
        "fingerprint": dataset_fingerprint(data),
        "summary": summary,
        "sources": sources,
        "versions": {
            "torch": str(torch.__version__),
            "numpy": np.__version__,
            "torch_geometric": torch_geometric.__version__,
        },
    }
    tensors = {field: getattr(data, field) for field in TENSOR_FIELDS}
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(dir=cache_path.parent, delete=False) as target:
            temporary_path = Path(target.name)
            torch.save({"tensors": tensors, "metadata": metadata}, target)
        os.replace(temporary_path, cache_path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    data.metadata = metadata
    data.num_classes = summary["num_classes"]
    logger.info("Prepared %s/%s: %s nodes", name, mode, summary["num_nodes"])
    return data
