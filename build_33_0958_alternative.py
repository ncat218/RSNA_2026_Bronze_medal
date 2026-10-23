"""Repackage the supplied R6/Raptor 0.958 candidate without changing scoring code."""

import copy
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "31_RSNA_0958_Rebuild_Strict.ipynb"
TARGET = ROOT / "33_RSNA_0958_R6_Raptor_Submit.ipynb"
MANIFEST = ROOT / "33_RSNA_0958_R6_Raptor_Manifest.json"
EXPECTED_SOURCE_SHA256 = "89afbd46977d4676a1c5f3344b930b7fe06e4032f3ac326034bb9777c4cda244"


def code_hashes(notebook):
    return [hashlib.sha256("".join(c["source"]).encode()).hexdigest()
            for c in notebook["cells"] if c["cell_type"] == "code"]


def main():
    source_bytes = SOURCE.read_bytes()
    source_hash = hashlib.sha256(source_bytes).hexdigest()
    if source_hash != EXPECTED_SOURCE_SHA256:
        raise RuntimeError("The supplied alternative changed; inspect it before repackaging")
    source = json.loads(source_bytes)
    if len(source["cells"]) != 9 or len(code_hashes(source)) != 7:
        raise RuntimeError("Expected the nine-cell R6/Raptor notebook")
    rebuilt = copy.deepcopy(source)
    for cell in rebuilt["cells"]:
        cell["metadata"].pop("papermill", None)
        if cell["cell_type"] == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
    rebuilt["metadata"].pop("widgets", None)
    kaggle = rebuilt["metadata"]["kaggle"]
    kaggle["accelerator"] = "nvidiaTeslaT4"
    kaggle["isGpuEnabled"] = True
    kaggle["isInternetEnabled"] = False
    if len(kaggle["dataSources"]) != 4:
        raise RuntimeError("Expected competition plus three pinned data sources")
    rebuilt["cells"][0]["source"] = (
        "# 33 — RSNA Knee R6 + Raptor 384/224: Public 0.958 candidate\n\n"
        "User-reported Public score: 0.958. This is the separately supplied nine-cell "
        "R6/Raptor notebook, repackaged for submission. All seven code cells and "
        "all scoring weights are byte-identical to the supplied source. The Kaggle "
        "setting now enables two T4 GPUs; Internet remains off. Saved three-study "
        "outputs are cleared. Attachments remain the competition and the same "
        "three dataset versions.\n\n"
        "The supplied `results 31.zip` came from a different, 55-cell v6/CoAt "
        "notebook and contains only a complete three-study run. It does not "
        "identify the hidden-rerun exception.\n\n"
        "---\n\n" + "".join(source["cells"][0]["source"])
    ).splitlines(keepends=True)
    if code_hashes(rebuilt) != code_hashes(source):
        raise RuntimeError("The repackaging changed executable source")
    TARGET.write_text(json.dumps(rebuilt, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    manifest = {
        "source": SOURCE.name,
        "source_sha256": source_hash,
        "output": TARGET.name,
        "output_sha256": hashlib.sha256(TARGET.read_bytes()).hexdigest(),
        "code_sha256": code_hashes(source),
        "scoring_code_changed": False,
        "gpu_metadata_enabled": True,
        "user_reported_public_score": 0.958,
        "hidden_submission_verified": False,
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(TARGET)


if __name__ == "__main__":
    main()
