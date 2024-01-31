from collections import defaultdict
from typing import Any, Dict, List, Optional
import pickle

def clip_aggregation(loader_output: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    with open("./loader_output_mapdataset_v1_q90_th95_all_vids.pkl", "wb") as f:
        pickle.dump(loader_output, f)

    return loader_output

def frame_aggregation(loader_output: List[Dict[str, Any]], name: Optional[str] = None) -> List[Dict[str, Any]]:
    model_name = 'torchscript_ensemble'
    save_name = f"./loader_output_{model_name}_{name}.pkl" if name is not None else f"./loader_output_{model_name}.pkl"
    with open(save_name, "wb") as f:
        pickle.dump(loader_output, f)

    return loader_output
