import numpy as np
import pandas as pd

# measures the mean distance from 0.5 (multiplied by 2 to scale to 0-1)
_confident = lambda p: np.mean(np.abs(p - 0.5) * 2) >= 0.7

# _confident = lambda p: (np.mean(np.square(p - 0.5)) * 4) > 0.8
# slightly increase max_prediction
_label_spread = lambda x: x - np.log10(x) if x >= 0.8 else x

# aggregate per-frame scores to single score for the component
# if more than 90% of the frames are in one category
# then return mean of than category
# else just return mean
# works with both frame and clip scores
def clip_aggregation(clip_preds: pd.Series, threshold=0.8) -> float:
    clip_preds = clip_preds.to_numpy()
    fakes = clip_preds[clip_preds >= threshold]
    reals = clip_preds[clip_preds <= (1 - threshold)]
    # if more than 90% of preds are fake
    # if len(fakes) >= int(len(preds) * 0.9):
    #     return np.mean(fakes)
    # if len(reals) >= int(len(preds) * 0.9):
    #     return np.mean(reals)
    if len(fakes) > (2 * len(reals)):
        return np.mean(fakes)
    elif len(reals) > (2 * len(fakes)):
        return np.mean(reals)
    return np.mean(clip_preds)

# aggregate per-component scores to single score for the segment
def component_aggregation(comp_preds: pd.Series, threshold=0.8) -> float:
    comp_preds = comp_preds.to_numpy()
    if len(comp_preds) == 0:
        # return more appripriate error, such as
        # no valid components in this segment
        return np.nan
    comp_preds = np.array(comp_preds)
    max_comp = np.max(comp_preds)
    # If there is a fake id and we're _confident,
    # return spreaded fake score, otherwise return
    # the original fake score.
    if max_comp >= threshold:
        if _confident(comp_preds):
            # if _confident then slightly increase max_comp
            return _label_spread(max_comp)
        # else just return max_comp
        return max_comp
    # If everyone is real and we're _confident return
    # the minimum real score, otherwise return the
    # mean of all predictions.
    if _confident(comp_preds):
        return np.min(comp_preds)
    return np.mean(comp_preds)

# aggregate per-segment scores to single score for the video
def segment_aggregation(segment_preds: pd.Series) -> float:
    segment_preds = segment_preds.to_numpy()
    return np.max(segment_preds)
