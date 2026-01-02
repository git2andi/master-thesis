from typing import List, Tuple, Dict, Any
import random

# Select positive and negative frame indices based on a negative_ratio policy.
# Always keeps all positive frames (indices in frames_wbox_indexes)
# For Negative Frames: 
#   Fraction of negative frames to keep per video, in [0,1].
#       0.0 > no negatives
#       1.0 > keep all negatives
#       -1 > match num of negatives with positives (so 1:1 ratio)

def select_pos_neg_frames(
    frames_wbox_indexes: List[int],
    frames_nobox_indexes: List[int],
    negative_ratio: float,
    seed: int = 1000,
) -> Tuple[List[int], List[int], Dict[str, Any]]:

    selected_pos_idxs = sorted(frames_wbox_indexes)
    num_pos = len(selected_pos_idxs)
    num_neg = len(frames_nobox_indexes)

    # Decide how many negatives to keep
    if negative_ratio == -1.0:
        # SPECIAL CASE: match number of negatives to number of positives (as far as possible)
        to_keep = min(num_neg, num_pos)
        mode = "match_pos_neg"
    else:
        # ORIGINAL BEHAVIOR: fraction of all negatives
        if not (0.0 <= negative_ratio <= 1.0):
            raise ValueError(
                f"Invalid 'negative_ratio' {negative_ratio}, must be in [0,1] "
                f"or == {-1} for match_pos_neg"
            )
        to_keep = int(negative_ratio * num_neg)
        # clamp to valid range
        to_keep = max(0, min(num_neg, to_keep))
        mode = "fraction_of_negatives"

    # Sample negatives
    if to_keep > 0 and num_neg > 0:
        random.seed(seed)
        selected_neg_idxs = random.sample(frames_nobox_indexes, to_keep)
        selected_neg_idxs = sorted(selected_neg_idxs)
    else:
        selected_neg_idxs = []

    info = {
        "mode": mode,
        "num_pos": num_pos,
        "num_neg": num_neg,
        "num_neg_kept": len(selected_neg_idxs),
        "negative_ratio": negative_ratio,
    }

    return selected_pos_idxs, selected_neg_idxs, info
