from itertools import combinations
from typing import Union
from math import comb
import numpy as np
from tqdm import tqdm

def _choose_k_bits_from_n_exhaustive_generator(n: int, k: int):
    for combo in map(list, combinations(range(n), k)):
        yield np.array(combo)
    yield None


def _choose_k_bits_from_n_random_generator(n: int, k:int, np_rng: np.random.Generator):
    range_list=range(n)
    while True:
        yield np_rng.choice(range_list, size=k, replace=False)
        
def permutation_test(
    query_group: np.ndarray,
    candidate_group: np.ndarray,
    n_iters: int=10000,
    random_state: Union[int, np.random.Generator] = 7,
    verbose: bool = False,
):
    if not isinstance(random_state, (np.random.Generator, int)):
        raise ValueError(f"random_state should be an int or a np.random.Generator, it was a '{type(random_state)}'")
    if isinstance(random_state, int):
        random_state = np.random.default_rng(random_state)
    len_qg = len(query_group)
    len_cg = len(candidate_group)
    
    stacked_data=np.vstack([query_group, candidate_group])
    
    n_combinations=comb(len_qg+len_cg, len_qg)
    
    orig_d = np.linalg.norm(np.mean(query_group, axis=0) - np.mean(candidate_group, axis=0))
    n_gt_or_equal = 1
    full_range=np.array(range(len_qg+len_cg), dtype=int)
    
    if n_combinations>n_iters: # Must randomly sample
        if verbose:
            print(f"Sampling {n_iters} times from {n_combinations} possible label combinations")
        cur_iter = 1
        generator = _choose_k_bits_from_n_random_generator(len_qg+len_cg, len_qg, random_state)
        pbar=tqdm(total=n_iters, disable=not verbose)
        while cur_iter< n_iters:
            generated_indexes= next(generator)
            if np.all(np.array(generated_indexes)<len_qg):
                continue
            cur_iter+=1
            pbar.update(1)
            d=np.linalg.norm(np.mean(stacked_data[generated_indexes], axis=0)-np.mean(stacked_data[np.setxor1d(generated_indexes, full_range)], axis=0))
            if d >= orig_d:
                n_gt_or_equal+=1
    else: # Exhastively enumerate all
        if verbose:
            print(f"Enumerating {n_combinations} possible label combinations")
        n_iters=n_combinations
        generator=_choose_k_bits_from_n_exhaustive_generator(len_qg+len_cg, len_qg)
        generated_indexes=next(generator)
        pbar=tqdm(total=n_iters, disable=not verbose)
        pbar.update(1)
        while generated_indexes is not None:
            if np.all(generated_indexes<len_qg): # If correct, ignore
                generated_indexes=next(generator)
                continue
            pbar.update(1)
            d=np.linalg.norm(np.mean(stacked_data[generated_indexes], axis=0)-np.mean(stacked_data[np.setxor1d(generated_indexes, full_range)], axis=0))
            if d >= orig_d:
                n_gt_or_equal+=1
            generated_indexes=next(generator)
    return n_gt_or_equal/n_iters
