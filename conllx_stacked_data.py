__author__ = 'max'

import random
import numpy as np
import torch
from .conllx_data import _buckets, PAD_ID_WORD, PAD_ID_CHAR, PAD_ID_TAG, UNK_ID
from .conllx_data import NUM_SYMBOLIC_TAGS
from .conllx_data import create_alphabets
from . import utils
from .reader import CoNLLXReader



def _obtain_child_index_for_left2right(heads):
    child_ids = [[] for _ in range(len(heads))]
    # skip the symbolic root.
    for child in range(1, len(heads)):
        head = heads[child]
        child_ids[head].append(child)
    return child_ids


def _obtain_child_index_for_inside_out(heads):
    child_ids = [[] for _ in range(len(heads))]
    for head in range(len(heads)):
        # first find left children inside-out
        for child in reversed(range(1, head)):
            if heads[child] == head:
                child_ids[head].append(child)
        # second find right children inside-out
        for child in range(head + 1, len(heads)):
            if heads[child] == head:
                child_ids[head].append(child)
    return child_ids


def _obtain_child_index_for_depth(heads, reverse):
    def calc_depth(head):
        children = child_ids[head]
        max_depth = 0
        for child in children:
            depth = calc_depth(child)
            child_with_depth[head].append((child, depth))
            max_depth = max(max_depth, depth + 1)
        child_with_depth[head] = sorted(child_with_depth[head], key=lambda x: x[1], reverse=reverse)
        return max_depth

    child_ids = _obtain_child_index_for_left2right(heads)
    child_with_depth = [[] for _ in range(len(heads))]
    calc_depth(0)
    return [[child for child, depth in child_with_depth[head]] for head in range(len(heads))]


def _generate_stack_inputs(heads, types, prior_order):
    """
    if prior_order == 'deep_first':
        child_ids = _obtain_child_index_for_depth(heads, True)
    elif prior_order == 'shallow_first':
        child_ids = _obtain_child_index_for_depth(heads, False)
    elif prior_order == 'left2right':
        child_ids = _obtain_child_index_for_left2right(heads)
    elif prior_order == 'inside_out':
        child_ids = _obtain_child_index_for_inside_out(heads)
    else:
        raise ValueError('Unknown prior order: %s' % prior_order)
    """

    debug=False
    
    stacked_heads = []
    children = [0 for _ in range(len(heads)-1)]
    siblings = []
    stacked_types = []
    skip_connect = []
    prev = [0 for _ in range(len(heads))]
    sibs = [0 for _ in range(len(heads))]
    stack = [1]
    position = 1

    for child in range(len(heads)):
        if child == 0: continue
        stacked_heads.append(child)
        head=heads[child]
        
        siblings.append(sibs[head])
        skip_connect.append(prev[head])
        prev[head] = position
        children[child-1]=head
        sibs[head] = child
        stacked_types.append(types[child])
        position += 1

        if debug:
            print 'stacked_heads', stacked_heads
            print 'stacked_types', stacked_types
            print 'children', children

    if debug:exit(0) 

    return stacked_heads, children, siblings, stacked_types, skip_connect


def read_stacked_data(source_path, word_alphabet, char_alphabet, pos_alphabet, type_alphabet, max_size=None, normalize_digits=True, prior_order='deep_first'):
    data = [[] for _ in _buckets]
    max_char_length = [0 for _ in _buckets]
    print('Reading data from %s' % source_path)
    counter = 0
    reader = CoNLLXReader(source_path, word_alphabet, char_alphabet, pos_alphabet, type_alphabet)
    inst = reader.getNext(normalize_digits=normalize_digits, symbolic_root=True, symbolic_end=False)
    while inst is not None and (not max_size or counter < max_size):
        counter += 1
        if counter % 10000 == 0:
            print("reading data: %d" % counter)

        inst_size = inst.length()
        sent = inst.sentence
        for bucket_id, bucket_size in enumerate(_buckets):
            if inst_size < bucket_size:
                stacked_heads, children, siblings, stacked_types, skip_connect = _generate_stack_inputs(inst.heads, inst.type_ids, prior_order)
                data[bucket_id].append([sent.word_ids, sent.char_id_seqs, inst.pos_ids, inst.heads, inst.type_ids, stacked_heads, children, siblings, stacked_types, skip_connect])
                max_len = max([len(char_seq) for char_seq in sent.char_seqs])
                if max_char_length[bucket_id] < max_len:
                    max_char_length[bucket_id] = max_len
                break

        inst = reader.getNext(normalize_digits=normalize_digits, symbolic_root=True, symbolic_end=False)
    reader.close()
    print("Total number of data: %d" % counter)
    return data, max_char_length

def read_stacked_data_to_tensor(source_path, word_alphabet, char_alphabet, pos_alphabet, type_alphabet,
                                max_size=None, normalize_digits=True, prior_order='deep_first', device=torch.device('cpu')):
    """
    Read stacked data from source_path and convert to tensors.

    Args:
        source_path (str): Path to data file.
        word_alphabet (Alphabet): Word alphabet.
        char_alphabet (Alphabet): Character alphabet.
        pos_alphabet (Alphabet): Part-of-speech alphabet.
        type_alphabet (Alphabet): Dependency type alphabet.
        max_size (int, optional): Maximum size of data. Defaults to None.
        normalize_digits (bool, optional): Normalize digits. Defaults to True.
        prior_order (str, optional): Prior order ('deep_first' or 'left_first'). Defaults to 'deep_first'.
        device (torch.device, optional): Device. Defaults to torch.device('cpu').

    Returns:
        tuple: Data tensor and bucket sizes.
    """

    # Read data and max char length from source_path
    data, max_char_length = read_stacked_data(source_path, word_alphabet, char_alphabet, pos_alphabet, type_alphabet, 
                                              max_size=max_size, normalize_digits=normalize_digits, prior_order=prior_order)

    bucket_sizes = [len(data[b]) for b in range(len(_buckets))]
    data_tensor = []

    for bucket_id in range(len(_buckets)):
        bucket_size = bucket_sizes[bucket_id]
        if bucket_size == 0:
            data_tensor.append((1, 1))
            continue

        bucket_length = _buckets[bucket_id]
        char_length = min(utils.MAX_CHAR_LENGTH, max_char_length[bucket_id] + utils.NUM_CHAR_PAD)

        # Initialize numpy arrays
        wid_inputs = np.empty([bucket_size, bucket_length], dtype=np.int64)
        cid_inputs = np.empty([bucket_size, bucket_length, char_length], dtype=np.int64)
        pid_inputs = np.empty([bucket_size, bucket_length], dtype=np.int64)
        hid_inputs = np.empty([bucket_size, bucket_length], dtype=np.int64)
        tid_inputs = np.empty([bucket_size, bucket_length], dtype=np.int64)
        masks_e = np.zeros([bucket_size, bucket_length], dtype=np.float32)
        single = np.zeros([bucket_size, bucket_length], dtype=np.int64)
        lengths_e = np.empty(bucket_size, dtype=np.int64)
        stack_hid_inputs = np.empty([bucket_size, bucket_length - 1], dtype=np.int64)
        chid_inputs = np.empty([bucket_size, bucket_length - 1], dtype=np.int64)
        ssid_inputs = np.empty([bucket_size, bucket_length - 1], dtype=np.int64)
        stack_tid_inputs = np.empty([bucket_size, bucket_length - 1], dtype=np.int64)
        skip_connect_inputs = np.empty([bucket_size, bucket_length - 1], dtype=np.int64)
        masks_d = np.zeros([bucket_size, bucket_length - 1], dtype=np.float32)
        lengths_d = np.empty(bucket_size, dtype=np.int64)

        for i, inst in enumerate(data[bucket_id]):
            wids, cid_seqs, pids, hids, tids, stack_hids, chids, ssids, stack_tids, skip_ids = inst

            # Replace None values with default padding values
            wids = [wid if wid is not None else PAD_ID_WORD for wid in wids]
            cid_seqs = [[cid if cid is not None else PAD_ID_CHAR for cid in cids] if cids is not None else [PAD_ID_CHAR] * char_length for cids in cid_seqs]
            pids = [pid if pid is not None else PAD_ID_TAG for pid in pids]
            hids = [hid if hid is not None else PAD_ID_TAG for hid in hids]
            tids = [tid if tid is not None else PAD_ID_TAG for tid in tids]
            stack_hids = [hid if hid is not None else PAD_ID_TAG for hid in stack_hids]
            chids = [chid if chid is not None else PAD_ID_TAG for chid in chids]
            ssids = [ssid if ssid is not None else PAD_ID_TAG for ssid in ssids]
            stack_tids = [tid if tid is not None else PAD_ID_TAG for tid in stack_tids]
            skip_ids = [skip_id if skip_id is not None else PAD_ID_TAG for skip_id in skip_ids]

            # Calculate inst_size
            inst_size = min(bucket_length, len(wids))

                        # Pad sequences
            wid_inputs[i, :inst_size] = wids[:inst_size]
            wid_inputs[i, inst_size:] = PAD_ID_WORD

            for c, cids in enumerate(cid_seqs):
                if c < bucket_length:
                    cid_inputs[i, c, :len(cids)] = cids[:min(char_length, len(cids))]
                    cid_inputs[i, c, len(cids):] = PAD_ID_CHAR
            cid_inputs[i, inst_size:, :] = PAD_ID_CHAR

            pid_inputs[i, :inst_size] = pids[:inst_size]
            pid_inputs[i, inst_size:] = PAD_ID_TAG

            tid_inputs[i, :inst_size] = tids[:inst_size]
            tid_inputs[i, inst_size:] = PAD_ID_TAG

            hid_inputs[i, :inst_size] = hids[:inst_size]
            hid_inputs[i, inst_size:] = PAD_ID_TAG

            masks_e[i, :inst_size] = 1.0

            for j, wid in enumerate(wids):
                if word_alphabet.is_singleton(wid):
                    single[i, j] = 1

            lengths_e[i] = inst_size

            # Decoder inputs
            inst_size_decoder = min(bucket_length - 1, len(stack_hids))
            stack_hid_inputs[i, :inst_size_decoder] = stack_hids[:inst_size_decoder]
            stack_hid_inputs[i, inst_size_decoder:] = PAD_ID_TAG

            chid_inputs[i, :inst_size_decoder] = chids[:inst_size_decoder]
            chid_inputs[i, inst_size_decoder:] = PAD_ID_TAG

            ssid_inputs[i, :inst_size_decoder] = ssids[:inst_size_decoder]
            ssid_inputs[i, inst_size_decoder:] = PAD_ID_TAG

            stack_tid_inputs[i, :inst_size_decoder] = stack_tids[:inst_size_decoder]
            stack_tid_inputs[i, inst_size_decoder:] = PAD_ID_TAG

            skip_connect_inputs[i, :inst_size_decoder] = skip_ids[:inst_size_decoder]
            skip_connect_inputs[i, inst_size_decoder:] = PAD_ID_TAG

            masks_d[i, :inst_size_decoder] = 1.0
            lengths_d[i] = inst_size_decoder

        # Convert numpy arrays to PyTorch tensors
        words = torch.from_numpy(wid_inputs).to(device)
        chars = torch.from_numpy(cid_inputs).to(device)
        pos = torch.from_numpy(pid_inputs).to(device)
        heads = torch.from_numpy(hid_inputs).to(device)
        types = torch.from_numpy(tid_inputs).to(device)
        masks_e = torch.from_numpy(masks_e).to(device)
        single = torch.from_numpy(single).to(device)
        lengths_e = torch.from_numpy(lengths_e).to(device)
        stacked_heads = torch.from_numpy(stack_hid_inputs).to(device)
        children = torch.from_numpy(chid_inputs).to(device)
        siblings = torch.from_numpy(ssid_inputs).to(device)
        stacked_types = torch.from_numpy(stack_tid_inputs).to(device)
        skip_connect = torch.from_numpy(skip_connect_inputs).to(device)
        masks_d = torch.from_numpy(masks_d).to(device)
        lengths_d = torch.from_numpy(lengths_d).to(device)

        data_tensor.append((words, chars, pos, heads, types, masks_e, single, lengths_e, 
                            stacked_heads, children, siblings, stacked_types, skip_connect, masks_d, lengths_d))

    return data_tensor, bucket_sizes


def get_batch_stacked_tensor(data, batch_size, unk_replace=0.):
    data_tensor, bucket_sizes = data
    total_size = float(sum(bucket_sizes))
    buckets_scale = [sum(bucket_sizes[:i + 1]) / total_size for i in range(len(bucket_sizes))]

    random_number = np.random.random_sample()
    bucket_id = min([i for i in range(len(buckets_scale)) if buckets_scale[i] > random_number])
    bucket_length = _buckets[bucket_id]

    words, chars, pos, heads, types, masks_e, single, lengths_e, \
    stacked_heads, children, siblings, stacked_types, skip_connect, masks_d, lengths_d = data_tensor[bucket_id]

    data_encoder = (words, chars, pos, heads, types, masks_e, single, lengths_e)
    data_decoder = (stacked_heads, children, siblings, stacked_types, skip_connect, masks_d, lengths_d)

    bucket_size = bucket_sizes[bucket_id]
    batch_size = min(bucket_size, batch_size)
    index = torch.randperm(bucket_size).long()[:batch_size]
    index = index.to(words.device)

    words = words[index]
    if unk_replace:
        ones = single.new_ones(batch_size, bucket_length)
        noise = masks_e.new_empty(batch_size, bucket_length).bernoulli_(unk_replace).long()
        words = words * (ones - single[index] * noise)

    return (words, chars[index], pos[index], heads[index], types[index], masks_e[index], lengths_e[index]), \
           (stacked_heads[index], children[index], siblings[index], stacked_types[index], skip_connect[index], masks_d[index], lengths_d[index])


def iterate_batch_stacked_variable(data, batch_size, unk_replace=0., shuffle=False):
    data_tensor, bucket_sizes = data

    bucket_indices = np.arange(len(_buckets))
    if shuffle:
        np.random.shuffle((bucket_indices))

    for bucket_id in bucket_indices:
        bucket_size = bucket_sizes[bucket_id]
        bucket_length = _buckets[bucket_id]
        if bucket_size == 0:
            continue
        words, chars, pos, heads, types, masks_e, single, lengths_e, \
        stacked_heads, children, siblings, stacked_types, skip_connect, masks_d, lengths_d = data_tensor[bucket_id]

        data_encoder = (words, chars, pos, heads, types, masks_e, single, lengths_e)
        data_decoder = (stacked_heads, children, siblings, stacked_types, skip_connect, masks_d, lengths_d)

        if unk_replace:
            ones = single.new_ones(bucket_size, bucket_length)
            noise = masks_e.new_empty(bucket_size, bucket_length).bernoulli_(unk_replace).long()
            words = words * (ones - single * noise)

        indices = None
        if shuffle:
            indices = torch.randperm(bucket_size).long()
            indices = indices.to(words.device)
        for start_idx in range(0, bucket_size, batch_size):
            if shuffle:
                excerpt = indices[start_idx:start_idx + batch_size]
            else:
                excerpt = slice(start_idx, start_idx + batch_size)
            yield (words[excerpt], chars[excerpt], pos[excerpt], heads[excerpt], types[excerpt], masks_e[excerpt], lengths_e[excerpt]), \
                  (stacked_heads[excerpt], children[excerpt], siblings[excerpt], stacked_types[excerpt], skip_connect[excerpt], masks_d[excerpt], lengths_d[excerpt])