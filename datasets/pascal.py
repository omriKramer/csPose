import json
from pathlib import Path

import numpy as np
import torch
from scipy.io import loadmat

from datasets.broden_adapter import BrodenAdapter


def load_part2ind(part2ind_filename):
    """
    Parses the part2ind.m file from the PASCAL Parts distribution
    just as a series of line patterns, returning a map from (object, part)
    number paris to (objectname, partname) names.
    """
    import re
    from collections import OrderedDict
    with open(part2ind_filename, 'r') as part2ind_file:
        lines = part2ind_file.readlines()
    result = OrderedDict()
    for line in lines:
        # % [aeroplane]
        m = re.match('^% \[([^\]]*)\]', line)
        if m:
            object_name = m.group(1)
            continue
        # pimap{1}('lwing')       = 3;                % left wing
        m = re.match(r'''(?x)^
            pimap\{(\d+)\}            # group 1: the object index
            \('([^']*)'\)             # group 2: the part short name
            \s*=\s*                   # equals
            (\d+);                    # group 3: the part number
            (?:\s*%\s*(\w[\w ]*\w))?  # group 4: the part long name
            ''', line)
        if m:
            part_name = m.group(2)
            readable_name = m.group(4) or part_name
            object_index = int(m.group(1))
            part_index = int(m.group(3))
            # ('aeroplane', 'left wing')
            result[(object_index, part_name)] = (
                object_name, readable_name)
            continue
        # for ii = 1:10
        m = re.match(r'for ii = 1:(\d+)*', line)
        if m:
            iteration_count = int(m.group(1))
            continue

        # pimap{1}(sprintf('engine_%d', ii)) = 10+ii; % multiple engines
        m = re.match(r'''(?x)^\s*
            pimap\{ (\d+) \}          # group 1: the object index
            \(sprintf\('
                ([^']*)_%d            # group 2: the short part name
            ',\s*ii\)\)
            \s*=\s*                   # equals
            (\d+)                     # group 3: the part number
            \s*\+\s*ii\s*;            # plus indexing
            (?:\s*%\s*(\w[\w ]*\w))?  # group 4: the multiple-part name
            ''', line)
        if m:
            # Deal with ranges
            part_name = m.group(2)
            # Take a multiple name if it does not say 'multiple' in it.
            readable_name = part_name
            if m.group(4) and 'multiple' not in m.group(4):
                readable_name = m.group(4)
            object_index = int(m.group(1))
            first = int(m.group(3)) + 1  # one-based indexing
            for part_index in range(1, 1 + iteration_count):
                result[(object_index, part_name + '_%d' % part_index)] = (
                    object_name, readable_name)
            continue

        # % only has sihouette mask
        m = re.match(r'% only has si', line)  # (misspelled in file)
        if m:
            object_index += 1
            result[(object_index, 'silhouette')] = (
                object_name, 'silhouette')

        # keySet = keys(pimap{8});
        # valueSet = values(pimap{8});
        m = re.match(r'''(?x)^
            (key|value)Set\s*=\s*\1s\(pimap{ # group 1: key or value
            (\d+)                            # group 2: old object number
            }\);
            ''', line)
        if m:
            object_to_copy = int(m.group(2))
            continue

        # pimap{12} = containers.Map(keySet, valueSet);
        m = re.match(r'''(?x)^
            pimap{
            (\d+)                           # group 1: new object number
            }\s*=\s*containers.Map\(keySet,\s*valueSet\);
            ''', line)
        if m:
            object_index = int(m.group(1))
            for (other_obj, part_index), val in list(result.items()):
                if other_obj == object_to_copy:
                    result[(object_index, part_index)] = (object_name, val[1])
            continue
        # recognize other lines that can be ignored
        m = re.match(r'''(?x)^(?:
                function|
                pimap\ =|
                \s*pimap\{ii\}|
                end|
                \s*%|
                remove|
                $)''', line)
        if m:
            continue
        print('unrecognized line', line)
        import sys
        sys.exit(1)
    return result


def normalize_all_readable(raw_keys, collapse_adjectives):
    return dict((k, (normalize_readable(c, collapse_adjectives),
                     normalize_readable(p, collapse_adjectives)))
                for k, (c, p) in list(raw_keys.items()))


def normalize_readable(name, collapse_adjectives):
    # Long names for short part names that are unexplained in the file.
    decoded_names = dict(
        lfho='left front hoof',
        rfho='right front hoof',
        lbho='left back hoof',
        rbho='right back hoof',
        fwheel='front wheel',
        bwheel='back wheel',
        frontside='front side',
        leftside='left side',
        rightside='right side',
        backside='back side',
        roofside='roof side',
        leftmirror='left mirror',
        rightmirror='right mirror'
    )
    if name in decoded_names:
        name = decoded_names[name]
    # Other datasets use 'airplane'.
    name = name.replace('aeroplane', 'airplane')
    # If we need to remove adjectives like 'left' and 'right', do so now.h
    if collapse_adjectives is not None:
        name = ' '.join(n for n in name.split() if n not in collapse_adjectives)
    return name


def normalize_part_key(raw_names):
    """
    Enforces a coding policy: all the multiple part names such as 'engine 3'
    are just called 'engine', and they are aliased down to the same part code.
    Returns names of object classes, parts, and a part_key mapping raw
    (object, part) number pair tuples to the canonical integer part code.
    """
    # Identify the maximum object and part indexes used.
    object_class_count = max(c for c, p in raw_names) + 1
    # Create a list of object names, a list of part names, and a part key
    object_class_names = ['-'] * object_class_count
    part_class_names = ['-']
    # alias_key map (collapsed obj name, collapsed part name) -> assigned part index.
    alias_key = {}
    part_key = {}

    for (c, p) in sorted(list(raw_names.keys())):
        n = raw_names[(c, p)]
        object_class_names[c] = n[0]
        # Alias codes that have the same name such as 'airplane engine' (1,2,3)
        if n in alias_key:
            code = alias_key[n]
        else:
            if n[1] in part_class_names:
                code = part_class_names.index(n[1])
            else:
                code = len(part_class_names)
                part_class_names.append(n[1])
            alias_key[n] = code
        part_key[(c, p)] = code
    return object_class_names, part_class_names, part_key


def load_context_labels(labels_filename):
    """
    Parses labels.txt from the PASCAL Context distribution;
    this contains an "index: name" mapping.
    """
    with open(labels_filename, 'r') as labels_file:
        lines = [s.strip().split(': ') for s in labels_file.readlines()]
    pairs = [(int(i), n) for i, n in lines]
    object_names = ['-'] * (max(i for i, n in pairs) + 1)
    for i, n in pairs:
        object_names[i] = n
    return object_names


class PascalAdapter(BrodenAdapter):
    collapse_adjectives = {'left', 'right', 'front', 'back', 'upper', 'lower', 'side'}

    def __init__(self, root):
        self.root = Path(root)
        codes = load_part2ind(self.root / 'part2ind.m')
        # Normalized names
        self.codes = normalize_all_readable(codes, self.collapse_adjectives)
        self.part_object_names, self.part_names, self.part_key = normalize_part_key(self.codes)
        # Load the PASCAL context segmentation labels
        self.object_names = load_context_labels(root / 'context_labels.txt')
        self.unknown_label = self.object_names.index('unknown')
        self.object_names[self.unknown_label] = '-'  # normalize unknown
        with (self.root / 'pascal_index_mapping.json').open() as f:
            pascal2broden = json.load(f)
        super().__init__(pascal2broden['object'], pascal2broden['part'])

    def load_parts_segmentation(self, filename):
        """
        Processes a single Annotations_Part annotation into two arrays:
        an object class segmentation, and a part segmentation.  We discard
        instance information for now.  If no objects are present, returns (None, None).
        """
        d = loadmat(filename)
        instance_count = d['anno'][0, 0]['objects'].shape[1]
        # We need at least one instance annotated
        if not instance_count:
            return None, None

        mask_shape = d['anno'][0, 0]['objects'][0, 0]['mask'].shape
        # We will merge all objects and parts into these two layers
        object_seg = np.zeros(mask_shape, dtype=np.int16)
        part_seg = np.zeros(mask_shape, dtype=np.int16)
        for i in range(instance_count):
            obj = d['anno'][0, 0]['objects'][0, i]
            object_ind = obj['class_ind'][0, 0]
            object_seg[obj['mask'].astype(np.bool)] = object_ind
            part_count = obj['parts'].shape[1]
            for j in range(part_count):
                part = obj['parts'][0, j]
                part_name = part['part_name'][0]
                part_code = self.part_key[(object_ind, part_name)]
                part_seg[part['mask'].astype(np.bool)] = part_code
        return object_seg, part_seg

    def get_obj_mask(self, obj_fn):
        obj = loadmat(obj_fn)['LabelMap']
        return obj

    def get_part_mask(self, part_fn):
        _, part = self.load_parts_segmentation(part_fn)
        return part
