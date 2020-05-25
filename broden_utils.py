import numpy as np


def MydrawMask(img, masks, lr=(None, None), clrs=None):
    n, h, w = masks.shape[0], masks.shape[1], masks.shape[2]
    if lr[0] is None:
        lr = (0, n)
    alpha = [.6, .6, .6]
    if clrs is None:
        clrs = np.zeros((n, 3)).astype(np.float)
        for i in range(n):
            for j in range(3):
                clrs[i][j] = np.random.random() * .6 + .4

    for i in range(max(0, lr[0]), min(n, lr[1])):
        M = masks[i].reshape(-1)
        B = np.zeros(h * w, dtype=np.int8)
        ix, ax, iy, ay = 99999, 0, 99999, 0
        for y in range(h - 1):
            for x in range(w - 1):
                k = y * w + x
                if M[k] == 1:
                    ix = min(ix, x)
                    ax = max(ax, x)
                    iy = min(iy, y)
                    ay = max(ay, y)
                if M[k] != M[k + 1]:
                    B[k], B[k + 1] = 1, 1
                if M[k] != M[k + w]:
                    B[k], B[k + w] = 1, 1
                if M[k] != M[k + 1 + w]:
                    B[k], B[k + 1 + w] = 1, 1
        M.shape = (h, w)
        B.shape = (h, w)
        for j in range(3):
            O, c, a = img[:, :, j], clrs[i][j], alpha[j]
            am = a * M
            O = O - O * am + c * am * 255
            img[:, :, j] = O * (1 - B) + c * B
    return img


def maskrcnn_colorencode(img, label_map, color_list):
    # do not modify original list
    label_map = np.array(np.expand_dims(label_map, axis=0), np.uint8)
    label_list = list(np.unique(label_map))
    out_img = img.copy()
    for i, label in enumerate(label_list):
        if label == 0:
            continue
        this_label_map = (label_map == label)
        alpha = [0, 0, 0]
        o = i
        if o >= 6:
            o = np.random.randint(1, 6)
        o_lst = [o % 2, (o // 2) % 2, o // 4]
        for j in range(3):
            alpha[j] = np.random.random() * 0.5 + 0.45
            alpha[j] *= o_lst[j]
        out_img = MydrawMask(out_img, this_label_map, clrs=np.expand_dims(color_list[label], axis=0))
    return out_img


def visualize_result(img, object_result, part_result, tree):
    np.random.seed(233)
    color_list = np.random.rand(1000, 3) * .7 + .3

    # object
    object_result_colored = maskrcnn_colorencode(img, object_result, color_list)

    # part
    img_part_pred = img.copy()
    valid_object = np.zeros_like(object_result)
    present_obj_labels = np.unique(object_result)
    for obj_part_index, object_label in enumerate(tree.obj_with_parts):
        if object_label not in present_obj_labels:
            continue
        object_mask = (object_result == object_label)
        valid_object += object_mask
        part_result_masked = part_result[obj_part_index] * object_mask
        present_part_label = np.unique(part_result_masked)
        if len(present_part_label) == 1:
            continue
        img_part_pred = maskrcnn_colorencode(
            img_part_pred, part_result_masked + object_mask, color_list)

    return object_result_colored, img_part_pred
