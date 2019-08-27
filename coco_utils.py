def decode_keypoints(keypoints):
    x = keypoints[::3]
    y = keypoints[1::3]
    v = keypoints[2::3]
    return x, y, v
