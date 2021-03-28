import numpy as np


def max_regarding_to_abs(a, b):
    c = np.zeros(a.shape)
    for i in range(len(a)):
        for j in range(len(a[0])):
            if abs(a[i][j]) >= abs(b[i][j]):
                c[i][j] = a[i][j]
            else:
                c[i][j] = b[i][j]
    return c


def reverse_preprocess_imagenet_image(data):
    # Showing the image itself
    data_np = data.clone().detach().cpu().numpy()
    data_np[0][0] = data_np[0][0] * 0.229 + 0.485
    data_np[0][1] = data_np[0][1] * 0.224 + 0.456
    data_np[0][2] = data_np[0][2] * 0.225 + 0.406
    return np.asarray(data_np.squeeze(0).transpose((1, 2, 0)))