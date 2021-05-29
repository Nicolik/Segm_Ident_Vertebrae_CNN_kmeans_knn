def get_cmap():
    import matplotlib
    cmap = matplotlib.colors.ListedColormap([(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 0, 1), (1, 1 , 0), (0, 1, 1), (1, 0, 1), (0, 0, 205/255), (205/255, 133/255, 63/255),
                                             (210/255, 180/255, 140/255), (102/255, 205/255, 170/255), (0, 0, 128/255), (0, 139/255, 139/255),
                                             (46/255, 139/255, 87/255), (1, 228/255, 225/255), (106/255, 90/255, 205/255), (221/255, 160/255, 221/255),
                                             (233/255, 150/255, 122/255), (165/255, 42/255, 42/255), (1, 250/255, 250/255), (147/255, 112/255, 219/255),
                                             (218/255, 112/225, 214/255), (75/255, 0, 130/255), (1, 182/255, 193/255)], name='from_list', N=None)
    return cmap


def get_input_map():
    keys = ["C" + str(i) for i in range(1, 8)]
    keys.extend(["T" + str(i) for i in range(1, 13)])
    keys.extend(["L" + str(i) for i in range(1, 7)])
    keys.append("T13")
    input_map = {}

    for i, k in enumerate(keys):
        input_map[k] = i + 1
    return input_map
