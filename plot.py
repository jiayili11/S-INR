import matplotlib.pyplot as plt
import scipy
import numpy as np
import matplotlib
methods = ["_dt", "_knr", "_rf", "_siren", "_sacinr"]  # , "_dt", "_knr", "_rf"
cases = [0.75]  # 0.75, 0.8, 0.85,

for types, name, color in zip([2, 3, 4, 5, 6], ['Prcp', 'SWE', 'VP', 'Tmax', 'Tmin'],
                              [(0.51, 0.69, 0.82), (0.56, 0.81, 0.79), (1, 0.75, 0.48), (1, 0.50, 0.44), (0.75, 0.72, 0.86)]):
    for method in methods:
        for Case in cases:
            data = scipy.io.loadmat("./mat_2d/data1_" + str(Case) + method)["clean_data"]
            fig = plt.figure(figsize=(12, 10))
            ax = plt.axes(projection='3d')
            ax.scatter3D(data[:, 0], data[:, 1], data[:, types], color=color)
            # 蓝 0.51 0.69 0.82 绿 0.56 0.81 0.79 黄 1 0.75 0.48 红 1 0.50 0.44
            # ax.grid(None)
            # ax.axis('equal')  # {equal, scaled}
            # ax.axis('off')
            ax.set_xlabel('Longitude', fontweight='bold')
            ax.set_ylabel('Latitude', fontweight='bold')
            ax.set_zlabel(name, fontweight='bold', rotation=270, ha='right')

            # ax.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
            # plt.gca().set_aspect('equal', adjustable='box')
            ax.view_init(elev=30, azim=-60)
            plt.savefig('./final_eps/' + "data1_" + str(Case) + str(types) + method + ".png", format='png', bbox_inches="tight")
            #plt.show()

            train = scipy.io.loadmat("./mat_2d/data_1_train_"+str(Case))["train"]
            test = scipy.io.loadmat("./mat_2d/data_1_test_"+str(Case))["test"]
            data = np.concatenate([test, train], axis=0)
            # fig = plt.figure(figsize=(12, 10))
            ax = plt.axes(projection='3d')
            ax.scatter3D(data[:, 0], data[:, 1], data[:, types], color=color)
            # 蓝 0.51 0.69 0.82 绿 0.56 0.81 0.79 黄 1, 0.75, 0.48 红 1, 0.50, 0.44 紫 0.75, 0.72, 0.86
            # ax.grid(None)
            # ax.axis('equal')  # {equal, scaled}
            # ax.axis('off')
            ax.set_xlabel('Longitude', fontweight='bold')
            ax.set_ylabel('Latitude', fontweight='bold')
            ax.set_zlabel(name, fontweight='bold', rotation=270, ha='right')

            # ax.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
            # plt.gca().set_aspect('equal', adjustable='box')
            ax.view_init(elev=30, azim=-60)
            plt.savefig('./final_eps/' + "data1_" + str(Case) + str(types) + "original" + ".png", format='png', bbox_inches="tight")

            train = scipy.io.loadmat("./mat_2d/data_1_train_"+str(Case))["train"]
            data = train
            # fig = plt.figure(figsize=(12, 10))
            ax = plt.axes(projection='3d')
            ax.scatter3D(data[:, 0], data[:, 1], data[:, types], color=color)
            # 蓝 0.51 0.69 0.82 绿 0.56 0.81 0.79 黄 1 0.75 0.48 红 1 0.50 0.44
            # ax.grid(None)
            # ax.axis('equal')  # {equal, scaled}
            # ax.axis('off')
            ax.set_xlabel('Longitude', fontweight='bold')
            ax.set_ylabel('Latitude', fontweight='bold')
            ax.set_zlabel(name, fontweight='bold', rotation=270, ha='right')

            # ax.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
            # plt.gca().set_aspect('equal', adjustable='box')
            ax.view_init(elev=30, azim=-60)
            plt.savefig('./final_eps/' + "data1_" + str(Case) + str(types) + "observe" + ".png", format='png', bbox_inches="tight")
            #plt.show()





# fig = plt.figure(figsize=(12, 10))
# ax = plt.axes(projection='3d')
# colors = ['red', 'green', 'blue', 'orange', 'purple', 'brown', 'teal', 'gray', 'red', 'green', 'blue', 'orange']
# for i, cluster in enumerate(clusters):
#     x = cluster[:, 0].cpu().numpy()
#     y = cluster[:, 1].cpu().numpy()
#     z = cluster[:, 2].cpu().numpy()
#
#     ax.scatter3D(x, y, z, color=colors[i])
#
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_zlabel('z')
# #
# plt.show()

