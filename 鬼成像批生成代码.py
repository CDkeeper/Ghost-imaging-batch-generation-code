import numpy as np
import cv2 as cv
import torch
import time

batch_size = 3000  # 根据需要更改每批的图像数目
image_size = 64  # 128 图片的大小
iteration = 4000  # 每张图片的帧数

pathsrc = "C:/Users/chenda/Desktop/MNIST_use/train/"
pathtar = "C:/Users/chenda/Desktop/MNIST_use/train-test-GI-tmp/"
# cuda = torch.device('cuda')
# cuda0 = torch.device('cuda:0')

# 鬼成像
num = 0  # 第几批次？

for now in range(1):  # 注意这里是使得iteration跑now遍，那么结果的帧率就是iteration * now帧了，是跑不出iteration * now帧的无奈之举，之前是10改为1
    print("now = %d" % (now + 1))
    with torch.cuda.device(0):
        st = time.time()

        # [512, 128, 128]的数组
        image = np.zeros((batch_size, image_size, image_size))
        # 读取图片
        for i in range(batch_size):
            img = cv.imread(pathsrc + str(i + 1 + num * batch_size) + ".jpg", 0)
            image[i] = img

        # 把image转换为张量,size = [512,128,128]
        image = torch.from_numpy(image).cuda()
        # [512]
        sum_I1 = torch.zeros(batch_size).cuda()
        # [512, 128, 128]
        sum_I2 = torch.zeros(image.shape).cuda()
        # [512, 128, 128]
        sum_ans = torch.zeros(image.shape).cuda()

        # [512, 128, 128] -> [128, 128, 512]
        sum_I2 = sum_I2.permute(1, 2, 0)
        sum_ans = sum_ans.permute(1, 2, 0)

        # [512]
        I1 = torch.zeros(batch_size)
        # 提前就初始化好了iteration个随机矩阵[4000,128,128]
        field = np.random.normal(0, 1, [iteration, image_size, image_size])
        field = torch.from_numpy(field).cuda()

        for k in range(iteration):
            if (k + 1) % 100 == 0:
                print('Completion: {:.2%}'.format((k + 1) / iteration))
            # 重复第k个热光矩阵，赋值给tmp,size = [512,128,128]
            temp = field[k].repeat(batch_size, 1, 1)
            # I1是一个 512 元素的序列，每个元素是桶测量值，这是在同时对512张图片进行操作
            I1 = torch.mul(temp, image).sum(2).sum(1)
            # [512,128,128] -> [128,128,512]
            temp = temp.permute(1, 2, 0)

            # 桶测量值求和
            sum_I1 = sum_I1 + I1
            # 热光矩阵求和
            sum_I2 = sum_I2 + temp
            # 桶测量值乘热光矩阵求和
            sum_ans = sum_ans + temp * I1

        sum_I1 = sum_I1 / iteration
        sum_I2 = sum_I2 / iteration
        sum_ans = sum_ans / iteration
        # [512, 128, 128]
        ans = sum_ans - sum_I1 * sum_I2

        ed = time.time()
        print(ed - st)

        # 把GPU数据转换为CPU数据，并把张量转化为numpy数组
        tmp = ans.cpu().numpy()
        np.save(pathtar + str(now + 1) + ".npy", tmp)

img = np.zeros((image_size, image_size, batch_size))

for i in range(1):  # 10
    img = img + np.load(pathtar + str(i + 1) + ".npy")

img = img / 1  # 10
np.save(pathtar + str(1 + num * batch_size) + " - " + str((num + 1) * batch_size) + ".npy", img)

image = image.cpu().numpy()

for i in range(batch_size):
    res = img[:, :, i]
    mx = res.max()
    mi = res.min()
    res = 255 * (res - mi) / (mx - mi)

    up = image[i] - res
    down = res - image[i]

    up[up < 0] = 0
    down[down < 0] = 0
    cv.imwrite("C:/Users/chenda/Desktop/MNIST_use/train-test2-GI/" + str(i + 1 + num * batch_size) + ".jpg", res)
#     cv.imwrite("C:/Users/86188/Desktop/Res_up/" + str(i + 1 + num * batch_size) + ".jpg", up)
#     cv.imwrite("C:/Users/86188/Desktop/Res_down/" + str(i + 1 + num * batch_size) + ".jpg", down)
# # print(img[:,:,0])
# import matplotlib.pyplot as plt
# # tt = sum_I2.cpu()
# # plt.imshow(tt[:,:,0])
# print(type(res))

# up = image[i] - res
# Down = res - image[i]
print(type(res))
