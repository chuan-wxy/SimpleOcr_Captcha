"""
    此文件用于数据的生成
"""
import random
import time

captcha_array = list("0123456789abcdefghijklmnopqrstuvwxyz")
captcha_size = 4

from captcha.image import ImageCaptcha
if __name__ == '__main__':
    for i in range(200):    #生成验证码数量
        image = ImageCaptcha()
        image_text = "".join(random.sample(captcha_array, captcha_size))
        image_path = "./datasets/test/{}_{}.png".format(image_text, int(time.time())) #验证码存放的位置
        image.write(image_text,image_path)

