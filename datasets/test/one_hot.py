import common
import torch
import torch.nn.functional as F
def text2vec(text):
    vectors=torch.zeros((common.captcha_size,common.captcha_array.__len__()))

    for i in range(len(text)):
        vectors[i,common.captcha_array.index(text[i])]=1
    return vectors
def vec2text(vec):

    vec=torch.argmax(vec,dim=1)

    text=""
    for i in vec:
        text+=common.captcha_array[i]
    return  text

if __name__ == '__main__':
    vec=text2vec("aaab")

    print(vec.shape)
    print(vec2text(vec))