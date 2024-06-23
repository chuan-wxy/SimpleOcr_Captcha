import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision  import transforms
from torch.utils.tensorboard import SummaryWriter
import one_hot
class mydatasets(Dataset):
    def __init__(self,root_dir):
       super(mydatasets, self).__init__()
       self.image_path=[ os.path.join(root_dir,image_name) for image_name in os.listdir(root_dir)]

       self.transforms=transforms.Compose([
           transforms.Resize((60,160)),
           transforms.ToTensor(),
           transforms.Grayscale()

       ])
    def __getitem__(self, index):
        image_path = self.image_path[index]
        image=self.transforms(Image.open(image_path))
        label=image_path.split("/")[-1]
        label=label.split("_")[0]
        label_tensor=one_hot.text2vec(label)
        label_tensor=label_tensor.view(1,-1)[0]
        return image,label_tensor
    def __len__(self):
        return self.image_path.__len__()



if __name__ == '__main__':
    writer = SummaryWriter("logs")
    train_data=mydatasets("./datasets/train/")
    img,label= train_data[0]
    print(img.shape,label.shape)
    writer.add_image("img",img,1)
    writer.close()
