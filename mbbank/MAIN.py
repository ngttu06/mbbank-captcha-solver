#coding:utf8
import shortuuid
import torch
import torch.nn as nn
from torch.autograd import Variable
try:
    from .datasets import CaptchaData,CaptchaDataOne
    from .CONFIG import LettersInt
except:
    from datasets import CaptchaData,CaptchaDataOne
    from CONFIG import LettersInt
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Resize
import time
from PIL import Image
import os, requests
import base64
from io import BytesIO


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


class CNN(nn.Module):
    def __init__(self, range_len=LettersInt.range_len, pic_name_len=LettersInt.PIC_NAME_LEN):
        super(CNN, self).__init__()
        self.range_len = range_len
        self.pic_name_len = pic_name_len

        IMG_WIDTH = LettersInt.IMG_WIDTH
        IMG_HEIGHT = LettersInt.IMG_HEIGHT

        IMG = 3  # RGB
        input1 = 3
        out1 = 16
        maxpool2d1 = 2
        IMG_HEIGHT = int(IMG_HEIGHT / maxpool2d1)
        IMG_WIDTH = int(IMG_WIDTH / maxpool2d1)
        input2 = 16
        out2 = 64
        maxpool2d2 = 2
        IMG_HEIGHT = int(IMG_HEIGHT / maxpool2d2)
        IMG_WIDTH = int(IMG_WIDTH / maxpool2d2)
        input3 = 64
        out3 = 512
        maxpool2d3 = 2
        IMG_HEIGHT = int(IMG_HEIGHT / maxpool2d3)
        IMG_WIDTH = int(IMG_WIDTH / maxpool2d3)

        self.linear = out3 * IMG_WIDTH * IMG_HEIGHT
        self.conv = nn.Sequential(
            # batch*3*180*100 # IMG_WIDTH,IMG_HEIGHT=100,30
            nn.Conv2d(input1, out1, IMG, padding=(1, 1)),
            # 参数分别对应着输入的通道数3，输出通道数16,卷积核大小为3（长宽都为3）adding为（1， 1）可以保证输入输出的长宽不变
            nn.MaxPool2d(maxpool2d1, maxpool2d1),
            nn.BatchNorm2d(out1),
            nn.ReLU(),
            # batch*16*90*50
            nn.Conv2d(input2, out2, IMG, padding=(1, 1)),
            nn.MaxPool2d(maxpool2d2, maxpool2d2),
            nn.BatchNorm2d(out2),
            nn.ReLU(),
            # batch*64*45*25
            nn.Conv2d(input3, out3, IMG, padding=(1, 1)),
            nn.MaxPool2d(maxpool2d3, maxpool2d3),
            nn.BatchNorm2d(out3),
            nn.ReLU(),
            # batch*512*22*12
            # nn.Conv2d(input4, out4, IMG, padding=(1, 1)),
            # nn.MaxPool2d(maxpool2d4, maxpool2d4),
            # nn.BatchNorm2d(out4),
            # nn.ReLU(),
            # batch*512*11*6
        )
        self.fc = nn.Linear(self.linear, self.range_len * self.pic_name_len)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, self.linear)
        x = self.fc(x)
        return x

class CrackLettesInt4:

    def __init__(self):
        self.batch_size = 128
        self.base_lr = 0.001
        self.max_epoch = 15
        self.num_workers = 0
        self.model_path = LettersInt.model_path
        self.restor = False
        self.range_len = LettersInt.range_len

    def calculat_acc(self, output, target):
        output = output.view(-1, LettersInt.range_len)
        target = target.view(-1, LettersInt.range_len)
        output = torch.argmax(output, dim=1).view(-1, LettersInt.PIC_NAME_LEN)
        target = torch.argmax(target, dim=1).view(-1, LettersInt.PIC_NAME_LEN)
        correct = (output == target).all(dim=1).float().mean()
        return correct.item()

    def train(self, train_folder, val_split=0.1):
        device = get_device()
        print('Using device:', device)
        transforms = Compose([Resize((LettersInt.IMG_HEIGHT, LettersInt.IMG_WIDTH)), ToTensor()])
        full_dataset = CaptchaData(train_folder, transform=transforms)
        n_val = max(1, int(len(full_dataset) * val_split))
        n_train = len(full_dataset) - n_val
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset, [n_train, n_val],
            generator=torch.Generator().manual_seed(42))
        train_data_loader = DataLoader(train_dataset, batch_size=self.batch_size,
                                       shuffle=True, num_workers=self.num_workers, drop_last=True)
        val_data_loader = DataLoader(val_dataset, batch_size=self.batch_size,
                                     shuffle=False, num_workers=self.num_workers, drop_last=False)

        cnn = CNN().to(device)
        optimizer = torch.optim.Adam(cnn.parameters(), lr=self.base_lr)
        criterion = nn.MultiLabelSoftMarginLoss()

        best_val_acc = 0.0
        for epoch in range(self.max_epoch):
            start_ = time.time()
            cnn.train()
            loss_history = []
            acc_history = []
            for img, target in train_data_loader:
                img = img.to(device)
                target = target.to(device)
                output = cnn(img)
                loss = criterion(output, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                acc = self.calculat_acc(output.detach(), target)
                loss_history.append(loss.item())
                acc_history.append(acc)

            train_loss = sum(loss_history) / len(loss_history)
            train_acc = sum(acc_history) / len(acc_history)

            cnn.eval()
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for img, target in val_data_loader:
                    img = img.to(device)
                    target = target.to(device)
                    output = cnn(img)
                    acc = self.calculat_acc(output, target)
                    val_correct += acc * img.size(0)
                    val_total += img.size(0)
            val_acc = val_correct / val_total if val_total else 0.0

            print('epoch: {}|train_loss: {:.4f}|train_acc: {:.4f}|val_acc: {:.4f}|time: {:.1f}s'.format(
                epoch, train_loss, train_acc, val_acc, time.time() - start_))

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(cnn.state_dict(), self.model_path)
                print('  -> saved best model (val_acc={:.4f})'.format(val_acc))

        print('Training done. best_val_acc={:.4f}'.format(best_val_acc))


    def predict_one(self, image_path):
        device = get_device()
        img = Image.open(image_path)
        img = img.convert('RGB')
        img = img.resize((LettersInt.IMG_WIDTH, LettersInt.IMG_HEIGHT))
        transforms = Compose([ToTensor()])
        img = transforms(img)
        cnn = CNN().to(device)
        cnn.eval()
        cnn.load_state_dict(torch.load(self.model_path, map_location=torch.device('cpu')))
        img = img.view(1, 3, LettersInt.IMG_HEIGHT, LettersInt.IMG_WIDTH).to(device)
        with torch.no_grad():
            output = cnn(img)
        output = output.view(-1, self.range_len)
        output = torch.argmax(output, dim=1)
        output = output.view(-1, LettersInt.PIC_NAME_LEN)[0]
        pred = ''.join([LettersInt.content_range[i] for i in output.cpu().numpy()])
        print('result', pred)
        return pred

    @classmethod
    def down_img(cls,url,c):
        for i in range(c):
            res = requests.get(url).content
            g = open('./data/test/%s.jpg' % int(time.time()*1000),'wb')
            g.write(res)
            g.close()
            print(i)

    @classmethod
    def check_name(cls):
        pp = os.path.dirname(os.path.abspath(__file__))
        for d in os.listdir('./data/train'):
            if len(d) != 8:
                print(d)
                os.remove(pp+'\\data\\train\\'+d)

    def check_all(self,folder):
        for d in os.listdir(folder):
            f = './data/test/%s' % d
            self.predict_one(f)


def getCaptcha():
    url = 'https://ebanking.baca-bank.vn/IBSRetail/servlet/ImageServlet'
    headers = {}
    response = requests.get(url, headers=headers, timeout=15)
    return base64.b64encode(response.content).decode('utf-8')


def img_base4_save(img_base64, savePath):
    '''
    Base64图片保存本地
    '''
    # if not img_base64.startswith('data:image/png;base64,'):
    #     return

    # 去除前缀信息（如'data:image/png;base64,'）
    if img_base64.startswith('data:'):
        _, data = img_base64.split(',', maxsplit=1)
    else:
        data = img_base64

    # 解码Base64数据
    decoded_data = base64.b64decode(data)
    # 创建BytesIO对象来读取二进制数据
    buffered_io = BytesIO()
    buffered_io.write(decoded_data)
    buffered_io.seek(0)
    # 打开图像文件
    img = Image.open(buffered_io)
    # 指定保存路径及名称
    # 保存图像到本地
    img.save(savePath)
    return True


def ttt():
    from PIL import Image
    # 打开图片
    img = Image.open('1709659296.jpg')

    # 获取图片的宽度和高度
    width, height = img.size

    # 计算每个子图的尺寸
    sub_width = width // 4

    # 分割图片
    for j in range(4):
        if j == 4:
            box = (j * sub_width, 0, (j + 1) * sub_width+3, height)
        elif j == 5:
            box = (j * sub_width+3, 0, (j + 1) * sub_width, height)
        else:
            box = (j * sub_width, 0, (j + 1) * sub_width, height)
        sub_img = img.crop(box)
        sub_img.save(f'./t/sub_img_{j}.jpg')



def getImg():
    for i in range(20):
        print(i)
        url = 'https://online.vietbank.com.vn/ibk/vn/login/capcha.jsp'
        headers = {}
        response = requests.get(url, headers=headers, timeout=20)
        with open('./data/test/' + shortuuid.uuid() + '.jpg', 'wb') as wf:
            wf.write(response.content)



if __name__ == "__main__":
    CrackLettesInt4().train(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mbbank_dataset')
    )

