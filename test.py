from models.yolo import Model, Detect
from models.common import *
import train


def fuse_model(self):
    for x in self.model:
        if type(x)==Conv:
            torch.quantization.fuse_modules(x, ['conv', 'bn'], inplace=True)
        elif type(x)==C3:
            for k in x.children():
                if type(k)==Conv:
                    torch.quantization.fuse_modules(k, ['conv', 'bn'], inplace=True)
                else:
                    for g in k.children():
                        for j in g.children():
                            if type(j)==Conv:
                                torch.quantization.fuse_modules(j, ['conv', 'bn'], inplace=True)
        elif type(x)==SPPF:
            for u in x.children():
                if type(u)==Conv:
                    torch.quantization.fuse_modules(u, ['conv', 'bn'], inplace=True)
        else:
            continue

model = train(data='data.yaml', cfg='models/yolov5s.yaml', img=256, epochs=1, batch=100)
print(model)
