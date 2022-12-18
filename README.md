# bwh1-review


0. Сперва попробовала использовать свою нейросеть, которая выдала мне без аугментаций(ну точнее, просто с T.Resize(224) и normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])). Сама нейросеть:

```
class BasicBlockNet(nn.Module):
    def __init__(self):
        super().__init__()
        # <your code here>
        self.layer_1 = nn.Sequential(
          nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
          nn.BatchNorm2d(num_features=32),
          nn.ReLU(),
          nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
          nn.BatchNorm2d(num_features=32)
        )
        self.conv1 = nn.Sequential(
                        nn.Conv2d(in_channels=3, out_channels=, kernel_size = 3, stride = 1, padding = 1),
                        nn.BatchNorm2d(num_features=32))
        
        self.layer_2 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=1)
        self.layer_3 = nn.ReLU()
        self.layer_4 = nn.AvgPool2d(kernel_size=8)
        self.layer_5 = nn.Linear(in_features=512, out_features=10)

        # self.layer_3 = nn.AvgPool2d(kernel_size=8)

        # self.layer_5 = nn.ReLU()

        # self.layer_4 = nn.Linear(in_features=512, out_features=10)

    def forward(self, x):
        out1 = self.layer_1(x)
        # out = out.reshape(10, -1)
        # torch.reshape(out, (-1,))
        out2 = self.layer_2(x)
        out = out1 + out2
        out = self.layer_3(out)
        out = self.layer_4(out)
        out = torch.flatten(out, start_dim=1)
        out = self.layer_5(out)
        return out

```
потом решила, что лучше использовать торчовые модели из-под коробки, потому что они энивей будут работать сильно лучше чем то, что чисто теоретически можно написать самостоятельно



1. Какие у меня были аугментации + гиперпараметры на самых ранних эпохах, и на этом удалось выбить скор 0.43

вплоть до 5 пункта я обучала вот так:

```
model = mobilenet_v2(num_classes=200).to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0001)
criterion = torch.nn.CrossEntropyLoss()
scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.01, max_lr=0.1, step_size_up=5, mode='triangular2')
```

```
normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

test_transform = T.Compose([
    T.Resize(224),
    T.ToTensor(),
    normalize,
])

train_transform = T.Compose([
    T.Resize(224),
    T.ToTensor(),
    normalize,
])

```
не добавляли никакие шумы, просто вот с такими базовыми (даже наверное это не аугментации особо) обучили нашу модель, было приятно, что эпоха отрабатывает за 7 минут. 
так я обучалась эпох 15 наверное, но скор получился хороший все-таки

2. Потом добавила больше аугментаций + добавила шумы, чтобы наша выборка увеличилась
```
normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

train_transform = T.Compose([
    T.Resize(224),
    T.ColorJitter(brightness=0.5, contrast=0.6),
    T.RandomRotation(degrees=45),
    T.RandomHorizontalFlip(p=0.5),  # Flips the image horizontally with probability 0.5
    T.RandomVerticalFlip(p=0.05),
    T.RandomGrayscale(p=0.2),
    T.ToTensor(),
    normalize,
])
```
#noise
вот такое добавила в функцию train_epoch, чтобы добавить шумы к картинке
```
a = 5*torch.ones(image.shape)
p = torch.poisson(a)
p_norm = p/p.max()
image = (image + p_norm).clip(0,1)

```
с такими аугментациями + с шумами одна эпоха обучалась 30 минут, но даже скор не удалось выбить особо большой, поэтому быстро оставили эту идею - примерно через пару эпох.

еще у меня была такая ошибка, что я применяла аугментации к валидационной выборке, а так делать плохо, но я почему-то об этом забыла. вот так вот, признаю свою большую ошибку. можем вместе понаблюдать как развивался скор из-за этого:( 

![Image](/images/proeb.png)


4. обучила на 50 эпох с такими аугментациями + убрала шумы, дошла до скора 0.5, еще уменьшила выборку на 40% и брала трейн и валидацию в пропорции 9:1.

```
train_transform = T.Compose([
    T.Resize(224),
    T.RandomPosterize(bits=2),
    T.RandAugment(),
    T.ToTensor(),
    normalize,
])
```

добавляю графики для accuracy и loss-a, здесь немного странная тенденция, у меня немного скакало туда-сюда во время обучения, но в итоге вышло более-менее приемлемо. не смотрите на то, что в самом конце accuracy упало, это я уже начала обучать новую модель с другими аугментациями.
![Image](/images/50epochs_acc.png)
![Image](/images/50epochs_loss.png)

5. Дальше пошло другое обучение, решила немного повертеть с аугментациями и подольше пообучать модельку, надеюсь, что выбьет хороший скор) +убрала шумы, потому что они как будто скорее ухудшали качество модели судя по скору на валидационной выборке

в этот раз поменяла немного weight_decay в оптимизаторе с $10^{-4}$ на $2*10^{-4}$ + поменяла шкедулер на CosineAnnealingLR


```
train_transform = T.Compose([
    T.Resize(224),
    T.RandomChoice([
        T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomPosterize(bits=2),
            T.RandAugment(),
        ]),
        T.RandomChoice([
             T.ColorJitter(brightness=0.5, contrast=0.6),
             T.RandomRotation(degrees=45),
             T.RandomHorizontalFlip(p=0.5),  # Flips the image horizontally with probability 0.5
             T.RandomGrayscale(p=0.2),
        ]),
        T.TrivialAugmentWide(31)
        ], p=[0.2, 0.4, 0.4]),
    T.ToTensor(),
    normalize,
])

```

в итоге вышло так, что у меня колаб умер где-то посередине обучения, но план был такой: 100 эпох обучаю с такими аугментациями, потом меняю weight_decay на $5*10^{-4}$ + меняю шкедулер на REDUCELRONPLATEAU, чтобы до конца добить до хорошего скора. жаль, конечно, что не получилось дообучать до конца из-за проблем с датасферой, но в целом опыт нормальный. могла бы конечно и пару ночей поспать, если бы не эта домашка, но в целом довольно терпимо. но вообще вот картинка того, какие у меня были скоры пока обучение не приостановилось, в целом тенденция довольно перспективная.
![Image](/images/scores.png)

