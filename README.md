# bwh1-review

1. Какие у меня были аугментации на самых ранних эпохах, и на этом удалось выбить скор 0.43

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

добавляю графики для accuracy и loss-a, здесь немного странная тенденция, у меня немного скакало туда-сюда во время обучения, но в итоге вышло более-менее приемлемо.
![Image](/images/50epochs_acc.png)
![Image](/images/50epochs_loss.png)

5. Дальше пошло другое обучение, решила немного повертеть с аугментациями и подольше пообучать модельку, надеюсь, что выбьет хороший скор)

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

