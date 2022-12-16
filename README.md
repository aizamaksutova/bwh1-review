# bwh1-review

1. Какие у меня были аугментации на самых ранних эпохах, и на этом удалось выбить скор 0.43

```
normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

test_transform = T.Compose([
    T.Resize(224),
    T.ToTensor(),
    normalize,
])

```

так я обучалась эпох 15 наверное

2. Потом добавила больше аугментаций + добавила шумы, чтобы наша выборка увеличилась
```
normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

test_transform = T.Compose([
    T.Resize(224),
    T.ColorJitter(brightness=0.5, contrast=0.6),
    T.RandomRotation(degrees=45),
    T.RandomHorizontalFlip(p=0.5),  # Flips the image horizontally with probability 0.5
    T.RandomVerticalFlip(p=0.05),
    T.RandomGrayscale(p=0.2),
    T.ToTensor(),
    normalize,
])

#noise


```

3. 
```
train_transform = T.Compose([
    T.Resize(224),
    T.RandomPosterize(bits=2),
    T.RandAugment(),
    # T.ColorJitter(brightness=0.5, contrast=0.6),
    # T.RandomRotation(degrees=45),
    # T.RandomHorizontalFlip(p=0.5),  # Flips the image horizontally with probability 0.5
    # T.RandomVerticalFlip(p=0.4),
    # T.RandomGrayscale(p=0.2),
    T.ToTensor(),
    normalize,
]) 
```

proeb1
