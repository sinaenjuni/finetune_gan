import pytorch_lightning as pl
from torchvision import transforms
from data_module.imbalance_cifar import Imbalanced_CIFAR10
from torch.utils.data import Sampler, DataLoader
import numpy as np
import random

class BalancedSampler(Sampler):
    def __init__(self, buckets, retain_epoch_size=False):
        for bucket in buckets:
            random.shuffle(bucket)

        self.bucket_num = len(buckets)
        self.buckets = buckets
        self.bucket_pointers = [0 for _ in range(self.bucket_num)]
        self.retain_epoch_size = retain_epoch_size

    def __iter__(self):
        count = self.__len__()
        print('sampler', count)
        while count > 0:
            yield self._next_item()
            count -= 1

    def _next_item(self):
        bucket_idx = random.randint(0, self.bucket_num - 1)
        bucket = self.buckets[bucket_idx]
        item = bucket[self.bucket_pointers[bucket_idx]]
        self.bucket_pointers[bucket_idx] += 1
        if self.bucket_pointers[bucket_idx] == len(bucket):
            self.bucket_pointers[bucket_idx] = 0
            random.shuffle(bucket)
        return item

    def __len__(self):
        if self.retain_epoch_size:
            return sum([len(bucket) for bucket in self.buckets])  # AcruQRally we need to upscale to next full batch
        else:
            return max([len(bucket) for bucket in
                        self.buckets]) * self.bucket_num  # Ensures every instance has the chance to be visited in an epoch



class ImbalancedMNISTDataModule(pl.LightningDataModule):
    def __init__(self, image_size, batch_size, imb_factor, balanced, retain_epoch_size, augmentation):
        super().__init__()
        self.save_hyperparameters()

        # self.image_size = image_size
        self.batch_size = batch_size
        self.balanced = balanced
        self.retain_epoch_size = retain_epoch_size

        normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                         std=[0.2023, 0.1994, 0.2010])
        if augmentation:
            train_transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.RandomCrop(image_size, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ToTensor(),
                normalize])
        else:
            train_transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.ToTensor(),
                normalize])

        print("Train dataloader")
        print(train_transform)

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize])
        print("Test dataloader")
        print(test_transform)

        self.train_dataset = Imbalanced_CIFAR10("~/data/", train=True, download=False, transform=train_transform, imb_factor=imb_factor)
        self.test_dataset = Imbalanced_CIFAR10("~/data/", train=False, download=False, transform=test_transform)

        self.num_classes = len(np.unique(self.train_dataset.targets))

        self.train_cls_num_list = [0] * self.num_classes
        for label in self.train_dataset.targets:
            self.train_cls_num_list[label] += 1

        print(self.train_cls_num_list)

        if self.balanced:
            buckets = [[] for _ in range(self.num_classes)]
            for idx, label in enumerate(self.train_dataset.targets):
                buckets[label].append(idx)
            self.sampler = BalancedSampler(buckets, self.retain_epoch_size)

    def train_dataloader(self):
        if self.balanced:
            return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=False, sampler=self.sampler, num_workers=4, persistent_workers=True)
        else:
            return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, sampler = None, num_workers=4, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4, persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4, persistent_workers=True)



if __name__ == "__main__":
    data_loader = ImbalancedMNISTDataModule(image_size=32, batch_size=128, imb_factor=0.01, balanced=False, retain_epoch_size=False, augmentation=True)


    count = {i:0 for i in range(10)}
    for image, label in data_loader.train_dataloader():
        for l in label.tolist():
            count[l] += 1

    print(count)

    count = {i:0 for i in range(10)}
    for image, label in data_loader.val_dataloader():
        for l in label.tolist():
            count[l] += 1

    print(count)

    count = {i:0 for i in range(10)}
    for image, label in data_loader.test_dataloader():
        for l in label.tolist():
            count[l] += 1

    print(count)