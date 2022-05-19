import os
import torch
import tensorboard as tb
PATH = "/home/dblab/git/finetuning_gan/src/train/cls/tb_logs"
log_dirs = os.listdir(PATH)
log_dirs = os.listdir(f"{PATH}/{log_dirs[0]}")
raw = tb.data.experimental

df = raw.get_scalars()


import pandas as pd

from collections import defaultdict
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


PATH10 = {
    "ORI_False":
    "/home/dblab/git/finetuning_gan/src/train/cls/tb_logs/resnet18_original_cifar10_0.1_False_False/version_0/events.out.tfevents.1652413253.ckserver.2384.0",
    "ORI_True":
    "/home/dblab/git/finetuning_gan/src/train/cls/tb_logs/resnet18_original_cifar10_0.1_True_False/version_0/events.out.tfevents.1652422155.ckserver.24656.0",
    "GAN_False_False":
    "/home/dblab/git/finetuning_gan/src/train/cls/tb_logs/resnet18_fine_cifar10_0.1_False_False-gen_False_False/version_0/events.out.tfevents.1652688125.ckserver.34643.0",
    "GAN_True_False":
    "/home/dblab/git/finetuning_gan/src/train/cls/tb_logs/resnet18_fine_cifar10_0.1_True_False-gen_False_False/version_0/events.out.tfevents.1652688334.ckserver.40332.0",
    "GAN_False_True":
    "/home/dblab/git/finetuning_gan/src/train/cls/tb_logs/resnet18_fine_cifar10_0.1_False_False-gen_True_False/version_0/events.out.tfevents.1652689488.ckserver.14044.0",
    "GAN_True_True":
    "/home/dblab/git/finetuning_gan/src/train/cls/tb_logs/resnet18_fine_cifar10_0.1_True_False-gen_True_False/version_0/events.out.tfevents.1652689670.ckserver.19687.0"
}

PATH100 = {
    "ORI_False":
        "/home/dblab/git/finetuning_gan/src/train/cls/tb_logs/resnet18_original_cifar10_0.01_False_False/version_0/events.out.tfevents.1652421829.ckserver.19084.0",
    "ORI_True":
        "/home/dblab/git/finetuning_gan/src/train/cls/tb_logs/resnet18_original_cifar10_0.01_True_False/version_0/events.out.tfevents.1652422491.ckserver.30330.0",
    "GAN_False_False":
        "/home/dblab/git/finetuning_gan/src/train/cls/tb_logs/resnet18_fine_cifar10_0.01_False_False-gen_False_False/version_0/events.out.tfevents.1652688578.ckserver.46351.0",
    "GAN_True_False":
        "/home/dblab/git/finetuning_gan/src/train/cls/tb_logs/resnet18_fine_cifar10_0.01_True_False-gen_False_False/version_0/events.out.tfevents.1652688743.ckserver.3713.0",
    "GAN_False_True":
        "/home/dblab/git/finetuning_gan/src/train/cls/tb_logs/resnet18_fine_cifar10_0.01_False_False-gen_True_False/version_0/events.out.tfevents.1652789668.ckserver.44487.0",
    "GAN_True_True":
        "/home/dblab/git/finetuning_gan/src/train/cls/tb_logs/resnet18_fine_cifar10_0.01_True_False-gen_True_False/version_0/events.out.tfevents.1652789809.ckserver.1426.0"
}


for name, path in PATH10.items():
    event = EventAccumulator(path).Reload()

    ret = pd.DataFrame()
    col = {}

    tags = event.Tags()['scalars']
    for tag in tags:
        if "val" not in tag or "loss" in tag:
            continue
        # print(tag)
        # steps = [e.step for e in event.Scalars(tag)]
        val = pd.Series([e.value for e in event.Scalars(tag)])
        # col = pd.Series({tag: val})
        ret[tag] = val
    prt = name
    for v_ in ret.iloc[ret['acc/val'].argmax()].to_list():
        prt += f"\t{v_:.2}"
    print(prt)

steps = [e.step for e in event.Scalars(tags[0])]
value = [e.value for e in event.Scalars(tags[0])]

for i in d:
    print(i)

summary_iterators = [ for dname in os.listdir(dpath)]


def tabulate_events(dpath):
    summary_iterators = [EventAccumulator(os.path.join(dpath, dname)).Reload() for dname in os.listdir(dpath)]

    tags = summary_iterators[0].Tags()['scalars']

    for it in summary_iterators:
        assert it.Tags()['scalars'] == tags

    out = defaultdict(list)
    steps = []

    for tag in tags:
        steps = [e.step for e in summary_iterators[0].Scalars(tag)]

        for events in zip(*[acc.Scalars(tag) for acc in summary_iterators]):
            assert len(set(e.step for e in events)) == 1

            out[tag].append([e.value for e in events])

    return out, steps


def to_csv(dpath):
    dirs = os.listdir(dpath)

    d, steps = tabulate_events(dpath)
    tags, values = zip(*d.items())
    np_values = np.array(values)

    for index, tag in enumerate(tags):
        df = pd.DataFrame(np_values[index], index=steps, columns=dirs)
        df.to_csv(get_file_path(dpath, tag))


def get_file_path(dpath, tag):
    file_name = tag.replace("/", "_") + '.csv'
    folder_path = os.path.join(dpath, 'csv')
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    return os.path.join(folder_path, file_name)


if __name__ == '__main__':
    path = "path_to_your_summaries"
    to_csv(path)

if __name__ == "__main__":
    print(torch.cuda.is_available())