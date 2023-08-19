# -*- coding: UTF-8 -*-
import os
import random


def split_dataset(dataset_root, ratio: list):
    """
    to split the dataset of SA1B
    Parameters
    ----------
    dataset_root: str
        path to the root of dataset
    ratio: list
        dataset split ratio [train, val, test]
    """
    data_root = os.path.join(dataset_root, 'data')
    file_names = os.listdir(data_root)
    image_list = []
    # collect the name of labeled data
    for file_name in file_names:
        if file_name.split('.')[-1] == 'json':
            index = file_name.split('.')[0]
            image_path = os.path.join(data_root, index + '.jpg')
            if os.path.exists(image_path):
                image_list.append(index)
    # split the test data
    total_num = len(image_list)
    test_num = int(total_num * ratio[2])
    random.shuffle(image_list)
    test_list = image_list[:test_num]
    untest_list = image_list[test_num:]

    # split the train data and validation data
    train_num = int(total_num * ratio[0])
    random.shuffle(untest_list)
    train_list = image_list[:train_num]
    val_list = image_list[train_num:]
    val_num = len(val_list)

    # wirte results in txt
    generate_txt(txt_path=os.path.join(dataset_root, 'test.txt'), image_list=test_list)
    generate_txt(txt_path=os.path.join(dataset_root, 'train.txt'), image_list=train_list)
    generate_txt(txt_path=os.path.join(dataset_root, 'val.txt'), image_list=val_list)
    return [total_num, train_num, val_num, test_num]


def generate_txt(txt_path, image_list):
    if os.path.exists(txt_path):
        with open(txt_path, 'a+') as f:
            f.truncate(0)

    with open(txt_path, 'w') as f:
        for line in image_list:
            f.write(line+'\n')


if __name__ == '__main__':
    dataset_root = './SA1B'
    ratio = [0.6, 0.2, 0.2]
    numbers = split_dataset(dataset_root, ratio)
    print('Total number of the Dataset is {}.\n'
          ' Dataset is splited into {} trains, {} vals, {} tests'.format(*numbers))
