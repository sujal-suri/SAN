import os
import cv2
import sys
import torch
import argparse

from utils import load_config, load_checkpoint
from infer.Backbone import Backbone
from dataset import Words

def convert(nodeid, gtd_list):
    isparent = False
    child_list = []
    for i in range(len(gtd_list)):
        if gtd_list[i][2] == nodeid:
            isparent = True
            child_list.append([gtd_list[i][0], gtd_list[i][1], gtd_list[i][3]])
    if not isparent:
        return [gtd_list[nodeid][0]]
    else:
        if gtd_list[nodeid][0] == '\\frac':
            return_string = [gtd_list[nodeid][0]]
            for i in range(len(child_list)):
                if child_list[i][2] == 'Above':
                    return_string += ['{'] + convert(child_list[i][1], gtd_list) + ['}']
            for i in range(len(child_list)):
                if child_list[i][2] == 'Below':
                    return_string += ['{'] + convert(child_list[i][1], gtd_list) + ['}']
            for i in range(len(child_list)):
                if child_list[i][2] == 'Right':
                    return_string += convert(child_list[i][1], gtd_list)
            for i in range(len(child_list)):
                if child_list[i][2] not in ['Right', 'Above', 'Below']:
                    return_string += ['illegal']
        else:
            return_string = [gtd_list[nodeid][0]]
            for i in range(len(child_list)):
                if child_list[i][2] in ['l_sup']:
                    return_string += ['['] + convert(child_list[i][1], gtd_list) + [']']
            for i in range(len(child_list)):
                if child_list[i][2] == 'Inside':
                    return_string += ['{'] + convert(child_list[i][1], gtd_list) + ['}']
            for i in range(len(child_list)):
                if child_list[i][2] in ['Sub', 'Below']:
                    return_string += ['_', '{'] + convert(child_list[i][1], gtd_list) + ['}']
            for i in range(len(child_list)):
                if child_list[i][2] in ['Sup', 'Above']:
                    return_string += ['^', '{'] + convert(child_list[i][1], gtd_list) + ['}']
            for i in range(len(child_list)):
                if child_list[i][2] in ['Right']:
                    return_string += convert(child_list[i][1], gtd_list)
        return return_string

def make_predictions(image_path):
    config_path = "14.yaml"

    params = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    params['device'] = device

    words = Words(params['word_path'])
    params['word_num'] = len(words)
    params['struct_num'] = 7
    params['words'] = words

    model = Backbone(params).to(device)
    load_checkpoint(model, None, params['checkpoint'])
    model.eval()

    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    image = torch.Tensor(img) / 255
    image = image.unsqueeze(0).unsqueeze(0).to(device)  # [1, 1, H, W]

    image_mask = torch.ones(image.shape).to(device)
    """
    img = cv2.imread(args.image, cv2.IMREAD_GRAYSCALE)  
    if img.mean() > 127:
        img = cv2.bitwise_not(img)
    _, img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)
    image = torch.tensor(img, dtype=torch.float32) / 255.0
    image = image.unsqueeze(0).unsqueeze(0).to(device)  
    image_mask = torch.ones(image.shape).to(device)
    """
    with torch.no_grad():
        prediction = model(image, image_mask)
        latex_tokens = convert(1, prediction)
        latex_string = ' '.join(latex_tokens)
        print(latex_string)
        return latex_string

if __name__ == "__main__":
    image_path = sys.argv[1]
    make_predictions(image_path)

