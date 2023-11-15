from torchmetrics.functional.multimodal import clip_score
from functools import partial
import torch
import os
import numpy as np
from PIL import Image
from datetime import datetime

def calculate_clip_score(images, prompts):
        #print(np.shape(images))
        images = images.astype("uint8")
        clip_score = clip_score_fn(torch.from_numpy(images).permute(0, 3, 1, 2), prompts).detach()
        return round(float(clip_score), 4)
        

def cmd_cal_clip_score(model, style, prompt_id):    
    #변환할 이미지 목록 불러오기
    clip_score_lst = []
    
    for index in prompt_id.keys():
        if len(prompt_id[str(index)]) > 77:
            continue
        image_path = f'./output_img/{model}/{style}/{index}'

        img_list = os.listdir(image_path) #디렉토리 내 모든 파일 불러오기
        img_list_png = [img for img in img_list if img.endswith(".png")] #지정된 확장자만 필터링
        img_len = np.shape(img_list_png)[0]
        prompts = []
        
        for _ in range(img_len):
            prompts.append(prompt_id[index])
        img_list_png = np.array(img_list_png)
        
        img_list_np = []

        for i in img_list_png:
            img = Image.open(image_path + '/' + i)
            img_array = np.array(img)
            img_list_np.append(img_array)
        
        img_np = np.array(img_list_np)
        sd_clip_score = calculate_clip_score(img_np, prompts)
        
        clip_score_lst.append(sd_clip_score)
    
    return clip_score_lst   
        
        
if __name__=="__main__":
    clip_score_fn = partial(clip_score, model_name_or_path="openai/clip-vit-base-patch16")
    prompt_id = dict()
    with open('./prompt.txt','r') as f:
        prompt_lst = f.readlines()

        for i in range(1,len(prompt_lst)+1):
            
            prompt_id[str(i)] = prompt_lst[i-1].rstrip('\n')
            
    model_dic = {'0' : 're_img_0_500', # person + back
                '1' : 're_img_0_750',
                '2' : 're_img_0_1000', 
                '3' : 're_img_0_2000',
                '4' : 're_img_0_3000',
                
                '5' : 'in_img_face_with_background_500',
                '6' : 'in_img_face_with_background_750',
                '7' : 'in_img_face_with_background_1000',
                '8' : 'in_img_face_with_background_2000',
                '9' : 'in_img_face_with_background_3000', 
                
                '10' : 'realistic_re_img_test', # step : 500, real : 20 
                '11': 'realistic_re_img_test2', # step : 500, all : 146
                '12': 'realistic_re_img_test3', # step : 500, real : 5
                '13': 'realistic_re_img_test4', # step : 500, real : 5 (back)
                '14': 'realistic_re_img_test5', # step : 500, real : 20 (back)

                '15': 'anime_re_img_test6', # step : 1000, anime : 10 (back)
                '16': 'mid_re_img_test7', # step : 750, mid : 8 (back)
                '17': 'realistic_re_img_test8', # step : 500, mid : 5 (back)
                
                '18': 'anime_re_img_test9', # step : 1000, anime : 10 (person)
                #instance image configuration
                '19': 'only_in_img_back',
                '20': 'only_in_img_person',
                
                #comparision
                '21': 'in_img_50_compare',
                '22': 'in_img_50_compare2',
                
                #style test <화풍>
                '23': 'style_test', # re image : 40 
                '24': 'style_test2', # re image : 20
                
                '25': 're_mid_test', # re mid image : 20 person
                '26': 're_mid_test2', # re mid image : 50 person 
                
                #style test2 <Digital art & Digital painting>
                '27': 'digital_person_test', # re image : 20 person
                '28': 'digital_back_test', # re image : 20 back
                '29': 'digital_all_test', # re image : 10 person + 10 back
                '30': 'digital_person_test2', # re image : 20 person
                '31': 'digital_person_anime_test', # re image : 10 person
                '32': 'digital_person_test3', # re image : another person image
                '33': 'digital_person_anime_test2', # re image : new digital anime art 20 person
            }
    instance_style = { '1' : 'mid-journey', 
                      '2' : 'anime',
                      '3' : 'realistic'}
    
    Comparative_group_one = True
    if Comparative_group_one:
        for i in range(3,3+1):
            style = instance_style[str(i)]
            for j in range(4,4+1):
                model = model_dic[str(j)]   
                
                clip_score_lst = cmd_cal_clip_score(model, style, prompt_id)
                sum_of_score = sum(clip_score_lst)/len(clip_score_lst)
                all_imgs = os.listdir(f'./output_img/{model}/{style}')
                
                imgs = [file for file in all_imgs if not file.endswith(".npy")]
                total_len = 0
                for pth in imgs:
                    if not pth=='all_images':
                        file_path = f'./output_img/{model}/{style}/{pth}'
                        total_len = total_len + len(os.listdir(file_path))
                
                with open('clip_score.txt','a') as f:
                    now = datetime.now()
                    f.write(f'\ndate : {now.date()} {now.time()}\nmodel : {model}\t|style : {style}\t|score : {clip_score_lst}\t|sum of score : {sum_of_score}\
                        \n|# of image : {total_len}')