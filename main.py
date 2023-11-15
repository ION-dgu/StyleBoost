import SD
import cmd_txt2img
import argparse

def cmd_generation_img(model_dic, model_style):
    for model in model_dic:
                #style만큼 실행
                for style in model_style:
                        # prompt 실험 개수만큼 실행
                        index = 1    
                        for prompt_index in prompt_id:
                            prompt = prompt_id[prompt_index]
                            prompt =  'a photo of zwx style, '+prompt
                            cmd_txt2img.main_process(prompt, negative_prompt, model, style, int(prompt_index), index, gpu)
#output_img -> model_name -> prompt_id path에 저장
if __name__=="__main__":

    prompt_id = dict()
    with open('./prompt.txt','r') as f:
        prompt_lst = f.readlines()
    
        for i in range(len(prompt_lst)):
            
            prompt_id[str(i+1)] = prompt_lst[i].rstrip('\n')
        
    negative_prompt = '(worst quality, low quality:1.2), canvas frame, ((disfigured)), ((bad art)), ((deformed)),((extra limbs)), weird colors, blurry, (((duplicate))), ((morbid)), ((mutilated)), [out of frame], extra fingers, mutated hands, ((poorly drawn hands)), ((poorly drawn face)), (((mutation))), 3d render'
    gpu = 3
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--iter", type= int, default= 1)  
    args = parser.parse_args()
    
    for i in range(args.iter):
        
        model_dic = ['example_dic'] 
        model_style = ['example_style']  
            
        cmd_generation_img(model_dic, model_style)
                            