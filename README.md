# Generative AI: Text-to-Image Models
*students: Beomseok Ko, Junseo Park*

## Description
- For training,   
    - training_code.py
        - train_dreambooth.py
            - Training Stable Diffusion using DreamBooth
    
    - concepts_list.json: Path of images to be used for learning

- For inference,
    - main.py
        - cmd_txt2img.py: Actual image generation code using stable diffusion
        - prompt.txt: Text of images to be created
    
- For evalution,
    - KID_score_measurement.py: KID evaluation
    - FID_score_measurement.py: FID evaluation
    - CLIP_score.py: CLIP evaluation


- img_rename.py: Create and set a path to save the created image

- convert_original_stable_diffusion_to_diffuers.py: convert .ckpt to diffusers format
    - v1-inference.yaml: configuration file