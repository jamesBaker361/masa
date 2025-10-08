import os
import argparse
from experiment_helpers.gpu_details import print_details
from accelerate import Accelerator
import time
import torch
import numpy as np
from datasets import load_dataset,Dataset
from transformers import AutoProcessor, CLIPModel
#import ImageReward as RM
from style_rl.image_utils import concat_images_horizontally
from style_rl.eval_helpers import DinoMetric
from style_rl.prompt_list import real_test_prompt_list
from masactrl.diffuser_utils import MasaCtrlPipeline
from masactrl.masactrl_utils import AttentionBase
from masactrl.masactrl_utils import regiter_attention_editor_diffusers,postprocess_image
from masactrl.masactrl import MutualSelfAttentionControl
import wandb
from transformers import BlipProcessor, BlipForConditionalGeneration

from diffusers import DDIMScheduler

parser=argparse.ArgumentParser()

parser.add_argument("--mixed_precision",type=str,default="fp16")
parser.add_argument("--project_name",type=str,default="masa")
parser.add_argument("--src_dataset",type=str, default="jlbaker361/mtg")
parser.add_argument("--num_inference_steps",type=int,default=20)
parser.add_argument("--limit",type=int,default=-1)
parser.add_argument("--size",type=int,default=512)
parser.add_argument("--object",type=str, default="person")
parser.add_argument("--dest_dataset",type=str,default="jlbaker361/masa")
parser.add_argument("--dim",type=int,default=256)
parser.add_argument("--resize",action="store_true")



def main(args):
    accelerator=Accelerator(log_with="wandb",mixed_precision=args.mixed_precision)
    accelerator.init_trackers(project_name=args.project_name,config=vars(args))
    torch_dtype={
        "no":torch.float32,
        "fp16":torch.float16,
        "bf16":torch.bfloat16
    }[args.mixed_precision]
    device=accelerator.device

    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
    

    blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base", torch_dtype=torch_dtype).to(device)

    dino_metric=DinoMetric(accelerator.device)

    

    '''try:
        output_dict=load_dataset(args.dest_dataset,split="train").to_dict()
        skip=len(output_dict["image"])
    except:'''
    output_dict={
        "image":[],
        "augmented_image":[],
        "text_score":[],
        "image_score":[],
        "dino_score":[],
        "prompt":[]
    }
    skip=0

    print("skipping ",skip)
    model_path = "CompVis/stable-diffusion-v1-4"
    # model_path = "runwayml/stable-diffusion-v1-5"
    scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
    model = MasaCtrlPipeline.from_pretrained(model_path, scheduler=scheduler).to(device)


    data=load_dataset(args.src_dataset, split="train")

    background_data=load_dataset("jlbaker361/real_test_prompt_list",split="train")
    background_dict={row["prompt"]:row["image"] for row in background_data}

    text_score_list=[]
    image_score_list=[]
    image_score_background_list=[]
    #ir_score_list=[]
    dino_score_list=[]


    for k,row in enumerate(data):
        if k==args.limit:
            break
        if k<skip:
            continue
        
        prompt=real_test_prompt_list[k%len(real_test_prompt_list)]
        background_image=background_dict[prompt]
        image=row["image"]

        image=image.resize((args.dim,args.dim))

        blip_inputs = blip_processor(image, return_tensors="pt").to(device, torch_dtype)
        blip_out = blip_model.generate(**blip_inputs)
        src_caption=blip_processor.decode(blip_out[0], skip_special_tokens=True)

        object=args.object
        if "object" in row:
            object=row["object"]

        source_prompt=src_caption
        target_prompt=f"{src_caption} {prompt}"
        prompts = [source_prompt, target_prompt]

        preprocessed_image=model.image_processor.preprocess(image).to(device)

        [height,width]=preprocessed_image.size()[-2:]

        start_code, latents_list = model.invert(preprocessed_image,
                                        "",
                                        guidance_scale=7.5,
                                        num_inference_steps=50,
                                        return_intermediates=True)
        start_code = start_code.expand(len(prompts), -1, -1, -1)

        STEP = 4
        LAYPER = 10

        # hijack the attention module
        editor = MutualSelfAttentionControl(STEP, LAYPER)
        regiter_attention_editor_diffusers(model, editor)

        # inference the synthesized image
        raw_image,augmented_image= model(prompts,
                            latents=start_code,
                            guidance_scale=7.5,height=height,width=width)
        
        augmented_image=augmented_image[0]

        # Note: querying the inversion intermediate features latents_list
        # may obtain better reconstruction and editing results
        '''image_latents,_ = model(prompts,
                               latents=start_code,
                                guidance_scale=7.5,
                                ref_intermediate_latents=latents_list)'''
        raw_image=postprocess_image(raw_image)[0]
        
        
        
        concat=concat_images_horizontally([row["image"],raw_image])

        accelerator.log({
            f"image_{k}":wandb.Image(concat)
        })
        with torch.no_grad():
            inputs = processor(
                    text=[prompt], images=[row["image"],raw_image,background_image], return_tensors="pt", padding=True
            )

            outputs = clip_model(**inputs)
            image_embeds=outputs.image_embeds.detach().cpu()
            text_embeds=outputs.text_embeds.detach().cpu()
            logits_per_text=torch.matmul(text_embeds, image_embeds.t())[0]
        #accelerator.print("logits",logits_per_text.size())

        image_similarities=torch.matmul(image_embeds,image_embeds.t()).numpy()[0]

        [_,text_score,__]=logits_per_text
        [_,image_score,image_score_background]=image_similarities
        #ir_score=ir_model.score(prompt,augmented_image)
        dino_score=dino_metric.get_scores(image, [raw_image])

        text_score_list.append(text_score.detach().cpu().numpy())
        image_score_list.append(image_score)
        image_score_background_list.append(image_score_background)
       # ir_score_list.append(ir_score)
        dino_score_list.append(dino_score)

        output_dict["augmented_image"].append(raw_image)
        output_dict["image"].append(image)
        output_dict["dino_score"].append(dino_score)
        output_dict["image_score"].append(image_score)
        output_dict["text_score"].append(text_score)
        output_dict["prompt"].append(prompt)

        Dataset.from_dict(output_dict).push_to_hub(args.dest_dataset)

       
    accelerator.log({
        "text_score_list":np.mean(text_score_list),
        "image_score_list":np.mean(image_score_list),
        "image_score_background_list":np.mean(image_score_background_list),
       # "ir_score_list":np.mean(ir_score_list),
        "dino_score_list":np.mean(dino_score_list)
    })

    

        


if __name__=='__main__':
    print_details()
    start=time.time()
    args=parser.parse_args()
    print(args)
    main(args)
    end=time.time()
    seconds=end-start
    hours=seconds/(60*60)
    print(f"successful generating:) time elapsed: {seconds} seconds = {hours} hours")
    print("all done!")