import argparse
import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import gradio as gr

from bliva.common.config import Config
from bliva.common.dist_utils import get_rank
from bliva.common.registry import registry
from bliva.conversation.conversation import Chat, CONV_VISION, CONV_DIRECT

# imports modules for registration

from bliva.models import *
from bliva.processors import *
from bliva.models import load_model_and_preprocess
from evaluate import disable_torch_init

def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--model_name",default='bliva_vicuna', type=str, help='model name')
    parser.add_argument("--gpu_id", type=int, default=0, help="specify the gpu to load the model.")
    args = parser.parse_args()
    return args

# ========================================
#             Model Initialization
# ========================================

print('Initializing Chat')
args = parse_args()

if torch.cuda.is_available():
    device='cuda:{}'.format(args.gpu_id)
else:
    device=torch.device('cpu')

disable_torch_init()
if args.model_name == "blip2_vicuna_instruct":
    model, vis_processors, _ = load_model_and_preprocess(name=args.model_name, model_type="vicuna7b", is_eval=True, device=device)
elif args.model_name == "bliva_vicuna":
    model, vis_processors, _ = load_model_and_preprocess(name=args.model_name, model_type="vicuna7b", is_eval=True, device=device)
elif args.model_name == "bliva_flant5":
    model, vis_processors, _ = load_model_and_preprocess(name=args.model_name, model_type="flant5xxl", is_eval=True, device=device)
else:
    print("Model not found")    
    
vis_processor = vis_processors["eval"]


# vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
# vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
chat = Chat(model, vis_processor, device=device)
print('Initialization Finished')

# ========================================
#             Gradio Setting
# ========================================

def gradio_reset(chat_state, img_list):
    if chat_state is not None:
        chat_state.messages = []
    if img_list is not None:
        img_list = []
    return None, gr.update(value=None, interactive=True), gr.update(placeholder='Please upload your image first', interactive=False),gr.update(value="Upload & Start Chat", interactive=True), chat_state, img_list

def upload_img(gr_img, text_input, chat_state):
    if gr_img is None:
        return None, None, gr.update(interactive=True), chat_state, None
    chat_state = CONV_DIRECT.copy()   #CONV_VISION.copy()
    img_list = []
    llm_message = chat.upload_img(gr_img, chat_state, img_list)
    return gr.update(interactive=False), gr.update(interactive=True, placeholder='Type and press Enter'), gr.update(value="Start Chatting", interactive=False), chat_state, img_list

def gradio_ask(user_message, chatbot, chat_state):
    if len(user_message) == 0:
        return gr.update(interactive=True, placeholder='Input should not be empty!'), chatbot, chat_state
    chat.ask(user_message, chat_state)
    chatbot = chatbot + [[user_message, None]]
    return '', chatbot, chat_state


def gradio_answer(chatbot, chat_state, img_list, num_beams, temperature):
    llm_message = chat.answer(conv=chat_state,
                              img_list=img_list,
                              num_beams=num_beams,
                              temperature=temperature,
                              max_new_tokens=300,
                              max_length=2000)[0]
    chatbot[-1][1] = llm_message[0]
    return chatbot, chat_state, img_list

title = """<h1 align="center">Demo of BLIVA</h1>"""
description = """<h3>This is the demo of BLIVA. Upload your images and start chatting!</h3>"""
article = """<p><a href='https://gordonhu608.github.io/bliva/'><img src='https://img.shields.io/badge/Project-Page-Green'></a></p><p><a href='https://github.com/mlpc-ucsd/BLIVA'><img src='https://img.shields.io/badge/Github-Code-blue'></a></p><p><a href='https://raw.githubusercontent.com/'><img src='https://img.shields.io/badge/Paper-PDF-red'></a></p>
"""

#TODO show examples below

with gr.Blocks() as demo:
    gr.Markdown(title)
    gr.Markdown(description)
    gr.Markdown(article)

    with gr.Row():
        with gr.Column(scale=0.5):
            image = gr.Image(type="pil")
            upload_button = gr.Button(value="Upload & Start Chat", interactive=True, variant="primary")
            clear = gr.Button("Restart 🔄")
            
            num_beams = gr.Slider(
                minimum=1,
                maximum=10,
                value=5,
                step=1,
                interactive=True,
                label="beam search numbers)",
            )
            
            temperature = gr.Slider(
                minimum=0.1,
                maximum=2.0,
                value=1.0,
                step=0.1,
                interactive=True,
                label="Temperature",
            )

        with gr.Column():
            chat_state = gr.State()
            img_list = gr.State()
            chatbot = gr.Chatbot(label='BLIVA')
            text_input = gr.Textbox(label='User', placeholder='Please upload your image first', interactive=False)
            
            gr.Examples(examples=[
                [f"images/example.jpg", "What is this image about?"],
                [f"images/img3.jpg", "What is this image about?"],
                [f"images/img4.jpg", "What is the title of this movie?"],
            ], inputs=[image, text_input])          
            
    upload_button.click(upload_img, [image, text_input, chat_state], [image, text_input, upload_button, chat_state, img_list])
    
    text_input.submit(gradio_ask, [text_input, chatbot, chat_state], [text_input, chatbot, chat_state]).then(
        gradio_answer, [chatbot, chat_state, img_list, num_beams, temperature], [chatbot, chat_state, img_list]
    )
    clear.click(gradio_reset, [chat_state, img_list], [chatbot, image, text_input, upload_button, chat_state, img_list], queue=False)

demo.launch(share=True, enable_queue=True)