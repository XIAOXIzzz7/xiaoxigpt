import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList, \
    TextIteratorStreamer
from threading import Thread

tokenizer = AutoTokenizer.from_pretrained("lmsys/vicuna-7b-v1.3")
model = AutoModelForCausalLM.from_pretrained("lmsys/vicuna-7b-v1.3", torch_dtype=torch.float16, device_map="auto")
model = model.to('cuda:0')


class StopOnTokens(StoppingCriteria):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        stop_ids = [29, 0]
        for stop_id in stop_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False


def predict(message, history):
    history_transformer_format = history + [[message, ""]]
    stop = StopOnTokens()

    messages = "".join(["".join(["\n<human>:" + item[0], "\n<bot>:" + item[1]])  # curr_system_message +
                        for item in history_transformer_format])

    model_inputs = tokenizer([messages], return_tensors="pt").to("cuda")
    streamer = TextIteratorStreamer(tokenizer, timeout=10., skip_prompt=True, skip_special_tokens=True)
    generate_kwargs = dict(
        model_inputs,
        streamer=streamer,
        max_new_tokens=300,
        do_sample=True,
        top_p=0.5,
        top_k=20,
        temperature=0.7,
        num_beams=1,
        stopping_criteria=StoppingCriteriaList([stop])
    )
    t = Thread(target=model.generate, kwargs=generate_kwargs)
    t.start()

    partial_message = ""
    for new_token in streamer:
        if new_token != '<':
            partial_message += new_token
            print(partial_message)
            yield partial_message


gr.ChatInterface(predict).queue().launch(server_name="0.0.0.0", server_port=8811)