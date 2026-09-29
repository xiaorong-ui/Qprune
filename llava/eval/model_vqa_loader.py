import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from torch.utils.data import Dataset, DataLoader

from PIL import Image
import math


def configure_qficr_debug_stats(answers_file):
    value = os.environ.get("EC_QFID_DEBUG_STATS_JSONL", "").strip()
    if not value:
        return
    if value.lower() in {"1", "true", "yes", "on"}:
        value = os.path.join(os.path.dirname(os.path.abspath(answers_file)), "debug_qficr_stats.jsonl")
        os.environ["EC_QFID_DEBUG_STATS_JSONL"] = value
    if os.path.exists(value):
        os.remove(value)


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, questions, image_folder, tokenizer, image_processor, model_config):
        self.questions = questions
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config

    def __getitem__(self, index):
        line = self.questions[index]
        image_file = line["image"]
        qs = line["text"]
        if self.model_config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        image = Image.open(os.path.join(self.image_folder, image_file)).convert('RGB')
        image_tensor = process_images([image], self.image_processor, self.model_config)[0]

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')

        return input_ids, image_tensor, image.size

    def __len__(self):
        return len(self.questions)


def collate_fn(batch):
    input_ids, image_tensors, image_sizes = zip(*batch)
    input_ids = torch.stack(input_ids, dim=0)
    image_tensors = torch.stack(image_tensors, dim=0)
    return input_ids, image_tensors, image_sizes


# DataLoader
def create_data_loader(questions, image_folder, tokenizer, image_processor, model_config, batch_size=1, num_workers=4):
    assert batch_size == 1, "batch_size must be 1"
    dataset = CustomDataset(questions, image_folder, tokenizer, image_processor, model_config)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, collate_fn=collate_fn)
    return data_loader


def eval_model(args):
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)

    # Data
    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answer_mode = "w"
    if args.resume and os.path.exists(answers_file):
        completed = []
        with open(answers_file, "r") as existing_file:
            for line_number, line in enumerate(existing_file, 1):
                if not line.strip():
                    continue
                try:
                    completed.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise RuntimeError(
                        f"Cannot resume from invalid JSON at {answers_file}:{line_number}"
                    ) from exc
        if len(completed) > len(questions):
            raise RuntimeError(
                f"Resume file has {len(completed)} rows, but shard has only {len(questions)} questions"
            )
        for index, answer in enumerate(completed):
            expected_id = str(questions[index]["question_id"])
            actual_id = str(answer.get("question_id"))
            if actual_id != expected_id:
                raise RuntimeError(
                    f"Resume prefix mismatch at row {index + 1}: "
                    f"expected question_id={expected_id}, found {actual_id}"
                )
        questions = questions[len(completed):]
        answer_mode = "a"
        print(f"Resuming {answers_file}: {len(completed)} complete, {len(questions)} remaining")

    if not questions:
        print(f"Already complete: {answers_file}")
        return

    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    configure_qficr_debug_stats(answers_file)

    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, args.model_base, model_name,
        visual_token_num=args.visual_token_num,
    )
    ans_file = open(answers_file, answer_mode)

    if 'plain' in model_name and 'finetune' not in model_name.lower() and 'mmtag' not in args.conv_mode:
        args.conv_mode = args.conv_mode + '_mmtag'
        print(f'It seems that this is a plain model, but it is not using a mmtag prompt, auto switching to {args.conv_mode}.')

    data_loader = create_data_loader(questions, args.image_folder, tokenizer, image_processor, model.config)

    data_bar = tqdm(zip(data_loader, questions), total=len(questions))
    for (input_ids, image_tensors, image_sizes), line in data_bar:
        idx = line["question_id"]
        cur_prompt = line["text"]
        os.environ["EC_QFID_DEBUG_QUESTION_ID"] = str(idx)
        os.environ["EC_QFID_DEBUG_QUESTION_TEXT"] = str(cur_prompt)

        question = cur_prompt
        # question = question.split("\nReference OCR token")[0]
        question = question.replace("\nAnswer the question using a single word or phrase.", "")

        input_ids = input_ids.to(device='cuda', non_blocking=True)
        image_tensors = image_tensors.to(dtype=torch.float16, device='cuda', non_blocking=True)
        decoder_feedback_query_ids = tokenizer(
            question, add_special_tokens=False, return_tensors='pt'
        ).input_ids.to(device='cuda', non_blocking=True)

        with torch.inference_mode():
            output_ids, visual_token_num = model.generate(
                input_ids,
                images=image_tensors,
                image_sizes=image_sizes,
                texts=question,
                decoder_feedback_query_ids=decoder_feedback_query_ids,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                use_cache=True)
            if hasattr(model.model, 'visual_token_num'):
                visual_token_num = model.model.visual_token_num
            data_bar.set_postfix(vtn=f"{visual_token_num}")

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": cur_prompt,
                                   "text": outputs,
                                   "answer_id": ans_id,
                                   "model_id": model_name,
                                   "metadata": {}}) + "\n")
        ans_file.flush()
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--visual_token_num", type=int, default=576)
    parser.add_argument("--resume", action="store_true",
                        help="Validate and keep the completed prefix in answers-file, then run only remaining questions")
    args = parser.parse_args()

    eval_model(args)
