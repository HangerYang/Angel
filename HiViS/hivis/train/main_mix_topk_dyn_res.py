import argparse
from pathlib import Path

parser = argparse.ArgumentParser(description='sp')

TRAIN_DIR = Path(__file__).resolve().parent


parser.add_argument('--basepath', type=str, default='llava-hf/llava-v1.6-vicuna-7b-hf')
parser.add_argument(
    '--configpath',
    type=str,
    default=str(TRAIN_DIR / 'vicuna_7B_config.json'),
)
parser.add_argument('--lr', type=float, default=3e-5)
parser.add_argument('--bs', type=int, default=2)
parser.add_argument('--gradient-accumulation-steps', type=int, default=1)
parser.add_argument(
    '--text-data-dir',
    '--tmpdir',
    dest='text_data_dir',
    type=str,
    default='eval_data/generated/llava16_7b/sharegpt/non_code',
    help='Directory containing second-stage text training samples.',
)
parser.add_argument(
    '--multimodal-data-dir',
    type=str,
    default='eval_data/generated/llava16_7b/llava_v1_5_mix665k',
    help='Directory containing second-stage multimodal training samples.',
)
parser.add_argument('--outdir', type=str, default='outdir1')
parser.add_argument('--cpdir', type=str, default='checkpoints/stage2')
parser.add_argument('--topk', type=int, default=10)
parser.add_argument('--topk_w', type=float, default=1.0)
parser.add_argument('--forward_num_total', type=int, default=3)
parser.add_argument('--ckpt_path', type=str, default='checkpoints/stage1')


args = parser.parse_args()
if args.forward_num_total < 1:
    parser.error("--forward_num_total must be positive")


train_config={
    "lr":args.lr,
    "bs":args.bs,
    "gradient_accumulation_steps":args.gradient_accumulation_steps,
    "text_datapath": args.text_data_dir,
    "multimodal_datapath": args.multimodal_data_dir,
    "is_warmup":True,
    "num_epochs":10,
    "num_warmup_steps":2000,
    "total_steps":800000,
    "p_w":0.1,
    "v_w":1.0,
    "topk_w": args.topk_w,
    "head_w":0.1,
    "num_workers":2,
    "embeding":True,
    "act":"No",
    "data_noise":True,
    "noise":"uniform",
    "mean":0.0,
    "std":0.2,
    "residual":"true,norm",
    "max_len":4096,
    "config_path":args.configpath,
    "b1":0.9,
    "b2": 0.95,
    "grad_clip": 0.5,
    "save_freq": 5
}


import json
from safetensors import safe_open
import os
import torch

torch.backends.cuda.matmul.allow_tf32 = True
from accelerate import Accelerator
from accelerate.utils import set_seed

set_seed(0)
accelerator = Accelerator(mixed_precision='bf16',
                          gradient_accumulation_steps=train_config["gradient_accumulation_steps"])

from .cnets_dyn_res import Model
from .model_paths import is_qwen_model, resolve_base_model_path

from ..model.configs import EConfig
from typing import Any, Dict, List

from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
from transformers import get_linear_schedule_with_warmup, AutoConfig


def list_files(text_data_dir, multimodal_data_dir):
    datapath = []
    for label, directory in (
        ("text", text_data_dir),
        ("multimodal", multimodal_data_dir),
    ):
        root = Path(directory).expanduser()
        if not root.is_dir():
            raise FileNotFoundError(f"{label} training data directory does not exist: {root}. Pass --{label}-data-dir with the generated dataset path.")
        files = sorted(str(path) for path in root.rglob("*") if path.is_file())
        if not files:
            raise ValueError(f"{label} training data directory is empty: {root}")
        datapath.extend(files)
        if accelerator.is_main_process:
            print(f"Loaded {len(files)} {label} training files from {root}")
    if accelerator.is_main_process:
        print(f"Loaded {len(datapath)} training files in total")
    return datapath


datapath = list_files(
    train_config["text_datapath"],
    train_config["multimodal_datapath"],
)


base_model_path = resolve_base_model_path(args.basepath)
baseconfig = AutoConfig.from_pretrained(base_model_path)
use_qwen_position_ids = is_qwen_model(baseconfig)
if accelerator.is_main_process:
    model_family = "Qwen" if use_qwen_position_ids else "non-Qwen"
    print(f"Detected {model_family} base model ({baseconfig.model_type})")


try:
    with open(os.path.join(base_model_path, "model.safetensors.index.json"), "r") as f:
        index_json = json.loads(f.read())
        head_path = index_json["weight_map"]["language_model.lm_head.weight"]
    with safe_open(os.path.join(base_model_path, head_path),
                   framework="pt",
                   device="cpu") as f:
        tensor_slice = f.get_slice("language_model.lm_head.weight")
        vocab_size, hidden_dim = tensor_slice.get_shape()
        tensor = tensor_slice[:, :hidden_dim].float()
except:
    with open(os.path.join(base_model_path, "model.safetensors.index.json"), "r") as f:
        index_json = json.loads(f.read())
        head_path = index_json["weight_map"]["lm_head.weight"]
    with safe_open(os.path.join(base_model_path, head_path),
                   framework="pt",
                   device="cpu") as f:
        tensor_slice = f.get_slice("lm_head.weight")
        vocab_size, hidden_dim = tensor_slice.get_shape()
        tensor = tensor_slice[:, :hidden_dim].float()

head = torch.nn.Linear(tensor.shape[1], tensor.shape[0], bias=False)
head.weight.data = tensor
head.eval()

config = EConfig.from_pretrained(train_config["config_path"])

for param in head.parameters():
    param.requires_grad = False

for param in head.parameters():
    param.requires_grad = False


from torch.nn.utils.rnn import pad_sequence



class AddGaussianNoise:
    def __init__(self, mean=0.0, std=0.0):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        tensor = data["hidden_state_big"]
        noise = torch.randn(tensor.size()) * self.std + self.mean
        noisy_tensor = tensor + noise
        data["hidden_state_big"] = noisy_tensor
        return data


class AddUniformNoise:
    def __init__(self, std=0.0):
        self.std = std

    def __call__(self, data):
        tensor = data["hidden_state_big"]
        noise = (torch.rand_like(tensor) - 0.5) * self.std * 512 / tensor.shape[1]
        noisy_tensor = tensor + noise
        data["hidden_state_big"] = noisy_tensor
        return data


class CustomDataset(Dataset):
    def __init__(self, datapath, transform=None):
        self.data = datapath
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # try:
        data = torch.load(self.data[index])
        new_data = {}
        hidden_state = data['target'][:train_config["max_len"]][None, :]
        input_ids = data['input_ids'][:train_config["max_len"]][None, :]
        loss_mask = data["loss_mask"][:train_config["max_len"]][None, :]
        if use_qwen_position_ids:
            if "position_ids" not in data:
                raise KeyError(f"Qwen training sample is missing 'position_ids': {self.data[index]}")
            position_ids = data["position_ids"][:train_config["max_len"]][None, :]


        length = hidden_state.shape[1]
        attention_mask = [1] * length
        loss_mask = loss_mask[0].tolist()
        loss_mask[-1] = 0

        input_ids_target = input_ids[:, 1:]
        zeropadding = torch.tensor([[0]])
        input_ids_target = torch.cat((input_ids_target, zeropadding), dim=1)

        target = hidden_state[:, 1:, :]
        zeropadding = torch.zeros(1, 1, target.shape[2])
        target = torch.cat((target, zeropadding), dim=1)
        loss_mask[-1] = 0
        new_data["attention_mask"] = attention_mask
        new_data["loss_mask"] = loss_mask
        new_data["target"] = target
        new_data["hidden_state_big"] = hidden_state
        new_data["input_ids"] = input_ids_target
        if use_qwen_position_ids:
            new_data["position_ids"] = position_ids


        if self.transform:
            new_data = self.transform(new_data)

        return new_data


class DataCollatorWithPadding:

    def paddingtensor(self, intensors, N):
        B, n, S = intensors.shape
        # padding_tensor = torch.zeros(B, N - n, S,dtype=intensors.dtype)
        padding_tensor = torch.zeros(B, N - n, S)
        outtensors = torch.cat((intensors, padding_tensor), dim=1)
        return outtensors

    def paddingtensor2D(self, intensors, N):
        B, n = intensors.shape
        padding_tensor = torch.zeros(B, N - n, dtype=intensors.dtype)
        outtensors = torch.cat((intensors, padding_tensor), dim=1)
        return outtensors

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        max_length = max(item['hidden_state_big'].shape[1] for item in features)
        batch_input_ids = torch.cat([self.paddingtensor2D(item['input_ids'], max_length) for item in features])
        batch_hidden_states = torch.cat([self.paddingtensor(item['hidden_state_big'], max_length) for item in features])
        batch_target = torch.cat([self.paddingtensor(item['target'], max_length) for item in features])
        batch_loss_mask = torch.tensor(
            [item['loss_mask'] + [0] * (max_length - len(item['loss_mask'])) for item in features])
        batch_attention_mask = torch.tensor(
            [item['attention_mask'] + [0] * (max_length - len(item['attention_mask'])) for item in features])
        # batch_loss_mask = torch.ones_like(batch_loss_mask)
        # batch_attention_mask=torch.ones_like(batch_attention_mask)
        batch = {
            "input_ids": batch_input_ids,
            "hidden_states": batch_hidden_states,
            "target": batch_target,
            "attention_mask": batch_attention_mask,
            "loss_mask": batch_loss_mask,
        }
        if use_qwen_position_ids:
            batch["position_ids"] = torch.cat([self.paddingtensor2D(item["position_ids"], max_length) for item in features])
        return batch


def top_accuracy(output, target, topk=(1,)):
    # output.shape (bs, num_classes), target.shape (bs, )
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k)
        return res
    

def compute_loss(target, target_p, predict, loss_mask):
    out_head = head(predict)
    out_logp = nn.LogSoftmax(dim=2)(out_head)

    plogp = target_p * out_logp
    ploss = -torch.sum(torch.sum(loss_mask * plogp, 2)) / (loss_mask.sum() + 1e-5)

    vloss = criterion(predict, target)
    vloss = torch.sum(torch.mean(loss_mask * vloss, 2)) / (loss_mask.sum() + 1e-5)

    topk_mask = torch.topk(target_p, k=args.topk, dim=2).indices
    topk_loss = -torch.sum(torch.sum(loss_mask * plogp.gather(dim=2, index=topk_mask), 2)) / (loss_mask.sum() + 1e-5)

    return vloss, ploss, topk_loss, out_head


@torch.no_grad()
def getkacc(model, data, head, max_length=5):
    hidden_states = data["hidden_states"]
    input_ids = data["input_ids"]
    # attention_mask=data["attention_mask"]
    loss_mask = data["loss_mask"]
    # sample_mask=data["sample_mask"]
    target = data["target"]
    total = [0 for _ in range(max_length)]
    correct = [0 for _ in range(max_length)]
    bs, sl = hidden_states.shape[0], hidden_states.shape[1]
    target_headout = head(target)
    hidden_states_headout = head(hidden_states)

    for i in range(bs):
        for j in range(sl):

            single_hidden_states = hidden_states[i, :j]
            single_input_ids = input_ids[i, :j]

            single_hidden_states = single_hidden_states[None, :, :]
            single_input_ids = single_input_ids[None, :]
            for k in range(max_length):
                if loss_mask[i, single_hidden_states.shape[1] - 1] == 0:
                    break
                tmp_in_target_headout = hidden_states_headout[i, single_hidden_states.shape[1] - 1]
                tmp_out_target_headout = target_headout[i, single_hidden_states.shape[1] - 1]
                target_in_token = torch.argmax(tmp_in_target_headout)
                target_out_token = torch.argmax(tmp_out_target_headout)
                tmp_token = input_ids[i, single_hidden_states.shape[1] - 1]
                # tmp_sample_mask=sample_mask[i,single_hidden_states.shape[1]-1]
                if not (target_in_token == tmp_token):
                    break
                out_hidden = model(single_hidden_states, input_ids=single_input_ids)
                last_hidden = out_hidden[:, -1]
                last_headout = head(last_hidden)
                token = torch.argmax(last_headout)
                total[k] += 1
                if token == target_out_token:
                    correct[k] += 1
                else:
                    for kk in range(k + 1, max_length):
                        total[kk] += 1
                    break

                single_hidden_states = torch.cat((single_hidden_states, out_hidden[:, -1:]), dim=1)
                single_input_ids = torch.cat((single_input_ids, torch.tensor([[token]]).to(single_input_ids.device)),
                                             dim=1)

    acc = [correct[i] / total[i] if total[i] else 0 for i in range(len(correct))]
    return acc


if train_config["data_noise"]:
    if train_config["noise"] == "uniform":
        aug = AddUniformNoise(std=train_config["std"])
    else:
        aug = AddGaussianNoise(mean=train_config["mean"], std=train_config["std"])
else:
    aug = None

traindatapath = datapath[:int(len(datapath) * 0.95)]
testdatapath = datapath[int(len(datapath) * 0.95):]
# print('td',train_config["datapath"])
# print(datapath)
# exit()
traindataset = CustomDataset(traindatapath, transform=aug)
testdataset = CustomDataset(testdatapath)
train_loader = DataLoader(traindataset, batch_size=train_config["bs"], shuffle=True,
                          collate_fn=DataCollatorWithPadding(), num_workers=train_config["num_workers"],
                          pin_memory=True)
test_loader = DataLoader(testdataset, batch_size=train_config["bs"], shuffle=False,
                         collate_fn=DataCollatorWithPadding(), num_workers=train_config["num_workers"], pin_memory=True)
# for batch_data in train_loader:
#     print(batch_data)

if accelerator.is_main_process:
    if not os.path.exists(args.cpdir):
        os.makedirs(args.cpdir)

config = EConfig.from_pretrained(train_config["config_path"])
model = Model(
    config,
    load_emb=True,
    path=base_model_path,
    forward_num=args.forward_num_total,
)

import safetensors

if args.ckpt_path is not None: 
    ea_model_path = args.ckpt_path
    
    load_model_path=os.path.join(ea_model_path, "pytorch_model.bin")
    if os.path.exists(load_model_path):
        ea_layer_state_dict = torch.load(load_model_path, map_location="cuda")
    else:
        load_model_path = os.path.join(ea_model_path, "model.safetensors")
        ea_layer_state_dict = safetensors.torch.load_file(load_model_path)

    if "adaptive_embedding" in ea_layer_state_dict:
        ea_layer_state_dict["residual"] = ea_layer_state_dict.pop(
            "adaptive_embedding"
        )
    if "residual" in ea_layer_state_dict:
        residual = ea_layer_state_dict["residual"]  # shape: [1, hidden_size]
        if residual.shape[0] == 1:
            ea_layer_state_dict["residual"] = residual.expand(
                model.residual.shape[0], -1
            ).clone()
        elif residual.shape[0] != model.residual.shape[0]:
            raise ValueError(f"Checkpoint has {residual.shape[0]} residuals, but --forward_num_total={args.forward_num_total} requires {model.residual.shape[0]}")

    model.load_state_dict(ea_layer_state_dict, strict=True)

    print(f"load model from {load_model_path}")

criterion = nn.SmoothL1Loss(reduction="none")
optimizer = optim.AdamW(model.parameters(), lr=train_config["lr"], betas=(train_config["b1"], train_config["b2"]))


num_epochs = train_config["num_epochs"]
num_warmup_steps = train_config["num_warmup_steps"]
total_steps = train_config["total_steps"]
is_warmup = train_config["is_warmup"]

if is_warmup:
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps,
                                                num_training_steps=total_steps)

    model, head, optimizer, train_loader, test_loader, scheduler = accelerator.prepare(
        model, head, optimizer, train_loader, test_loader, scheduler
    )
else:
    model, head, optimizer, train_loader, test_loader = accelerator.prepare(
        model, head, optimizer, train_loader, test_loader
    )

import time
for epoch in range(num_epochs + 1):
    top_3acc = [0 for _ in range(3)]
    correct = 0
    total = 0
    epoch_loss = 0
    num_batches = 0
    epoch_start_time = time.time()

    model.train()
    for batch_idx, data in enumerate(tqdm(train_loader)):

        with accelerator.accumulate(model):
            optimizer.zero_grad()
            hidden_states, input_ids, attention_mask, target, loss_mask = data["hidden_states"], data["input_ids"], data["attention_mask"], data["target"], data["loss_mask"][..., None]
            loss = 0
            with torch.no_grad():
                target_head = head(target)
                target_p = nn.Softmax(dim=2)(target_head)
                target_p = target_p.detach()
                token = torch.argmax(target_p,dim=-1)[..., None].detach()


            q_hidden_states = None
            loss_mask_forward = loss_mask

            for forward_idx in range(args.forward_num_total):
                model_inputs = {"q_hidden_states": q_hidden_states}
                if use_qwen_position_ids:
                    model_inputs["position_ids"] = data["position_ids"]
                predict = model(hidden_states, input_ids, attention_mask, **model_inputs)

                if q_hidden_states is None:
                    q_hidden_states = torch.cat([hidden_states[:, :1, :], predict[:, :-1, :]], dim=1)[None, :, :, :]
                else:
                    new_q_hidden_states = torch.cat([q_hidden_states[-1][:, :1, :], predict[:, :-1, :]], dim=1)[None, :, :, :]
                    q_hidden_states = torch.cat([q_hidden_states, new_q_hidden_states], dim=0)
                q_hidden_states = q_hidden_states.detach()

                vloss, ploss, topk_loss, out_head = compute_loss(target, target_p, predict, loss_mask_forward)
                total_loss = train_config["p_w"] * ploss  + train_config["v_w"] * vloss  + train_config["topk_w"] * topk_loss
                loss += total_loss
                accelerator.backward(total_loss)
                if forward_idx == args.forward_num_total-1:
                    break

                with torch.no_grad():

                    out_head = head(predict)
                    _, draft_topk = torch.topk(out_head, k=5, dim=2)

                    if forward_idx>0:
                        loss_mask_shift = torch.cat([torch.zeros(loss_mask_forward.shape[0], 1, 1).to(loss_mask_forward.device), loss_mask_forward[:,:-1,:]], dim=1)
                        loss_mask_forward[mask == 0] = 1
                        mask = (draft_topk == token).any(dim=-1)[..., None]
                        out_mask = mask * loss_mask_shift * loss_mask_forward
                        loss_mask_forward = out_mask * loss_mask

                    else:
                        mask = (draft_topk == token).any(dim=-1)[..., None]
                        loss_mask_forward = mask * loss_mask



            accelerator.clip_grad_value_(model.parameters(), train_config["grad_clip"])
            optimizer.step()
            optimizer.zero_grad()
            loss /= args.forward_num_total
            if is_warmup:
                scheduler.step()

        with torch.no_grad():
            _, predicted = torch.max(out_head, 2)
            _, target = torch.max(target_head, 2)
            ct = loss_mask_forward.sum().item()
            cc = ((predicted == target) * loss_mask_forward.squeeze()).sum().item()
            out_head = out_head.view(-1, target_head.shape[-1])[loss_mask_forward.view(-1) == 1]
            target = target.view(-1)[loss_mask_forward.view(-1) == 1]
            topkacc = top_accuracy(out_head, target, (1, 2, 3))
            for top_i in range(len(topkacc)):
                top_3acc[top_i] += topkacc[top_i]
            total += ct
            correct += cc
        if accelerator.is_main_process and ct != 0:
            logdict = {"train/lr": optimizer.optimizer.param_groups[0]["lr"], "train/vloss": vloss.item(),
                       "train/ploss": ploss.item(), "train/topkloss": topk_loss.item(), "train/loss": loss.item(), "train/acc": cc / ct}
            for id, i in enumerate(top_3acc):
                logdict[f'train/top_{id + 1}_acc'] = topkacc[id].item() / ct

        del ploss, vloss
        epoch_loss += loss.item()
        num_batches += 1
    
    epoch_time = time.time() - epoch_start_time
    batches_per_sec = num_batches / epoch_time if epoch_time > 0 else 0.0

    correct, total = torch.tensor(correct).cuda(), torch.tensor(total).cuda()
    correct, total = accelerator.gather_for_metrics((correct, total))
    correct, total = correct.sum().item(), total.sum().item()
    epoch_loss /= num_batches
    top_3acc = accelerator.gather_for_metrics(top_3acc)
    # if accelerator.is_local_main_process:
    #     for id, i in enumerate(top_3acc):
    #         wandb.log({f'train/epochtop_{id + 1}_acc': i.sum().item() / total})
    if accelerator.is_local_main_process:
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, epoch_loss))
        print('Train Accuracy: {:.2f}%'.format(100 * correct / total))
        print(args.cpdir)
        print(f'Train speed: {batches_per_sec:.2f} batches/s')

        # wandb.log({"train/epochacc": correct / total, "train/epochloss": epoch_loss})
        accelerator.save_state(output_dir=f"{args.cpdir}/state_{epoch}")


    # if (epoch + 1) % train_config["save_freq"]:
        # top_3acc = [0 for _ in range(3)]
        # correct = 0
        # total = 0
        # epoch_loss = 0
        # num_batches = 0
        # model.eval()

        # k_acc = [[] for i in range(5)]
        # for batch_idx, data in enumerate(tqdm(test_loader)):
        #     with torch.no_grad():
        #         if batch_idx < 10:
        #             acces = getkacc(model, data, head, max_length=5)
        #             for i in range(len(acces)):
        #                 k_acc[i].append(acces[i])
        #         predict = model(data["hidden_states"], input_ids=data["input_ids"],
        #                         attention_mask=data["attention_mask"])
        #         target_head = head(data["target"])
        #         target_p = nn.Softmax(dim=2)(target_head)
        #         target_p = target_p.detach()
        #         out_head = head(predict)
        #         out_logp = nn.LogSoftmax(dim=2)(out_head)
        #         loss_mask = data["loss_mask"][:, :, None]
        #         plogp = target_p * out_logp
        #         ploss = -torch.sum(torch.sum(loss_mask * plogp, 2)) / (loss_mask.sum()+1e-5)
        #         vloss = criterion(predict, data["target"])
        #         vloss = torch.sum(torch.mean(loss_mask * vloss, 2)) / (loss_mask.sum()+1e-5)
        #         loss = train_config["v_w"] * vloss + train_config["p_w"] * ploss
        #         _, predicted = torch.max(out_head, 2)
        #         _, target = torch.max(target_head, 2)
        #         ct = loss_mask.sum().item()
        #         cc = ((predicted == target) * loss_mask.squeeze()).sum().item()
        #         out_head = out_head.view(-1, target_head.shape[-1])[loss_mask.view(-1) == 1]
        #         target = target.view(-1)[loss_mask.view(-1) == 1]
        #         topkacc = top_accuracy(out_head, target, (1, 2, 3))
        #         for top_i in range(len(topkacc)):
        #             top_3acc[top_i] += topkacc[top_i]
        #         total += ct
        #         correct += cc
        #     epoch_loss += loss.item()
        #     num_batches += 1

        # mean_acces = []
        # for id, i in enumerate(k_acc):
        #     mean_acc = np.array(i).mean()
        #     mean_acc = torch.tensor(mean_acc).cuda()
        #     mean_acces.append(mean_acc)

        # mean_acces = accelerator.gather_for_metrics(mean_acces)
        # if accelerator.is_local_main_process:
        #     for id, i in enumerate(mean_acces):
        #         mean_acc = i.mean().item()
        #         wandb.log({f"test/{id}_acc": mean_acc})

        # correct, total = torch.tensor(correct).cuda(), torch.tensor(total).cuda()
        # correct, total = accelerator.gather_for_metrics((correct, total))
        # correct, total = correct.sum().item(), total.sum().item()
        # top_3acc = accelerator.gather_for_metrics(top_3acc)
        # if accelerator.is_local_main_process:
        #     for id, i in enumerate(top_3acc):
        #         wandb.log({f'test/top_{id + 1}_acc': i.sum().item() / total})
        # epoch_loss /= num_batches
        # if accelerator.is_local_main_process:
        #     print('Test Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, epoch_loss))
        #     print('Test Accuracy: {:.2f}%'.format(100 * correct / total))
        #     wandb.log({"test/epochacc": correct / total, "test/epochloss": epoch_loss})
        #     # accelerator.save_model(model, f"checkpoints/model_{epoch}")
        #     # accelerator.save_state(output_dir=f"{args.outdir}/state_{epoch}")
        #     # os.system(f"cp -r {args.outdir} {args.cpdir}")
        #     accelerator.save_state(output_dir=f"{args.cpdir}/state_{epoch}")
