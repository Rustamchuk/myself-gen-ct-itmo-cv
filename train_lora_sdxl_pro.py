import os
import warnings

warnings.filterwarnings("ignore", message=".*OpenSSL.*", module="urllib3")
warnings.filterwarnings("ignore", message=".*device_type of 'cuda'.*", module="torch.amp")
warnings.filterwarnings("ignore", message=".*Palette images with Transparency.*", module="PIL")
warnings.filterwarnings("ignore", category=FutureWarning, module="diffusers")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "0")
if os.environ.get("PYTORCH_MPS_HIGH_WATERMARK_RATIO") is None:
    os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

import argparse
import math
from pathlib import Path
import torch
import torch.nn.functional as F

if not torch.cuda.is_available():
    _orig_autocast = torch.amp.autocast
    def _autocast_mps_fix(device_type="cuda", *args, **kwargs):
        if device_type == "cuda" or kwargs.get("device_type") == "cuda":
            kwargs = {k: v for k, v in kwargs.items() if k != "device_type"}
            return _orig_autocast("cpu", enabled=False, *args, **kwargs)
        return _orig_autocast(device_type, *args, **kwargs)
    torch.amp.autocast = _autocast_mps_fix
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from accelerate import Accelerator
from diffusers import (
    DDPMScheduler,
    StableDiffusionXLPipeline,
)
from diffusers.utils import convert_state_dict_to_diffusers
from peft import LoraConfig, get_peft_model_state_dict
from transformers import CLIPTextModel, CLIPTokenizer
import json


class FaceDataset(Dataset):
    def __init__(self, root, tokenizer, size=1024, instance_repeats=1, use_fixed_prompt=True, use_augment=False):
        self.root = Path(root)
        self.tokenizer = tokenizer
        self.use_fixed_prompt = use_fixed_prompt
        self.size = size

        images = []
        for ext in ["jpg", "jpeg", "png", "webp"]:
            images.extend(self.root.glob(f"*.{ext}"))
        self.images = []
        for _ in range(max(1, instance_repeats)):
            self.images.extend(images)
        if not self.images:
            raise ValueError(f"В {root} не найдено фото (jpg, jpeg, png, webp).")

        transform_list = [
            transforms.Resize(size, interpolation=transforms.InterpolationMode.LANCZOS),
            transforms.CenterCrop(size),
        ]
        if use_augment:
            transform_list += [
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.05, 0.05, 0.05, 0.02),
            ]
        transform_list += [transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        self.transform = transforms.Compose(transform_list)

    def load_caption(self, img):
        if self.use_fixed_prompt:
            return ""
        txt = img.with_suffix(".txt")
        if txt.exists():
            return txt.read_text().strip()
        return ""

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        img_path = self.images[i]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        caption = self.load_caption(img_path)
        return {"pixel_values": image, "caption": caption}



def build_prompt(identity_token, caption):
    base = f"photo of {identity_token} person, face of {identity_token}, portrait of {identity_token}"
    if caption:
        return base + ", " + caption
    return base



def train(args):
    use_fp16 = torch.cuda.is_available()
    mixed_precision = "fp16" if use_fp16 else "no"
    dtype = torch.float16 if use_fp16 else torch.float32

    accelerator = Accelerator(
        gradient_accumulation_steps=args.grad_accum,
        mixed_precision=mixed_precision,
    )

    device = accelerator.device

    pipe = StableDiffusionXLPipeline.from_pretrained(
        args.base_model,
        torch_dtype=dtype,
    )

    tokenizer = pipe.tokenizer
    tokenizer_2 = pipe.tokenizer_2
    text_encoder = pipe.text_encoder
    text_encoder_2 = pipe.text_encoder_2
    vae = pipe.vae
    unet = pipe.unet
    scheduler = DDPMScheduler.from_pretrained(args.base_model, subfolder="scheduler")

    vae = vae.to(device, dtype=dtype)
    text_encoder = text_encoder.to(device, dtype=dtype)
    text_encoder_2 = text_encoder_2.to(device, dtype=dtype)

    train_size = getattr(args, "resolution", 512)
    original_size = target_size = (train_size, train_size)
    crops_coords_top_left = (0, 0)

    placeholder_token = getattr(args, "placeholder_token", None) or args.token
    add_placeholder = getattr(args, "add_placeholder_token", True)
    placeholder_token_id_1 = None
    placeholder_token_id_2 = None
    placeholder_emb_1 = None
    placeholder_emb_2 = None

    if add_placeholder:
        for tok, enc, name in [
            (tokenizer, text_encoder, "tokenizer"),
            (tokenizer_2, text_encoder_2, "tokenizer_2"),
        ]:
            if placeholder_token in tok.get_vocab():
                add_placeholder = False
                print(f"Токен «{placeholder_token}» уже в {name}, placeholder отключён.")
                break
        if add_placeholder:
            n1 = tokenizer.add_tokens([placeholder_token], special_tokens=False)
            n2 = tokenizer_2.add_tokens([placeholder_token], special_tokens=False)
            if n1 == 0 or n2 == 0:
                add_placeholder = False
            else:
                text_encoder.resize_token_embeddings(len(tokenizer))
                text_encoder_2.resize_token_embeddings(len(tokenizer_2))
                pid1 = tokenizer.convert_tokens_to_ids(placeholder_token)
                pid2 = tokenizer_2.convert_tokens_to_ids(placeholder_token)
                emb1 = text_encoder.get_input_embeddings().weight.data
                emb2 = text_encoder_2.get_input_embeddings().weight.data
                for enc, emb, pid in [(tokenizer, emb1, pid1), (tokenizer_2, emb2, pid2)]:
                    person_ids = enc.encode("person", add_special_tokens=False)
                    src_id = person_ids[0] if person_ids else enc.unk_token_id
                    if src_id != enc.unk_token_id:
                        emb[pid].copy_(emb[src_id])
                placeholder_token_id_1 = pid1
                placeholder_token_id_2 = pid2
                placeholder_emb_1 = torch.nn.Parameter(emb1[pid1].clone().detach().to(device, dtype=dtype))
                placeholder_emb_2 = torch.nn.Parameter(emb2[pid2].clone().detach().to(device, dtype=dtype))
                print(f"Добавлен placeholder «{placeholder_token}» (id1={pid1}, id2={pid2}). Промпт обучения: «{build_prompt(placeholder_token, '')}» — такой же формат используйте при генерации.")

    dataset = FaceDataset(
        args.data,
        tokenizer,
        size=train_size,
        instance_repeats=getattr(args, "instance_repeats", 100),
        use_fixed_prompt=getattr(args, "fixed_prompt", True),
        use_augment=getattr(args, "use_augment", False),
    )
    n_photos = len(set(dataset.images)) if dataset.images else 0
    print(f"Датасет: {len(dataset)} сэмплов ({n_photos} фото × {getattr(args, 'instance_repeats', 10)} повторов), "
          f"промпт: {'один на все (как train_lora.py)' if getattr(args, 'fixed_prompt', True) else 'из .txt'}, "
          f"CenterCrop.")
    loader = DataLoader(dataset, batch_size=args.batch, shuffle=True, num_workers=0)


    train_text_encoder = getattr(args, "train_text_encoder", True)

    lora_config_unet = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank,
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        lora_dropout=0.05,
        bias="none",
        init_lora_weights="gaussian",
    )
    unet.enable_gradient_checkpointing()
    unet.add_adapter(lora_config_unet)
    trainable_params = [p for p in unet.parameters() if p.requires_grad]

    te_rank = getattr(args, "te_rank", min(16, args.rank))
    if train_text_encoder:
        lora_config_te = LoraConfig(
            r=te_rank,
            lora_alpha=te_rank,
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
            lora_dropout=0.0,
            bias="none",
            init_lora_weights="gaussian",
        )
        text_encoder.add_adapter(lora_config_te)
        text_encoder_2.add_adapter(lora_config_te)
        trainable_params = trainable_params + [p for p in text_encoder.parameters() if p.requires_grad]
        trainable_params = trainable_params + [p for p in text_encoder_2.parameters() if p.requires_grad]
        print("Включена LoRA на обоих текстовых энкодерах — привязка токена к лицу.")

    te_lr = getattr(args, "text_encoder_lr", 1e-5)
    placeholder_lr = getattr(args, "placeholder_lr", 5e-5)
    opt_groups = [{"params": [p for p in unet.parameters() if p.requires_grad], "lr": args.lr}]
    if train_text_encoder:
        te_params = [p for p in text_encoder.parameters() if p.requires_grad] + [p for p in text_encoder_2.parameters() if p.requires_grad]
        opt_groups.append({"params": te_params, "lr": te_lr})
    if placeholder_emb_1 is not None and placeholder_emb_2 is not None:
        opt_groups.append({"params": [placeholder_emb_1, placeholder_emb_2], "lr": placeholder_lr})
    optimizer = torch.optim.AdamW(opt_groups)

    unet, optimizer, loader = accelerator.prepare(
        unet, optimizer, loader
    )

    phase1_steps = getattr(args, "phase1_steps", 250)
    if placeholder_emb_1 is not None and phase1_steps > 0:
        print(f"Двухфазное обучение: Фаза 1 (шаги 1–{phase1_steps}) — только эмбеддинг «{args.token}» (UNet и TE заморожены). Фаза 2 — полное обучение. Так модель гарантированно запоминает лицо.")

    global_step = 0

    for epoch in range(1000):

        for batch in loader:

            with accelerator.accumulate(unet):

                images = batch["pixel_values"].to(device)

                with torch.no_grad():
                    latents = vae.encode(images).latent_dist.sample()
                    latents *= 0.18215

                noise = torch.randn_like(latents)
                num_train_timesteps = getattr(scheduler.config, "num_train_timesteps", scheduler.num_train_timesteps)
                timesteps = torch.randint(
                    0,
                    num_train_timesteps,
                    (latents.shape[0],),
                    device=device
                ).long()

                noisy_latents = scheduler.add_noise(latents, noise, timesteps)

                prompts = [
                    build_prompt(args.token, c)
                    for c in batch["caption"]
                ]

                tokens1 = tokenizer(
                    prompts,
                    padding="max_length",
                    max_length=tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )
                tokens2 = tokenizer_2(
                    prompts,
                    padding="max_length",
                    max_length=tokenizer_2.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )
                input_ids_1 = tokens1.input_ids.to(device)
                input_ids_2 = tokens2.input_ids.to(device)

                if placeholder_emb_1 is not None:
                    text_encoder.get_input_embeddings().weight.data[placeholder_token_id_1].copy_(placeholder_emb_1.data)
                    text_encoder_2.get_input_embeddings().weight.data[placeholder_token_id_2].copy_(placeholder_emb_2.data)

                if train_text_encoder:
                    out1 = text_encoder(input_ids_1, output_hidden_states=True)
                    out2 = text_encoder_2(input_ids_2, output_hidden_states=True)
                else:
                    with torch.no_grad():
                        out1 = text_encoder(input_ids_1, output_hidden_states=True)
                        out2 = text_encoder_2(input_ids_2, output_hidden_states=True)
                hidden1 = out1.hidden_states[-2]
                hidden2 = out2.hidden_states[-2]
                encoder_hidden_states = torch.cat([hidden1, hidden2], dim=-1)
                add_text_embeds = getattr(out2, "text_embeds", getattr(out2, "pooler_output", out2[0]))  # pooled

                add_time_ids = pipe._get_add_time_ids(
                    original_size,
                    crops_coords_top_left,
                    target_size,
                    dtype=dtype,
                    text_encoder_projection_dim=text_encoder_2.config.projection_dim,
                )
                add_time_ids = add_time_ids.to(device).repeat(encoder_hidden_states.shape[0], 1)
                added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}

                noise_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    added_cond_kwargs=added_cond_kwargs,
                ).sample

                loss = F.mse_loss(noise_pred, noise)

                accelerator.backward(loss)

                if placeholder_emb_1 is not None:
                    w1 = text_encoder.get_input_embeddings().weight
                    w2 = text_encoder_2.get_input_embeddings().weight
                    if w1.grad is not None:
                        placeholder_emb_1.grad = w1.grad[placeholder_token_id_1].clone().detach()
                    if w2.grad is not None:
                        placeholder_emb_2.grad = w2.grad[placeholder_token_id_2].clone().detach()

                if global_step < phase1_steps and placeholder_emb_1 is not None:
                    for p in unet.parameters():
                        if p.grad is not None:
                            p.grad.zero_()
                    for p in text_encoder.parameters():
                        if p.grad is not None:
                            p.grad.zero_()
                    for p in text_encoder_2.parameters():
                        if p.grad is not None:
                            p.grad.zero_()
                    if accelerator.is_main_process and global_step == 0:
                        print("Phase 1: только placeholder — шаг 0")

                optimizer.step()
                optimizer.zero_grad()
                if placeholder_emb_1 is not None:
                    if text_encoder.get_input_embeddings().weight.grad is not None:
                        text_encoder.get_input_embeddings().weight.grad.zero_()
                    if text_encoder_2.get_input_embeddings().weight.grad is not None:
                        text_encoder_2.get_input_embeddings().weight.grad.zero_()

                global_step += 1

                if accelerator.is_main_process and global_step % 50 == 0:
                    phase = "phase1" if global_step < phase1_steps else "phase2"
                    print(f"step {global_step}  loss {loss.item():.4f}  [{phase}]")
                if accelerator.is_main_process and global_step == phase1_steps and phase1_steps > 0:
                    print(f"Фаза 2: полное обучение (UNet + TE + placeholder) с шага {phase1_steps}")

                if global_step >= args.steps:
                    break

        if global_step >= args.steps:
            break

    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        unet = accelerator.unwrap_model(unet)
        unet_lora = convert_state_dict_to_diffusers(get_peft_model_state_dict(unet))
        save_kw = {"unet_lora_layers": unet_lora}
        if train_text_encoder:
            te1_sd = get_peft_model_state_dict(text_encoder)
            te2_sd = get_peft_model_state_dict(text_encoder_2)
            te1_lora = convert_state_dict_to_diffusers(te1_sd)
            te2_lora = convert_state_dict_to_diffusers(te2_sd)
            def strip_peft_prefix(d):
                return {k.replace("base_model.model.", ""): v for k, v in d.items()}
            save_kw["text_encoder_lora_layers"] = strip_peft_prefix(te1_lora)
            save_kw["text_encoder_2_lora_layers"] = strip_peft_prefix(te2_lora)
        pipe.save_lora_weights(args.output, **save_kw)
        if placeholder_emb_1 is not None and placeholder_emb_2 is not None:
            from safetensors.torch import save_file
            out_path = Path(args.output)
            ph_dir = out_path / "placeholder"
            ph_dir.mkdir(parents=True, exist_ok=True)
            save_file(
                {
                    "embedding_1": placeholder_emb_1.detach().cpu(),
                    "embedding_2": placeholder_emb_2.detach().cpu(),
                },
                ph_dir / "placeholder_embeddings.safetensors",
            )
            with open(ph_dir / "placeholder_token.json", "w", encoding="utf-8") as f:
                json.dump({"placeholder_token": placeholder_token}, f, ensure_ascii=False)
            tokenizer.save_pretrained(ph_dir / "tokenizer")
            tokenizer_2.save_pretrained(ph_dir / "tokenizer_2")
            print("Placeholder-токен и эмбеддинги сохранены в", ph_dir)
        print("LoRA saved to:", args.output)



def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--data", required=True)
    parser.add_argument("--output", default="./lora_face")
    parser.add_argument("--base_model", default="stabilityai/stable-diffusion-xl-base-1.0")

    parser.add_argument("--token", default="ohwx", help="Токен идентичности в промпте (photo of TOKEN person)")
    parser.add_argument("--add_placeholder_token", action="store_true", default=True, help="Добавить обучаемый эмбеддинг токена = сильнее сходство лица")
    parser.add_argument("--no_add_placeholder_token", action="store_false", dest="add_placeholder_token")
    parser.add_argument("--placeholder_token", default=None, help="Строка placeholder-токена (по умолчанию = --token)")
    parser.add_argument("--placeholder_lr", type=float, default=1e-4, help="LR эмбеддинга placeholder (выше = жёстче привязка лица)")

    parser.add_argument("--batch", type=int, default=1, help="Размер батча (1 рекомендуется для MPS)")
    parser.add_argument("--steps", type=int, default=600, help="Всего шагов (phase1 + phase2)")
    parser.add_argument("--phase1_steps", type=int, default=250, help="Фаза 1: только эмбеддинг токена — модель «запоминает» лицо, UNet/TE заморожены")
    parser.add_argument("--lr", type=float, default=5e-5, help="UNet LR (5e-5 как в train_lora.py — лучше для лиц)")
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--te_rank", type=int, default=16, help="Ранг LoRA для текстовых энкодеров (меньше = меньше памяти)")
    parser.add_argument("--resolution", type=int, default=512, help="Разрешение обучения (512 экономит память на MPS, 1024 для CUDA)")
    parser.add_argument("--instance_repeats", type=int, default=20, help="Повторений каждого фото (больше = модель чаще видит лицо)")
    parser.add_argument("--fixed_prompt", action="store_true", default=True, help="Один промпт «photo of TOKEN person» для всех фото (сильнее привязка лица)")
    parser.add_argument("--no_fixed_prompt", action="store_false", dest="fixed_prompt", help="Использовать подписи из .txt")
    parser.add_argument("--use_augment", action="store_true", default=False, help="RandomHorizontalFlip + ColorJitter (для разнообразия, может ослабить сходство)")
    parser.add_argument("--train_text_encoder", action="store_true", default=True, help="Обучать LoRA на текстовых энкодерах (привязка токена к лицу)")
    parser.add_argument("--no_train_text_encoder", action="store_false", dest="train_text_encoder", help="Отключить обучение TE (экономия памяти, слабее сходство)")

    parser.add_argument("--text_encoder_lr", type=float, default=1e-5, help="Learning rate для текстовых энкодеров")
    parser.add_argument("--grad_accum", type=int, default=4)

    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()

