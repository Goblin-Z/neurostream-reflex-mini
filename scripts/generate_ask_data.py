"""
用 teacher 生成"澄清式对话"数据——训练模型主动提问、组织语言的核心数据。

目标（与训练核心目的对齐）：模型能够提出逻辑顺畅的问题，进行问答和对话。
数据形态：4 轮对话 = 用户模糊请求 → 助手澄清提问（1-3 个问句）→ 用户补充
        → 助手完整回答。模型在 SFT 中学会"信息不足时主动提问，而不是瞎猜"。

用法:
  python scripts/generate_ask_data.py \
      --teacher /root/autodl-tmp/data/qwen2.5-1.5b-instruct \
      --output /root/autodl-tmp/data/clarify_30k.jsonl \
      --max-samples 30000 --batch-size 4 --device cuda
"""
import os, sys, json, random, re, argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# ── 模糊请求池（信息不足场景，迫使模型提问）────────────────────
TOPICS = [
    "产品推广", "旅游攻略", "健身计划", "学习计划", "公司年会", "团建活动",
    "简历", "辞职信", "个人理财", "装修", "宠物养护", "孩子的教育",
    "开源项目", "软件选型", "服务器配置", "算法题", "论文", "研究课题",
    "餐厅", "电影", "小说", "游戏", "手机", "笔记本电脑", "耳机",
    "汽车", "保险", "贷款", "租房", "购房", "投资", "股票基金",
]

AMBIGUOUS_TEMPLATES = [
    "帮我写一份关于{topic}的方案",
    "帮我推荐一个{topic}",
    "给我解释一下{topic}",
    "我最近想搞{topic}，有什么建议吗",
    "帮我写一段关于{topic}的文字",
    "介绍一下{topic}",
    "我想了解一下{topic}",
    "帮我规划一下{topic}",
    "{topic}怎么做比较好",
    "有没有关于{topic}的好办法",
]

# 少样本示例（教 teacher 输出格式 + 提问风格：问句结尾、逻辑清晰、一次问全）
FEWSHOT = """示例：
用户：帮我写一份关于产品推广的方案
助手：好的，可以帮你写。在动笔之前，我想先确认几点：1）这份方案主要给谁看（老板、客户还是内部执行团队）？2）推广的核心目标是什么（品牌曝光、拉新还是转化）？3）大概的预算和时间范围？
用户：给老板看的，目标是把新产品推出去，预算 50 万，三个月内完成。
助手：好的，明白了。下面是这份产品推广方案的大纲：一、目标：三个月内完成新产品市场导入，覆盖目标客户群 100 万……（正文略）"""

PROMPT = """你是一个对话数据生成器。请根据下面的用户请求，生成一段 4 轮对话，必须严格使用这种格式：

用户：{request}
助手：（先提出 1-3 个澄清问题，必须以问号结尾，逻辑清晰，一次问全关键信息）
用户：（对澄清问题给出具体、合理的补充回答）
助手：（基于补充信息，给出完整、有条理、结构化的最终回答，200 字以上）

要求：
- 助手第一轮必须提问，不能直接回答（因为信息不足）
- 问题要具体、覆盖请求中缺失的关键信息，不能是空泛的"能说详细点吗"
- 用户补充要自然，像真实对话
- 最终回答要组织有序（分点、分步骤），语言流畅

{fewshot}

现在请生成：
用户：{request}
"""


def build_request(rng):
    t = rng.choice(AMBIGUOUS_TEMPLATES)
    topic = rng.choice(TOPICS)
    return t.format(topic=topic)


def parse_dialogue(text, request):
    """从 teacher 输出解析 (q1, a1, q2, a2)。失败返回 None。"""
    lines = [ln.strip() for ln in text.split('\n') if ln.strip()]
    turns = []  # [(role, content)]
    cur_role, cur_buf = None, []
    for ln in lines:
        m = re.match(r'^(用户|助手|user|assistant)[：:]\s*(.*)$', ln)
        if m:
            if cur_role is not None and cur_buf:
                turns.append((cur_role, ' '.join(cur_buf)))
            cur_role = 'user' if m.group(1) in ('用户', 'user') else 'assistant'
            cur_buf = [m.group(2).strip()]
        else:
            if cur_role is not None:
                cur_buf.append(ln)
    if cur_role is not None and cur_buf:
        turns.append((cur_role, ' '.join(cur_buf)))

    # 需要 4 轮：user(请求) assistant(提问) user(补充) assistant(回答)
    if len(turns) < 4:
        return None
    u1, a1, u2, a2 = turns[0][1], turns[1][1], turns[2][1], turns[3][1]
    if not (u1 and a1 and u2 and a2):
        return None
    # 助手第一轮必须含问号（提问行为）
    if '？' not in a1 and '?' not in a1:
        return None
    # 过滤脏回答（重复/过短）
    if len(a2) < 60 or re.search(r'(.)\1{4,}', a2):
        return None
    return [u1, a1, u2, a2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--teacher', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--max-samples', type=int, default=30000)
    ap.add_argument('--batch-size', type=int, default=4)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--max-new-tokens', type=int, default=400)
    args = ap.parse_args()

    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f'Loading teacher: {args.teacher}')
    tok = AutoTokenizer.from_pretrained(args.teacher, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = 'left'
    model = AutoModelForCausalLM.from_pretrained(
        args.teacher, trust_remote_code=True,
        dtype=torch.bfloat16 if device != 'cpu' else torch.float32,
    ).to(device).eval()

    rng = random.Random(42)
    written = 0
    seen = set()
    with open(args.output, 'w', encoding='utf-8') as f:
        while written < args.max_samples:
            requests = [build_request(rng) for _ in range(args.batch_size)]
            prompts = [PROMPT.format(request=r, fewshot=FEWSHOT) for r in requests]
            enc = tok(prompts, return_tensors='pt', padding=True).to(device)
            with torch.no_grad():
                out = model.generate(
                    **enc, max_new_tokens=args.max_new_tokens,
                    temperature=0.7, top_p=0.9, do_sample=True,
                )
            for i, r in enumerate(requests):
                inp_len = enc['input_ids'][i].shape[0]
                text = tok.decode(out[i][inp_len:], skip_special_tokens=True).strip()
                parsed = parse_dialogue(text, r)
                if parsed is None:
                    continue
                key = (parsed[0], parsed[2])
                if key in seen:
                    continue
                seen.add(key)
                u1, a1, u2, a2 = parsed
                conv = [
                    {'from': 'human', 'value': u1},
                    {'from': 'assistant', 'value': a1},
                    {'from': 'human', 'value': u2},
                    {'from': 'assistant', 'value': a2},
                ]
                f.write(json.dumps({'conversations': conv},
                                   ensure_ascii=False) + '\n')
                written += 1
                if written % 2000 == 0:
                    print(f'  {written}/{args.max_samples} clarify dialogues')
                if written >= args.max_samples:
                    break
    print(f'Done: {written} clarify dialogues -> {args.output}')


if __name__ == '__main__':
    main()
