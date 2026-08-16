"""
用 teacher (Qwen2.5-1.5B-Instruct) 自问自答生成 QA 蒸馏数据。

原理: KD 只能教"数据里出现的输入"。当前 20 万条任务数据里常识问答 <0.1%，
student 学不到"中国的首都是北京"。此脚本让 teacher 对海量开放问题生成回答，
把 teacher 的知识倒出来作为训练数据。

用法:
  python scripts/generate_qa.py \
      --teacher /root/autodl-tmp/data/qwen2.5-1.5b-instruct \
      --output /root/autodl-tmp/data/qa_50k.jsonl \
      --max-samples 50000 \
      --batch-size 4
"""
import os, sys, json, random, argparse, re

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# ── 脏输出过滤 ──────────────────────────────────────────────────────
_REPEAT_CHAR = re.compile(r'(.)\1{4,}')          # 同一字符重复 5+（"对对对对"）
_REPEAT_WORD = re.compile(r'(.{2,8})\1{3,}')     # 短词重复 4+（"哈哈哈哈"）
_SYMBOL_ONLY = re.compile(r'^[\W_]+$')           # 纯符号


def filter_answer(q: str, ans: str) -> bool:
    """Return True if the answer is acceptable (keep), False if dirty."""
    ans = ans.strip()
    if not (5 <= len(ans) <= 500):
        return False
    if _REPEAT_CHAR.search(ans):
        return False
    if _REPEAT_WORD.search(ans):
        return False
    if _SYMBOL_ONLY.match(ans):
        return False
    # emoji/特殊符号占比过高（>25%）
    nonsymbol = sum(1 for ch in ans if ch.isalnum() or '\u4e00' <= ch <= '\u9fff')
    if nonsymbol / max(1, len(ans)) < 0.75:
        return False
    # 答案直接复述问题开头（如"什么是光合作用是…"）
    if len(q) >= 6 and ans.startswith(q[:6]):
        return False
    # 明确拒绝/无法回答（这类占比例控制：保留但标注？这里直接过滤）
    if re.match(r'^(抱歉|对不起|我不(知道|确定|清楚)|无法回答)', ans):
        return False
    return True


# ── 问题模板 × 实体词（覆盖常识问答） ──
TEMPLATES = [
    "{}的首都是什么？",
    "什么是{}？",
    "请介绍一下{}。",
    "{}是谁？",
    "{}是什么时候出现的？",
    "{}有什么作用？",
    "{}是怎么产生的？",
    "{}的意义是什么？",
    "{}有哪些特点？",
    "为什么要研究{}？",
    "{}最早出现在哪里？",
    "{}对人类社会有什么影响？",
    "{}和{}有什么区别？",
    "{}为什么重要？",
    "请举例说明{}。",
    "{}在哪个国家？",
    "{}的主要成分是什么？",
    "{}的工作原理是什么？",
    "{}有哪些类型？",
    "{}是怎么发展起来的？",
    "{}为什么是重要的？",
    "{}的起源是什么？",
    "{}有哪些应用？",
    "{}和{}有什么关系？",
    "{}的优缺点是什么？",
    "{}是怎么被发现的？",
    "{}的历史是什么？",
    "{}的代表人物是谁？",
    "{}出现在哪些地方？",
    "{}对人类的影响是什么？",
    "{}是由什么组成的？",
    "{}的由来是什么？",
    "{}有什么传说？",
    "{}的现状如何？",
    "{}为什么有意义？",
    "{}在不同文化中是什么样？",
    "{}和{}哪个更重要？",
    "{}为什么著名？",
    "{}是如何工作的？",
    "{}有哪些有趣的方面？",
]

ENTITIES = [
    # 国家/首都/城市
    "中国", "北京", "美国", "华盛顿", "日本", "东京", "法国", "巴黎",
    "英国", "伦敦", "俄罗斯", "莫斯科", "德国", "柏林", "意大利", "罗马",
    "埃及", "开罗", "印度", "新德里", "澳大利亚", "加拿大", "巴西", "韩国", "首尔",
    "上海", "广州", "成都", "西安", "深圳", "南京", "杭州", "武汉", "重庆", "昆明",
    "纽约", "洛杉矶", "芝加哥", "新加坡", "悉尼", "莫斯科", "伊斯坦布尔",
    # 名人
    "李白", "杜甫", "苏轼", "鲁迅", "曹雪芹", "孔子", "老子", "庄子", "孟子",
    "屈原", "陶渊明", "白居易", "王维", "李清照", "辛弃疾", "毛泽东", "周恩来",
    "牛顿", "爱因斯坦", "伽利略", "达尔文", "爱迪生", "居里夫人", "诺贝尔",
    "特斯拉", "霍金", "图灵", "瓦特", "哥白尼", "开普勒",
    # 科学概念
    "光合作用", "重力", "原子", "DNA", "基因", "蛋白质", "量子力学",
    "相对论", "进化论", "生态系统", "气候变暖", "可再生能源", "能量守恒",
    "电磁波", "黑洞", "细胞", "病毒", "细菌", "遗传", "神经网络",
    # 历史
    "辛亥革命", "第一次世界大战", "第二次世界大战", "文艺复兴",
    "丝绸之路", "长城", "兵马俑", "造纸术", "指南针", "火药", "印刷术",
    "唐朝", "宋朝", "明朝", "清朝", "秦朝", "三国", "五四运动",
    # 社会文化
    "汉字", "京剧", "唐诗", "宋词", "红楼梦", "西游记", "三国演义", "水浒传",
    "春节", "中秋节", "端午节", "故宫", "旗袍", "功夫", "围棋", "象棋",
    "茶", "瓷器", "书法", "中医", "春节联欢晚会",
    # 科技
    "云计算", "人工智能", "区块链", "大数据", "5G", "互联网",
    "电动汽车", "太阳能", "核能", "手机", "电脑", "机器人", "无人机",
    "自动驾驶", "物联网", "虚拟现实", "芯片", "卫星", "高铁",
    # 自然
    "大熊猫", "鲸鱼", "老虎", "竹子", "沙漠", "海洋", "珠穆朗玛峰",
    "亚马逊雨林", "珊瑚礁", "企鹅", "北极", "臭氧层", "水资源", "土壤",
    # 医学
    "疫苗", "抗生素", "维生素", "免疫系统", "心脏", "大脑", "糖尿病", "感冒",
]


def build_questions(seed: int = 0) -> list:
    rng = random.Random(seed)
    qs = []
    for t in TEMPLATES:
        if "{}和{}" in t:
            for _ in range(100):
                a, b = rng.sample(ENTITIES, 2)
                qs.append(t.format(a, b))
        else:
            for e in ENTITIES:
                qs.append(t.format(e))
    rng.shuffle(qs)
    return qs


# ── 多轮对话追问池（含长程引用——训练跨轮检索） ──
# 2026-08 扩充：原 15 条导致多轮数据模式单一（对话能力不足的根因之一）。
# 按"信息请求/深化/对比/情境化/反事实/元对话"六类扩充至 45 条。
FOLLOWUPS = [
    # 信息请求（原池）
    "能不能说得更详细一点？",
    "这有什么实际例子吗？",
    "为什么这么重要？",
    "那它是什么时候出现的？",
    "它对普通人有什么影响？",
    "还有别的类似的东西吗？",
    "现在的情况怎么样？",
    "这有什么有趣的故事吗？",
    "你觉得值得深入了解吗？",
    "和我想的不太一样，能再解释一下吗？",
    "最早是谁发现的？",
    "有哪些常见的误解？",
    "给我一个简单的比喻说明。",
    "未来会怎么发展？",
    "它在生活中哪里用得到？",
    # 深化
    "这背后的原理是什么？",
    "能举一个反例吗？",
    "这个说法有没有什么争议？",
    "它的优点和缺点分别是什么？",
    "你刚才说的那个结论，依据是什么？",
    "如果从另一个角度看，会有什么不同？",
    "这件事和我的专业领域有什么关系？",
    # 对比
    "它和传统做法比有什么优势？",
    "和其他类似的东西比，它好在哪？",
    "东西方在这个问题上的看法有什么不同？",
    "过去和现在相比，变化大吗？",
    "它在不同年龄段的人眼里有什么差别？",
    # 情境化
    "如果是我这种情况，应该怎么应用？",
    "我刚开始接触，应该从哪里入手？",
    "如果时间有限，最重要的是先做什么？",
    "预算不多的话，有什么替代方案？",
    "放在我们公司（学校）的场景下，具体怎么落地？",
    # 反事实/边界
    "如果当初没有它，会怎么样？",
    "它有没有什么不适用的情况？",
    "做到什么程度就算合格了？",
    "有没有什么常见错误是新手容易犯的？",
    "这件事有什么前提条件吗？",
    # 元对话/收束
    "你说的和我理解的是一个意思吗？",
    "我需要记住的最关键一点是什么？",
    "关于这个话题，还有什么值得我继续了解的？",
    "如果我想深入学习，应该看什么资料？",
    "今天聊的这些，你能帮我总结一下吗？",
]

# 长程引用追问（记忆微调关键：第 N 轮引用第 1 轮信息）
REFERENCE_FOLLOWUPS = [
    "你刚才说的那个名字是什么来着？",
    "我最早提到的那个概念，你能再说说吗？",
    "第一轮你说的那个东西是什么？",
    "我们最开始讨论的是什么？",
    "你前面提到的那个术语叫什么？",
    "我记得你之前说过一个关键点，是什么来着？",
    "回到我们最开始的话题，那个事情怎么样了？",
    "你刚才提到的细节，能重复一下吗？",
    "最初的问题是什么来着？",
    "第一轮你回答的核心内容是什么？",
    "我们刚开头时你提到的那组数据是多少？",
    "你最开始举的例子，能再讲一遍吗？",
    "还记得我一开始说的背景吗？它和现在聊的有什么关系？",
    "第一轮里你说要注意什么来着？",
    "开头我们确认过的那个目标，现在怎么评估？",
    "我之前提到过我自己的情况，你还记得吗？",
    "最开始你说这件事有三个要点，是哪三个？",
    "我们第一轮聊的那个方案，现在可以补充细节吗？",
    "你最早给的建议里，哪一条最重要？",
    "还记得我们是怎么开始聊这个话题的吗？",
]


def generate_multi_turn(model, tok, device, topic, max_new_tokens=100):
    """Generate a 3-4 turn conversation about a topic using the teacher."""
    msgs = []
    # 首轮 user 提问
    q1 = random.choice([
        f'你能介绍一下{topic}吗？', f'什么是{topic}？',
        f'关于{topic}，你知道些什么？', f'请讲讲{topic}。',
    ])
    for _ in range(3):  # 3-4 轮
        msgs.append({'role': 'user', 'content': q1})
        prompt = tok.apply_chat_template(msgs, tokenize=False,
                                         add_generation_prompt=True)
        ids = tok(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model.generate(
                **ids, max_new_tokens=max_new_tokens,
                temperature=0.6, top_p=0.9, do_sample=True,
            )
        ans = tok.decode(out[0][ids['input_ids'].shape[1]:],
                         skip_special_tokens=True).strip()
        if not filter_answer(q1, ans):
            return None
        msgs.append({'role': 'assistant', 'content': ans})
        # 追问
        q1 = random.choice(FOLLOWUPS)
    # 转成 conversations 格式
    return [{'from': 'human' if m['role'] == 'user' else 'assistant',
             'value': m['content']} for m in msgs]


def batch_generate(model, tok, device, prompts, max_new_tokens,
                   temperature=0.6):
    """Batch-generate answers for a list of prompts (much faster than serial).

    Pads prompts to common length; attention_mask excludes padding so
    generation is parallel across the batch.
    """
    enc = tok(prompts, return_tensors='pt', padding=True).to(device)
    with torch.no_grad():
        out = model.generate(
            **enc, max_new_tokens=max_new_tokens,
            temperature=temperature, top_p=0.9, do_sample=True,
        )
    results = []
    for i in range(len(prompts)):
        inp_len = enc['input_ids'][i].shape[0]
        results.append(tok.decode(out[i][inp_len:],
                                  skip_special_tokens=True).strip())
    return results


def self_instruct_questions(model, tok, device, topic, n_q=15, max_tokens=256):
    """让 teacher 生成关于某主题的多样化问题（self-instruct 风格）。

    Returns a list of question strings.
    """
    prompt = (f'请生成{n_q}个关于“{topic}”的不同类型的常见问题'
              f'（包括事实类、理解类、应用类、对比类），'
              f'每行一个，只输出问题本身，不要编号以外的其他内容：')
    msgs = [{'role': 'user', 'content': prompt}]
    p = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = tok(p, return_tensors='pt').to(device)
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=max_tokens,
                             temperature=0.8, top_p=0.9, do_sample=True)
    text = tok.decode(out[0][ids['input_ids'].shape[1]:],
                      skip_special_tokens=True).strip()
    qs = []
    for line in text.split('\n'):
        line = re.sub(r'^\s*\d+[\.、)\s]*', '', line).strip()
        line = re.sub(r'^[-\*]\s*', '', line).strip()
        if 6 <= len(line) <= 80 and '？' in line or 6 <= len(line) <= 80:
            qs.append(line)
        if len(qs) >= n_q:
            break
    return qs


def _generate_self_instruct(model, tok, device, args):
    """teacher 自产问题 -> 生成回答（问题池无限，可支撑大数量）。"""
    topics = list(ENTITIES)
    written = 0
    seen_q = set()
    q_buf = []  # 待回答的问题缓冲
    with open(args.output, 'w', encoding='utf-8') as f:
        while written < args.max_samples:
            # 问题池补充（batch 生成问题，多个主题并行）
            if len(q_buf) < args.batch_size:
                t_topics = [topics[(written + i) % len(topics)]
                            for i in range(args.batch_size)]
                prompts = [(
                    '请生成10个关于"' + t + '"的不同类型的常见问题'
                    '（包括事实类、理解类、应用类、对比类），'
                    '每行一个，只输出问题本身：') for t in t_topics]
                texts = batch_generate(model, tok, device, prompts,
                                       256, temperature=0.8)
                for text in texts:
                    for line in text.split('\n'):
                        line = re.sub(r'^\s*\d+[\.、)\s]*', '', line).strip()
                        line = re.sub(r'^[-\*]\s*', '', line).strip()
                        if 6 <= len(line) <= 80:
                            q_buf.append(line)
            # batch 生成回答
            batch_q = []
            while len(batch_q) < args.batch_size and q_buf:
                q = q_buf.pop(0)
                if q not in seen_q:
                    seen_q.add(q)
                    batch_q.append(q)
            if not batch_q:
                continue
            prompts = [tok.apply_chat_template(
                [{'role': 'user', 'content': q}],
                tokenize=False, add_generation_prompt=True) for q in batch_q]
            answers = batch_generate(model, tok, device, prompts,
                                     args.max_new_tokens, temperature=0.6)
            for q, a in zip(batch_q, answers):
                if not filter_answer(q, a):
                    continue
                f.write(json.dumps({'instruction': q, 'input': '', 'output': a},
                                   ensure_ascii=False) + '\n')
                written += 1
                if written % 2000 == 0:
                    print(f'  {written}/{args.max_samples} self-instruct QA')
                if written >= args.max_samples:
                    break
    print(f'Done: {written} self-instruct QA -> {args.output}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--teacher', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--max-samples', type=int, default=50000)
    ap.add_argument('--batch-size', type=int, default=4)
    ap.add_argument('--max-new-tokens', type=int, default=120)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--mode', choices=['qa', 'multi-turn', 'self-instruct'], default='qa',
                    help='qa=模板问答; multi-turn=多轮对话; self-instruct=教师自产问题')
    ap.add_argument('--memory-tune', action='store_true',
                    help='记忆微调模式：多轮 7 轮 + 长程引用追问（训练跨轮检索）')
    args = ap.parse_args()

    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f'Loading teacher: {args.teacher}')
    tok = AutoTokenizer.from_pretrained(args.teacher, trust_remote_code=True)
    # decoder-only 必须 left-padding（right-padding 会让短样本在尾部 pad，
    # 生成质量受损 + transformers 警告刷屏）
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = 'left'
    model = AutoModelForCausalLM.from_pretrained(
        args.teacher, trust_remote_code=True,
        dtype=torch.bfloat16 if device != 'cpu' else torch.float32,
    ).to(device).eval()

    questions = build_questions()
    print(f'Question pool: {len(questions)} (will sample up to {args.max_samples})')

    if args.mode == 'multi-turn':
        _generate_multi_turn(model, tok, device, args)
        return
    if args.mode == 'self-instruct':
        _generate_self_instruct(model, tok, device, args)
        return

    rng = random.Random(42)
    written = 0
    seen = set()

    with open(args.output, 'w', encoding='utf-8') as f:
        while written < args.max_samples:
            batch_q = []
            while len(batch_q) < args.batch_size and len(questions) > 0:
                q = questions.pop()
                batch_q.append(q)
            if not batch_q:
                questions = build_questions()  # 池耗尽则重新洗牌再抽
                continue

            prompts = [tok.apply_chat_template(
                [{'role': 'user', 'content': q}],
                tokenize=False, add_generation_prompt=True) for q in batch_q]
            answers = batch_generate(model, tok, device, prompts,
                                     args.max_new_tokens, temperature=0.6)
            for q, ans in zip(batch_q, answers):
                if not filter_answer(q, ans):
                    continue
                key = (q, ans[:50])
                if key in seen:
                    continue
                seen.add(key)
                f.write(json.dumps(
                    {'instruction': q, 'input': '', 'output': ans},
                    ensure_ascii=False) + '\n')
                written += 1
                if written % 1000 == 0:
                    print(f'  {written}/{args.max_samples} generated')
                if written >= args.max_samples:
                    break

    print(f'Done: {written} QA samples -> {args.output}')


def _generate_multi_turn(model, tok, device, args):
    """Generate N multi-turn conversations (batched by round).

    记忆微调版：6-8 轮（默认 7），随机插入 1-2 次"引用早期信息"的追问
    （REFERENCE_FOLLOWUPS）——训练跨轮检索能力。
    """
    topics = list(ENTITIES)
    rng = random.Random(42)
    rng.shuffle(topics)
    written = 0
    seen = set()
    n_rounds = getattr(args, 'memory_tune', False) and 7 or 4
    n_rounds = max(n_rounds, 4)

    def _round_prompt(msgs, q):
        return tok.apply_chat_template(
            msgs + [{'role': 'user', 'content': q}],
            tokenize=False, add_generation_prompt=True)

    with open(args.output, 'w', encoding='utf-8') as f:
        while written < args.max_samples:
            # 建 batch 个对话
            convs = []
            for _ in range(args.batch_size):
                topic = topics[written % len(topics)]
                q1 = rng.choice([
                    f'你能介绍一下{topic}吗？', f'什么是{topic}？',
                    f'关于{topic}，你知道些什么？', f'请讲讲{topic}。',
                ])
                convs.append({'topic': topic, 'msgs': [], 'q': q1, 'done': False,
                              'ref_used': 0})
            for rnd in range(n_rounds):  # 每轮批量推进
                pending = [c for c in convs if not c['done']]
                if not pending:
                    break
                prompts = [_round_prompt(c['msgs'], c['q']) for c in pending]
                answers = batch_generate(model, tok, device, prompts,
                                         args.max_new_tokens, temperature=0.6)
                for c, a in zip(pending, answers):
                    if not filter_answer(c['q'], a):
                        c['done'] = True
                        continue
                    c['msgs'].append({'role': 'user', 'content': c['q']})
                    c['msgs'].append({'role': 'assistant', 'content': a})
                    # 追问选择：长程引用（随机插入 1-2 次）+ 普通追问
                    if (rnd >= 1 and len(c['msgs']) >= 4
                            and c['ref_used'] < 2 and rng.random() < 0.35):
                        c['q'] = rng.choice(REFERENCE_FOLLOWUPS)
                        c['ref_used'] += 1
                    else:
                        c['q'] = rng.choice(FOLLOWUPS)
            # 写完成（>=2 轮）的对话
            for c in convs:
                if c['done'] or len(c['msgs']) < 4:
                    continue
                conv = [{'from': 'human' if m['role'] == 'user' else 'assistant',
                         'value': m['content']} for m in c['msgs']]
                key = tuple((t['from'], t['value'][:30]) for t in conv)
                if key in seen:
                    continue
                seen.add(key)
                f.write(json.dumps({'conversations': conv},
                                   ensure_ascii=False) + '\n')
                written += 1
                if written % 1000 == 0:
                    print(f'  {written}/{args.max_samples} multi-turn convs')
                if written >= args.max_samples:
                    break
    print(f'Done: {written} multi-turn conversations -> {args.output}')


if __name__ == '__main__':
    main()
