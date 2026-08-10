import math
import torch
import torch.nn.functional as F

from config.model_config import ReflexConfig
from interaction.feedback import StructuredFeedback
from interaction.manager import InteractionManager
from learn.hebbian_update import focal_update


class ReflexPipeline:
    """
    User interaction pipeline with InteractionManager protocol.

    Two distinct input paths:
      PATH A (feedback):  User input arrives while manager is AWAITING_ANSWER
                          → processed as answer to the active question
      PATH B (new query): User input arrives while manager is IDLE
                          → processed as a normal conversation query

    The question-answer protocol ensures:
      - Exactly one active question at any time
      - User input is never confused between "answer to question" and "new query"
      - The correct expert receives targeted feedback
    """

    def __init__(self, model, config: ReflexConfig = None):
        self.model = model
        self.config = config or model.config
        self.tokenizer = None
        self.feedback_processor = StructuredFeedback(config)
        self.interaction = InteractionManager(config)
        self.max_new_tokens = getattr(self.config, 'max_new_tokens', 256)
        # 会话记忆：多轮对话历史（显式 token 级拼接，与 SFT 多轮训练格式一致）
        self._chat_history = []          # list of {'role': 'user'/'assistant', 'content': str}
        self._max_history_turns = 8      # 最多保留 8 轮
        self._max_history_chars = 2000   # 历史总字符上限（防止超长）

    def set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer
        self.model._decode_tokenizer = tokenizer

    def process_text(self, text, max_length=None):
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not set.")

        # ── Route: feedback path (PATH A) or new query path (PATH B) ──
        if self.interaction.is_expecting_answer:
            print(f"  [FEEDBACK] User answer received, state=AWAITING_ANSWER")
            return self._process_feedback(text)
        else:
            print(f"  [NEW QUERY] state={self.interaction.state}")
            return self._process_new_query(text, max_length)

    # ── PATH A: Feedback ───────────────────────────────────────────

    def _process_feedback(self, text):
        """
        User is answering an active verification question.
        Retrieve the question context, process the answer as feedback,
        train the confused expert, then generate the response.
        """
        question = self.interaction.retrieve_answer()
        if question is None:
            return self._process_new_query(text)

        feedback_ctx = self.feedback_processor.process(
            question, text, self.tokenizer, self.model
        )

        question_ids = question.get('query_ids')
        expert_id = question.get('expert_id')

        input_ids = self._build_chat_input(text)

        with self.model._lock:
            response_ids = self.model.generate(
                input_ids,
                max_new_tokens=self.max_new_tokens,
                temperature=self.config.sampling_temperature,
                repetition_penalty=self.config.sampling_repetition_penalty,
                top_k=self.config.sampling_top_k,
                top_p=self.config.sampling_top_p,
            )

            eos_id = self.tokenizer.eos_token_id
            segments = []
            if question_ids is not None and question_ids.numel() > 0:
                q_ids = question_ids.to(input_ids.device)
                if q_ids.dim() == 1:
                    q_ids = q_ids.unsqueeze(0)
                segments.append(q_ids)
                if eos_id is not None:
                    segments.append(
                        torch.tensor([[eos_id]], device=input_ids.device)
                    )
            segments.append(input_ids)
            segments.append(response_ids[:, input_ids.size(1):])
            if eos_id is not None:
                segments.append(
                    torch.tensor([[eos_id]], device=input_ids.device)
                )
            full_ids = torch.cat(segments, dim=-1)
            # I2: 回复起点 = 除 response/eos 外的所有前缀段
            prefix_len = sum(seg.size(1) for seg in segments[:-2])
            self._train_on_full(full_ids, feedback_ctx,
                                reply_start=prefix_len)

        self.interaction.finalize_feedback(
            feedback_ctx, self.model._internal_step_count
        )

        response_text = self.tokenizer.decode(
            response_ids[0, input_ids.size(1):], skip_special_tokens=True
        )
        return response_text

    # ── PATH B: New query ──────────────────────────────────────────

    def _trim_history(self):
        """限制历史轮数与总字符（保持最近对话，防超长）。"""
        total = sum(len(m['content']) for m in self._chat_history)
        while (len(self._chat_history) > self._max_history_turns
               or total > self._max_history_chars):
            removed = self._chat_history.pop(0)
            total -= len(removed['content'])
            if len(self._chat_history) <= 1:
                break

    def _inject_dialog_memory(self, input_ids, response_ids=None):
        """L1 短期语义记忆：对话 embedding 混合进 h_t（方向 C）。

        模型内在记忆——h_t 随后自然影响 Router 状态门控、
        SelfModel 演化、AttnRes，零外部检索代码。
        """
        if not getattr(self.config, 'memory_enabled', True):
            return
        sm = getattr(self.model, 'self_model', None)
        h = getattr(self.model, '_h_state', None)
        if sm is None or h is None:
            return
        try:
            with torch.no_grad():
                emb = self.model.token_embedding(input_ids).mean(dim=1)  # [1, d]
                if response_ids is not None:
                    r_emb = self.model.token_embedding(
                        response_ids).mean(dim=1)
                    emb = 0.5 * emb + 0.5 * r_emb
                # RISK-3: 归一化对话向量——防 embedding 量级污染 GRU 状态
                emb_norm = emb.norm(dim=-1, keepdim=True).clamp(min=1e-6)
                emb = emb / emb_norm * (self.config.d_model ** 0.5)
                alpha = getattr(self.config, 'dialog_memory_alpha', 0.3)
                new_h = (alpha * emb + (1 - alpha) * h.to(emb.device))
                self.model._h_state = new_h.detach()
                # 同步到 endosphere（内循环 get_latest 可读取）
                self.model.endosphere.push(emb[0].detach().cpu(), sigma=0.5)
                # RISK-5: KV→语义槽固化链——对话轮次语义沉淀为长期记忆
                mb = getattr(self.model, 'memory_bank', None)
                if mb is not None:
                    mb.write(emb[0].detach(), lr=0.02)
        except Exception:
            pass  # 记忆注入是 best-effort，永不打断对话

    def _store_round_memory(self):
        """L4 内容记忆：收集本轮各层 KV 存入 MemoryBank（FIFO）。"""
        mb = getattr(self.model, 'memory_bank', None)
        if mb is None:
            return
        try:
            layer_kvs = []
            for layer in self.model.layers:
                attn = layer.attention
                if getattr(attn, '_last_kv', None) is None:
                    return  # 本轮 forward 未缓存 KV（非训练模式）
                k, v = attn._last_kv   # [B, n_kv, T, hd]
                layer_kvs.append(
                    (k[0].float().cpu(), v[0].float().cpu()))
            mb.add_round_kv(layer_kvs, text='round')
        except Exception:
            pass

    def _build_mem_kv(self):
        """L4 读取：从 MemoryBank 构建 {layer_idx: (mem_k, mem_v)}。"""
        mb = getattr(self.model, 'memory_bank', None)
        if mb is None or not getattr(self.config, 'memory_enabled', True):
            return None
        try:
            mem_kv = {}
            for i, layer in enumerate(self.model.layers):
                kv = mb.get_kv(i)
                if kv is None:
                    return None
                mem_k, mem_v = kv
                if mem_k.size(1) == 0:
                    return None
                mem_kv[i] = (mem_k.unsqueeze(0).to(
                    next(self.model.parameters()).device),
                    mem_v.unsqueeze(0).to(
                        next(self.model.parameters()).device))
            return mem_kv
        except Exception:
            return None

    def _build_chat_input(self, text):
        """Tokenize user text, matching SFT training format (no chat template)."""
        ids = self.tokenizer.encode(str(text), add_special_tokens=True,
                                     max_length=self.config.max_seq_len, truncation=True)
        return torch.tensor([ids], dtype=torch.long).to(
            next(self.model.parameters()).device
        )

    def _process_new_query(self, text, max_length=None):
        """
        Normal conversation query with multi-turn memory.
        History + current input are joined via the Qwen chat template
        (same format as SFT multi-turn training), so the model can
        reference previous turns.
        """
        # ── 构造带历史的输入 ──
        self._chat_history.append({'role': 'user', 'content': str(text)})
        self._trim_history()

        if getattr(self.tokenizer, 'chat_template', None):
            prompt_str = self.tokenizer.apply_chat_template(
                self._chat_history, tokenize=False,
                add_generation_prompt=True)
        else:
            prompt_str = text
        input_ids = self.tokenizer(
            prompt_str, return_tensors='pt',
            max_length=self.config.max_seq_len, truncation=True)['input_ids']
        input_ids = input_ids.to(next(self.model.parameters()).device)
        if input_ids.size(1) == 0:
            return ""

        with self.model._lock:
            # L4: 开启 KV 缓存 + 构建记忆 KV
            for layer in self.model.layers:
                layer.attention._kv_cache_enabled = True
            mem_kv = self._build_mem_kv()
            # 内外循环一致化：生成带内循环最新状态（Router 状态门控）
            h_state = getattr(self.model, '_h_state', None)
            response_ids = self.model.generate(
                input_ids,
                max_new_tokens=self.max_new_tokens,
                temperature=self.config.sampling_temperature,
                repetition_penalty=self.config.sampling_repetition_penalty,
                top_k=self.config.sampling_top_k,
                top_p=self.config.sampling_top_p,
                mem_kv=mem_kv,
                h_state=h_state,
            )

            eos_id = self.tokenizer.eos_token_id
            full_ids = torch.cat(
                [input_ids, response_ids[:, input_ids.size(1):]], dim=-1
            )
            if eos_id is not None:
                full_ids = torch.cat(
                    [full_ids,
                     torch.tensor([[eos_id]], device=input_ids.device)],
                    dim=-1
                )
            self._train_on_full(full_ids, feedback_ctx=None,
                                reply_start=input_ids.size(1))

        response_text = self.tokenizer.decode(
            response_ids[0, input_ids.size(1):], skip_special_tokens=True
        )

        # ── 记录 assistant 回复到历史 ──
        if response_text.strip():
            self._chat_history.append(
                {'role': 'assistant', 'content': response_text.strip()})
            self._trim_history()

        # ── L1: 对话注入 h_t（短期语义记忆）──
        self._inject_dialog_memory(input_ids, response_ids[:, input_ids.size(1):])

        # ── L4: 轮次内容记忆（KV 缓存写入）──
        if getattr(self.config, 'memory_enabled', True):
            self._store_round_memory()

        # ── Check for pending question from interaction manager ──
        if self.interaction.state == InteractionManager.QUESTION_PENDING:
            q = self.interaction.active_question
            if q is not None:
                # Prioritize confused_text (the generated question)
                # over query_ids (which is just training context,
                # i.e. the user's last input — NOT the question text)
                q_text = q.get('confused_text')
                if not q_text:
                    # Fallback: decode query_ids if no question text
                    q_ids = q.get('query_ids')
                    if q_ids is not None:
                        if q_ids.dim() == 1:
                            q_ids = q_ids.unsqueeze(0)
                        q_text = self.tokenizer.decode(
                            q_ids[0], skip_special_tokens=True
                        )

                if q_text and len(q_text.strip()) >= 1:
                    response_text += f"\n\n[QUESTION] {q_text}"
                    self.interaction.notify_question_displayed()
                else:
                    self.interaction.cancel_question()

        return response_text

    # ── Training ───────────────────────────────────────────────────

    def _train_on_full(self, full_ids, feedback_ctx=None, reply_start=None):
        # H5 fix: truncate to max_seq_len to prevent position embedding overflow
        max_len = self.config.max_seq_len
        if full_ids.size(1) > max_len:
            full_ids = full_ids[:, -max_len:]
        self.model.train()
        # 在线训练同样带记忆 + 状态（L4 + 内外循环一致）：
        # 梯度经过记忆注意力与状态门控路径，模型被训练成
        # "何时回忆、如何带着想法思考"
        mem_kv = self._build_mem_kv()
        h_state = getattr(self.model, '_h_state', None)
        logits = self.model(full_ids, mem_kv=mem_kv, h_state=h_state)

        shift = full_ids[:, 1:]
        # I2 fix: 只监督模型回复段——prompt/用户文本作上下文不参与 CE
        # （labels[t]=input[t+1]；回复起点 reply_start 对应的监督位置是 reply_start-1）
        labels = shift.clone()
        if reply_start is not None and reply_start > 1:
            labels[:, :reply_start - 1] = -100
        loss = F.cross_entropy(
            logits[:, :-1, :].contiguous().view(-1, logits.size(-1)),
            labels.contiguous().view(-1),
            ignore_index=-100,
        )
        loss = loss.clamp(max=getattr(self.config, 'internal_loss_clip', 10.0))

        # ── Sigma calibration loss: uncertainty_head learns to reflect
        # actual prediction uncertainty (high CE -> high sigma, low CE -> low sigma).
        # Small weight (0.05) as auxiliary signal, not dominant.
        # Target: tanh(ce_loss) in [0, 1] -- normalized uncertainty estimate.
        learnable = getattr(self.model, '_learnable_sigmas', None)
        if self.model.training and learnable is not None:
            try:
                ce_uncertainty = math.tanh(loss.item())
                target_sigma = torch.tensor(
                    ce_uncertainty, dtype=torch.float32,
                    device=logits.device,
                ).detach()
                sigma_cal_loss = F.mse_loss(
                    learnable.mean(),
                    target_sigma,
                )
                sigma_cal_weight = getattr(
                    self.config, 'sigma_calibration_weight', 0.05
                )
                loss = loss + sigma_cal_weight * sigma_cal_loss
            except Exception:
                pass

        # ── Alignment loss: direct correction toward user's answer ──
        # CE loss teaches "predict conversation", alignment loss teaches
        # "match the semantic direction of the user's correction".
        # Small weight (0.2) so it complements CE, not competes with it.
        if (feedback_ctx is not None
                and feedback_ctx.get('focal_boost') is not None
                and feedback_ctx.get('answer_text') is not None
                and feedback_ctx.get('expert_id') is not None):
            try:
                align_loss = self._compute_alignment_loss(feedback_ctx)
                if align_loss is not None:
                    align_weight = getattr(
                        self.config, 'feedback_align_loss_weight', 0.2
                    )
                    loss = loss + align_weight * align_loss
            except Exception:
                pass  # alignment is best-effort, never break training

        self._update_experts(loss, feedback_ctx)

    def _compute_alignment_loss(self, feedback_ctx):
        """Compute alignment between confused expert output and user answer."""
        answer_text = feedback_ctx.get('answer_text')
        expert_id = feedback_ctx.get('expert_id')
        if not answer_text or not expert_id:
            return None

        expert = self.model.get_expert_by_id(expert_id)
        if expert is None or expert._output is None:
            return None

        with torch.no_grad():
            answer_ids = self.tokenizer(
                answer_text, return_tensors='pt',
                max_length=64, truncation=True,
            )['input_ids'].to(next(self.model.parameters()).device)
            answer_emb = self.model.token_embedding(answer_ids).mean(dim=1)

        # Expert output from the forward pass: [B, T, d_model]
        expert_out = expert._output
        if expert_out.dim() == 3:
            expert_out = expert_out.mean(dim=1)  # [B, d_model]
        elif expert_out.dim() == 2:
            expert_out = expert_out.mean(dim=0, keepdim=True)

        # Alignment: 1 - cosine_similarity (0 = perfect alignment, 2 = opposite)
        align = 1.0 - F.cosine_similarity(expert_out, answer_emb.detach()).mean()
        return align

    def _update_experts(self, loss, feedback_ctx=None):
        sig = getattr(self.model, '_last_sigma_aggregate', 0.3)
        base_gamma = min(max(
            sig / max(self.config.verify_threshold, 0.01), 0.5
        ), 2.0)

        focal_boost = None
        target_expert_id = None
        if feedback_ctx is not None:
            focal_boost = feedback_ctx.get('focal_boost')
            target_expert_id = feedback_ctx.get('expert_id')

        # Batch gradient computation per layer (O(n) not O(n^2))
        from loop.gradient_manager import GradientManager
        gm = GradientManager(len(self.model.layers))

        target_found = False

        for layer_idx, layer, retain in gm.iterate_layers(
                loss, self.model.layers
        ):
            captured = [(e, e._output) for e in layer.all_experts
                         if e._output is not None and e._output.requires_grad]
            if not captured:
                continue

            outputs = [out for _, out in captured]
            grads = torch.autograd.grad(
                loss, outputs,
                retain_graph=retain,
                allow_unused=True,
            )

            for (expert, _), grad_y in zip(captured, grads):
                if grad_y is None:
                    continue

                boost = 1.0
                if (focal_boost is not None
                        and expert.id == target_expert_id):
                    boost = focal_boost
                    target_found = True

                focal_update(grad_y, expert, expert.effective_lr,
                             base_gamma, focus_boost=boost)

        # If the target expert was not activated in this forward pass,
        # apply a direct gradient nudge using a proxy gradient.
        # C3 fix: the computation graph may already be freed by
        # autograd.grad(retain_graph=False) in the main loop above.
        # Use try/except to gracefully skip on freed graph.
        if (focal_boost is not None and not target_found
                and target_expert_id is not None):
            target_expert = self.model.get_expert_by_id(target_expert_id)
            if target_expert is not None:
                try:
                    final_hidden = getattr(
                        self.model, '_last_layer_outputs', {}
                    ).get('hidden_states')
                    if final_hidden is not None and final_hidden.requires_grad:
                        proxy_grad = torch.autograd.grad(
                            loss, final_hidden,
                            retain_graph=False,
                            allow_unused=True,
                        )[0]
                        if proxy_grad is not None:
                            d_model = proxy_grad.size(-1)
                            proxy_grad_2d = proxy_grad.reshape(-1, d_model)
                            scaled_grad = proxy_grad_2d.mean(0, keepdim=True)
                            dummy_hidden = torch.ones(
                                1, target_expert.w_down.in_features,
                                device=scaled_grad.device,
                            )
                            delta_w_down = scaled_grad.t() @ dummy_hidden
                            update_scale = (
                                target_expert.effective_lr
                                * base_gamma * focal_boost * 0.01
                            )
                            if torch.isfinite(delta_w_down).all():
                                target_expert.w_down.weight.data -= (
                                    update_scale * delta_w_down
                                )
                except RuntimeError:
                    # Graph already freed - skip proxy gradient.
                    # The feedback for this inactive expert is dropped
                    # this step; it will be retried on the next
                    # interaction where the expert is activated.
                    pass

        # Push focal boost to the internal loop so the dialectic signal
        # persists beyond this external forward pass.  The internal loop
        # (Stage B) will consume and clear it on the next contemplation step,
        # strengthening the targeted expert's Hebbian update.
        if focal_boost is not None and target_expert_id is not None:
            self.model.push_focal_boost(target_expert_id, focal_boost)
