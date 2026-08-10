"""
Interaction Manager — sigma-driven emergent Q/A protocol.

Key philosophical departure from the previous version:
    NO hardcoded cooldown. The "don't ask again immediately" behavior
    EMERGES from the learning dynamics:

        User answers question → model learns → sigma drops →
        sigma < threshold → can_ask returns False → no new question

    The model's own uncertainty IS the cooldown mechanism.
    When sigma genuinely rises again (new confusion), the model
    naturally becomes ready to ask.

    This is closer to how human curiosity works: you don't ask about
    the same thing twice in a row because your first question (and the
    answer) resolved the confusion. You ask about something NEW only
    when you encounter NEW confusion.
"""

import threading


class InteractionManager:
    """
    Emergent Q/A protocol with sigma-driven gating.

    States:
      IDLE             — no active question
      QUESTION_PENDING — question submitted, not yet displayed
      AWAITING_ANSWER  — displayed, waiting for user
      PROCESSING_ANSWER — processing feedback
    """

    IDLE = 'idle'
    QUESTION_PENDING = 'question_pending'
    AWAITING_ANSWER = 'awaiting_answer'
    PROCESSING_ANSWER = 'processing_answer'

    def __init__(self, config=None):
        self._state = self.IDLE
        self._lock = threading.RLock()
        self._active_question = None

        # Session limits (soft caps, not hard cooldowns)
        self._questions_this_session = 0
        self._max_questions_per_session = 5

        # Sigma tracking for emergent cooldown
        self._current_model_sigma = 0.0
        self._sigma_threshold = getattr(config, 'verify_threshold', 0.5) if config else 0.5
        self._post_feedback_sigma = 0.0  # sigma right after feedback processed
        # 追问冷却（修复）：问完冷却 _ask_cooldown 步（防边缘抖动），
        # 冷却结束后完全按 sigma 判断——疑问未解决（sigma 仍高）可追问 ✓
        self._ask_cooldown = 0
        self._ask_cooldown_max = 50

        # Stats
        self._total_questions_asked = 0
        self._total_feedback_received = 0
        self._last_feedback_reward = 0.0

    # ── Properties ──

    @property
    def state(self):
        with self._lock:
            return self._state

    @property
    def active_question(self):
        with self._lock:
            return self._active_question

    @property
    def is_expecting_answer(self):
        with self._lock:
            return self._state == self.AWAITING_ANSWER

    @property
    def can_ask(self):
        """
        Emergent gating: can ask only when:
          1. No active question (state check)
          2. Session limit not exceeded
          3. Model's current sigma exceeds threshold (genuine confusion)

        Condition 3 is the emergent cooldown:
        after asking + getting answer → model learns → sigma drops →
        sigma < threshold → can_ask = False → no new question.
        When sigma rises again (new confusion) → can_ask = True.

        追问冷却（修复）: 问完冷却 _ask_cooldown_max 步（防抖动），
        冷却结束后完全按 sigma 判断——疑问未解决（sigma 仍高）可追问。
        """
        with self._lock:
            if self._ask_cooldown > 0:
                return False
            return (self._state == self.IDLE
                    and self._questions_this_session < self._max_questions_per_session
                    and self._current_model_sigma > self._sigma_threshold)

    def update_sigma(self, sigma):
        """
        Called by the internal loop after each forward pass.
        Updates the model's current uncertainty level.
        """
        with self._lock:
            self._current_model_sigma = sigma
            if self._ask_cooldown > 0:
                self._ask_cooldown -= 1  # 冷却按内循环步数递减

    # ── Question lifecycle ──

    def submit_question(self, question_data, current_step=None):
        """Called by internal loop when sigma is high enough."""
        with self._lock:
            if self._state != self.IDLE:
                return False
            if self._questions_this_session >= self._max_questions_per_session:
                return False
            if self._current_model_sigma <= self._sigma_threshold:
                return False

            self._active_question = question_data
            self._state = self.QUESTION_PENDING
            self._total_questions_asked += 1
            self._questions_this_session += 1
            self._ask_cooldown = self._ask_cooldown_max  # 防抖动冷却
            return True

    def notify_question_displayed(self):
        """Called by pipeline after showing question to user."""
        with self._lock:
            if self._state == self.QUESTION_PENDING:
                self._state = self.AWAITING_ANSWER

    def retrieve_answer(self):
        """Called by pipeline when user input arrives in AWAITING_ANSWER state."""
        with self._lock:
            if self._state != self.AWAITING_ANSWER:
                return None
            q = self._active_question
            self._state = self.PROCESSING_ANSWER
            return q

    def finalize_feedback(self, feedback_result, current_step=None):
        """
        Called after feedback processing. Does NOT set a cooldown timer.
        The emergent cooldown comes from sigma dropping after learning.
        """
        with self._lock:
            self._active_question = None
            self._state = self.IDLE
            self._last_feedback_reward = feedback_result.get('reward', 0.0)
            if feedback_result.get('reward') is not None:
                self._total_feedback_received += 1
            # Record post-feedback sigma — will be compared against future sigma
            self._post_feedback_sigma = self._current_model_sigma

    def cancel_question(self):
        """Cancel active question without feedback."""
        with self._lock:
            self._active_question = None
            self._state = self.IDLE
            self._questions_this_session = max(0, self._questions_this_session - 1)

    def reset_session(self, current_step=None):
        """Reset per-session counters."""
        with self._lock:
            self._questions_this_session = 0
            if self._state in (self.QUESTION_PENDING, self.AWAITING_ANSWER):
                self._active_question = None
                self._state = self.IDLE

    def get_stats(self):
        with self._lock:
            return {
                'state': self._state,
                'sigma': self._current_model_sigma,
                'sigma_threshold': self._sigma_threshold,
                'can_ask': self.can_ask,
                'questions_this_session': self._questions_this_session,
                'total_asked': self._total_questions_asked,
                'total_feedback': self._total_feedback_received,
                'last_reward': self._last_feedback_reward,
            }
