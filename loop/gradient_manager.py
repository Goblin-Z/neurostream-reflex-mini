import torch


class GradientManager:
    """
    Per-layer gradient management.

    Instead of retaining the full computation graph across ALL experts
    in ALL layers (SEQ: 32 experts * retain_graph=True), retains the
    graph across ONE layer's experts at a time.

    Usage:
        gm = GradientManager()
        for layer_idx, (loss, layer) in gm.iterate_layers(loss, layers):
            for expert, grad_y in gm.layer_gradients(loss, layer, layer_idx):
                expert.update_local(grad_y, ...)
    """

    def __init__(self, num_layers: int):
        self.num_layers = num_layers
        self._grads_cache = {}

    def iterate_layers(self, loss, layers):
        """
        Generator that yields (layer_idx, layers[layer_idx]) from last to first,
        managing graph retention so each layer's graph is released immediately
        after its experts have been updated.

        The loss must be computed from a forward pass through ALL layers.
        """
        for layer_idx in range(len(layers) - 1, -1, -1):
            retain = layer_idx > 0
            yield layer_idx, layers[layer_idx], retain

    def compute_expert_gradients(self, loss, layer, retain):
        """
        For a given layer, compute gradients for all active experts.
        If retain=True, keeps the graph alive for earlier layers.
        Captures expert outputs at call time to avoid race conditions
        with concurrent forward passes overwriting _output.
        """
        captured = [(e, e._output) for e in layer.all_experts
                    if e._output is not None and e._output.requires_grad]
        if not captured:
            return

        outputs = [out for _, out in captured]
        grads = torch.autograd.grad(
            loss, outputs,
            retain_graph=retain,
            allow_unused=True,
        )
        for (expert, _), grad_y in zip(captured, grads):
            if grad_y is not None:
                yield expert, grad_y

    def compute_focal_gradient(self, loss, expert, retain):
        """Single-expert gradient with graph retention flag."""
        if expert._output is None or not expert._output.requires_grad:
            return None
        (grad_y,) = torch.autograd.grad(
            loss, expert._output,
            retain_graph=retain,
            allow_unused=True,
        )
        return grad_y
