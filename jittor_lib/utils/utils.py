import jittor
import numpy as np
import time


def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            grad = param.opt_grad(optimizer)
            if grad is not None:
                grad.assign(grad.clamp(-grad_clip, grad_clip))


def adjust_lr(optimizer, init_lr, epoch, decay_rate=0.1, decay_epoch=30):
    decay = decay_rate ** (epoch // decay_epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = decay * init_lr
    return param_group['lr']



class AvgMeter(object):
    def __init__(self, num=40):
        self.num = num
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.losses = []

    def update(self, val, n=1):
        if isinstance(val, jt.Var):
            val = val.item()

        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.losses.append(val)

    def show(self):
        # Use numpy for slicing and mean calculation
        recent_losses = self.losses[-self.num:] if len(self.losses) > self.num else self.losses
        return np.mean(recent_losses)


def CalParams(model, input_tensor):
    """
    Calculate FLOPs and parameters for Jittor models
    """
    # Calculate parameters
    total_params = sum([np.prod(p.shape) for p in model.parameters()])

    # Reset and calculate FLOPs
    jt.flags.reset_flags()  # Reset all counters
    jt.sync(True)  # Ensure all operations are completed
    jt.flags.auto_convert_64_to_32 = 1  # Reduce precision for FLOPs calculation

    # Warm-up run
    model.eval()
    with jt.no_grad():
        model(input_tensor)
        jt.sync(True)

    # Reset FLOPs counter
    jt.flags.reset_flags()

    # Actual FLOPs measurement
    with jt.no_grad():
        model(input_tensor)
        jt.sync(True)
        flops = jt.flops_counter

    # Format results
    def format_size(size):
        if size < 1e3:
            return f"{size:.2f}"
        elif size < 1e6:
            return f"{size / 1e3:.2f}K"
        elif size < 1e9:
            return f"{size / 1e6:.2f}M"
        else:
            return f"{size / 1e9:.2f}G"

    flops_str = format_size(flops)
    params_str = format_size(total_params)

    print('[Jittor Statistics Information]\nFLOPs: {}\nParams: {}'.format(flops_str, params_str))