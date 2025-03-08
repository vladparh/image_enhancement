import math

import torch


class Enhancer:
    def __init__(self, model, crop_size=256, max_minibatch=None):
        self.model = model
        self.crop_size = crop_size
        self.max_minibatch = max_minibatch

    def feed_data(self, data):
        self.lq = data
        self.device = data.device

    def predict_with_crop(self, lq):
        b, c, h, w = lq.size()
        device = lq.device
        preds = torch.zeros((b, c, h, w))
        count_mt = torch.zeros((b, 1, h, w))

        # adaptive step_i, step_j
        num_row = (h - 1) // self.crop_size + 1
        num_col = (w - 1) // self.crop_size + 1

        step_j = (
            self.crop_size
            if num_col == 1
            else math.ceil((w - self.crop_size) / (num_col - 1) - 1e-8)
        )
        step_i = (
            self.crop_size
            if num_row == 1
            else math.ceil((h - self.crop_size) / (num_row - 1) - 1e-8)
        )

        i = 0  # 0~h-1
        last_i = False
        while i < h and not last_i:
            j = 0
            if i + self.crop_size >= h:
                i = h - self.crop_size
                last_i = True

            last_j = False
            while j < w and not last_j:
                if j + self.crop_size >= w:
                    j = w - self.crop_size
                    last_j = True
                preds[:, :, i : i + self.crop_size, j : j + self.crop_size] += (
                    self.model(
                        lq[:, :, i : (i + self.crop_size), j : (j + self.crop_size)]
                    )
                    .detach()
                    .cpu()
                )
                count_mt[:, 0, i : i + self.crop_size, j : j + self.crop_size] += 1.0
                j = j + step_j
            i = i + step_i

        output = (preds / count_mt).to(device)
        return output

    def predict(self, lq, use_split=False):
        self.model.eval()
        with torch.no_grad():
            n = len(lq)
            outs = []
            m = n if self.max_minibatch is None else self.max_minibatch
            i = 0
            while i < n:
                j = i + m
                if j >= n:
                    j = n
                if use_split:
                    pred = self.predict_with_crop(lq[i:j])
                else:
                    pred = self.model(lq[i:j])
                if isinstance(pred, list):
                    pred = pred[-1]
                outs.append(pred.detach())
                i = j

            output = torch.cat(outs, dim=0)
        return output
