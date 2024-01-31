import io
import itertools
from lightning.pytorch.callbacks import Callback
import subprocess
from matplotlib import pyplot as plt
from torchmetrics import Metric, Recall
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import wandb
from PIL import Image


class WandbArtifactCacheClean(Callback):
    def clean_cache(self):
        subprocess.call("wandb artifact cache cleanup 10GB", shell=True)

    def on_fit_start(self, trainer, pl_module):
        self.clean_cache()

    def on_fit_end(self, trainer, pl_module):
        self.clean_cache()

    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch % 20 == 0:
            self.clean_cache()


class FalsePositiveRate(Metric):
    def __init__(self, threshold=0.5, eps=1e-8, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.threshold = threshold
        self.eps = eps

        self.add_state("false_positives", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("negatives", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        preds_binary = (preds >= self.threshold).int()
        self.false_positives += torch.sum((preds_binary == 1) & (target == 0))
        self.negatives += torch.sum(target == 0)

    def compute(self):
        return self.false_positives.float() / (self.negatives.float() + self.eps)


class FalseNegativeRate(Metric):
    def __init__(self, threshold=0.5, eps=1e-8, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.threshold = threshold
        self.eps = eps

        self.add_state("false_negatives", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("positives", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        preds_binary = (preds >= self.threshold).int()
        self.false_negatives += torch.sum((preds_binary == 0) & (target == 1))
        self.positives += torch.sum(target == 1)

    def compute(self):
        return self.false_negatives.float() / (self.positives.float() + self.eps)


class BalancedAccuracy(Metric):
    def __init__(
        self,
        threshold: float = 0.5,
        num_classes: int = 2,
        average: str = "macro",
        dist_sync_on_step: bool = False,
    ):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.threshold = threshold
        self.recall = Recall(
            num_classes=num_classes, average=average, task="multiclass"
        )

        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        # Convert probabilities to binary predictions
        preds_binary = (preds >= self.threshold).int()
        self.recall.update(preds_binary, target)

    def compute(self):
        # Compute the final balanced accuracy (recall) score
        return self.recall.compute()

    def reset(self):
        # Reset the recall metric
        self.recall.reset()


class TradesLoss(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        natural_loss: nn.Module,
        criterion_kl=nn.KLDivLoss(reduction="su,"),
        distance="l_inf",
        perturb_steps=10,
        step_size=0.003,
        epsilon=0.031,
        beta=6.0,
        random_smoothing=True,
    ):
        super(TradesLoss, self).__init__()

        self.model = model
        self.natural_loss = natural_loss
        self.criterion_kl = criterion_kl
        self.distance = distance
        self.perturb_steps = perturb_steps
        self.step_size = step_size
        self.epsilon = epsilon
        self.beta = beta
        self.random_smoothing = random_smoothing

    def forward(self, input, target, optimizer, mask=None, noise=None):
        self.model.eval()
        batch_size = len(input)
        if mask is None:
            mask = self.mask_maker(input)

        # generate adversarial example
        input = input.mul(mask)
        input_adv = input.detach() + 0.001 * torch.randn(input.shape).cuda().detach()
        input_adv = input_adv.mul(mask)

        if self.random_smoothing:
            assert noise is not None, "noise needs to be included for random smoothing"
            noise = noise.mul(mask).detach()

        if self.distance == "l_inf":
            for _ in range(self.perturb_steps):
                input_adv.requires_grad_()
                with torch.enable_grad():
                    loss_kl = self.calc_divergence(input, input_adv, noise)
                grad = torch.autograd.grad(loss_kl, [input_adv])[0]
                input_adv = input_adv.detach() + self.step_size * torch.sign(
                    grad.detach()
                )
                input_adv = torch.min(
                    torch.max(input_adv, input - self.epsilon), input + self.epsilon
                )
                input_adv = torch.clamp(input_adv, 0.0, 1.0)
                input_adv = input_adv.mul(mask)
        elif self.distance == "l_2":
            delta = 0.001 * torch.randn(input.shape).cuda().mul(mask).detach()
            delta = Variable(delta.data, requires_grad=True)

            # Setup optimizers
            optimizer_delta = optim.SGD(
                [delta], lr=self.epsilon / self.perturb_steps * 2
            )

            for _ in range(self.perturb_steps):
                input_adv = input + delta

                # optimize
                optimizer_delta.zero_grad()
                with torch.enable_grad():
                    loss = (-1) * self.calc_divergence(input, input_adv, noise)
                loss.backward()
                # renorming gradient
                grad_norms = delta.grad.view(batch_size, -1).norm(p=2, dim=1)  # type: ignore
                delta.grad.div_(grad_norms.view(-1, 1, 1, 1))  # type: ignore
                # avoid nan or inf if gradient is 0
                if (grad_norms == 0).any():
                    delta.grad[grad_norms == 0] = torch.randn_like(  # type: ignore
                        delta.grad[grad_norms == 0]  # type: ignore
                    )
                optimizer_delta.step()

                # projection
                delta.data.add_(input)
                delta.data.clamp_(0, 1).sub_(input)
                delta.data.mul(mask)
                delta.data.renorm_(p=2, dim=0, maxnorm=self.epsilon)
            input_adv = Variable(input + delta, requires_grad=False)
        else:
            input_adv = torch.clamp(input_adv, 0.0, 1.0)

        self.model.train()
        input_adv = Variable(torch.clamp(input_adv, 0.0, 1.0), requires_grad=False)
        # zero gradient
        optimizer.zero_grad()

        # natural loss
        if self.random_smoothing:
            logits = self.model(input + noise)
        else:
            logits = self.model(input)
        loss_natural = self.natural_loss(logits, target)

        # calculate robust loss
        loss_robust = (1.0 / batch_size) * self.calc_divergence(input, input_adv, noise)
        loss = loss_natural + self.beta * loss_robust

        return loss, logits

    @staticmethod
    def mask_maker(x):
        """
        The function outputs an all-1 mask. One can adapt the function to creating other masks.
        """
        mask = torch.ones_like(x)
        return mask

    def calc_divergence(self, input, adv, noise):
        if self.random_smoothing:
            adv_logits = self.model(adv + noise)
            logits = self.model(input + noise)
        else:
            adv_logits = self.model(adv)
            logits = self.model(input)

        return self.criterion_kl(
            F.log_softmax(adv_logits, dim=1),
            F.softmax(logits, dim=1),
        )

    @staticmethod
    def squared_l2_norm(x):
        flattened = x.view(x.unsqueeze(0).shape[0], -1)
        return (flattened**2).sum(1)

    def l2_norm(self, x):
        return self.squared_l2_norm(x).sqrt()


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25):
        """
        :param gamma: focal loss power parameter, a float scalar.
        :param alpha: weight of the class indicated by 1, a float scalar.
        """
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, inputs, targets):
        """
        :param bce_loss: Binary Cross Entropy loss, a torch tensor.
            Input and target are of shape (batch_size, num_classes)
        :param targets: a torch tensor containing the ground truth, 0s and 1s.
        """
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets)
        p_t = torch.exp(-bce_loss)
        alpha_tensor = (1 - self.alpha) + targets * (
            2 * self.alpha - 1
        )  # alpha if target = 1 and 1 - alpha if target = 0
        f_loss = alpha_tensor * (1 - p_t) ** self.gamma * bce_loss
        # return sigmoid_focal_loss(inputs, targets, self.gamma, self.alpha, reduction="mean")
        return f_loss.mean()


def wandb_define_metrics():
    wandb.define_metric("train/loss_epoch", summary="min")
    wandb.define_metric("train/BinaryAccuracy", summary="max")
    wandb.define_metric("train/MulticlassRecall", summary="max")
    wandb.define_metric("train/BinaryAUROC", summary="max")
    wandb.define_metric("train/BinaryF1Score", summary="max")
    wandb.define_metric("train/BinaryPrecision", summary="max")
    wandb.define_metric("train/BinaryRecall", summary="max")
    wandb.define_metric("train/BinarySpecificity", summary="max")
    wandb.define_metric("train/FalsePositiveRate", summary="min")
    wandb.define_metric("train/FalseNegativeRate", summary="min")

    wandb.define_metric("val/loss_epoch", summary="min")
    wandb.define_metric("val/BinaryAccuracy", summary="max")
    wandb.define_metric("val/MulticlassRecall", summary="max")
    wandb.define_metric("val/BinaryAUROC", summary="max")
    wandb.define_metric("val/BinaryF1Score", summary="max")
    wandb.define_metric("val/BinaryPrecision", summary="max")
    wandb.define_metric("val/BinaryRecall", summary="max")
    wandb.define_metric("val/BinarySpecificity", summary="max")
    wandb.define_metric("val/FalsePositiveRate", summary="min")
    wandb.define_metric("val/FalseNegativeRate", summary="min")


def plot_confusion_matrix(cm, class_names):
    """
    Returns a matplotlib figure containing the plotted confusion matrix with actual numbers.

    Args:
      cm (Tensor): a confusion matrix of integer classes
      class_names (array, shape = [n]): String names of the integer classes
    """
    cm = cm.cpu().numpy()  # Convert to numpy array
    figure = plt.figure(figsize=(16, 16))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Normalize the confusion matrix.
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(
            j,
            i,
            f"{cm_normalized[i, j]:.2f} ({cm[i, j]})",
            horizontalalignment="center",
            color=color,
        )

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    return figure


def convert_figure_to_tensor(figure):
    """
    Converts a matplotlib figure to a 3D tensor normalized to [0, 1].
    """
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close(figure)
    buf.seek(0)
    image = Image.open(buf)
    image_tensor = torch.tensor(np.array(image)).float() / 255.0  # Normalize to [0, 1]
    return image_tensor
