import timm
from typing import Optional
from models.base_model import BaseModel
import torch
from models import utils
from art.attacks.evasion import (
    BoundaryAttack,
    HopSkipJump,
    ProjectedGradientDescentPyTorch,
)
from art.estimators.classification import PyTorchClassifier
from neptune.types import File
from aggregation import frame_aggregation
from lightning.pytorch.utilities.rank_zero import rank_zero_only


class BaseFrameModel(BaseModel):
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)

    def training_step(self, batch, batch_idx):
        if isinstance(batch, dict):
            x = torch.cat([v[0] for k, v in batch.items() if v is not None])
            y = torch.cat([v[1] for k, v in batch.items() if v is not None])
        else:
            x, y = batch[0]

        # y = y.view(-1, 1)
        y = y.long()
        noise = None
        if self.random_smoothing:
            noise = (
                torch.randn_like(x, device=self.device) * self.random_smoothing_noise_sd
            )
        if isinstance(self.loss, utils.TradesLoss):
            loss_value, logits = self.loss(x, y, self.optimizers(), noise=noise)
        else:
            if noise is not None:
                x = x + noise
            logits = self.model(x)
            loss_value = self.loss(logits, y)  # type: ignore
        self.log(
            "train/loss",
            loss_value,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
            batch_size=self.trainer.datamodule.batch_size,  # type: ignore
        )

        # log images
        if batch_idx == 0 and self.logger is not None:
            sample = x[0].cpu().permute(1, 2, 0)  # (C, H, W) -> (H, W, C)
            sample = self._normalize_tensor(sample)
            self.logger.experiment["train/samples"].append(  # type: ignore
                File.as_image(sample), step=self.global_step, name=y[0].cpu().item()
            )

        return {"loss": loss_value, "logits": logits, "y": y}

    def on_train_batch_end(self, output: dict, batch, batch_idx, dataloader_idx=None):
        preds = self.act(output["logits"])
        self.train_metrics.update(preds, output["y"])
        self.log_dict(
            self.train_metrics,  # type: ignore
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            sync_dist=True,
            batch_size=self.trainer.datamodule.batch_size,  # type: ignore
        )

        # self.cm_train.update(preds, output["y"])

    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        x, y = batch
        # y = y.view(-1, 1)
        y = y.long()
        if self.random_smoothing:
            noise = (
                torch.randn_like(x, device=self.device) * self.random_smoothing_noise_sd
            )
            x = x + noise

        logits = self.model(x)
        if isinstance(self.loss, utils.TradesLoss):
            loss_value = self.loss.natural_loss(logits, y)
        else:
            loss_value = self.loss(logits, y)  # type: ignore

        self.log(
            "val/loss",
            loss_value,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
            batch_size=self.trainer.datamodule.batch_size,  # type: ignore
        )

        # log images
        if batch_idx == 0 and self.logger is not None:
            sample = x[0].cpu().permute(1, 2, 0)  # (C, H, W) -> (H, W, C)
            sample = self._normalize_tensor(sample)
            self.logger.experiment["val/samples"].append(  # type: ignore
                File.as_image(sample), step=self.global_step, name=y[0].cpu().item()
            )

        return {"loss": loss_value, "logits": logits, "y": y}

    def on_validation_batch_end(
        self, output: dict, batch, batch_idx, dataloader_idx=None
    ):
        preds = self.act(output["logits"])
        self.val_metrics.update(preds, output["y"])
        self.log_dict(
            self.val_metrics,  # type: ignore
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            sync_dist=True,
            batch_size=self.trainer.datamodule.batch_size,  # type: ignore
        )
        # self.cm_val.update(preds, output["y"])

    def on_test_start(self):
        self.test_step_outputs = {
            i: [] for i in range(len(self.trainer.datamodule.test_data))  # type: ignore
        }

        if len(self.adv_attacks) > 0:
            input_shape = (
                3,
                self.trainer.datamodule.rs_size,  # type: ignore
                self.trainer.datamodule.rs_size,  # type: ignore
            )
            self.art_classifier = PyTorchClassifier(
                model=self,
                loss=torch.nn.BCEWithLogitsLoss(),
                input_shape=input_shape,
                nb_classes=2,
                clip_values=(0.0, 1.0),
                device_type="gpu",
            )
            self.adv_metrics = {
                name: self.metrics.clone(prefix=f"adv_{name}/")
                for name in self.adv_attacks
            }

        self.attacks = {}
        for name in self.adv_attacks:
            if name == "hsj":
                self.attacks[name] = HopSkipJump(
                    classifier=self.art_classifier,
                    targeted=False,
                )
            elif name == "ba":
                self.attacks[name] = BoundaryAttack(
                    estimator=self.art_classifier,
                    targeted=False,
                )
            elif name == "pgd":
                self.attacks[name] = ProjectedGradientDescentPyTorch(
                    estimator=self.art_classifier,
                    targeted=False,
                )
            else:
                raise ValueError(f"Unknown attack: {name}")

    def test_step(self, batch, batch_idx, dataloader_idx=None):
        x, y = batch
        # y = y.view(-1, 1)
        y = y.long()
        if self.random_smoothing:
            noise = (
                torch.randn_like(x, device=self.device) * self.random_smoothing_noise_sd
            )
            x = x + noise

        logits = self.model(x)
        if isinstance(self.loss, utils.TradesLoss):
            loss_value = self.loss.natural_loss(logits, y)
        else:
            loss_value = self.loss(logits, y)  # type: ignore

        preds = self.act(logits)

        if dataloader_idx is None:
            dataloader_idx = 0

        if len(self.adv_attacks) > 0:
            x = x.detach().cpu().numpy()
            adv_preds = {}
            for name, attack in self.attacks.items():
                with torch.inference_mode(False):
                    x_adv = attack.generate(x)
                adv_pred = self(torch.from_numpy(x_adv).to(self.device))
                adv_preds[name] = adv_pred

            self.test_step_outputs[dataloader_idx].append(  # type: ignore
                {
                    "preds": preds,
                    "adv_preds": adv_preds,
                    "loss": loss_value,
                    "logits": logits,
                    "y": y,
                }
            )
            return {
                "preds": preds,
                "adv_preds": adv_preds,
                "loss": loss_value,
                "logits": logits,
                "y": y,
            }

        self.test_step_outputs[dataloader_idx].append(  # type: ignore
            {"preds": preds, "loss": loss_value, "logits": logits, "y": y}
        )
        return {"preds": preds, "loss": loss_value, "logits": logits, "y": y}

    def on_test_batch_end(self, output: dict, batch, batch_idx, dataloader_idx=None):
        # loss = self.loss(output['logits'], output['y'])
        self.log(
            "test/loss",
            output["loss"],
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            sync_dist=True,
            batch_size=self.trainer.datamodule.batch_size,  # type: ignore
        )
        self.cm_test.update(output["preds"], output["y"])

    def on_test_epoch_end(self):
        if self.aggregate_predict_results:
            self.test_step_outputs = {
                k: frame_aggregation(v, name=self.trainer.datamodule.test_data[k])
                for k, v in self.test_step_outputs.items()
            }

        for i, loader_output in self.test_step_outputs.items():  # type: ignore
            test_metrics = self.test_metrics.clone(
                prefix=f"test_{self.trainer.datamodule.test_data[i]}/"  # type: ignore
            )
            preds = torch.vstack([step["preds"] for step in loader_output])
            y = torch.cat([step["y"] for step in loader_output], dim=0)
            self.log_dict(
                test_metrics(preds, y),
                logger=True,
                sync_dist=True,
                batch_size=self.trainer.datamodule.batch_size,  # type: ignore
            )

            # log confusion matrix
            test_cm = self.cm_test.clone()
            cm = test_cm(preds, y)
            fig = utils.plot_confusion_matrix(cm, self.trainer.datamodule.class_names)
            tensor_image = utils.convert_figure_to_tensor(fig)
            self.logger.experiment[
                f"test_{self.trainer.datamodule.test_data[i]}/cm"
            ].append(File.as_image(tensor_image), step=self.global_step)

            for name in self.adv_attacks:
                adv_attack_metrics = self.test_metrics.clone(
                    prefix=f"test_{self.trainer.datamodule.test_data[i]}/adv_{name}/"  # type: ignore
                )
                adv_preds = torch.vstack(
                    [step["adv_preds"][name] for step in loader_output]
                )
                self.log_dict(
                    adv_attack_metrics(adv_preds, y),
                    logger=True,
                    sync_dist=True,
                    batch_size=self.trainer.datamodule.batch_size,  # type: ignore
                )

        self.test_step_outputs.clear()

        # @rank_zero_only
        # def log_confusion_matrix(self, cm, name):
        #     cm = cm.compute()
        #     fig = utils.plot_confusion_matrix(cm, self.trainer.datamodule.class_names)
        #     tensor_image = utils.convert_figure_to_tensor(fig)
        #     self.logger.experiment[f"{name}/cm"].append(
        #         File.as_image(tensor_image), step=self.global_step
        #     )
        #     print(f"{name} cm: {cm}")


class FrameModel(BaseFrameModel):
    def __init__(
        self,
        model_name: str,
        pretrained: bool = True,
        drop_path_rate: Optional[float] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.save_hyperparameters()

        self.model: torch.nn.Module = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=self.num_classes,
            drop_path_rate=drop_path_rate,
        )
