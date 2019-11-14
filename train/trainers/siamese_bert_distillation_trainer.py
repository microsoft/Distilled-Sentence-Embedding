import utils.module as module_utils
from evaluation.evaluators.evaluator import VoidEvaluator
from train.trainer import Trainer


class SiameseBertDistillationTrainer(Trainer):
    """
    Trainer for bert distillation training.
    """

    def __init__(self, model, optimizer, loss_fn, distillation_coeff=0, gradient_accumulation_steps=1,
                 train_evaluator=VoidEvaluator(), val_evaluator=VoidEvaluator(), callback=None,
                 device=module_utils.get_device()):
        super().__init__(model, optimizer, train_evaluator, val_evaluator, callback, device)
        self.loss_fn = loss_fn
        self.distillation_coeff = distillation_coeff

        if gradient_accumulation_steps < 1:
            raise ValueError("Gradient accumulation steps must be >= 1")

        self.gradient_accumulation_steps = gradient_accumulation_steps

    def batch_update(self, batch_num, batch):
        train_first_input_ids, train_first_input_mask, train_second_input_ids, train_second_input_mask, y, distillation_logits = batch
        train_first_input_ids = train_first_input_ids.to(self.device)
        train_first_input_mask = train_first_input_mask.to(self.device)
        train_second_input_ids = train_second_input_ids.to(self.device)
        train_second_input_mask = train_second_input_mask.to(self.device)
        y = y.to(self.device)
        distillation_logits = distillation_logits.to(self.device)

        y_pred = self.model(train_first_input_ids, train_first_input_mask, train_second_input_ids, train_second_input_mask)

        loss = self.loss_fn(y_pred, y)
        if self.distillation_coeff != 0:
            distillation_loss = self.__calc_distillation_loss(y_pred, distillation_logits)
            loss = (1 - self.distillation_coeff) * loss + self.distillation_coeff * distillation_loss

        loss /= self.gradient_accumulation_steps
        loss.backward()

        if (batch_num + 1) % self.gradient_accumulation_steps == 0:
            self.optimizer.step()
            self.optimizer.zero_grad()

        return {
            "loss": loss.item(),
            "y_pred": y_pred.detach(),
            "y": y,
        }

    def __calc_distillation_loss(self, y_pred, distillation_logits):
        return ((y_pred - distillation_logits) ** 2).sum(dim=1).mean()
