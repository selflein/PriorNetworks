from typing import Dict, Any
import sys
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_

from prior_networks.training import Trainer, calc_accuracy_torch
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal


class TrainerWithOOD(Trainer):
    def __init__(self, model, criterion,
                 train_dataset, ood_dataset, test_dataset,
                 optimizer,
                 scheduler=None,
                 optimizer_params: Dict[str, Any] = None,
                 scheduler_params: Dict[str, Any] = None,
                 batch_size=50,
                 device=None,
                 log_interval: int = 100,
                 test_criterion=None,
                 num_workers=4,
                 clip_norm: float = 10.0,
                 pin_memory=False,
                 checkpoint_path='./',
                 checkpoint_steps=0):
        super().__init__(model=model, criterion=criterion, train_dataset=train_dataset,
                         test_dataset=test_dataset, optimizer=optimizer, scheduler=scheduler,
                         optimizer_params=optimizer_params, scheduler_params=scheduler_params,
                         batch_size=batch_size, device=device, log_interval=log_interval,
                         test_criterion=test_criterion, num_workers=num_workers,
                         pin_memory=pin_memory, clip_norm=clip_norm,
                         checkpoint_path=checkpoint_path, checkpoint_steps=checkpoint_steps)

        assert len(train_dataset) == len(ood_dataset)
        self.oodloader = DataLoader(ood_dataset, batch_size=batch_size,
                                    shuffle=True, num_workers=1, pin_memory=self.pin_memory)

    def _train_single_epoch(self):
        # Set model in train mode
        self.model.train()

        for i, (data, ood_data) in enumerate(
                zip(self.trainloader, self.oodloader), 0):
            # Get inputs
            inputs, labels = data
            ood_inputs, _ = ood_data
            if self.device is not None:
                # Move data to adequate device
                inputs, labels, ood_inputs = map(lambda x: x.to(self.device,
                                                                non_blocking=self.pin_memory),
                                                 (inputs, labels, ood_inputs))

            # zero the parameter gradients
            self.optimizer.zero_grad()

            inputs = torch.cat((inputs, ood_inputs), dim=0)
            outputs = self.model(inputs)
            id_outputs, ood_outputs = torch.chunk(outputs, 2, dim=0)
            loss = self.criterion((id_outputs, ood_outputs), (labels, None))
            assert torch.isnan(loss) == torch.tensor([0], dtype=torch.uint8).to(self.device)
            loss.backward()
            clip_grad_norm_(self.model.parameters(), self.clip_norm)
            self.optimizer.step()

            # Update the number of steps
            self.steps += 1

            # log statistics
            if self.steps % self.log_interval == 0:
                probs = F.softmax(id_outputs, dim=1)
                self.train_accuracy.append(
                    calc_accuracy_torch(probs, labels, self.device).item())
                self.train_loss.append(loss.item())
                self.train_eval_steps.append(self.steps)

            if self.checkpoint_steps > 0:
                if self.steps % self.checkpoint_steps == 0:
                    self._save_checkpoint(save_at_steps=True)
        return


class TrainerWithOODJoint(Trainer):
    def __init__(self, model, criterion,
                 train_dataset, test_dataset,
                 optimizer,
                 scheduler=None,
                 optimizer_params: Dict[str, Any] = None,
                 scheduler_params: Dict[str, Any] = None,
                 batch_size: int = 50,
                 device=None,
                 log_interval: int = 100,
                 test_criterion=None,
                 pin_memory: bool = False,
                 clip_norm: float = 10.0,
                 num_workers: int = 4,
                 checkpoint_path='./',
                 checkpoint_steps: int = 0):
        super().__init__(model=model, criterion=criterion, train_dataset=train_dataset,
                         test_dataset=test_dataset, optimizer=optimizer, scheduler=scheduler,
                         optimizer_params=optimizer_params, scheduler_params=scheduler_params,
                         batch_size=batch_size, device=device, log_interval=log_interval,
                         test_criterion=test_criterion, num_workers=num_workers,
                         pin_memory=pin_memory, clip_norm=clip_norm,
                         checkpoint_path=checkpoint_path, checkpoint_steps=checkpoint_steps)

    def _train_single_epoch(self):
        # Set model in train mode
        self.model.train()

        id_alpha_0, ood_alpha_0 = 0.0, 0.0
        train_loss, accuracy = 0.0, 0.0
        init_steps = self.steps
        for i, data in enumerate(self.trainloader, 0):
            # Get inputs
            inputs, labels = data
            if self.device is not None:
                # Move data to adequate device
                inputs, *labels = map(lambda x: x.to(self.device, non_blocking=self.pin_memory),
                                      (inputs, *labels))

            # zero the parameter gradients
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, *labels)
            assert torch.isnan(loss) == torch.tensor([0], dtype=torch.uint8).to(self.device)
            loss.backward()
            clip_grad_norm_(self.model.parameters(), self.clip_norm)
            self.optimizer.step()
            train_loss += loss.item()
            weights = labels[1] / torch.max(labels[1])
            weights = weights.to(dtype=torch.float32)
            ood_weights = 1.0 - weights
            alpha_0 = torch.sum(torch.exp(outputs), dim=1)
            id_alpha_0 += (torch.sum(alpha_0 * weights) / torch.sum(weights)).item()
            ood_alpha_0 += (torch.sum(alpha_0 * ood_weights) / torch.sum(ood_weights)).item()

            # Update the number of steps
            self.steps += 1

            # log statistics
            if self.steps % self.log_interval == 0:
                probs = F.softmax(outputs, dim=1)
                weights = labels[1] / torch.max(labels[1])
                accuracy = calc_accuracy_torch(probs, labels[0], self.device, weights=weights).item()
                self.train_accuracy.append(accuracy)
                self.train_loss.append(loss.item())
                self.train_eval_steps.append(self.steps)

            if self.checkpoint_steps > 0:
                if self.steps % self.checkpoint_steps == 0:
                    self._save_checkpoint(save_at_steps=True)

        train_loss = train_loss / (self.steps - init_steps)
        id_alpha_0 = id_alpha_0 / (self.steps - init_steps)
        ood_alpha_0 = ood_alpha_0 / (self.steps - init_steps)
        print(f"Train Loss: {np.round(train_loss, 3)}; "
              f"Train Error: {np.round(100.0 * (1.0 - accuracy), 1)}; "
              f"Train ID precision: {np.round(id_alpha_0, 1)}; "
              f"Train OOD precision: {np.round(ood_alpha_0, 1)}")

        with open('./LOG.txt', 'a') as f:
            f.write(f"Train Loss: {np.round(loss.item(), 3)}; "
                    f"Train Error: {np.round(100.0 * (1.0 - accuracy), 1)}; "
                    f"Train ID precision: {np.round(id_alpha_0, 1)}; "
                    f"Train OOD precision: {np.round(ood_alpha_0, 1)}; ")

        return

    def test(self, time):
        """
        Single evaluation on the entire provided test dataset.
        Return accuracy, mean test loss, and an array of predicted probabilities
        """
        test_loss, accuracy = 0.0, 0.0
        id_alpha_0, ood_alpha_0 = 0.0, 0.0

        domain_labels = []
        logits = []
        # Set model in eval mode
        self.model.eval()
        with torch.no_grad():
            for i, data in enumerate(self.testloader, 0):
                # Get inputs
                inputs, labels = data
                if self.device is not None:
                    inputs, *labels = map(lambda x: x.to(self.device, non_blocking=self.pin_memory),
                                          (inputs, *labels))
                outputs = self.model(inputs)
                weights = labels[1] / torch.max(labels[1])
                weights = weights.to(dtype=torch.float32)
                ood_weights = 1.0 - weights
                alpha_0 = torch.sum(torch.exp(outputs), dim=1)
                id_alpha_0 += (torch.sum(alpha_0 * weights) / torch.sum(weights)).item()
                ood_alpha_0 += (torch.sum(alpha_0 * ood_weights) / torch.sum(ood_weights)).item()
                test_loss += self.test_criterion(outputs, *labels).item()
                probs = F.softmax(outputs, dim=1)
                accuracy += calc_accuracy_torch(probs, labels[0], self.device, weights=weights).item()

                logits.append(outputs.cpu().numpy())
                domain_labels.append(ood_weights.cpu().numpy())

        logits = np.stack(logits, axis=0)
        domain_labels = np.stack(domain_labels, axis=0)
        print(logits.shape, domain_labels.shape)

        id_alpha_0 = id_alpha_0 / len(self.testloader)
        ood_alpha_0 = ood_alpha_0 / len(self.testloader)
        test_loss = test_loss / len(self.testloader)
        accuracy = accuracy / len(self.testloader)

        print(f"Test Loss: {np.round(test_loss, 3)}; "
              f"Test Error: {np.round(100.0 * (1.0 - accuracy), 1)}%; "
              f"Test ID precision: {np.round(id_alpha_0, 1)}; "
              f"Test OOD precision: {np.round(ood_alpha_0, 1)}; "
              f"Time Per Epoch: {np.round(time / 60.0, 1)} min")

        with open('./LOG.txt', 'a') as f:
            f.write(f"Test Loss: {np.round(test_loss, 3)}; "
                    f"Test Error: {np.round(100.0 * (1.0 - accuracy), 1)}; "
                    f"Test ID precision: {np.round(id_alpha_0, 1)}; "
                    f"Test OOD precision: {np.round(ood_alpha_0, 1)}; "
                    f"Time Per Epoch: {np.round(time / 60.0, 1)} min.\n")
        # Log statistics
        self.test_loss.append(test_loss)
        self.test_accuracy.append(accuracy)
        self.test_eval_steps.append(self.steps)
        return


class TrainerWithAdv(Trainer):
    def __init__(self,
                 model,
                 criterion,
                 adv_criterion,
                 train_dataset,
                 test_dataset,
                 optimizer,
                 adv_noise,
                 scheduler=None,
                 optimizer_params: Dict[str, Any] = None,
                 scheduler_params: Dict[str, Any] = None,
                 batch_size=50,
                 device=None,
                 log_interval: int = 100,
                 test_criterion=None,
                 clip_norm=10.0,
                 pin_memory=False,
                 num_workers=4,
                 checkpoint_path='./',
                 checkpoint_steps=0):
        super().__init__(model=model,
                         criterion=criterion,
                         train_dataset=train_dataset,
                         test_dataset=test_dataset,
                         optimizer=optimizer,
                         scheduler=scheduler,
                         optimizer_params=optimizer_params,
                         scheduler_params=scheduler_params,
                         batch_size=batch_size,
                         device=device,
                         log_interval=log_interval,
                         clip_norm=clip_norm,
                         test_criterion=test_criterion,
                         num_workers=num_workers,
                         pin_memory=pin_memory,
                         checkpoint_path=checkpoint_path,
                         checkpoint_steps=checkpoint_steps)

        self.adv_criterion = adv_criterion
        self.adv_noise = adv_noise

    def _construct_FGSM_attack(self, inputs, labels):

        adv_inputs = inputs.clone()
        adv_inputs.requires_grad = True
        self.model.eval()

        with torch.enable_grad():
            outputs = self.model(adv_inputs)

            probs = torch.ones(size=[outputs.size()[1]]) / outputs.size()[1]
            target_sampler = Categorical(probs=probs)
            targets = target_sampler.sample(torch.Size([outputs.size()[0]]))

            epsilon_sampler = Normal(0, self.adv_noise)
            epsilon = epsilon_sampler.sample(torch.Size([outputs.size()[0]]))
            epsilon = torch.abs(epsilon) + 0.004
            epsilon = epsilon.view([outputs.size()[0], 1, 1, 1])

            if self.device is not None:
                targets = targets.to(self.device, non_blocking=self.pin_memory)
                epsilon = epsilon.to(self.device, non_blocking=self.pin_memory)

            loss = self.adv_criterion(outputs, targets, mean=False)
            assert torch.equal(torch.isnan(loss),
                               torch.zeros_like(loss, dtype=torch.uint8))

            loss = torch.where(torch.eq(targets, labels), -loss, loss)

            grad_outputs = torch.ones(loss.shape)
            if self.device is not None:
                grad_outputs = grad_outputs.to(self.device, non_blocking=self.pin_memory)

            grads = torch.autograd.grad(loss,
                                        adv_inputs,
                                        grad_outputs=grad_outputs,
                                        only_inputs=True)[0]

            update = epsilon * grads.sign()

            perturbed_image = torch.clamp(adv_inputs - update, 0, 1)
            adv_inputs.data = perturbed_image

        # Return the perturbed image
        self.model.train()
        return adv_inputs

    def _train_single_epoch(self):
        # Set model in train mode
        self.model.train()

        for i, data in enumerate(self.trainloader, 0):
            # Get inputs
            inputs, labels = data
            if self.device is not None:
                # Move data to adequate device
                inputs, labels = map(lambda x: x.to(self.device,
                                                    non_blocking=self.pin_memory),
                                     (inputs, labels))

            adv_inputs = self._construct_FGSM_attack(labels=labels,
                                                     inputs=inputs)
            self.model.zero_grad()
            cat_inputs = torch.cat([inputs, adv_inputs], dim=1).view(
                torch.Size([2 * inputs.size()[0]]) + inputs.size()[1:])
            logits = self.model(cat_inputs).view([inputs.size()[0], -1])
            logits, adv_logits = torch.chunk(logits, 2, dim=1)
            loss = self.criterion([logits, adv_logits], [labels, labels])
            assert torch.isnan(loss) == torch.tensor([0], dtype=torch.uint8).to(self.device)
            loss.backward()
            clip_grad_norm_(self.model.parameters(), self.clip_norm)
            # zero the parameter gradients
            self.optimizer.step()

            # Update the number of steps
            self.steps += 1

            # log statistics
            if self.steps % self.log_interval == 0:
                labels = torch.cat([labels, labels], dim=0)
                probs = F.softmax(torch.cat([logits, adv_logits], dim=0), dim=1)

                self.train_accuracy.append(calc_accuracy_torch(probs,
                                                               labels,
                                                               self.device).item())
                self.train_loss.append(loss.item())
                self.train_eval_steps.append(self.steps)

            if self.checkpoint_steps > 0:
                if self.steps % self.checkpoint_steps == 0:
                    self._save_checkpoint(save_at_steps=True)
        return
