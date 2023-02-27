import jax
import jax.numpy as jnp

import flax
import flax.linen as nn
from flax.training import train_state, checkpoints
import optax

from torch.utils import tensorboard
import os
import typing as tp
import numpy as np

from absl import flags
from tqdm import tqdm
import collections

FLAGS = flags.FLAGS


class TrainState(train_state.TrainState):
    batch_stats: tp.Any

class TrainerModule:


    def __init__(self, model_name: str, model_class: nn.Module, 
                 model_hparams:  dict, optimizer_name: str, 
                 optimizer_hparams: dict, exmp_imgs: tp.Any, seed: int = 42, chekpoint_path: str = "./assets") -> None:
        
        super().__init__()
        self.model_name = model_name
        self.model_class = model_class
        self.model_hparams = model_hparams
        self.optimizer_name = optimizer_name
        self.optimizer_hparams = optimizer_hparams
        self.seed = seed
        self.model = self.model_class(**self.model_hparams)

        self.checkpoint_path = chekpoint_path
        self.log_dir = os.path.join(chekpoint_path, self.model_name)
        self.logger = tensorboard.SummaryWriter(log_dir=self.log_dir)

        self.create_functions()
        self.init_model(exmp_imgs)

    def create_functions(self):
        def calc_loss(params, batch_stats, batch, train):
            imgs, labels = batch
            outs = self.model.apply(
                {'params': params, 'batch_stats': batch_stats}, imgs, mutable=['batch_stats'] if train else False
            )
            logits, new_model_state = outs if train else (outs, None)
            loss = optax.softmax_cross_entropy(logits, labels).mean()
            acc = (logits.argmax(axis=-1) == labels).mean()
            return loss, (acc, new_model_state)
        
        def train_step(state, batch):
            loss_fn = lambda params: calc_loss(params, state.batch_stats, batch, train=True)
            ret, grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
            loss, acc, new_model_state = ret[0], *ret[1]
            state = state.apply_gradients(grads=grads, batch_stats=new_model_state['batch_stats'])

            return state, loss, acc

        def eval_step(state, batch):
            _, (acc, _) = calc_loss(state.params, state.batch_stats, batch, train=False)
            return acc

        self.train_step = jax.jit()
        self.eval_step = jax.jit()

    def init_model(self, exmp_imgs):
        init_rng = jax.random.PRNGKey(FLAGS.seed)
        variables = self.model.init(init_rng, exmp_imgs, train=True)
        self.init_params, self.init_batch_stats = variables['params'], variables['batch_stats']
        self.state = None

    def init_optimizer(self, num_epochs, num_steps_per_epoch):
        opt_class = optax.adamw
        lr_schedule = optax.piecewise_constant_schedule(
            init_value=self.optimizer_hparams.pop('lr'),
            boundaries_and_scales=
                {int(num_steps_per_epoch*num_epochs*0.6): 0.1,
                 int(num_steps_per_epoch*num_epochs*0.85): 0.1}
        )
        optimizer = optax.chain(optax.clip(1.0), opt_class(lr_schedule, **self.optimizer_hparams))
        self.state = TrainState.create(apply_fn=self.model.apply,
                                       params=self.init_params if self.state is None else self.state.params,
                                       batch_stats=self.init_batch_stats if self.state is None else self.state.batch_stats,
                                       tx=optimizer)
        

    def train_model(self, train_loader, val_loader, num_epochs=200):
        self.init_optimizer(num_epochs, len(train_loader))
        best_eval = 0.0
        for epoch_idx in tqdm(range(1, num_epochs+1)):
            self.train_epoch(train_loader, epoch=epoch_idx)
            if epoch_idx % 2 == 0:
                eval_acc = self.eval_model(val_loader)
                self.logger.add_scalar('val/acc', eval_acc, global_step=epoch_idx)
                if eval_acc >= best_eval:
                    best_eval = eval_acc
                    self.save_model(step=epoch_idx)
                self.logger.flush()


    def train_epoch(self, train_loader, epoch):
        metrics = collections.defaultdict(list)
        for batch in tqdm(train_loader, desc='Training', leave=False):
            self.state, loss, acc = self.train_step(self.state, batch)
            metrics['loss'].append(loss)
            metrics['acc'].append(acc)

        for key in metrics:
            avg_val = np.stack(jax.device_get(metrics[key])).mean()
            self.logger.add_scalar('train/'+key, avg_val, global_step=epoch)
    
    def eval_model(self, data_loader):
        correct_class, count = 0, 0
        for batch in data_loader:
            acc = self.eval_step(self.state, batch)
            correct_class += acc * batch[0].shape[0]
            count += batch[0].shape[0]
        eval_acc = (correct_class / count).item()
        return eval_acc
    
    def save_model(self, step=0):
        checkpoints.save_checkpoint(ckpt_dir=self.log_dir,
                                    target={'params': self.state.params,
                                            'batch_stats': self.state.batch_stats},
                                    step=step,
                                   overwrite=True)
        
    def load_model(self, pretrained=False):
        if not pretrained:
            state_dict = checkpoints.restore_checkpoint(ckpt_dir=self.log_dir, target=None)
        else:
            state_dict = checkpoints.restore_checkpoint(ckpt_dir=os.path.join(self.checkpoint_path, f'{self.model_name}.ckpt'), target=None)
        self.state = TrainState.create(apply_fn=self.model.apply,
                                       params=state_dict['params'],
                                       batch_stats=state_dict['batch_stats'],
                                       tx=self.state.tx if self.state else optax.sgd(0.1)   # Default optimizer
                                      )
    def checkpoint_exists(self):
        return os.path.isfile(os.path.join(self.checkpoint_path, f'{self.model_name}.ckpt'))