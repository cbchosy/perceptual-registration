from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import torch
from torch import optim
import torch.nn.functional as F
import lpips
from kornia import geometry as KG
from kornia.geometry.transform.image_registrator import build_pyramid
from kornia.core import Tensor


if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


class ClassicalRegistrator(KG.ImageRegistrator):
    def __init__(
    self,
    model_type='similarity',
    optimizer=optim.Adam,
    loss_fn = F.l1_loss,
    pyramid_levels=3,
    lr=1e-3,
    num_iterations=100,
    tolerance=1e-4,
    warper=None,
    ) -> None:
        super().__init__(model_type, optimizer, loss_fn, pyramid_levels, lr, num_iterations, tolerance, warper)
        self.model.to(device)

    def register(
        self, src_img: Tensor, dst_img: Tensor, verbose: bool = False, output_intermediate_models: bool = False, output_intermediate_loss: bool = False, gamma=0.95):
        r"""Estimate the tranformation' which warps src_img into dst_img by gradient descent. The shape of the
        tensors is not checked, because it may depend on the model, e.g. volume registration.

        Args:
            src_img: Input image tensor.
            dst_img: Input image tensor.
            verbose: if True, outputs loss every 10 iterations.
            output_intermediate_models: if True with intermediate models
            output_intermediate_loss: if True, outputs total vector of loss

        Returns:
            the transformation between two images, shape depends on the model,
            typically [1x3x3] tensor for string model_types.
        """
        self.reset_model()
        _opt_args: Dict[str, Any] = {}
        _opt_args["lr"] = self.lr
        # compute the gaussian pyramids
        # [::-1] because we have to register from coarse to fine
        img_src_pyr = build_pyramid(src_img, self.pyramid_levels)[::-1]
        img_dst_pyr = build_pyramid(dst_img, self.pyramid_levels)[::-1]
        prev_loss = 1e10
        aux_models = []
        intermediate_loss = torch.zeros((self.pyramid_levels, self.num_iterations))
        if len(img_dst_pyr) != len(img_src_pyr):
            raise ValueError("Cannot register images of different sizes")
        for j, pyr in enumerate(zip(img_src_pyr, img_dst_pyr)):
            img_src_level, img_dst_level = pyr
            opt = self.optimizer(self.model.parameters(), **_opt_args)
            scheduler = optim.lr_scheduler.ExponentialLR(opt, gamma=gamma)
            for i in range(self.num_iterations):
                # compute gradient and update optimizer parameters
                opt.zero_grad()
                loss = self.get_single_level_loss(img_src_level, img_dst_level, self.model()) + self.get_single_level_loss(img_dst_level, img_src_level, self.model.forward_inverse())
                current_loss = loss.item()
                if output_intermediate_loss:
                    intermediate_loss[j, i] = current_loss
                if abs(current_loss - prev_loss) < self.tolerance:
                    break
                prev_loss = current_loss
                loss.backward()
                if verbose and (i % 10 == 0):
                    print(f"Loss = {current_loss:.4f}, iter={i}, lr={scheduler.get_lr()[0]:.4f}")
                opt.step()
                if i % 10 == 0:
                    scheduler.step()
            if output_intermediate_models:
                aux_models.append(self.model().clone().detach())
        if output_intermediate_models:
            return self.model(), aux_models
        if output_intermediate_loss:
            return self.model(), intermediate_loss
        return self.model()

    def get_single_level_loss(self, img_src: Tensor, img_dst: Tensor, transform_model: Tensor) -> Tensor:
        """Warp img_src into img_dst with transform_model and returns loss."""
        if img_src.shape != img_dst.shape:
            raise ValueError(
                f"Cannot register images of different shapes\
                              {img_src.shape} {img_dst.shape:} "
            )
        _height, _width = img_dst.shape[-2:]
        warper = self.warper(_height, _width)
        img_src_to_dst = warper(img_src, transform_model).requires_grad_()
        # compute and mask loss
        loss = self.loss_fn(img_src_to_dst, img_dst, reduction='none')  # 1xCxHxW
        ones = warper(torch.ones_like(img_src), transform_model)
        loss = loss.masked_select(ones > 0.9).mean()
        return loss

class LPIPSRegistrator(KG.ImageRegistrator):
    def __init__(
    self,
    model_type='similarity',
    optimizer=optim.Adam,
    pyramid_levels=3,
    lr=1e-3,
    num_iterations=100,
    tolerance=1e-4,
    warper=None,
    ) -> None:
        loss_fn = lpips.LPIPS('alex').to(device).forward
        super().__init__(model_type, optimizer, loss_fn, pyramid_levels, lr, num_iterations, tolerance, warper)
        self.model.to(device)

    def register(
        self, src_img: Tensor, dst_img: Tensor, verbose: bool = False, output_intermediate_models: bool = False, output_intermediate_loss: bool = False, gamma = 0.95):
        r"""Estimate the tranformation' which warps src_img into dst_img by gradient descent. The shape of the
        tensors is not checked, because it may depend on the model, e.g. volume registration.

        Args:
            src_img: Input image tensor.
            dst_img: Input image tensor.
            verbose: if True, outputs loss every 10 iterations.
            output_intermediate_models: if True with intermediate models
            output_intermediate_loss: if True, outputs total vector of loss

        Returns:
            the transformation between two images, shape depends on the model,
            typically [1x3x3] tensor for string model_types.
        """
        self.reset_model()
        _opt_args: Dict[str, Any] = {}
        _opt_args["lr"] = self.lr
        # compute the gaussian pyramids
        # [::-1] because we have to register from coarse to fine
        img_src_pyr = build_pyramid(src_img, self.pyramid_levels)[::-1]
        img_dst_pyr = build_pyramid(dst_img, self.pyramid_levels)[::-1]
        prev_loss = 1e10
        aux_models = []
        intermediate_loss = torch.zeros((self.pyramid_levels, self.num_iterations))
        if len(img_dst_pyr) != len(img_src_pyr):
            raise ValueError("Cannot register images of different sizes")
        for j, pyr in enumerate(zip(img_src_pyr, img_dst_pyr)):
            img_src_level, img_dst_level = pyr
            opt = self.optimizer(self.model.parameters(), **_opt_args)
            scheduler = optim.lr_scheduler.ExponentialLR(opt, gamma=gamma)
            for i in range(self.num_iterations):
                # compute gradient and update optimizer parameters
                opt.zero_grad()
                loss = self.get_single_level_loss(img_src_level, img_dst_level, self.model()) + self.get_single_level_loss(img_dst_level, img_src_level, self.model.forward_inverse())
                current_loss = loss.item()
                if output_intermediate_loss:
                    intermediate_loss[j, i] = current_loss
                if abs(current_loss - prev_loss) < self.tolerance:
                    break
                prev_loss = current_loss
                loss.backward()
                if verbose and (i % 10 == 0):
                    print(f"Loss = {current_loss:.4f}, iter={i}, lr={scheduler.get_lr()[0]:.4f}")
                opt.step()
                if i % 10 == 0:
                    scheduler.step()
                if output_intermediate_models:
                    aux_models.append(self.model().clone().detach())
        outs = []
        if output_intermediate_models:
            outs.append(aux_models)
        if output_intermediate_loss:
            outs.append(intermediate_loss)
        return self.model(), *outs

    def get_single_level_loss(self, img_src: Tensor, img_dst: Tensor, transform_model: Tensor) -> Tensor:
        """Warp img_src into img_dst with transform_model and returns loss."""
        if img_src.shape != img_dst.shape:
            raise ValueError(
                f"Cannot register images of different shapes\
                              {img_src.shape} {img_dst.shape:} "
            )
        _height, _width = img_dst.shape[-2:]
        warper = self.warper(_height, _width)
        img_src_to_dst = warper(img_src, transform_model).requires_grad_()
        # compute and mask loss
        loss = self.loss_fn(img_src_to_dst, img_dst)  # 1xCxHxW
        ones = warper(torch.ones_like(img_src), transform_model)
        loss = loss.masked_select(ones > 0.9).mean()
        return loss