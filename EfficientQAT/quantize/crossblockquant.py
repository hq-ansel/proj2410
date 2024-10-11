import shutil
import time
import os
import copy
import pdb
import gc
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

import utils
import quantize.int_linear_fake as int_linear_fake
import quantize.int_linear_real as int_linear_real
from quantize.utils import (
    quant_parameters,weight_parameters,trainable_parameters,
    set_quant_state,quant_inplace,set_quant_parameters,
    set_weight_parameters,trainable_parameters_num,get_named_linears,set_op_by_name)
from datautils_block import BlockTrainDataset,OptimBlockTrainDataset


def update_dataset(layers, dataset, dev, attention_mask, position_ids):
    with torch.no_grad():
        with torch.cuda.amp.autocast():
            for index, inps in enumerate(dataset):
                inps = inps.to(dev)
                if len(inps.shape)==2:
                    inps = inps.unsqueeze(0)
                if isinstance(layers, nn.ModuleList):
                    for layer in layers:
                        inps = layer(inps, attention_mask=attention_mask,position_ids=position_ids)[0]
                else:
                    inps = layers(inps, attention_mask=attention_mask,position_ids=position_ids)[0]
                new_data = inps.to('cpu')
                dataset.update_data(index,new_data)

                    
def cross_block_quantization(
    model,
    args,
    trainloader,
    valloader,
    logger=None,
):
    logger.info("Starting ...")
    if args.off_load_to_disk:
        logger.info("offload the training dataset to disk, saving CPU memory, but may slowdown the training due to additional I/O...")
    
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_cache = model.config.use_cache
    model.config.use_cache = False
    
    # step 1: move embedding layer and first layer to target device, only suppress llama models now
    layers = model.model.layers
    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.norm = model.model.norm.to(dev)
    if hasattr(model.model, 'rotary_emb'):
        # for llama-3.1
        model.model.rotary_emb = model.model.rotary_emb.to(dev)
    layers[0] = layers[0].to(dev)
    dtype = torch.float16

    # step 2: init dataset
    flag = time.time()
    if args.off_load_to_disk:
        fp_train_cache_path = f'{args.cache_dir}/{flag}/block_training_fp_train'
        fp_val_cache_path = f'{args.cache_dir}/{flag}/block_training_fp_val'
        quant_train_cache_path = f'{args.cache_dir}/{flag}/block_training_quant_train'
        quant_val_cache_path = f'{args.cache_dir}/{flag}/block_training_quant_val'
        for path in [fp_train_cache_path,fp_val_cache_path,quant_train_cache_path,quant_val_cache_path]:
            if os.path.exists(path):
                shutil.rmtree(path)
    else:
        fp_train_cache_path = None
        fp_val_cache_path = None
        quant_train_cache_path = None
        quant_val_cache_path = None
    # fp_train_inps = BlockTrainDataset(args.train_size, args.training_seqlen, 
    #                             model.config.hidden_size, args.batch_size, dtype, cache_path=fp_train_cache_path,off_load_to_disk=args.off_load_to_disk)
    # fp_val_inps = BlockTrainDataset(args.val_size, args.training_seqlen, 
    #                             model.config.hidden_size, args.batch_size, dtype, cache_path=fp_val_cache_path,off_load_to_disk=args.off_load_to_disk)
    fp_train_inps = OptimBlockTrainDataset(args.train_size, args.training_seqlen, 
                                    model.config.hidden_size, args.batch_size, dtype, cache_path=fp_train_cache_path,off_load_to_disk=args.off_load_to_disk)
    fp_val_inps = OptimBlockTrainDataset(args.val_size, args.training_seqlen, 
                                    model.config.hidden_size, args.batch_size, dtype, cache_path=fp_val_cache_path,off_load_to_disk=args.off_load_to_disk)

    # step 3: catch the input of thefirst layer 
    class Catcher(nn.Module):
        def __init__(self, module, dataset):
            super().__init__()
            self.module = module
            self.dataset = dataset
            self.index = 0
            self.attention_mask = None
            self.position_ids = None

        def forward(self, inp, **kwargs):
            self.dataset.update_data(self.index, inp.squeeze(0).to('cpu'))
            self.index += 1
            if self.attention_mask is None:
                self.attention_mask = kwargs["attention_mask"]
            if self.position_ids is None:
                self.position_ids = kwargs["position_ids"]
            raise ValueError
    
    # step 3.1: catch the input of training set
    layers[0] = Catcher(layers[0],fp_train_inps)
    iters = len(trainloader)//args.batch_size
    with torch.no_grad():
        for i in range(iters):
            data = torch.cat([trainloader[j][0] for j in range(i*args.batch_size,(i+1)*args.batch_size)],dim=0)
            try:
                model(data.to(dev))
            except ValueError:
                pass
    layers[0] = layers[0].module

    # step 3.2: catch the input of validation set
    layers[0] = Catcher(layers[0],fp_val_inps)
    iters = len(valloader)//args.batch_size
    with torch.no_grad():
        for i in range(iters):
            data = torch.cat([valloader[j][0] for j in range(i*args.batch_size,(i+1)*args.batch_size)],dim=0)
            try:
                model(data.to(dev))
            except ValueError:
                pass
    attention_mask = layers[0].attention_mask
    position_ids = layers[0].position_ids
    layers[0] = layers[0].module
    if attention_mask is not None:
        attention_mask_batch = attention_mask.repeat(args.batch_size,1,1,1).float()
    else:
        logger.info(
            "No attention mask caught from the first layer."
            " Seems that model's attention works without a mask."
        )
        attention_mask_batch = None
    
    # step 4: move embedding layer and first layer to cpu
    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()
    if hasattr(model.model, 'rotary_emb'):
        # for llama-3.1
        model.model.rotary_emb = model.model.rotary_emb.cpu()
    torch.cuda.empty_cache()

    # step 5: copy fp input as the quant input, they are same at the first layer
    if args.off_load_to_disk:
        # copy quant input from fp input, they are same in first layer
        shutil.copytree(fp_train_cache_path, quant_train_cache_path)
        shutil.copytree(fp_val_cache_path, quant_val_cache_path)
        quant_train_inps = BlockTrainDataset(args.train_size, args.training_seqlen, 
                                    model.config.hidden_size, args.batch_size, dtype, cache_path=quant_train_cache_path,off_load_to_disk=args.off_load_to_disk)
        quant_val_inps = BlockTrainDataset(args.val_size, args.training_seqlen, 
                                    model.config.hidden_size, args.batch_size, dtype, cache_path=quant_val_cache_path,off_load_to_disk=args.off_load_to_disk)
        # quant_train_inps = OptimBlockTrainDataset(args.train_size, args.training_seqlen, 
        #                             model.config.hidden_size, args.batch_size, dtype, cache_path=quant_train_cache_path,off_load_to_disk=args.off_load_to_disk)
        # quant_val_inps = OptimBlockTrainDataset(args.val_size, args.training_seqlen, 
                                    # model.config.hidden_size, args.batch_size, dtype, cache_path=quant_val_cache_path,off_load_to_disk=args.off_load_to_disk)
    else:
        quant_train_inps = BlockTrainDataset(args.train_size, args.training_seqlen, 
                                    model.config.hidden_size, args.batch_size, dtype, cache_path=quant_train_cache_path,off_load_to_disk=args.off_load_to_disk)
        quant_val_inps = BlockTrainDataset(args.val_size, args.training_seqlen, 
                                    model.config.hidden_size, args.batch_size, dtype, cache_path=quant_val_cache_path,off_load_to_disk=args.off_load_to_disk)
        # quant_train_inps = OptimBlockTrainDataset(args.train_size, args.training_seqlen, 
        #                             model.config.hidden_size, args.batch_size, dtype, cache_path=quant_train_cache_path,off_load_to_disk=args.off_load_to_disk)
        # quant_val_inps = OptimBlockTrainDataset(args.val_size, args.training_seqlen, 
        #                             model.config.hidden_size, args.batch_size, dtype, cache_path=quant_val_cache_path,off_load_to_disk=args.off_load_to_disk)
        for index,data in enumerate(fp_train_inps):
            quant_train_inps.update_data(index, data)
        for index,data in enumerate(fp_val_inps):
            quant_val_inps.update_data(index, data)
    # stat memory of datasets 
    quant_train_inps_memory_porfile = utils.profile_memory(quant_train_inps)
    quant_val_inps_memory_porfile = utils.profile_memory(quant_val_inps)
    fp_train_inps_memory_porfile = utils.profile_memory(fp_train_inps)
    fp_val_inps_memory_porfile = utils.profile_memory(fp_val_inps)
    logger.info(f"Memory profile of quant_train_inps: {quant_train_inps_memory_porfile}")
    logger.info(f"Memory profile of quant_val_inps: {quant_val_inps_memory_porfile}")
    logger.info(f"Memory profile of fp_train_inps: {fp_train_inps_memory_porfile}")
    logger.info(f"Memory profile of fp_val_inps: {fp_val_inps_memory_porfile}")


    # step 6: start training    
    loss_func = torch.nn.MSELoss()

    # step 6.1.1: setup loss recorder
    loss_dir="/home/ubuntu/data/exp/proj2410/logs"
    loss_recorder = utils.BlockLossRecorder(file_path=os.path.join(loss_dir,f"Llama2-7b-crossblock-loss-b2.csv"),)
    
    num_layers = len(layers)
    slide_step = 1
    window_size = 1
    qlayers = torch.nn.ModuleList()
    is_quant_layer = [False]*num_layers

    for start_idx in range(0, num_layers, slide_step):
        end_idx = min(start_idx + window_size, num_layers)  # 窗口范围
        step = 1
        logger.info(f"=== Start quantize blocks {start_idx} to {end_idx - 1} ===")
        
        # if args.epochs > 0:
        #     fp_train_inps_before = copy.deepcopy(fp_train_inps)
        #     fp_val_inps_before = copy.deepcopy(fp_val_inps)


        # step 6.1.2: replace torch.nn.Linear with QuantLinear for QAT
        for block_index in range(start_idx, end_idx):
            if not is_quant_layer[block_index]:
                layer = layers[block_index].to(dev)
                qlayer = copy.deepcopy(layer)
                for name, module in qlayer.named_modules():
                    if isinstance(module,torch.nn.Linear):
                        quantlinear = int_linear_fake.QuantLinear(module, args.wbits, args.group_size)
                        set_op_by_name(qlayer, name, quantlinear)  
                        del module  
                qlayer.to(dev)
                qlayers.append(qlayer)
                is_quant_layer[block_index] = True
            else:
                qlayer = qlayers[block_index]
        # step 6.2: obtain output of full-precision model for MSE
        set_quant_state(qlayers[start_idx:end_idx],weight_quant=False) # deactivate quantization for obtaining ground truth
        if args.epochs > 0:
            update_dataset(qlayers[start_idx:end_idx],fp_train_inps,dev,attention_mask,position_ids)
            update_dataset(qlayers[start_idx:end_idx],fp_val_inps,dev,attention_mask,position_ids)
        set_quant_state(qlayers[start_idx:end_idx],weight_quant=True)  # activate quantization
        
        
        if args.epochs > 0:
            with torch.no_grad():
                for block_index in range(start_idx, end_idx):
                    qlayers[block_index].float()      # fp32 is required for AMP training
            # step 6.3: create optimizer and learning rate schedule
            param = []
            assert args.quant_lr > 0 or args.weight_lr > 0
            param_group_index = 0
            total_training_iteration = args.epochs * args.train_size / args.batch_size 

            # 配置训练所需要的参数
            for block_index in range(start_idx, end_idx):
                qlayer = qlayers[block_index]
                if args.quant_lr > 0:
                    set_quant_parameters(qlayer,True)
                    param.append({"params":quant_parameters(qlayer),"lr":args.quant_lr})
                    empty_optimizer_1 = torch.optim.AdamW([torch.tensor(0)], lr=args.quant_lr)
                    quant_scheduler = CosineAnnealingLR(empty_optimizer_1, T_max=total_training_iteration, eta_min=args.quant_lr/args.min_lr_factor)
                    quant_index = param_group_index
                    param_group_index += 1
                else:
                    set_quant_parameters(qlayer,False)
                    
                if args.weight_lr > 0:
                    set_weight_parameters(qlayer,True)
                    param.append({"params":weight_parameters(qlayer),"lr":args.weight_lr})
                    empty_optimizer_2 = torch.optim.AdamW([torch.tensor(0)], lr=args.weight_lr)
                    weight_scheduler = CosineAnnealingLR(empty_optimizer_2, T_max=total_training_iteration, eta_min=args.weight_lr/args.min_lr_factor)
                    weight_index = param_group_index
                    param_group_index += 1
                else:
                    set_weight_parameters(qlayer,False)
            optimizer = torch.optim.AdamW(param, weight_decay=args.wd,foreach=True)
            loss_scaler = utils.NativeScalerWithGradNormCount()
            trainable_number = trainable_parameters_num(qlayers[start_idx:end_idx])
            print(f"trainable parameter number: {trainable_number/1e6}M")

            # 检查模型，优化器各自占用多少内存
            logger.info(f"Memory profile of qlayers: {utils.profile_memory(qlayers[start_idx:end_idx])}")
            logger.info(f"Memory profile of optimizer: {utils.profile_memory(optimizer)}")

            best_val_loss = 1e6
            early_stop_flag = 0
            for epoch in range(args.epochs):
                # step: 6.4 training
                loss_list = []
                norm_list = []
                start_time = time.time()
                # used for debug
                torch.autograd.set_detect_anomaly(True)
                for index, (quant_inps, fp_inps) in enumerate(zip(quant_train_inps, fp_train_inps)):    
                    # obtain output of quantization model
                    with torch.cuda.amp.autocast():
                        hidden_states = quant_inps.to(dev)
                        label = fp_inps.to(dev)
                        for block_index in range(start_idx, end_idx):
                            if not math.isfinite(hidden_states.sum().item()):
                                logger.info("hidden_states is NAN, stopping training")
                                pdb.set_trace()
                            hidden_states = qlayers[block_index](hidden_states, attention_mask=attention_mask_batch,position_ids=position_ids)[0]
                        if not math.isfinite(hidden_states.sum().item()):
                            logger.info("hidden_states is NAN, stopping training")
                            pdb.set_trace()
                        quant_out = hidden_states
                        reconstruction_loss = loss_func(label, quant_out)
                        loss =  reconstruction_loss

                    if not math.isfinite(loss.item()):
                        logger.info("Loss is NAN, stopping training")
                        pdb.set_trace()
                    loss_recorder.record(f"crossblock_loss",step,reconstruction_loss.detach().cpu().item())
                    loss_list.append(reconstruction_loss.detach().cpu())
                    optimizer.zero_grad()
                    # # debug
                    # print(f"Loss dtype: {loss.dtype}")
                    # for param in qlayers[start_idx:end_idx].parameters():
                    #     print(f"Param dtype: {param.dtype}")
                    #     if param.grad is not None:
                    #         print(f"Param grad dtype: {param.grad.dtype}")

                    norm = loss_scaler(loss, optimizer,clip_grad=1.0,parameters=trainable_parameters(qlayers[start_idx:end_idx])).cpu()
                    norm_list.append(norm.data)

                    # adjust lr
                    if args.quant_lr > 0:
                        quant_scheduler.step()
                        optimizer.param_groups[quant_index]['lr'] = quant_scheduler.get_lr()[0]
                    if args.weight_lr >0 :
                        weight_scheduler.step()
                        optimizer.param_groups[weight_index]['lr'] = weight_scheduler.get_lr()[0]
                    step += 1

                # step 6.5: calculate validation loss
                val_loss_list = []
                for index, (quant_inps,fp_inps) in enumerate(zip(quant_val_inps, fp_val_inps)):  
                    # obtain output of quantization model
                    with torch.no_grad():
                        with torch.cuda.amp.autocast():
                            hidden_states = quant_inps.to(dev)
                            label = fp_inps.to(dev)
                            for block_index in range(start_idx, end_idx):
                                hidden_states = qlayers[block_index](hidden_states, attention_mask=attention_mask_batch,position_ids=position_ids)[0]
                            quant_out = hidden_states
                            reconstruction_loss = loss_func(label, quant_out)
                    val_loss_list.append(reconstruction_loss.cpu())
                 
                train_mean_num = min(len(loss_list),64) # calculate the average training loss of last train_mean_num samples
                loss_mean = torch.stack(loss_list)[-(train_mean_num-1):].mean()
                val_loss_mean = torch.stack(val_loss_list).mean()
                norm_mean = torch.stack(norm_list).mean()
                logger.info(f"blocks {block_index} epoch {epoch} recon_loss:{loss_mean} val_loss:{val_loss_mean} quant_lr:{quant_scheduler.get_lr()[0]} norm:{norm_mean:.8f} max memory_allocated {torch.cuda.max_memory_allocated(dev) / 1024**2} time {time.time()-start_time} ")
                if val_loss_mean < best_val_loss:
                    best_val_loss = val_loss_mean
                else:
                    early_stop_flag += 1
                    if args.early_stop > 0 and early_stop_flag >=args.early_stop:
                        break
            optimizer.zero_grad()
            del optimizer

        # step 6.6: directly replace the weight with fake quantization
        qlayers[start_idx:end_idx].half()
        quant_inplace(qlayers[start_idx:end_idx])
        set_quant_state(qlayers[start_idx:end_idx],weight_quant=False)  # weight has been quantized inplace

        # step 6.7: update inputs of quantization model
        if args.epochs>0:
            # fp_train_inps=fp_train_inps_before
            # fp_val_inps=fp_val_inps_before
            update_dataset(qlayers[start_idx:start_idx+slide_step],quant_train_inps,dev,attention_mask,position_ids)
            update_dataset(qlayers[start_idx:start_idx+slide_step],quant_val_inps,dev,attention_mask,position_ids)
            # update_dataset(qlayers[start_idx:start_idx+slide_step],fp_train_inps,dev,attention_mask,position_ids)
            # update_dataset(qlayers[start_idx:start_idx+slide_step],fp_val_inps,dev,attention_mask,position_ids)

        ori_layers = layers[start_idx:end_idx]
        for block_index in range(start_idx, end_idx):
            layers[block_index] = qlayers[block_index].to("cpu")

        loss_recorder.save_to_file()
        # step 7: pack quantized weights into low-bits format, note that this process is slow on poor CPU or busy CPU
        if args.real_quant:
            for qlayer in qlayers[start_idx:start_idx+slide_step]:
                named_linears = get_named_linears(qlayer, int_linear_fake.QuantLinear)
                for name, module in named_linears.items():
                    scales = module.weight_quantizer.scale.clamp(1e-4,1e4).detach()
                    zeros = module.weight_quantizer.zero_point.detach().cuda().round().cpu()
                    group_size = module.weight_quantizer.group_size
                    dim0 = module.weight.shape[0]
                    scales = scales.view(dim0,-1).transpose(0,1).contiguous()
                    zeros = zeros.view(dim0,-1).transpose(0,1).contiguous()
                    q_linear = int_linear_real.QuantLinear(args.wbits, group_size, module.in_features,module.out_features,not module.bias is None)
                    q_linear.pack(module.cpu(),  scales.float().cpu(), zeros.float().cpu())
                    set_op_by_name(qlayer, name, q_linear)       
                    logger.info(f"pack quantized {name} finished")
                    del module        
        del ori_layers
        torch.cuda.empty_cache()

    # delete cached dataset
    if args.off_load_to_disk:
        for path in [fp_train_cache_path,fp_val_cache_path,quant_train_cache_path,quant_val_cache_path]:
            if os.path.exists(path):
                shutil.rmtree(path)

    torch.cuda.empty_cache()
    gc.collect()                    
    model.config.use_cache = use_cache
    return model

