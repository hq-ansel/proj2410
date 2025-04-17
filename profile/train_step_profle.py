
import torch
import torch.nn as nn
import torch.amp
from torch.utils.data import Dataset
from torch.utils.checkpoint import checkpoint
from torch.amp import autocast, GradScaler
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader



def train_units_layers(args,
                      layers)
    total_training_iteration = args.epochs * args.train_size / args.batch_size
    step = 0
    param_groups = []
    param_group_index = 0
    qlayers = layers
    for qlayer in qlayers:
        set_quant_state(qlayer,True)
        if args.quant_lr > 0:
            set_quant_parameters(qlayer,True)
            param_groups.append({"params":quant_parameters(qlayer),
                                "lr":args.quant_lr})
            empty_optimizer_1 = torch.optim.AdamW([torch.tensor(0)],
                                                lr=args.quant_lr)
            quant_scheduler = CosineAnnealingLR(empty_optimizer_1,
                                        T_max=total_training_iteration,
                                        eta_min=args.quant_lr/args.min_lr_factor)
            quant_index = param_group_index
            param_group_index += 1
        else:
            set_quant_parameters(qlayer,False)
            
        if args.weight_lr > 0:
            set_weight_parameters(qlayer,True)
            param_groups.append({"params":weight_parameters(qlayer),
                                "lr":args.weight_lr})
            empty_optimizer_2 = torch.optim.AdamW([torch.tensor(0)],
                                                lr=args.weight_lr)
            weight_scheduler = CosineAnnealingLR(empty_optimizer_2,
                                            T_max=total_training_iteration,
                                            eta_min=args.weight_lr/args.min_lr_factor)
            weight_index = param_group_index
            param_group_index += 1
        else:
            set_weight_parameters(qlayer,False)
        
        optimizer =torch.optim.AdamW(param_groups,
                                    weight_decay=args.wd,
                                    foreach=True)
        
        loss_scaler= torch.amp.GradScaler(device=args.dev)
        trainable_number = trainable_parameters_num(qlayers)
        print(f"trainable parameter number: {trainable_number/1e6}M")
        best_val_loss = 1e6
        early_stop_flag = 0

        if args.get("gradual_quant",False) or args.get("interpolate",False):
            class GradualWarmupScheduler:
                def __init__(self,
                              linear_list:List[int_linear_fake.QuantLinear],
                              total_iteration:int,):
                    self.linear_list = linear_list
                    self.total_iteration = total_iteration
                    self.iteration = 0
                    self.update()
                def update(self):
                    self.iteration += 1
                    for linear in self.linear_list:
                        if self.iteration < (self.total_iteration/2.0):
                            ratio = self.iteration/(self.total_iteration/2.0)
                            if args.get("gradual_quant",False):
                                linear.update_position_ratio(ratio)
                            if args.get("interpolate", False):
                                linear.update_interpolate_ratio(1-ratio)
                        else:
                            linear.update_position_ratio(1.0)
                            if args.get("interpolate", False):
                                linear.update_interpolate_ratio(0)
            q_linear_list = []
            for i in trainable_layer_idx_list:
                for n,m in qlayers[i].named_modules():
                    if isinstance(m, int_linear_fake.QuantLinear):
                        q_linear_list.append(m)
            graualWarmupScheduler = GradualWarmupScheduler(
                q_linear_list,
                total_training_iteration,
            )
        # step 6.3: training loop
        position_ids = torch.arange(args.training_seqlen, dtype=torch.long, device=args.dev)
        position_ids = position_ids.unsqueeze(0).expand(args.batch_size, -1).contiguous()
        # print(f" data size {len(train_dataset)}")

        # try torch.compile
        qlayers = torch.nn.ModuleList(
            [torch.compile(qlayer) for qlayer in qlayers]
        )

        for epoch in range(args.epochs):
            loss_list = []
            norm_list = []
            start_time = time.time()
            # used for debug
            # torch.autograd.set_detect_anomaly(True)
            dataloader = DataLoader(train_dataset,
                                    batch_size=args.batch_size,
                                    num_workers=1,
                                    pin_memory=True,
                                    prefetch_factor=32,  
                                    shuffle=True
                                    )
            # step 6.4: training                   
            for index, input_data in enumerate(dataloader):
                optimizer.zero_grad()
                with torch.autocast(device_type=args.dev,
                                    enabled=amp_enabled,
                                    dtype=args.dtype if amp_enabled else torch.float32):
                    inp,target = input_data
                    hidden_state = inp
                    for layer_idx in range(len(qlayers)):
                        layer_outputs = qlayers[layer_idx](
                            hidden_states=hidden_state,
                            attention_mask=attention_mask,
                            position_ids=position_ids,
                            position_embeddings=(position_embeddings[0],position_embeddings[1])
                        )
                        hidden_state = layer_outputs[0]
                    loss = loss_func(hidden_state, trg)
                if not math.isfinite(loss.item()) or loss.item()==0:
                    logger.info("Loss is NAN, stopping training")
                    pdb.set_trace()
                if args.log_loss:
                    loss_recorder.record(f"blk{trainable_layer_idx_list}",
                                        step,
                                        loss.data.cpu().item())
                    
                if args.get("gradual_quant",False):
                    graualWarmupScheduler.update() 
                else: 
                    None
                loss_list.append(loss.data.cpu())
                if amp_enabled:
                    loss_scaler.scale(loss).backward()
                else:
                    loss.backward()
                if amp_enabled: loss_scaler.unscale_(optimizer)
                if args.clip_grad > 0:
                    norm = torch.nn.utils.clip_grad_norm_(trainable_parameters(selected_layers)
                                                            , args.clip_grad).cpu()
                    norm_list.append(norm.data)
                # 使用子空间优化
                if args.get("sub_space_grad_clean",False):
                    sub_space_clean(selected_layers)
                if amp_enabled:
                    loss_scaler.step(optimizer)
                    loss_scaler.update()
                else:
                    optimizer.step()
                
                # adjust lr
                if args.quant_lr > 0:
                    quant_scheduler.step()
                    optimizer.param_groups[quant_index]['lr'] = quant_scheduler.get_lr()[0]
                if args.weight_lr >0 :
                    weight_scheduler.step()
                    optimizer.param_groups[weight_index]['lr'] = weight_scheduler.get_lr()[0]
                step += 1

            # step 6.5: calculate validation loss
            with torch.no_grad():
                val_loss_list = []
                dataloader = DataLoader(val_dataset,
                                        batch_size=args.batch_size,
                                        num_workers=0,
                                        # pin_memory=True,
                                        # prefetch_factor=32,  
                                        shuffle=True
                                        )
                for index, input_data in enumerate(dataloader):  
                    # obtain output of quantization model
                    with torch.autocast(device_type=args.dev,
                                    enabled=amp_enabled,
                                    dtype=args.dtype if amp_enabled else torch.float32):
                        inp,target = input_data
                        hidden_state = inp.to(args.dev,dtype=args.dtype)
                        for layer_idx in range(len(qlayers)):
                            layer_outputs = qlayers[layer_idx](
                                hidden_states=hidden_state,
                                attention_mask=attention_mask.float(),
                                position_embeddings=(position_embeddings[0].float(),position_embeddings[1].float())
                            )
                            hidden_state = layer_outputs[0]
                        loss = loss_func(hidden_state, target.to(args.dev,dtype=torch.float32))
                    val_loss_list.append(loss.cpu())

                train_mean_num = min(len(loss_list),64) 
                # calculate the average training loss of last train_mean_num samples
                loss_mean = torch.stack(loss_list)[-(train_mean_num-1):].mean()
                val_loss_mean = torch.stack(val_loss_list).mean()
                norm_mean = torch.stack(norm_list).mean()
                logger.info(f"blocks {trainable_layer_idx_list} epoch {epoch} recon_loss:{loss_mean} val_loss:{val_loss_mean} ")
                logger.info(f"quant_lr:{quant_scheduler.get_lr()[0]} weight_lr:{weight_scheduler.get_lr()[0]} norm:{norm_mean:.8f}  ")
                logger.info(f"max memory_allocated {torch.cuda.max_memory_allocated(args.dev) / 1024**2} time {time.time()-start_time} ")
                if val_loss_mean < best_val_loss:
                    best_val_loss = val_loss_mean
                else:
                    early_stop_flag += 1
                    if args.early_stop > 0 and early_stop_flag >=args.early_stop:
                        break
            
        optimizer.zero_grad()
        del optimizer
        # step 7: pack quantized weights into low-bits format, note that this process is slow on poor CPU or busy CPU
        
    torch.cuda.empty_cache()
    gc.collect()
