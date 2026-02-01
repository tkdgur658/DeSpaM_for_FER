"""
main_abl.ipynb 전용 유틸리티 모듈
모든 함수와 로직을 여기에 정의하고 ipynb에서 import하여 사용
"""

import os
import sys

# src 폴더를 sys.path에 추가 (main_abl.py와 ipynb 모두 호환)
_src_dir = os.path.dirname(os.path.abspath(__file__))
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

import warnings
def warn(*args, **kwargs):
    pass
warnings.warn = warn
import time
from datetime import datetime
import importlib
from tqdm.notebook import tqdm
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data as data
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use('Agg')

eps = sys.float_info.epsilon

from dataset import load_dataset_info, create_datasets, get_transforms
from utils import create_experiment_folder, measure_model_complexity, measure_inference_speed, control_random_seed, THOP_AVAILABLE

# 모델별 동적 import를 위한 캐시
_model_imports = {}


def create_args(datasets, models, iterations, epochs):
    """실험 설정으로 args 객체 생성"""
    return argparse.Namespace(
        # 실험 설정
        datasets=datasets,
        models=models,
        iterations=iterations,
        epochs=epochs,

        # 기본값
        model_dir='src.models',
        data_path='../datasets',
        gpu='0',
        workers=8,
        batch_size=64,
        optimizer='adamw',
        lr=0.001,
        weight_decay=1e-4,
        test_size=0.2,
        val_size=0.25,
        early_stopping_patience=30,
        exp_name='',
        seed=None,

        # DAN 전용
        num_head=4,

        # Ada-DF 전용
        threshold=0.7,
        sharpen=False,
        T=1.2,
        alpha=None,
        beta=3,
        max_weight=1.0,
        min_weight=0.2,
        drop_rate=0.0,
        label_smoothing=0.0,
        tops=0.7,
        margin_1=0.07,

        # CSE-GResNet 전용
        CSE_model='GCN',
        GCN_is_maxpool=False,
        CSE_base_model='resnet18',
        TSM_position=['layer3'],
        TSM_div=8,
        TSM_module_insert='residual',
        TSM_channel_enhance=4,
        channel_block_position='stochastic',
        TSM_div_e=16,
        channel_block_position_e='stochastic',
        channel_enhance_init_strategy=3,
        channel_enhance_kernelsize=3,
        TSM_conv_insert_e='residual',
        fusion_SE='A',
        GConv_M=4,
    )


def parse_model_name(model_name):
    """모델 이름 파싱: 'ProposedNet_7D3' 또는 'Abl_ProposedNet_7D3' -> ('ProposedNet', '7D3')"""
    if model_name.startswith('Abl_ProposedNet_'):
        return 'ProposedNet', model_name[len('Abl_ProposedNet_'):]
    if model_name.startswith('ProposedNet_'):
        return 'ProposedNet', model_name[len('ProposedNet_'):]
    return model_name, None


def get_abl_model(abl_model_name, num_classes=7):
    """Ablation 모델 동적 import 및 생성"""
    if abl_model_name is None:
        return None

    if abl_model_name.startswith('Abl_ProposedNet_'):
        abl_model_name = abl_model_name[len('Abl_ProposedNet_'):]

    module_name = f'Abl_ProposedNet_{abl_model_name}'
    class_name = 'ProposedNet'

    try:
        module = importlib.import_module(f'models.ProposedNet.{module_name}')
        model_class = getattr(module, class_name)
        return model_class(num_classes=num_classes)
    except (ImportError, AttributeError) as e:
        print(f"Warning: Could not import {abl_model_name}: {e}")
        return None


def load_model_dependencies(models):
    """models 리스트의 의존성 동적 import"""
    global _model_imports

    base_models = set(parse_model_name(m)[0] for m in models)

    if 'DAN' in base_models and 'DAN' not in _model_imports:
        from models.DAN.dan import DAN
        from models.DAN.dan_strategy import DANStrategy
        _model_imports['DAN'] = DAN
        _model_imports['DANStrategy'] = DANStrategy
        print("Loaded: DAN")

    if 'POSTER' in base_models and 'POSTER' not in _model_imports:
        from models.POSTERV2.PosterV2_7cls import pyramid_trans_expr2 as poster_7cls
        from models.POSTERV2.PosterV2_8cls import pyramid_trans_expr2 as poster_8cls
        _model_imports['poster_7cls'] = poster_7cls
        _model_imports['poster_8cls'] = poster_8cls
        print("Loaded: POSTER")

    if 'AdaDF' in base_models and 'AdaDF' not in _model_imports:
        from models.AdaDF.model import create_model as create_adadf_model
        from models.AdaDF.adadf_strategy import AdaDFStrategy
        _model_imports['create_adadf_model'] = create_adadf_model
        _model_imports['AdaDFStrategy'] = AdaDFStrategy
        print("Loaded: AdaDF")

    if 'CSE-GResNet' in base_models and 'CSE-GResNet' not in _model_imports:
        from models.CSEGResNet.model import CSE_GResNet
        _model_imports['CSE_GResNet'] = CSE_GResNet
        print("Loaded: CSE-GResNet")

    if 'ProposedNet' in base_models and 'ProposedNet' not in _model_imports:
        _model_imports['ProposedNet'] = True
        print("Loaded: ProposedNet")

    if 'get_model_strategy' not in _model_imports:
        from models.base.model_strategy import get_model_strategy
        _model_imports['get_model_strategy'] = get_model_strategy

    return _model_imports


def create_model_instance(base_model, abl_variant, num_classes, args, input_size_hw=None):
    """모델 인스턴스 생성"""
    if base_model == 'ProposedNet':
        if abl_variant is None:
            # 기본 ProposedNet 모델 로드
            from models.ProposedNet.ProposedNet import ProposedNet
            return ProposedNet(num_classes=num_classes)
        else:
            return get_abl_model(abl_variant, num_classes)
    elif base_model == 'DAN':
        return _model_imports['DAN'](num_head=args.num_head, num_class=num_classes)
    elif base_model == 'POSTER':
        img_size = input_size_hw[0] if input_size_hw else 112
        if num_classes == 7:
            return _model_imports['poster_7cls'](img_size=img_size, num_classes=num_classes)
        elif num_classes == 8:
            return _model_imports['poster_8cls'](img_size=img_size, num_classes=num_classes)
    elif base_model == 'AdaDF':
        return _model_imports['create_adadf_model'](num_classes, args.drop_rate)
    elif base_model == 'CSE-GResNet':
        return _model_imports['CSE_GResNet'](num_classes=num_classes, args=args)
    return None


def train_unified(args, train_loader, val_loader, model, strategy,
                  optimizer, scheduler, device, epochs, patience,
                  iteration, checkpoint_dir, model_name, dataset_name,
                  training_start_time, results_dir, abl_variant=None):
    """통합 훈련 함수"""
    best_loss, best_acc, patience_counter = float('inf'), 0, 0
    progress_bar = tqdm(range(1, epochs + 1), desc=f"Training {model_name}/{dataset_name} Iter {iteration}")
    epoch_times, train_losses, val_losses = [], [], []
    save_path = ""
    run_name = f"ProposedNet_{abl_variant}" if abl_variant else model_name

    AdaDFStrategy = _model_imports.get('AdaDFStrategy')
    collect_outputs = AdaDFStrategy and isinstance(strategy, AdaDFStrategy)

    log_file_path = os.path.join(results_dir, f"{run_name}_{dataset_name}_iter{iteration}_epoch_log.csv")
    with open(log_file_path, 'w') as f:
        f.write("timestamp,model_name,ablation,epoch,train_loss,val_loss,val_acc,val_balanced_acc\n")

    for epoch in progress_bar:
        epoch_start_time = time.time()
        strategy.epoch = epoch

        model.train()
        current_train_loss = 0.0

        if collect_outputs:
            all_outputs_1, all_targets, all_weights = [], [], []

        for (imgs, targets) in train_loader:
            imgs, targets = imgs.to(device), targets.to(device)
            optimizer.zero_grad()

            model_output = strategy.forward_model(model, imgs)
            loss_result = strategy.compute_loss(model_output, targets)

            if collect_outputs and isinstance(loss_result, tuple):
                loss, extra_info = loss_result
                all_outputs_1.append(extra_info['outputs_1'])
                all_targets.append(extra_info['targets'])
                all_weights.append(extra_info['attention_weights'])
            else:
                loss = loss_result

            loss.backward()
            optimizer.step()
            current_train_loss += loss.item()

        if scheduler:
            scheduler.step()

        if collect_outputs and all_outputs_1:
            outputs_tensor = torch.cat(all_outputs_1, dim=0)
            targets_tensor = torch.cat(all_targets, dim=0)
            strategy.update_LD(outputs_tensor, targets_tensor, outputs_tensor.size(1))

        avg_train_loss = current_train_loss / (len(train_loader) + eps)
        val_loss, acc, balanced_acc = validate_unified(val_loader, model, strategy, device)

        train_losses.append(avg_train_loss)
        val_losses.append(val_loss)

        with open(log_file_path, 'a') as f:
            f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')},{model_name},{abl_variant},{epoch},{avg_train_loss:.5f},{val_loss:.5f},{acc:.5f},{balanced_acc:.5f}\n")

        epoch_times.append((time.time() - epoch_start_time) / 60)
        progress_bar.set_description(f"Epoch {epoch} | Train: {avg_train_loss:.3f} | Val: {val_loss:.3f} | Acc: {acc:.4f} | BAcc: {balanced_acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            save_path = os.path.join(checkpoint_dir, f"{training_start_time}_{run_name}_{dataset_name}_iter{iteration}.pth")
            model_state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            checkpoint = {'epoch': epoch, 'model_state_dict': model_state, 'state_dict': model_state,
                          'optimizer_state_dict': optimizer.state_dict(), 'acc': acc, 'best_acc': best_acc}
            if collect_outputs:
                checkpoint['class_distributions'] = strategy.LD.detach()
            torch.save(checkpoint, save_path)

        if best_loss > val_loss:
            best_loss, patience_counter = val_loss, 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("Early stopping triggered")
            break

    return save_path, best_acc, np.mean(epoch_times) if epoch_times else 0, train_losses, val_losses


def validate_unified(val_loader, model, strategy, device):
    """통합 검증 함수"""
    model.eval()
    val_loss, y_true, y_pred = 0.0, [], []

    with torch.no_grad():
        for (imgs, targets) in val_loader:
            imgs, targets = imgs.to(device), targets.to(device)
            model_output = strategy.forward_model(model, imgs)
            loss_result = strategy.compute_loss(model_output, targets)
            val_loss += (loss_result[0] if isinstance(loss_result, tuple) else loss_result).item()
            _, predicts = torch.max(strategy.get_predictions(model_output), 1)
            y_true.extend(targets.cpu().numpy())
            y_pred.extend(predicts.cpu().numpy())

    if not y_true:
        return 0.0, 0.0, 0.0

    val_loss /= len(val_loader)
    acc = np.mean(np.array(y_true) == np.array(y_pred))
    try:
        balanced_acc = balanced_accuracy_score(y_true, y_pred)
    except:
        balanced_acc = acc
    return val_loss, acc, balanced_acc


def test_unified(test_loader, model, checkpoint_path, device, strategy):
    """통합 테스트 함수"""
    try:
        loaded_data = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state_dict = loaded_data.get('model_state_dict') or loaded_data.get('state_dict') or loaded_data
        model_to_load = model.module if isinstance(model, nn.DataParallel) else model
        model_to_load.load_state_dict(state_dict, strict=False)
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return 0.0, 0.0, np.array([])

    model.eval()
    if hasattr(model_to_load, 'switch_to_deploy'):
        model_to_load.switch_to_deploy()

    y_true, y_pred = [], []
    with torch.no_grad():
        for (imgs, targets) in tqdm(test_loader, desc="Testing", leave=False):
            imgs, targets = imgs.to(device), targets.to(device)
            _, predicts = torch.max(strategy.get_predictions(strategy.forward_model(model, imgs)), 1)
            y_true.extend(targets.cpu().numpy())
            y_pred.extend(predicts.cpu().numpy())

    if not y_true:
        return 0.0, 0.0, np.array([])

    acc = np.mean(np.array(y_true) == np.array(y_pred))
    try:
        balanced_acc = balanced_accuracy_score(y_true, y_pred)
    except:
        balanced_acc = acc
    return acc, balanced_acc, confusion_matrix(y_true, y_pred)


def run_experiment(datasets, models, iterations, epochs):
    """메인 실험 실행 함수"""
    args = create_args(datasets, models, iterations, epochs)
    code_start_time = datetime.now().strftime("%y%m%d_%H%M%S")

    # GPU 설정
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device_ids = [int(id) for id in args.gpu.split(',') if id.strip()]
    device = torch.device("cuda:0") if torch.cuda.is_available() and device_ids else torch.device("cpu")
    print(f"Device: {device}")

    torch.backends.cudnn.benchmark = True
    if args.seed is not None:
        control_random_seed(args.seed)

    # 모델 의존성 로드
    print("\n--- Loading model dependencies ---")
    load_model_dependencies(args.models)

    experiment_paths = create_experiment_folder(args)
    final_csv_path = os.path.join(experiment_paths['results_dir'], f'all_results_{code_start_time}.csv')

    print(f"\nExperiment: {os.path.basename(experiment_paths['experiment_dir'])}")
    print(f"Datasets: {args.datasets} | Models: {args.models} | Iterations: {args.iterations}")

    all_run_results = []
    results_df = None

    # 모델 복잡도 캐시 (모델별로 한 번만 측정)
    model_complexity_cache = {}

    for dataset_name in args.datasets:
        print(f"\n{'='*30} Dataset: {dataset_name} {'='*30}")

        try:
            all_data_indices, all_labels, num_classes, use_stratify = load_dataset_info(dataset_name, args.data_path)
            print(f"Loaded {len(all_labels)} samples, {num_classes} classes")
        except Exception as e:
            print(f"Error: {e}. Skipping.")
            continue

        # Iteration 루프 (바깥)
        for iteration in range(args.iterations[0], args.iterations[1] + 1):
            print(f"\n{'-'*25} Iteration {iteration}/{args.iterations[1]} (Seed: {iteration}) {'-'*25}")

            # 데이터 준비 (iteration별로 한 번)
            stratify_array = all_labels if use_stratify else None
            train_val_indices, test_indices = train_test_split(all_data_indices, test_size=args.test_size, random_state=iteration, stratify=stratify_array)

            # Model 루프 (안쪽)
            for model_name_full in args.models:
                base_model, abl_variant = parse_model_name(model_name_full)
                training_start_time = datetime.now().strftime("%y%m%d_%H%M%S")
                print(f"\n--- Model: {model_name_full} ---")

                # 모델 복잡도 (캐시 사용)
                if model_name_full not in model_complexity_cache:
                    model_complexity = {"params_M": -1, "flops_G": -1}
                    try:
                        control_random_seed(42)
                        temp_model = create_model_instance(base_model, abl_variant, num_classes, args, (112, 112))
                        if temp_model:
                            temp_model = temp_model.to(device)
                            model_complexity = measure_model_complexity(temp_model, input_size=(3, 112, 112))
                            print(f"Params: {model_complexity['params_M']:.2f}M" + (f" | FLOPs: {model_complexity['flops_G']:.2f}G" if THOP_AVAILABLE else ""))
                            del temp_model
                            torch.cuda.empty_cache()
                    except Exception as e:
                        print(f"Complexity error: {e}")
                    model_complexity_cache[model_name_full] = model_complexity
                else:
                    model_complexity = model_complexity_cache[model_name_full]

                data_transforms, val_transforms, input_size_hw = get_transforms(base_model)

                temp_args = argparse.Namespace(**vars(args))
                temp_args.dataset = dataset_name
                try:
                    train_dataset, val_dataset, test_dataset = create_datasets(temp_args, train_val_indices, test_indices, all_data_indices, all_labels, use_stratify, iteration, data_transforms, val_transforms)
                except Exception as e:
                    print(f"Dataset error: {e}. Skipping.")
                    continue

                train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
                val_loader = data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
                test_loader = data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

                control_random_seed(iteration)

                # 훈련
                model, strategy, optimizer, scheduler = None, None, None, None
                best_checkpoint_path, avg_epoch_time_min, train_losses, val_losses = "", -1.0, [], []

                try:
                    model = create_model_instance(base_model, abl_variant, num_classes, args, input_size_hw)
                    if model is None:
                        print(f"Error: Could not create {model_name_full}. Skipping.")
                        continue

                    if len(device_ids) > 1:
                        model = nn.DataParallel(model, device_ids=device_ids)
                    model = model.to(device)

                    strategy = _model_imports['get_model_strategy'](base_model, args, device, num_classes)

                    DANStrategy = _model_imports.get('DANStrategy')
                    params = list(model.parameters()) + list(strategy.criterion_af.parameters()) if DANStrategy and isinstance(strategy, DANStrategy) else model.parameters()

                    if args.optimizer == 'adam':
                        optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
                    elif args.optimizer == 'adamw':
                        optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
                    else:
                        optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)

                    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

                    best_checkpoint_path, _, avg_epoch_time_min, train_losses, val_losses = train_unified(
                        args, train_loader, val_loader, model, strategy,
                        optimizer, scheduler, device, args.epochs, args.early_stopping_patience,
                        iteration, experiment_paths['checkpoints_dir'], model_name_full, dataset_name,
                        training_start_time, experiment_paths['results_dir'], abl_variant=abl_variant
                    )
                except Exception as e:
                    print(f"Training error: {e}")
                    import traceback
                    traceback.print_exc()

                # 테스트
                test_acc, test_balanced_acc, latency_gpu_ms, latency_cpu_ms = 0.0, 0.0, -1.0, -1.0

                if best_checkpoint_path and os.path.exists(best_checkpoint_path) and strategy:
                    test_acc, test_balanced_acc, _ = test_unified(test_loader, model, best_checkpoint_path, device, strategy)
                    try:
                        model_for_speed = model.module if isinstance(model, nn.DataParallel) else model
                        metrics = measure_inference_speed(model_for_speed, (3, input_size_hw[0], input_size_hw[1]))
                        latency_gpu_ms, latency_cpu_ms = metrics.get('latency_gpu_ms', -1.0), metrics.get('latency_cpu_ms', -1.0)
                    except:
                        pass

                print(f"Test Acc: {test_acc:.4f} | Balanced Acc: {test_balanced_acc:.4f}")

                # 결과 저장
                result_data = {
                    "code_start_time": code_start_time, "training_start_time": training_start_time,
                    "Model Name": model_name_full, "Model": base_model, "ablation": abl_variant,
                    "Dataset": dataset_name, "Iteration": iteration,
                    "Accuracy": test_acc, "test_balanced_accuracy": test_balanced_acc,
                    "latency_gpu_ms": latency_gpu_ms, "latency_cpu_ms": latency_cpu_ms,
                    "avg_epoch_time_min": avg_epoch_time_min,
                    "params_M": model_complexity.get('params_M', -1), "flops_G": model_complexity.get('flops_G', -1),
                    "final_train_loss": train_losses[-1] if train_losses else -1,
                    "final_val_loss": val_losses[-1] if val_losses else -1,
                    "best_val_loss": min(val_losses) if val_losses else -1,
                    "checkpoint_path": best_checkpoint_path,
                }
                all_run_results.append(result_data)

                try:
                    results_df = pd.DataFrame(all_run_results)
                    results_df.to_csv(final_csv_path, index=False)
                except:
                    pass

                # 메모리 정리
                del model, strategy, optimizer, scheduler, train_loader, val_loader, test_loader
                torch.cuda.empty_cache()

    # 최종 결과
    if all_run_results:
        print("\n" + "="*60 + "\nFINAL RESULTS\n" + "="*60)
        if results_df is None:
            results_df = pd.DataFrame(all_run_results)
        print(results_df.groupby(['Dataset', 'Model Name'])[['Accuracy', 'test_balanced_accuracy']].mean())
        print(f"\nSaved to: {final_csv_path}")

    print(f"\nExperiment {code_start_time} finished.")
    return results_df
