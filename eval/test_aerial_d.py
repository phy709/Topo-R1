import argparse
import os
import json
import random
import numpy as np
import cv2
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm  # 🌟 引入进度条

def parse_args():
    parser = argparse.ArgumentParser(description="Aerial-D Topology & Geometry Evaluation")
    parser.add_argument("--data-root", default="/data0/data/Aerial_R1_Dataset", help="数据集统一根目录")
    parser.add_argument("--json-file", default="/data0/data/Aerial_R1_Dataset/aerial-d/val/annotations.json", help="Aerial-D测试集JSON")
    parser.add_argument("--work-dir", default="./results/tdpo-8b-wo-sft/aerial_d_test", help="保存可视化和指标的目录")
    parser.add_argument("--model", default="./models/tdpo-8b-wo-sft", help="模型路径")
    parser.add_argument("--sample", type=int, default=-1, help="-1为测试全量数据")
    parser.add_argument("--mask-thr", type=float, default=0.5, help="二值化阈值")
    return parser.parse_args()

# ==========================================
# 🎯 核心拓扑与几何计算 (与训练筛选绝对对齐)
# ==========================================
def get_macro_corners(mask_bin):
    contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    corners = []
    for cnt in contours:
        if cv2.contourArea(cnt) < 50: continue
        epsilon = 0.01 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        for pt in approx: corners.append(pt[0])
    return np.array(corners) if len(corners) > 0 else None

def evaluate_topology(gt_bin, pred_bin):
    num_gt, _ = cv2.connectedComponents(gt_bin, connectivity=8)
    num_pred, _ = cv2.connectedComponents(pred_bin, connectivity=8)
    return num_gt - 1, num_pred - 1

def calculate_all_metrics(pred_mask, gt_mask):
    pred = pred_mask > 0
    gt = gt_mask > 0
    
    TP = np.logical_and(pred, gt).sum()
    FP = np.logical_and(pred, np.logical_not(gt)).sum()
    FN = np.logical_and(np.logical_not(pred), gt).sum()

    precision = TP / (TP + FP + 1e-6)
    recall = TP / (TP + FN + 1e-6)
    f1_score = 2 * precision * recall / (precision + recall + 1e-6)
    iou = TP / (TP + FP + FN + 1e-6)
    
    gt_uint8 = gt.astype(np.uint8)
    pred_uint8 = pred.astype(np.uint8)
    
    gt_b0, pred_b0 = evaluate_topology(gt_uint8, pred_uint8)
    betti_error = abs(gt_b0 - pred_b0)

    gt_corners = get_macro_corners(gt_uint8)
    pred_corners = get_macro_corners(pred_uint8)
    
    corner_recall = 1.0
    if gt_corners is not None and pred_corners is not None:
        matched = sum(1 for gc in gt_corners if np.min(np.linalg.norm(pred_corners - gc, axis=1)) <= 15.0)
        corner_recall = matched / len(gt_corners)
    elif gt_corners is not None and pred_corners is None:
        corner_recall = 0.0

    return {
        "iou": float(iou), "f1_score": float(f1_score), "precision": float(precision),
        "recall": float(recall), "betti_error": float(betti_error),
        "corner_recall": float(corner_recall), "gt_betti0": int(gt_b0), "pred_betti0": int(pred_b0)
    }

def binarize_and_resize(pred_mask, img_h, img_w, thr=0.5):
    if hasattr(pred_mask, "cpu"): mask_np = pred_mask.cpu().numpy()
    else: mask_np = np.array(pred_mask)
    if mask_np.ndim == 3: mask_np = mask_np.squeeze()
    if mask_np.shape != (img_h, img_w):
        mask_np = cv2.resize(mask_np, (img_w, img_h), interpolation=cv2.INTER_LINEAR)
    return (mask_np > thr).astype(np.uint8)

def main():
    cfg = parse_args()
    os.makedirs(cfg.work_dir, exist_ok=True)
    vis_dir = os.path.join(cfg.work_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)

    print(f"Loading annotations from: {cfg.json_file}")
    with open(cfg.json_file, 'r', encoding='utf-8') as f: dataset = json.load(f)
    if cfg.sample > 0: dataset = random.sample(dataset, cfg.sample)

    print(f"Loading Model: {cfg.model}")
    model = AutoModelForCausalLM.from_pretrained(cfg.model, dtype="auto", device_map="auto", trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model, trust_remote_code=True)

    all_metrics = {"iou": [], "f1_score": [], "precision": [], "recall": [], "betti_error": [], "corner_recall": []}
    detailed_results = []
    
    # 🌟 使用 tqdm 包装 dataset，去掉了乱刷屏的 print
    pbar = tqdm(dataset, desc="Evaluating Aerial-D")
    
    for idx, item in enumerate(pbar):
        img_rel_path = item["image"]
        mask_rel_path = item["mask"]
        convs = item.get("conversations", [])
        prompt = convs[0]["value"] if convs and convs[0]["from"] == "human" else "<image>\nPlease segment it."
        
        image_path = os.path.join(cfg.data_root, img_rel_path)
        gt_mask_path = os.path.join(cfg.data_root, mask_rel_path)
        if not os.path.exists(image_path) or not os.path.exists(gt_mask_path): continue

        image = Image.open(image_path).convert("RGB")
        img_w, img_h = image.size

        gt_mask = cv2.imread(gt_mask_path, cv2.IMREAD_GRAYSCALE)
        if gt_mask is None: continue
        gt_mask_bin = (gt_mask > 127).astype(np.uint8)

        try:
            result = model.predict_forward(image=image, text=prompt, tokenizer=tokenizer)
            pred_masks_raw = result.get("prediction_masks", [])
            if pred_masks_raw and len(pred_masks_raw) > 0:
                pred_mask_bin = binarize_and_resize(pred_masks_raw[0], img_h, img_w, cfg.mask_thr)
            else:
                pred_mask_bin = np.zeros((img_h, img_w), dtype=np.uint8)
        except Exception:
            pred_mask_bin = np.zeros((img_h, img_w), dtype=np.uint8)

        metrics = calculate_all_metrics(pred_mask_bin, gt_mask_bin)
        for k in all_metrics: all_metrics[k].append(metrics[k])
        
        # 🌟 核心：动态更新进度条后缀，只显示刚刚算完的这一张图的成绩
        pbar.set_postfix({
            "IoU": f"{metrics['iou']:.3f}",
            "BettiErr": f"{metrics['betti_error']:.0f}",
            "Corner": f"{metrics['corner_recall']:.3f}"
        })
        
        detailed_results.append({"image": os.path.basename(image_path), "prompt": prompt.replace("<image>\n", "").strip(), **metrics})

        if idx < 100: 
            overlay = np.array(image).copy()
            overlay[gt_mask_bin > 0] = overlay[gt_mask_bin > 0] * 0.5 + np.array([0, 255, 0]) * 0.5
            overlay[pred_mask_bin > 0] = overlay[pred_mask_bin > 0] * 0.5 + np.array([255, 0, 0]) * 0.5
            
            cv2.putText(overlay, f"IoU: {metrics['iou']:.3f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(overlay, f"Corner Recall: {metrics['corner_recall']:.3f}", (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(overlay, f"Betti Err: {metrics['betti_error']} (GT:{metrics['gt_betti0']} P:{metrics['pred_betti0']})", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 100, 100), 2)
            
            save_name = f"{idx:03d}_iou_{metrics['iou']:.2f}_err_{metrics['betti_error']:.0f}.png"
            cv2.imwrite(os.path.join(vis_dir, save_name), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    mean_metrics = {k: float(np.mean(v)) for k, v in all_metrics.items()}
    json_result_path = os.path.join(cfg.work_dir, "metrics_full.json")
    with open(json_result_path, "w", encoding="utf-8") as f:
        json.dump({"mean_metrics": mean_metrics, "details": detailed_results}, f, indent=2, ensure_ascii=False)

    print("\n" + "="*50)
    print("🌍 Aerial-D Test Set Evaluation Finished!")
    print(f"Mean IoU          : {mean_metrics['iou']:.4f}")
    print(f"Mean Corner Recall: {mean_metrics['corner_recall']:.4f}")
    print(f"Mean Betti Error  : {mean_metrics['betti_error']:.4f}")
    print("="*50)

if __name__ == "__main__":
    main()