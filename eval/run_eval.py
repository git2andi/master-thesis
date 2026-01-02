import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple
from dataclasses import asdict

from helper.io import load_json_any, make_output_dir, write_json
from helper.eval_config import get_dataset_cfg
from helper.realcolon import remap_realcolon_coco_gt_ids, remap_realcolon_ultralytics_preds_ids, remap_realcolon_detectron2_preds_ids
from helper.sun import remap_sun_coco_gt_ids, remap_sun_detectron2_preds_ids, remap_sun_ultralytics_preds_ids
from helper.piccolo import remap_piccolo_coco_gt_ids, remap_piccolo_detectron2_preds_ids, remap_piccolo_ultralytics_preds_ids

from helper.coco_metrics import compute_coco_bbox_metrics
from helper.detection_metrics import compute_detection_metrics
from helper.frame_metrics import compute_frame_metrics, compute_frame_tpr_fpr_sweep
from helper.lesion_metrics import compute_lesion_frame_detection_stats
from helper.froc_metrics import compute_froc_points_for_conf_thresholds, compute_afroc_points_for_conf_thresholds
from helper.plot_curves import plot_frame_tp_fp_bars_from_sweep_json,plot_frame_two_metric_smooth_from_sweep_json, plot_froc_curve, plot_afroc_curve

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Evaluation entrypoint")
    p.add_argument("--gt", type=Path, required=True, help="Path to COCO GT JSON")
    p.add_argument("--pred", type=Path, required=True, help="Path to predictions JSON")
    p.add_argument("--framework", type=str, required=True, choices=["ultralytics", "detectron2"])
    p.add_argument("--dataset", type=str, required=True, choices=["realcolon", "sun", "piccolo"])
    p.add_argument("--conf", type=float, required=False)
    return p.parse_args()

def load(gt_path: Path, pred_path: Path, framework: str, dataset: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    print(f"gt        : {gt_path}")
    print(f"pred      : {pred_path}")
    print(f"framework : {framework}")
    print(f"dataset   : {dataset}")

    coco_gt_raw: Any = load_json_any(gt_path)
    preds_raw: Any = load_json_any(pred_path)

    preds: List[Dict[str, Any]] = [p for p in preds_raw if isinstance(p, dict)]

    print(f"[INFO] GT frames/images (raw): {len(coco_gt_raw.get('images', []))}")
    print(f"[INFO] Predictions (raw):      {len(preds)}")


    coco_gt_remapped: Dict[str, Any] = coco_gt_raw
    preds_remapped: List[Dict[str, Any]] = preds
    gt_id_set = None
    n_missing_imgid = 0

    match framework:
        case "ultralytics":
            match dataset:
                case "realcolon":
                    coco_gt_remapped, gt_id_set = remap_realcolon_coco_gt_ids(coco_gt_raw)
                    preds_remapped, n_missing_imgid = remap_realcolon_ultralytics_preds_ids(preds)
                case "sun":
                    coco_gt_remapped, gt_id_set = remap_sun_coco_gt_ids(coco_gt_raw)
                    preds_remapped, n_missing_imgid = remap_sun_ultralytics_preds_ids(preds)
                case "piccolo":
                    coco_gt_remapped, gt_id_set = remap_piccolo_coco_gt_ids(coco_gt_raw)
                    preds_remapped, n_missing_imgid = remap_piccolo_ultralytics_preds_ids(preds)
                case _:
                    raise ValueError(f"Unsupported dataset: {dataset}")

        case "detectron2":
            match dataset:
                case "realcolon":
                    coco_gt_remapped, gt_id_set = remap_realcolon_coco_gt_ids(coco_gt_raw)
                    preds_remapped, n_missing_imgid, n_unknown_imgid = remap_realcolon_detectron2_preds_ids(
                        preds, coco_gt_raw, pred_image_id_key="image_id", strict=False
                    ) 
                case "sun":
                    coco_gt_remapped, gt_id_set = remap_sun_coco_gt_ids(coco_gt_raw)
                    preds_remapped, n_missing_imgid, n_unknown_imgid = remap_sun_detectron2_preds_ids(
                        preds, coco_gt_raw, pred_image_id_key="image_id", strict=False
                    )
                case "piccolo":
                    coco_gt_remapped, gt_id_set = remap_piccolo_coco_gt_ids(coco_gt_raw)
                    preds_remapped, n_missing_imgid, n_unknown_imgid = remap_piccolo_detectron2_preds_ids(
                        preds, coco_gt_raw, pred_image_id_key="image_id", strict=False
                    )
                    pass
                case _:
                    raise ValueError(f"Unsupported dataset: {dataset}")
        case _:
            raise ValueError(f"Unsupported framework: {framework}")


    print(f"[INFO] GT frames/images (remap): {len(coco_gt_remapped.get('images', []))}")
    print(f"[INFO] Predictions (remap):      {len(preds_remapped)}")

    if framework == "ultralytics" and dataset == "realcolon":
        print(f"[INFO] Predictions missing image_id: {n_missing_imgid} (dropped)")
        if gt_id_set is not None:
            n_match = sum(1 for d in preds_remapped if d.get("image_id") in gt_id_set)
            print(f"[INFO] Preds with GT-known image_id: {n_match}")

    return coco_gt_remapped, preds_remapped


def main() -> None:
    args = parse_args()

    coco_gt_remapped, preds_remapped = load(
        gt_path=args.gt,
        pred_path=args.pred,
        framework=args.framework,
        dataset=args.dataset,
    )

    print("[INFO] Computing COCO bbox metrics")
    coco_vals = compute_coco_bbox_metrics(coco_gt=coco_gt_remapped, preds=preds_remapped)

    cfg = get_dataset_cfg(args.dataset)
    iou_thr = 0.0
    conf = args.conf if args.conf is not None else 0.2

    det = None
    if cfg.do_detection_metrics:
        det = compute_detection_metrics(
            coco_gt=coco_gt_remapped,
            preds=preds_remapped,
            iou_thr=iou_thr,
            conf=conf,
        )

    frame = None
    if cfg.do_frame_metrics:
        frame = compute_frame_metrics(
            coco_gt=coco_gt_remapped,
            preds=preds_remapped,
            iou_thr=iou_thr,
            conf=conf,
        )
        
    frame_sweep = None
    if cfg.do_frame_sweep:
        FRAME_THRESHOLDS = [0.001, 0.005, 0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
        frame_sweep = compute_frame_tpr_fpr_sweep(
            coco_gt=coco_gt_remapped,
            preds=preds_remapped,
            iou_thr=iou_thr,
            selected_thresholds=FRAME_THRESHOLDS,
            dense_lo=0.01,
            dense_hi=0.10,
            dense_step=0.001,
            coarse_step=0.05,
        )


    lesion = None
    if cfg.do_lesion_metrics:
        lesion = compute_lesion_frame_detection_stats(
            coco_gt_remapped=coco_gt_remapped,
            preds_remapped=preds_remapped,
            iou_thr=iou_thr,
            conf=conf,
            enable_timing=cfg.lesion_enable_timing,
        )


    froc = None
    conf_grid: list[float] | None = None
    if cfg.do_froc:
        conf_grid = [0.001, 0.005, 0.01] + [i / 100 for i in range(5, 101, 5)]  # 0.05..1.00
        froc = compute_froc_points_for_conf_thresholds(
            coco_gt=coco_gt_remapped,
            preds=preds_remapped,
            iou_thr=iou_thr,
            conf_thresholds=conf_grid,
        )

    afroc = None
    if cfg.do_afroc:
        if conf_grid is None:
            conf_grid = [0.001, 0.005, 0.01] + [i / 100 for i in range(5, 101, 5)]
        afroc = compute_afroc_points_for_conf_thresholds(
            coco_gt=coco_gt_remapped,
            preds=preds_remapped,
            iou_thr=iou_thr,
            conf_thresholds=conf_grid,
        )



    iou_tag = f"{int(round(iou_thr * 100)):02d}"
    conf_tag = f"{int(round(conf * 100)):05d}"
    out_dir = make_output_dir(args.pred) / f"i{iou_tag}_c{conf_tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # results.json
    out_path = out_dir / "results.json"
    row: Dict[str, Any] = {
        "dataset": args.dataset,
        "framework": args.framework,
        "gt_path": str(args.gt),
        "pred_path": str(args.pred),
        "pred_parent": args.pred.parent.name,
        "pred_file": args.pred.name,
        "n_frames": len(coco_gt_remapped.get("images", [])),
        "n_preds": len(preds_remapped),
        "coco_metrics": coco_vals,
        "iou_thr": iou_thr,
        "conf": conf,
        "detection_metrics": det,
        "frame_metrics": asdict(frame) if frame is not None else None,
        "lesion_metrics": lesion,
    }
    result = write_json(out_dir, row, filename=out_path.name)
    print(f"[INFO] Wrote results: {result}")


    # TPR / FPR Sweep
    if frame_sweep is not None:
        frames_sweep_path = out_dir / "frame_threshold_sweep.json"
        write_json(out_dir, frame_sweep, filename=frames_sweep_path.name)
        print(f"[INFO] Wrote frame sweep: {frames_sweep_path}")

        x_min = float(frame_sweep.get("min_conf_available", 0.0))

        tp_fp_plot = out_dir / "tp_fp_bars.png"
        plot_frame_tp_fp_bars_from_sweep_json(
            sweep_json=frames_sweep_path,
            out_path=tp_fp_plot,
        )
        print(f"[INFO] Plotted: {tp_fp_plot}")
        
        f12_plot = out_dir / "f1_f2_smooth.png"
        plot_frame_two_metric_smooth_from_sweep_json(
            sweep_json=frames_sweep_path,
            out_path=f12_plot,
            y1_key="f1",
            y2_key="f2",
            y1_label="F1 score",
            y2_label="F2 score",
            title="Frame-level F-scores vs confidence threshold",
            x_min=x_min,
            x_max=0.3,
            smooth_window=1,
        )
        print(f"[INFO] Plotted: {f12_plot}")

        tpr_fpr_plot = out_dir / "tpr_fpr_smooth.png"
        plot_frame_two_metric_smooth_from_sweep_json(
            sweep_json=frames_sweep_path,
            out_path=tpr_fpr_plot,
            y1_key="tpr",
            y2_key="fpr",
            y1_label="True positive rate (TPR)",
            y2_label="False positive rate (FPR)",
            title="Frame-level TPR/FPR vs confidence threshold",
            x_min=x_min,
            x_max=0.3,
            smooth_window=1,
        )
        print(f"[INFO] Plotted: {tpr_fpr_plot}")


        sens_spec_plot = out_dir / "sens_spec_smooth.png"
        plot_frame_two_metric_smooth_from_sweep_json(
            sweep_json=frames_sweep_path,
            out_path=sens_spec_plot,
            y1_key="sensitivity",
            y2_key="specificity",
            y1_label="Sensitivity",
            y2_label="Specificity",
            title="Frame-level sensitivity/specificity vs confidence threshold",
            x_min=x_min,
            x_max=0.3,
            smooth_window=1,
        )
        print(f"[INFO] Plotted: {sens_spec_plot}")
    else:
        print("[INFO] frame_sweep is None -> skipping frame sweep JSON and plots")


    curves: dict[str, dict[str, Any]] = {}
    if froc is not None and conf_grid is not None:
        curves["froc"] = {
            "grid_key": "conf_thresholds",
            "grid": conf_grid,
            "payload": {
                "conf_thresholds": froc.get("conf_thresholds", []),
                "fp_per_image": froc.get("fp_per_image", []),
                "tpr_loc": froc.get("tpr_loc", []),
                "total_gt": froc.get("total_gt", None),
                "n_images": froc.get("n_images", None),
            },
            "plot": lambda out_path: plot_froc_curve(
                fp_per_image=froc["fp_per_image"],
                tpr_loc=froc["tpr_loc"],
                out_path=out_path,
                title=f"FROC (IoU>{iou_thr:.2f})",
                log_x=False,
                conf_thresholds=froc.get("conf_thresholds", None),
                highlight_conf=conf,  # NEW
            ),
            "plot_name": f"froc_curve_i{iou_tag}.png",
        }

    if afroc is not None and conf_grid is not None:
        curves["afroc"] = {
            "grid_key": "conf_thresholds",
            "grid": conf_grid,
            "payload": {
                "conf_thresholds": afroc.get("conf_thresholds", []),
                "fpf": afroc.get("fpf", []),
                "tpr_loc": afroc.get("tpr_loc", []),
                "n_neg_images": afroc.get("n_neg_images", None),
            },
            "plot": lambda out_path: plot_afroc_curve(
                fpf=afroc["fpf"],
                tpr_loc=afroc["tpr_loc"],
                out_path=out_path,
                title=f"AFROC (IoU={iou_thr:.2f})",
            ),
            "plot_name": f"afroc_curve_i{iou_tag}.png",
        }

    if curves:
        curve_payload: Dict[str, Any] = {
            "dataset": args.dataset,
            "framework": args.framework,
            "iou_thr": float(iou_thr),
        }

        for name, c in curves.items():
            curve_payload[c["grid_key"]] = c["grid"]
            curve_payload[name] = c["payload"]

        curve_path = out_dir / "curve.json"
        write_json(out_dir, curve_payload, filename=curve_path.name)
        print(f"[INFO] Wrote curves: {curve_path}")
    else:
        print("[INFO] No FROC/AFROC required > skip curve.json")

    for name, c in curves.items():
        plot_path = out_dir / c["plot_name"]
        c["plot"](plot_path)
        print(f"[INFO] Plotted: {plot_path}")



if __name__ == "__main__":
    main()

'''
RT-DETR
python run_eval.py --gt /data/local/aschwab/data/real_colon_allPos_allNeg_onlyPatient/test_ann2.json --framework ultralytics --dataset realcolon --pred /home/stud/aschwab/master-thesis/missing/rtdetr_rc_640_s42/eval/filtered_predictions_rtdetr_s42_newww.json

YOLOv8
python run_eval.py --gt /data/local/aschwab/data/real_colon_allPos_allNeg_onlyPatient/test_ann2.json --framework ultralytics --dataset realcolon --pred /home/stud/aschwab/master-thesis/best_epochs/cross_dataset/y8m_sun_realcolon.json

YOLOv11
python run_eval.py --gt /data/local/aschwab/data/real_colon_allPos_allNeg_onlyPatient/test_ann2.json --framework ultralytics --dataset realcolon --pred /home/stud/aschwab/master-thesis/best_epochs/cross_dataset/rtdetr_sun_realcolon.json

Faster-RCNN
python run_eval.py --gt /data/local/aschwab/data/real_colon_allPos_allNeg_onlyPatient/test_ann2.json --framework detectron2 --dataset realcolon --pred /home/stud/aschwab/master-thesis/best_epochs/cross_dataset/fasterrcnn_piccolo_realcolon.json

python run_eval.py --gt /data/local/aschwab/data/piccolo_split/coco_annotations_test.json --framework detectron2 --dataset piccolo --pred /home/stud/aschwab/master-thesis/best_epochs/cross_dataset/fasterrcnn_realcolon_piccolo.json --conf 0.2

python run_eval.py --gt /data/local/aschwab/data/sun_split/coco_annotations_test.json --framework detectron2 --dataset sun --pred /home/stud/aschwab/master-thesis/best_epochs/cross_dataset/rtdert_sun_realcolon.json --conf 0.2

'''