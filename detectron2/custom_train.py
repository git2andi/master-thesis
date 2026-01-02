# Suppress torch.amp warning
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="torch.amp.autocast_mode")
warnings.filterwarnings("ignore", message=".*torch.cuda.amp.autocast.*")

# Code
import os
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer, default_argument_parser, launch, default_setup
from detectron2.config import get_cfg
from detectron2.evaluation import COCOEvaluator
from detectron2.checkpoint import DetectionCheckpointer

ROOT_640 = "/data/local/aschwab/data/real_colon_allPos_allNeg_onlyPatient"
register_coco_instances("realcolon_patient_train", {}, f"{ROOT_640}/train_ann.json", f"{ROOT_640}/train_images")
register_coco_instances("realcolon_patient_val",   {}, f"{ROOT_640}/validation_ann.json",   f"{ROOT_640}/validation_images")
register_coco_instances("realcolon_patient_test",  {}, f"{ROOT_640}/test_ann.json",  f"{ROOT_640}/test_images")

ROOT_300 = "/data/local/aschwab/data/real_colon_allPos_allNeg"
register_coco_instances("realcolon_all_train", {}, f"{ROOT_300}/train_ann.json", f"{ROOT_300}/train_images")
register_coco_instances("realcolon_all_val",   {}, f"{ROOT_300}/validation_ann.json",   f"{ROOT_300}/validation_images")
register_coco_instances("realcolon_all_test",  {}, f"{ROOT_300}/test_ann.json",  f"{ROOT_300}/test_images")


ROOT_PICCOLO = "/data/local/aschwab/data/piccolo_split"
register_coco_instances("piccolo_split_train", {}, f"{ROOT_PICCOLO}/coco_annotations_train.json", f"{ROOT_PICCOLO}/images/train")
register_coco_instances("piccolo_split_val",   {}, f"{ROOT_PICCOLO}/coco_annotations_val.json",   f"{ROOT_PICCOLO}/images/val")
register_coco_instances("piccolo_split_test",  {}, f"{ROOT_PICCOLO}/coco_annotations_test.json",  f"{ROOT_PICCOLO}/images/test")

ROOT_SUN = "/data/local/aschwab/data/sun_split"
register_coco_instances("sun_split_train", {}, f"{ROOT_SUN}/coco_annotations_train.json", f"{ROOT_SUN}/images/train")
register_coco_instances("sun_split_val",   {}, f"{ROOT_SUN}/coco_annotations_val.json",   f"{ROOT_SUN}/images/val")
register_coco_instances("sun_split_test",  {}, f"{ROOT_SUN}/coco_annotations_test.json",  f"{ROOT_SUN}/images/test")


class ThesisTrainer(DefaultTrainer):

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, output_dir=output_folder, tasks=("bbox",))

def setup(args):
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)
    # eval
    if args.eval_only:
        model = ThesisTrainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = ThesisTrainer.test(cfg, model)
        return res

    # train
    trainer = ThesisTrainer(cfg) 
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()

if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )