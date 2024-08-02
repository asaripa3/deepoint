import numpy as np
import cv2
import torch
import dataset
from pathlib import Path
from tqdm import tqdm
import hydra
from torch.utils.data import DataLoader
from omegaconf import DictConfig, OmegaConf
from model import build_pointing_network
from draw_arrow import WIDTH, HEIGHT
from mmdet.apis import inference_detector, init_detector

@hydra.main(version_base=None, config_path="../conf", config_name="base")
def main(cfg: DictConfig) -> None:
    import logging

    logging.info(
        "Successfully loaded settings:\n"
        + "==================================================\n"
        + f"{OmegaConf.to_yaml(cfg)}"
        + "==================================================\n"
    )

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    if DEVICE == "cpu":
        logging.warning("Running DeePoint with CPU takes a long time.")

    assert (
        cfg.movie is not None
    ), "Please specify movie path as `movie=/path/to/movie.mp4`"

    assert (
        cfg.lr is not None
    ), "Please specify whether the pointing hand is left or right with `lr=l` or `lr=r`."

    assert cfg.ckpt is not None, "checkpoint should be specified for evaluation"

    cfg.hardware.bs = 2
    cfg.hardware.nworkers = 0
    ds = dataset.MovieDataset(cfg.movie, cfg.lr, cfg.model.tlength, DEVICE)
    dl = DataLoader(
        ds,
        batch_size=cfg.hardware.bs,
        num_workers=cfg.hardware.nworkers,
    )

    network = build_pointing_network(cfg, DEVICE)

    model_dict = torch.load(cfg.ckpt)["state_dict"]
    new_model_dict = dict()
    for k, v in model_dict.items():
        new_model_dict[k[6:]] = model_dict[k]
    model_dict = new_model_dict
    network.load_state_dict(model_dict)
    network.to(DEVICE)

    # Initialize MMDetection model for object detection
    detector = init_detector(cfg.obj_detect_cfg_file, cfg.obj_detect_checkpoint_file, device=DEVICE)

    Path("demo").mkdir(exist_ok=True)
    fps = 15
    out_green = cv2.VideoWriter(
        f"demo/{Path(cfg.movie).name}-processed-green-{cfg.lr}.mp4",
        cv2.VideoWriter_fourcc("m", "p", "4", "v"),
        fps,
        (WIDTH, HEIGHT),
    )
    out_greenred = cv2.VideoWriter(
        f"demo/{Path(cfg.movie).name}-processed-greenred-{cfg.lr}.mp4",
        cv2.VideoWriter_fourcc("m", "p", "4", "v"),
        fps,
        (WIDTH, HEIGHT),
    )

    prev_arrow_base = np.array((0, 0))

    for batch in tqdm(dl):
        result = network(batch)
        
        bs = batch["abs_joint_position"].shape[0]
        for i_bs in range(bs):
            joints = batch["abs_joint_position"][i_bs][-1].to("cpu").numpy()
            image = batch["orig_image"][i_bs].to("cpu").numpy() / 255

            direction = result["direction"][i_bs]
            prob_pointing = float(
                (result["action"][i_bs, 1].exp() / result["action"][i_bs].exp().sum())
            )
            print(f"{prob_pointing=}")

            ORIG_HEIGHT, ORIG_WIDTH = image.shape[:2]
            hand_idx = 9 if batch["lr"][i_bs] == "l" else 10
            if (joints[hand_idx] < 0).any():
                arrow_base = prev_arrow_base
            else:
                arrow_base = (
                    joints[hand_idx] / np.array((ORIG_WIDTH, ORIG_HEIGHT)) * 2 - 1
                )
                prev_arrow_base = arrow_base

            image_green = draw_arrow_on_image(
                image,
                (
                    arrow_base[0],
                    -arrow_base[1],
                    direction[0].cpu(),
                    direction[2].cpu(),
                    -direction[1].cpu(),
                ),
                dict(
                    acolor=(
                        0,
                        1,
                        0,
                    ),  # Green. OpenCV uses BGR
                    asize=0.05 * prob_pointing,
                    offset=0.02,
                ),
            )
            image_greenred = draw_arrow_on_image(
                image,
                (
                    arrow_base[0],
                    -arrow_base[1],
                    direction[0].cpu(),
                    direction[2].cpu(),
                    -direction[1].cpu(),
                ),
                dict(
                    acolor=(
                        0,
                        prob_pointing,
                        1 - prob_pointing,
                    ),  # Green to red. OpenCV uses BGR
                    asize=0.05 * prob_pointing,
                    offset=0.02,
                ),
            )

            # Perform object detection using MMDetection
            obj_detection_results = inference_detector(detector, image)
            # Compute the closest object to the arrow
            closest_object = compute_closest_object(obj_detection_results, direction, arrow_base)
            
            # Process detection results as needed
            # For example, you can filter out detections based on confidence scores
            
            # Visualize detection results on the image
            for class_id, class_result in enumerate(obj_detection_results):
                for obj in class_result:
                    if obj[4] > 0.5:  # Adjust threshold as needed
                        bbox = obj[:4].astype(np.int32)
                        cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                        class_name = MetadataCatalog.get(cfg_obj_detect.DATASETS.TRAIN[0]).thing_classes[class_id]
                        cv2.putText(image, class_name, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # You can use MMDetection's visualization tools or customize as needed
            # For example, you can draw bounding boxes and labels directly using OpenCV
            
            cv2.imshow("", image)
            cv2.waitKey(10)

            out_green.write((image_green * 255).astype(np.uint8))
            out_greenred.write((image_greenred * 255).astype(np.uint8))

    return

def draw_arrow_on_image(image, arrow_spec, kwargs):
    """
    Params:
    image: np.ndarray(height, width, 3), with dtype=float, value in the range of [0,1]
    arrow_spec, kwargs: options for render_frame
    Returns:
    image: np.ndarray(HEIGHT, WIDTH, 3), with dtype=float, value in the range of [0,1]
    """
    from draw_arrow import render_frame, WIDTH, HEIGHT

    ret_image = cv2.resize(image, (WIDTH, HEIGHT)).astype(float)
    img_arrow = render_frame(*arrow_spec, **kwargs).astype(float) / 255
    arrow_mask = (img_arrow.sum(axis=2) == 0.0).astype(float)[:, :, None]
    ret_image = arrow_mask * ret_image + (1 - arrow_mask) * img_arrow
    return ret_image

def compute_closest_object(instances, direction, arrow_base):
    min_distance = float('inf')
    closest_object = None
    direction_vec = np.array([direction[0].cpu(), direction[1].cpu(), direction[2].cpu()])

    for instance in instances:
        bbox = instance.pred_boxes.tensor[0].cpu().numpy()
        object_position = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2])
        object_direction_vec = object_position - arrow_base[:2]  
        distance = np.linalg.norm(object_direction_vec - direction_vec)

        if distance < min_distance:
            min_distance = distance
            closest_object = instance

    return closest_object

if __name__ == "__main__":
    main()
