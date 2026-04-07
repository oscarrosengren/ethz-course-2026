python scripts/record_teleop_demos.py

2026-03-23_18-51-40

cp -r datasets/raw/single_cube/teleop/2026-03-23_18-51-40 datasets/raw/single_cube/curated_teleop/
cp -r datasets/raw/single_cube/teleop/2026-03-23_20-04-30 datasets/raw/single_cube/curated_teleop/

cp -r datasets/raw/single_cube/teleop/2026-03-23_15-28-52 datasets/raw/single_cube/curated_teleop/


python scripts/compute_actions.py   --datasets-dir ./datasets/raw/single_cube/curated_teleop   --action-space ee
python scripts/compute_actions.py   --datasets-dir ./datasets/raw/single_cube/teleop/2026-03-24_21-17-50 --action-space ee

python scripts/train.py     --zarr ./datasets/processed/single_cube/processed_ee_xyz.zarr     --policy obstacle     --state-keys state_ee_xyz state_gripper "state_cube[:3]" state_obstacle     --action-keys action_ee_xyz action_gripper --name 2026-03-24_21-17-50


python scripts/eval.py     --checkpoint ./checkpoints/single_cube/best_model_2026-03-24_21-17-50_ee_xyz_obstacle.pt --adversarial
__
python student_eval/run_eval.py --exercise 1 --checkpoint checkpoints/single_cube/best_model_ee_xyz_obstacle.pt

python student_eval/run_eval.py --exercise 1 --checkpoint checkpoints/single_cube/best_model_2026-03-24_21-17-50_ee_xyz_obstacle.pt


2.

python scripts/dagger_eval.py     --checkpoint ./checkpoints/single_cube/best_model_2026-03-24_21-17-50_ee_xyz_obstacle.pt


python scripts/compute_actions.py   --datasets-dir ./datasets/raw/single_cube/dagger_curated5 --action-space ee --output ./datasets/raw/single_cube/dagger_curated5/processed.zarr

python scripts/train.py     --zarr ./datasets/raw/single_cube/dagger_curated5/processed.zarr     --policy obstacle     --state-keys state_ee_xyz state_gripper "state_cube[:3]" state_obstacle     --action-keys action_ee_xyz action_gripper --name 2026-03-25_09-06-26_dagger3


python scripts/dagger_eval.py     --checkpoint ./checkpoints/single_cube/best_model_ee_xyz_obstacle_dagger46ep.pt

python student_eval/run_eval.py --exercise 2 --checkpoint checkpoints/single_cube/best_model_ee_xyz_obstacle_dagger50ep.pt


3.

python scripts/compute_actions.py \
  --action-space ee \
  --datasets-dir ./datasets/raw/multi_cube

python scripts/train_multicube.py \
  --zarr datasets/processed/multi_cube/processed_ee_xyz.zarr \
  --policy multitask \
  --epochs 150 \
  --state-keys original_pos_cube_red original_pos_cube_green original_pos_cube_blue state_ee_xyz goal_pos state_goal state_gripper


  python scripts/train_multicube.py   --zarr datasets/processed/multi_cube/processed_ee_xyz.zarr   --policy multitask   --epochs 100   --state-keys original_pos_cube_red original_pos_cube_green original_pos_cube_blue state_ee_xyz goal_pos state_goal state_gripper


 python scripts/eval.py \
  --checkpoint checkpoints/multi_cube/best_model_multicube_v1.pt \
  --multicube \
  --num-episodes 20


  python student_eval/run_eval.py --exercise 3 --checkpoint checkpoints/multi_cube/best_model_multicube_v1.pt
