import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path
import shutil

def run(cmd, dry=False):
    print(cmd)
    if not dry:
        return subprocess.Popen(cmd, shell=True)


def read_hostfile(path):
    with open(path) as f:
        lines = [l.strip() for l in f if l.strip()]
    return lines


def split_json_list(input_path, out_dir, num_splits):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("input bench file must be a JSON list")
    n = len(data)
    per = math.ceil(n / num_splits)
    for i in range(num_splits):
        part = data[i*per:(i+1)*per]
        out_f = out_dir / f"split_{i:02d}.json"
        with open(out_f, 'w', encoding='utf-8') as fo:
            json.dump(part, fo, ensure_ascii=False, indent=2)
    return n, per


def build_ssh_cmd(host, workdir, logs_dir, env, cmd_body, background=True):
    # env e.g. {"CUDA_VISIBLE_DEVICES":"0,1"}
    env_str = ' '.join(f"{k}={v}" for k,v in env.items()) if env else ""
    # ensure logs dir exists on remote; cmd_body is executed from workdir
    full = f"cd {workdir} && " \
            f"export PYTHONPATH=. && " \
            f"export http_proxy=http://star-proxy.oa.com:3128 && " \
            f"export https_proxy=http://star-proxy.oa.com:3128 && " \
            f"{env_str} {cmd_body}"
    if background:
        # redirect stdout/stderr to log; TASK_LOG passed in env to name log
        logname = env.get('TASK_LOG','task')
        full = f"nohup {full} > {logs_dir}/{logname}.log 2>&1 < /dev/null &"
    return f"ssh -f {host} '{full}'"


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--hostfile', required=True)
    p.add_argument('--bench-file', required=True, help='full bench json (list) to split and evaluate')
    p.add_argument("--img-dir-base-path", type=str, required=True)
    p.add_argument('--gpus-per-machine', type=int, default=8)
    p.add_argument('--gpus-per-task', type=int, default=2)
    p.add_argument('--num-tasks', type=int, default=None, help='optional override total tasks')
    p.add_argument('--use-scp', action='store_true', help='if set, scp code+splits to each host')
    p.add_argument('--python-bin', default='python3', help='python binary on remote host')
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--model-name', type=str, default='qwen25_vl_3b', help='model name to pass to eval.py')
    p.add_argument("--save-dir", type=str, default="outputs")
    p.add_argument("--save-prefix", type=str, default=None)
    p.add_argument("--recycle-test", action="store_true")
    p.add_argument("--rgbt-transed-img-path", type=str, default=None)
    p.add_argument("--rgbt-pair", action="store_true")
    p.add_argument("--use-prior", action="store_true")
    p.add_argument("--think-mode", type=str, default="no_think")
    args = p.parse_args()

    hosts = read_hostfile(args.hostfile)
    num_machines = len(hosts)
    tasks_per_machine = args.gpus_per_machine // args.gpus_per_task
    if tasks_per_machine < 1:
        print('gpus_per_task is larger than gpus_per_machine', file=sys.stderr); sys.exit(1)
    total_tasks = num_machines * tasks_per_machine
    if args.num_tasks is not None:
        total_tasks = args.num_tasks
    print(f"num_machines={num_machines}, tasks_per_machine={tasks_per_machine}, total_tasks={total_tasks}")

    custom_save_prefix = args.save_prefix or args.model_name
    if args.think_mode == "think":
        custom_save_prefix += "_thinking"
    elif args.think_mode == "auto":
        custom_save_prefix += "_auto_thinking"
    if args.recycle_test:
        custom_save_prefix += "_recycle"
    else:
        custom_save_prefix += "_single"
    if args.rgbt_pair:
        custom_save_prefix += "_rgbt"
    if args.use_prior:
        custom_save_prefix += "_prior"
        
    savedir = os.path.join(args.save_dir, custom_save_prefix).rstrip('/')
    print(f"save_dir={savedir}")
    local_savedir = Path(savedir)
    local_savedir.mkdir(parents=True, exist_ok=True)

    workdir = os.path.dirname(os.path.abspath(__file__))
    print(f"workdir={workdir}")

    splits_dir = local_savedir / 'splits'
    logs_dir = local_savedir / 'logs'
    outputs_dir = local_savedir / 'outputs'
    answer_dir = local_savedir / 'answers'
    for d in (splits_dir, logs_dir, outputs_dir, answer_dir):
        d.mkdir(parents=True, exist_ok=True)

    # 1) split bench json into total_tasks parts
    n, per = split_json_list(args.bench_file, splits_dir, total_tasks)
    print(f"bench size={n}, approx per task={per}")

    # # 2) optional scp code+splits to each host
    # if args.use_scp:
    #     if not Path('eval.py').exists():
    #         print('eval.py not found in current dir for scp mode', file=sys.stderr); sys.exit(1)
    #     for host in hosts:
    #         run(f"ssh {host} 'mkdir -p {savedir}/splits {savedir}/logs {savedir}/outputs'", dry=args.dry_run)
    #         run(f"scp eval.py {args.bench_file} {host}:{workdir}/", dry=args.dry_run)
    #         run(f"scp -r {splits_dir}/* {host}:{savedir}/splits/", dry=args.dry_run)
    #         # optionally copy other modules (custom model loaders) if exists
    #         for fname in ['qwen25_vl_infer.py','qwen3_vl_infer.py','internvl_3_infer.py','internvl_35_infer.py','glm_v_infer.py','llava_infer.py','keye_vl_infer.py','qwen3_infer.py']:
    #             if Path(fname).exists():
    #                 run(f"scp {fname} {host}:{workdir}/", dry=args.dry_run)

    # 2) launch processes on each machine
    global_id = 0
    procs = []
    for mid, host in enumerate(hosts):
        for t in range(tasks_per_machine):
            if global_id >= total_tasks:
                break
            # compute local GPU indices for this process (relative to machine)
            gpu_base = t * args.gpus_per_task
            gpu_ids = list(range(gpu_base, gpu_base + args.gpus_per_task))
            cuda_visible = ','.join(str(x) for x in gpu_ids)

            split_file = f"{splits_dir}/split_{global_id:02d}.json"
            # choose per-task save prefix (useful to merge later)
            save_prefix = args.save_prefix or args.model_name
            out_file_prefix = f"{save_prefix}_task{global_id:02d}"

            cmd_body = (
                f"{args.python_bin} {workdir}/bench_evaluate_dist.py "
                f"--bench-file {split_file} "
                f"--img-dir-base-path {args.img_dir_base_path} "
                f"--task-id {global_id} "
                f"--model-name {args.model_name} "
                f"--save-dir {outputs_dir} "
                f"--save-prefix {save_prefix} "
                f"--recycle-test {args.recycle_test} "
                f"--rgbt-transed-img-path {args.rgbt_transed_img_path} "
                f"--rgbt-pair {args.rgbt_pair} "
                f"--use-prior {args.use_prior} "
                f"--think-mode {args.think_mode} "
            )
            env = {'CUDA_VISIBLE_DEVICES': cuda_visible, 'TASK_LOG': out_file_prefix}

            ssh_cmd = build_ssh_cmd(host, workdir, logs_dir, env, cmd_body, background=True)
            p = run(ssh_cmd, dry=args.dry_run)
            procs.append(p)
            
            global_id += 1 

    print(f'All tasks launched. Monitor {logs_dir} or {outputs_dir} to check progress.')
    print(f'When finished, run: python merge_results.py --result_dir {args.save_dir} --save_prefix ***')



if __name__ == '__main__':
    main()
