#!/usr/bin/python3

import os
import time
import sys
import socket
import argparse
import subprocess


def _find_free_port():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Binding to port 0 will cause the OS to find an available port for us
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    # NOTE: there is still a chance the port could be taken by other processes.
    return port


def init_workdir():
    REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
    os.chdir(REPO_ROOT)
    sys.path.insert(0, REPO_ROOT)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SimSeg Launcher')
    parser.add_argument('--launch', type=str, default=None,
                        help='Specify launcher script.')
    parser.add_argument('--task', type=str, default='clip',
                        help='Specify task type.')
    parser.add_argument('--dist', type=int, default=1,
                        help='Whether start by torch.distributed.launch.')
    parser.add_argument('--nproc_per_node', type=int, default=1,
                        help='number of processes per node')
    parser.add_argument('--master_port', type=int, default=-1,
                        help='master port for communication')
    parser.add_argument('--no-local-log', action='store_true',
                        help='disable local logging')
    parser.add_argument('--exp_name', type=str, default=None,
                        help='Specify experiment name.')
    args, other_args = parser.parse_known_args()

    init_workdir()

    launch = f'simseg/tasks/{args.task}/train.py'
    if args.launch:
        launch = args.launch

    # Get port
    if args.master_port > 0:
        master_port = args.port
    else:
        master_port = _find_free_port()

    if args.dist >= 1:
        print(f'Start {launch} by torch.distributed.launch with port {master_port}!', flush=True)
        cmd = f'python3 -m torch.distributed.launch \
                --nproc_per_node={args.nproc_per_node} \
                --master_port={master_port} \
                {launch}'
    else:
        print(f'Start {launch}!', flush=True)
        cmd = f'python3 {launch}'

    for argv in other_args:
        cmd += f' {argv}'

    try:
        os.mkdir('./output')
    except:
        pass

    if args.no_local_log is False:
        if args.exp_name is not None:
            exp_name = args.exp_name
        else:
            exp_name = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
        with open(f'./output/{exp_name}_log.txt', 'wb') as f:
            proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
            while True:
                text = proc.stdout.readline()
                f.write(text)
                f.flush()
                sys.stdout.buffer.write(text)
                sys.stdout.buffer.flush()
                exit_code = proc.poll()
                if exit_code is not None:
                    break
    else:
        exit_code = subprocess.call(cmd, shell=True)
    sys.exit(exit_code)
