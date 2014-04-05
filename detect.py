#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import queue
import scipy.ndimage
import sys
import threading


def parse_args():
    parser = argparse.ArgumentParser(
            description='Cloud detector')
    parser.add_argument('-a', '--algorithm', default='hyta', metavar='ALGO',
            help='algorithm to use for detecting')
    parser.add_argument('-o', '--options',
            help='options for tweaking algorithm, seperated by comma')
    parser.add_argument('-m', '--mask',
            help='mask image for filtering')
    parser.add_argument('-s', '--source', default='input', dest='src',
            help='image file or directory containing it to be processed')
    parser.add_argument('-d', '--destination', default='output', dest='dest',
            help='directory containing processed images')
    parser.add_argument('-j', '--jobs', default=1, type=int, metavar='N',
            help='number of jobs to be run in parallel')
    return parser.parse_args()

def get_sources(src, src_list=None):
    SUPPORT_FORMAT = [
            '.jpg', '.jpeg',
            '.bmp',
            '.png',
            ]

    if src_list == None:
        src_list = []

    if os.path.isfile(src):
        if os.path.splitext(src.lower())[1] in SUPPORT_FORMAT:
            src_list.append(src)
    elif os.path.isdir(src):
        for f in os.listdir(src):
            get_sources(os.path.join(src, f), src_list)

    return src_list

def gen_dir(root, leaf, dest):
    leaf_dir = os.path.dirname(leaf[len(os.path.commonprefix([root, leaf]))+1:])
    target_dir = os.path.join(dest, leaf_dir)
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

def worker(Algo, options, mask, src_pool, input_root, output_root):
    while True:
        input_path = src_pool.get()
        output_path = os.path.join(output_root, input_path[len(
            os.path.commonprefix([input_path, input_root]))+1:])

        print(input_path)

        algo = Algo(
                scipy.ndimage.imread(input_path),
                options=options,
                mask=mask if mask else None)
        img = algo.run()
        scipy.misc.imsave(output_path, img)

        src_pool.task_done()

def main():
    args = parse_args()
    src_pool = queue.Queue()

    for src in get_sources(os.path.abspath(args.src)):
        src_pool.put(src)
        gen_dir(os.path.abspath(args.src), src, os.path.abspath(args.dest))

    if not src_pool.qsize():
        print('Error: input is empty', file=sys.stderr)
        return -1

    if args.algorithm.lower() == 'hyta':
        import hyta
        Algo = hyta.HYTA

    mask = scipy.ndimage.imread(args.mask) if args.mask else None

    for _ in range(args.jobs):
        thread = threading.Thread(
                target=worker,
                args=(Algo, args.options, mask, src_pool,
                    os.path.abspath(args.src if os.path.isdir(args.src) else
                        os.path.dirname(args.src)),
                    os.path.abspath(args.dest)),
                daemon=True)
        thread.start()

    src_pool.join()

if __name__ == '__main__':
    sys.exit(main())
