import os
import logging
import argparse
from config import cfg


def main():

    if not os.path.exists(cfg.GLOBAL.MODEL_SAVE_DIR):
        os.makedirs(cfg.GLOBAL.MODEL_SAVE_DIR)
    if not os.path.exists(os.path.join(cfg.GLOBAL.MODEL_SAVE_DIR, 'result')):
        os.makedirs(os.path.join(cfg.GLOBAL.MODEL_SAVE_DIR, 'result'))

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        filename=os.path.join(cfg.GLOBAL.MODEL_SAVE_DIR, 'record.log'),
                        filemode='w')


if __name__ == '__main__':
    main()
