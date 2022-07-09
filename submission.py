import argparse
import torch
from tqdm import tqdm
import model.model as module_arch
from parse_config import ConfigParser

import pandas as pd
import numpy as np


def main(config):
    logger = config.get_logger('test')

    # setup data_loader instances
    # data_loader = getattr(module_data, config['data_loader']['type'])(
    #     config['data_loader']['args']['data_dir'],
    #     batch_size=512,
    #     shuffle=False,
    #     validation_split=0.2,
    #     training=False,
    #     num_workers=2
    # )

    # build model architecture
    model = config.init_obj('arch', module_arch)
    logger.info(model)

    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    checkpoint = torch.load(config.resume)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        print('Making submission file.......')
        X = pd.read_csv(args.data_path)
        X = X.to_numpy().reshape(-1, 1, 28, 28)
        X = X.repeat(3, 1)
        X = torch.FloatTensor(X) / 255
        X = X.to(device)

        outs = model(X)

        _, preds = torch.max(outs, dim=1)
        
        cols = ['ImageId', 'Label']
        sub = pd.DataFrame(columns=cols)
        sub['ImageId'] = np.arange(1, preds.size(0) + 1)
        sub['Label'] = preds.numpy()
        sub.to_csv('submission.csv', index=False)
        print('Done.')


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-p', '--path', default=None, type=str,
                      help='test path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    config = ConfigParser.from_args(args)
    main(config)
