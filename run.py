#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Messenger Runner.
"""
from parlai.core.params import ParlaiParser
#from parlai.chat_service.services.messenger.messenger_manager import MessengerManager
from chat_service_manager import ChatServiceManager
import parlai.chat_service.utils.config as config_utils


SERVICE_NAME = 'Chat'


def setup_args():
    # ?????
    """
    Set up args.
    """
    parser = ParlaiParser(False, False)
    parser.add_parlai_data_path()
    parser.add_messenger_args()
    return parser.parse_args()


def run(opt):
    """
    Run ChatServiceManager.
    """
    opt['service'] = SERVICE_NAME
    manager = ChatServiceManager(opt)
    try:
        manager.start_task()
    except BaseException:
        raise
    finally:
        manager.shutdown()


if __name__ == '__main__':
    opt = setup_args()
    config_path = opt.get('config_path')
    config = config_utils.parse_configuration_file(config_path)
    opt.update(config['world_opt'])
    opt['config'] = config
    run(opt)
