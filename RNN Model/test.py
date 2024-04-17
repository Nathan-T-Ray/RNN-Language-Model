from tests.init_test import testCellInit
from tests.init_test import testModelInit
from tests.forward_test import testCellForward
from tests.forward_test import testModelForward
import argparse

parser = argparse.ArgumentParser(prog='test.py', 
                                 description='Test your RNN')


parser.add_argument('--test',
                    default='all',
                    nargs='?',
                    choices=['init_cell', 'init_model', 'forward_cell', 
                             'forward_model', 'all'],
                    help='run test of cell init and forward, model init '\
                    'and forward, or all (default: all)')

args = parser.parse_args()

if args.test == 'init_cell' or args.test == 'all':
    print('Testing RNNCell init...')
    testCellInit()
    print('Passed!')

if args.test == 'init_model' or args.test == 'all':
    print('Testing RNNModel init...')
    testModelInit()
    print('Passed!')

if args.test == 'forward_cell' or args.test == 'all':
    print('Testing RNNCell.forward()...')
    testCellForward()
    print('Passed!')

if args.test == 'forward_model' or args.test == 'all':
    print('Testing RNNModel.forward()...')
    testModelForward()
    print('Passed!')
