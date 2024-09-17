'''
Script implementing the argument parses for customTorchTools library, first created by PeterC 23-07-2024
'''
import argparse

# Define parser
parser = argparse.ArgumentParser(description='Argument parser for customTorchTools library')

# Define arguments
parser.add_argument('--verbose', action='store_true', help='Enable verbose mode')

# TBD

# Function to parse arguments
def getArgs():
    return parser.parse_args()



