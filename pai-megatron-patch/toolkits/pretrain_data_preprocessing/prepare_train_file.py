import json
import argparse
import os

from file import list_dirs_by_tag

def get_args():
    parser = argparse.ArgumentParser()
    group = parser.add_argument_group(title='input data')
    group.add_argument('--input_list',  nargs='+', help='List of items')

    group.add_argument(
        '--clear_file',
        type=bool,
        help='output file data.',
    ) 
    group.add_argument(
        '--postfix',
        type=str,
        default='.audio2',
        help='output file data.',
    ) 

    group.add_argument(
        '--output_path',
        type=str,
        help='output file data.',
    ) 
    args = parser.parse_args()

    return args

def save_data(save_path, all_dirs_path):
    """_summary_

    Args:
        save_path (_type_): _description_
        all_dirs_path (_type_): _description_
    """

    with open(save_path, 'w', encoding='utf-8') as fout:
        for path in all_dirs_path:
            res = {}
            res['path'] = path
            fout.write(json.dumps(res, ensure_ascii=False) + '\n')



def get_data_dir(args):
    """_summary_

    Args:
        args (_type_): _description_
    """
    
    all_dirs_path = []
    print(args.input_list)

    for input_path in args.input_list:
        print(input_path, flush=True)
        if not os.path.isdir(input_path):
            print(f'input_path dir not exit:{input_path}', flush=True)
        
        dirs_path = list_dirs_by_tag(input_path, args.postfix)
        all_dirs_path.extend(dirs_path)
    
    all_dirs_path = list(set(all_dirs_path))
    save_data(args.output_path, all_dirs_path)


def main():
    """_summary_
    """
    args = get_args()
    get_data_dir(args)

if __name__ == '__main__':
    main()
