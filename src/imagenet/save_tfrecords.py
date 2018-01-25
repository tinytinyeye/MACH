import util

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', nargs=2, type=str, required=True, help='input filename and output filename')
    params = P()
    args = parser.parse_args(namespace=params)
    input_filename = params.file[0]
    output_filename = params.file[1]
    save_to_tfrecords_dense(input_filename, output_filename)

    sys.exit()

if __name__ == '__main__':
  main()