# split up the dataset so I don't have to read all 33k genomes at once

import os


if __name__ == "__main__":
    raise Exception("don't re-run this unless you want to overwrite all of them")

    data_dir = "ncbi_dataset/data/"
    input_fp = os.path.join(data_dir, "genomic.fna")
    with open(input_fp) as f:
        for line in f.readlines():
            line = line.strip()
            if line.startswith(">"):
                # new genome, the part right after the > is the code number, keep this line in each file as well
                virus_id = line.split()[0][1:]
                output_fp = os.path.join(data_dir, "individual_genomes", virus_id + ".txt")
                print("new output file: {}".format(output_fp))
            # now, whether we changed output_fp or not, append line to file
            with open(output_fp, "a") as f_out:
                f_out.write(line + "\n")
    print("done")
