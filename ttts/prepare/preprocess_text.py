import json
from collections import defaultdict
from random import shuffle
from typing import Optional
import os
from multiprocessing import Pool

from tqdm import tqdm
import click
from ttts.gpt.text.cleaner import clean_text

cleaned_text = []


def callback(line):
    global cleaned_text
    cleaned_text.append(line)


def clean_text(line):
    try:
        utt, spk, language, text = line.strip().split("|")
        norm_text, phones = clean_text(text, language)
        cleaned_line = "{}|{}|{}|{}|{}\n".format(
                utt,
                spk,
                language,
                norm_text,
                " ".join(phones),
            )
        return cleaned_line
    except Exception as e:
        print(line)
        print(f"生成训练集和验证集时发生错误！, 详细信息:\n{e}")


@click.command()
@click.option(
    "--transcription-path",
    default="data/all.txt",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
)
@click.option("--cleaned-path", default="data/all.txt.cleaned")
@click.option("--train-path", default="filelists/train.list")
@click.option("--val-path", default="filelists/val.list")
@click.option("--val-per-spk", default=5)
@click.option("--max-val-total", default=2000)
@click.option("--clean/--no-clean", default=True)
@click.option("--num-processes", default=5)
def preprocess(
    transcription_path: str,
    cleaned_path: Optional[str],
    train_path: str,
    val_path: str,
    val_per_spk: int,
    max_val_total: int,
    clean: bool,
    num_processes: int,
):
    if cleaned_path == "" or cleaned_path is None:
        cleaned_path = transcription_path + ".cleaned"

    global cleaned_text
    if clean:
        with open(transcription_path, "r", encoding="utf-8") as trans_file:
            lines = trans_file.readlines()
            # print(lines, ' ', len(lines))
            with Pool(processes=num_processes) as pool:
                if len(lines) != 0:
                    for line in tqdm(lines):
                        pool.apply_async(func=clean_text, args=(line,), callback=callback)
                        if lines % 2000 == 0:
                            with open(cleaned_path, "w", encoding="utf-8") as out_file:
                                for line in cleaned_text:
                                    out_file.write(line)
                            cleaned_text.clear()

    transcription_path = cleaned_path
    spk_utt_map = defaultdict(list)
    spk_id_map = {}
    current_sid = 0

    with open(transcription_path, "r", encoding="utf-8") as f:
        audioPaths = set()
        countSame = 0
        countNotFound = 0
        for line in f.readlines():
            utt, spk, language, text, phones = line.strip().split("|")
            if utt in audioPaths:
                # 过滤数据集错误：相同的音频匹配多个文本，导致后续bert出问题
                print(f"重复音频文本：{line}")
                countSame += 1
                continue
            if not os.path.isfile(utt):
                # 过滤数据集错误：不存在对应音频
                print(f"没有找到对应的音频：{utt}")
                countNotFound += 1
                continue
            audioPaths.add(utt)
            spk_utt_map[spk].append(line)

            if spk not in spk_id_map.keys():
                spk_id_map[spk] = current_sid
                current_sid += 1
        print(f"总重复音频数：{countSame}，总未找到的音频数:{countNotFound}")

    train_list = []
    val_list = []

    for spk, utts in spk_utt_map.items():
        shuffle(utts)
        val_list += utts[:val_per_spk]
        train_list += utts[val_per_spk:]

    if len(val_list) > max_val_total:
        train_list += val_list[max_val_total:]
        val_list = val_list[:max_val_total]

    traindir = os.path.dirname(train_path)
    if not os.path.exists(traindir):
        os.makedirs(traindir)  
    with open(train_path, "w", encoding="utf-8") as f:
        for line in train_list:
            f.write(line)

    valdir = os.path.dirname(val_path)
    if not os.path.exists(valdir):
        os.makedirs(valdir)  
    with open(val_path, "w", encoding="utf-8") as f:
        for line in val_list:
            f.write(line)
    print("训练集和验证集生成完成！")


if __name__ == "__main__":
    preprocess()
