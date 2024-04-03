import json
from collections import defaultdict
from random import shuffle
from typing import Optional
import os
from multiprocessing import Pool
from functools import partial

from tqdm import tqdm
import click
from ttts.gpt.text.cleaner import clean_text1, text_normalize

cleaned_text = []
cleaned_file_path = "data/all.txt.cleaned"
flag = "w"


def callback(line):
    global cleaned_text
    global cleaned_file_path
    global flag
    if line is not None:
        cleaned_text.append(line)
    #print(f"{len(cleaned_text)} {cleaned_file_path} {line}")
    
    if len(cleaned_text) >= 5000:
        with open(cleaned_file_path, flag, encoding="utf-8") as out_file:
            for line in cleaned_text:
                if line is not None:
                    out_file.write(line)
            cleaned_text.clear()
            flag = "a+"


def process_line(line, type):
    try:
        key, wav, spk, language, text = line.strip().split("|")
        if type == "clean":
            norm_text, phones = clean_text1(text, language)
            cleaned_line = "{}|{}|{}|{}|{}|{}\n".format(
                    key,
                    wav,
                    spk,
                    language,
                    norm_text,
                    " ".join(phones),
                )
        elif type == "norm":
            norm_text = text_normalize(text, language)
            cleaned_line = "{}|{}|{}|{}|{}\n".format(
                key,
                wav,
                spk,
                language,
                norm_text,
            )
        #print(cleaned_line, end="")
        return cleaned_line
    except Exception as e:
        print(line, end="")
        print(f"生成训练集和验证集时发生错误！, 详细信息: {e}")
        return None



def multiplication(num):
    return num*(num+1)



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
@click.option("--type", default="clean")  #clean|norm
def preprocess(
    transcription_path: str,
    cleaned_path: Optional[str],
    train_path: str,
    val_path: str,
    val_per_spk: int,
    max_val_total: int,
    clean: bool,
    num_processes: int,
    type: str,
):
    global cleaned_text
    global cleaned_file_path
    global flag
    if cleaned_path == "" or cleaned_path is None:
        cleaned_path = transcription_path + ".cleaned"

    cleaned_file_path = cleaned_path
    if clean:
        pool = Pool(processes=num_processes)
        with open(transcription_path, "r", encoding="utf-8") as trans_file:
            '''
            lines = trans_file.readlines()
            with Pool(processes=num_processes) as pool:
                for _ in tqdm(pool.imap_unordered(partial(process_line), lines), total=len(lines)):
                    pass
            '''
            if num_processes == 1:
                with open(cleaned_path, "w", encoding="utf-8") as out_file:
                    for line in tqdm(trans_file):
                        cleaned_line = process_line(line, type)
                        out_file.write(cleaned_line)
            else:
                for line in tqdm(trans_file):
                    pool.apply_async(func=process_line, args=(line, type,), callback=callback)
                pool.close()
                pool.join()
                with open(cleaned_path, flag, encoding="utf-8") as out_file:
                    for line in cleaned_text:
                        if line is not None:
                            out_file.write(line)

    transcription_path = cleaned_path
    spk_utt_map = defaultdict(list)
    spk_id_map = {}
    current_sid = 0

    with open(transcription_path, "r", encoding="utf-8") as f:
        audioPaths = set()
        countSame = 0
        countNotFound = 0
        for line in f.readlines():
            key, wav, spk, language, text, phones = line.strip().split("|")
            if wav in audioPaths:
                # 过滤数据集错误：相同的音频匹配多个文本，导致后续bert出问题
                print(f"重复音频文本：{line}")
                countSame += 1
                continue
            if not os.path.isfile(wav):
                # 过滤数据集错误：不存在对应音频
                print(f"没有找到对应的音频：{wav}")
                countNotFound += 1
                continue
            audioPaths.add(wav)
            spk_utt_map[spk].append(line)

            if spk not in spk_id_map.keys():
                spk_id_map[spk] = current_sid
                current_sid += 1
        print(f"总重复音频数：{countSame}，总未找到的音频数:{countNotFound}")

    train_list = []
    val_list = []

    for spk, wavs in spk_utt_map.items():
        shuffle(wavs)
        val_list += wavs[:val_per_spk]
        train_list += wavs[val_per_spk:]

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
