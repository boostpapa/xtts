
import torch
import torchaudio
import torch.nn.functional as F
from omegaconf import OmegaConf
import argparse
import os

from ttts.vqvae.xtts_dvae import DiscreteVAE
from ttts.gpt.model import UnifiedVoice
from ttts.vocoder.feature_extractors import MelSpectrogramFeatures
from ttts.utils.checkpoint import load_checkpoint

from ttts.diffusion.aa_model import AA_diffusion
from ttts.utils.diffusion import SpacedDiffusion, space_timesteps, get_named_beta_schedule
from ttts.diffusion.aa_model import do_spectrogram_diffusion, normalize_tacotron_mel
from vocos import Vocos

from ttts.gpt.text.cleaner import clean_text1, text_normalize, text_to_sentences
from ttts.gpt.voice_tokenizer import VoiceBpeTokenizer
import sentencepiece as spm
from ttts.utils.byte_utils import byte_encode
from ttts.utils.utils import tokenize_by_CJK_char


class TTSModel(torch.nn.Module):

    def __init__(self, args):
        super().__init__()
        self.cfg = OmegaConf.load(args.config)
        self.dvae = DiscreteVAE(**self.cfg.vqvae)
        dvae_path = self.cfg.dvae_checkpoint
        load_checkpoint(self.dvae, dvae_path)
        self.dvae.eval()
        print(">> vqvae weights restored from:", dvae_path)

        ## load gpt model ##
        self.gpt = UnifiedVoice(**self.cfg.gpt)
        gpt_path = self.cfg.gpt_checkpoint
        load_checkpoint(self.gpt, gpt_path)
        self.gpt.eval()
        print(">> GPT weights restored from:", gpt_path)
        self.gpt.post_init_gpt2_config(use_deepspeed=False, kv_cache=False, half=args.fp16)

        ## load diffusion model ##
        self.diffusion = AA_diffusion(self.cfg)
        diffusion_path = self.cfg.diffusion_checkpoint
        load_checkpoint(self.diffusion, diffusion_path)
        self.diffusion.eval()
        print(">> diffusion weights restored from:", diffusion_path)

        self.vocos = Vocos.from_pretrained(self.cfg.vocoder_model)
        self.diffuser = SpacedDiffusion(use_timesteps=space_timesteps(1000, [15]), model_mean_type='epsilon',
                                   model_var_type='learned_range', loss_type='mse',
                                   betas=get_named_beta_schedule('linear', 1000),
                                   conditioning_free=True, ramp_conditioning_free=False, conditioning_free_k=2., sampler='dpm++2m')

    def forward(self, cond_mel: torch.Tensor,
                text_tokens: torch.IntTensor, text_lens: torch.IntTensor):
        batch_size = text_tokens.shape[0]
        cond_mel_lengths = torch.tensor([cond_mel.shape[-1]]*batch_size, device=text_tokens.device)
        print(cond_mel_lengths)
        print(text_tokens)
        codes = self.gpt.inference_speech(cond_mel,
                                        text_tokens,
                                        cond_mel_lengths=cond_mel_lengths,
                                        do_sample=True,
                                        top_p=.8,
                                        top_k=30,
                                        temperature=0.5,
                                        num_return_sequences=1,
                                        length_penalty=0.0,
                                        num_beams=3,
                                        repetition_penalty=10.0,
                                        max_generate_length=600)
        #codes = codes[:, :-2]
        print(codes)
        print(f"codes shape: {codes.shape}")
        code_lens = []
        for code in codes:
                if self.cfg.gpt.stop_mel_token not in code:
                    code_lens.append(len(code))
                else:
                    #len_ = code.cpu().tolist().index(8193)+1
                    len_ = (code == 8193).nonzero(as_tuple=False)[0]+1
                    code_lens.append(len_-2)
        code_lens = torch.LongTensor(code_lens).cuda()
        print(f"code len: {code_lens}")

        latent = self.gpt(cond_mel,
                        text_tokens,
                        text_lens,
                        codes,
                        code_lens*self.gpt.mel_length_compression,
                        cond_mel_lengths=cond_mel_lengths,
                        return_latent=True, clip_inputs=False).transpose(1, 2)
        print(f"latent shape: {latent.shape}")

        diffusion_conditioning = normalize_tacotron_mel(cond_mel)
        upstride = self.gpt.mel_length_compression / 256
        mel = do_spectrogram_diffusion(self.diffusion, self.diffuser, latent, diffusion_conditioning,
                                       upstride, temperature=1.0)
        print(f"mel shape: {mel.shape}")
        wav = self.vocos.decode(mel)
        print(f"wav shape: {wav.shape}")
        wav = 32767 / max(0.01, torch.max(torch.abs(wav))) * 1.0 * wav
        torch.clip(wav, -32767.0, 32767.0)
        wavs = []
        for w, len_ in zip(wav, code_lens):                                                                                                                                                                                                                         
                w = w[:len_ * self.gpt.mel_length_compression]
                wavs.append(w.unsqueeze(0))
        wav = torch.cat(wavs, dim=1)
        return wav


def get_args():
    parser = argparse.ArgumentParser(description='export your script model')
    parser.add_argument('--config', type=str, required=True, help='config file')
    parser.add_argument('--fp16',
                        action='store_true',
                        help='whether to export fp16 model, default false')
    # args = parser.parse_args()
    args, _ = parser.parse_known_args()
    return args


cond_audio = '/speechwork/users/wd007/tts/xtts2/gpt/s2_v3/bzshort/split2_J5_TTS_女性_愤怒_4.wav'
cond_audio = '/speechwork/users/wd007/tts/xtts2/gpt/s2_v3/bzshort/erbaHappyLow.wav'
cond_audio = '/speechfs01/data/tts/opensource/baker_BZNSYP/Wave/002650.wav'
cond_audio = '/speechwork/users/wd007/tts/xtts2/gpt/s2_v3/bzshort/guzong.wav'
cond_audio = '/speechwork/users/wd007/tts/xtts2/gpt/s2_v3/bzshort/5639-40744-0020.wav'
cond_audio = '/speechwork/users/wd007/tts/data/bilibili/manual/22all/22/speak/ZH/wav/22-all_speak_ZH_YouYou_emotion_ZH_309自豪_20230613_20230627-0150729-0155966.wav'
cond_audio = '/cfs/import/tts/opensource/baker_BZNSYP/BZNSYP/Wave_22k/003261.wav'
cond_audio = '/speechwork/users/wd007/tts/xtts2/gpt/s2_v3/bzshort/xialiu-chuanpu.wav'
cond_audio = '/speechwork/users/wd007/tts/fishspeech/academiCodec/s1/test_wav/live_audio2_57.wav'
cond_audio = '/cfs/import/tts/opensource/baker_BZNSYP/BZNSYP/Wave_22k/008669.wav'
cond_audio = '/speechwork/users/wd007/tts/xtts2/gpt/s2_v3/bzshort/jincheng_dongbei.wav'
cond_audio = '/speechwork/users/wd007/tts/xtts2/gpt/s2_v3/bzshort/bjincheng.wav'
cond_audio = '/speechwork/users/wd007/tts/yourtts/zhibo/live_audio2/wavs/live_audio2_741.wav'
cond_audio = '/speechwork/users/wd007/tts/xtts2/gpt/s2_v3/bzshort/zhoujielun.wav'
cond_audio = '/speechwork/users/wd007/tts/xtts2/gpt/s2_v3/bzshort/xuyuanshen.wav'
cond_audio = '/speechwork/users/wd007/tts/xtts2/gpt/s2_v3/bzshort/guanguan.wav'
cond_audio = '/speechwork/users/wd007/tts/fishspeech/academiCodec/s1/test_wav/dengwei.wav'
cond_audio = '/speechwork/users/wd007/tts/xtts2/gpt/s2_v3/bzshort/ham_male1.wav'
cond_audio = '/speechwork/users/wd007/tts/xtts2/gpt/s2_v3/bzshort/magi.wav'
cond_audio = '/speechwork/users/wd007/tts/fishspeech/academiCodec/s1/test_wav/taylor1.wav'
cond_audio = '/speechwork/users/wd007/tts/xtts2/gpt/s2_v3/bzshort/duyujiao.wav'
cond_audio = '/speechwork/users/wd007/tts/fishspeech/academiCodec/s1/test_wav/dengwei1.wav'
cond_audio = '/speechwork/users/wd007/tts/data/bilibili/manual/MeiHuo/MeiHuo/speak/ZH/wav/002266.wav'
cond_audio = '/speechwork/users/wd007/tts/xtts2/gpt/s2_v3/bzshort/erba.wav'
cond_audio = '/speechwork/users/wd007/tts/xtts2/gpt/s2_v3/bzshort/格恩猫-demo.wav'
cond_audio = '/speechwork/users/wd007/tts/xtts2/gpt/s2_v3/bzshort/格恩猫.wav'
cond_audio = '/speechwork/users/wd007/tts/xtts2/gpt/s2_v3/bzshort/孙笑川.wav'
cond_audio = '/speechwork/users/wd007/tts/xtts2/gpt/s2_v3/bzshort/永雏塔菲.wav'
cond_audio = '/audionas/users/xuanwu/tts/data/bilibili/pgc/xialei/process/flac_cut/xialei3_262.flac'
cond_audio = '/speechwork/users/wd007/tts/xtts2/gpt/s2_v3/bzshort/siyi.wav'
cond_audio = '/speechwork/users/wd007/tts/xtts2/gpt/s2_v3/bzshort/xialei_vc.wav'
cond_audio = '/speechwork/users/wd007/tts/xtts2/gpt/s2_v3/bzshort/j5_angry_2.wav'
cond_audio = '/speechwork/users/wd007/tts/data/bilibili/manual/jiachun/jiachun/speak/ZH/wav/00000001_000019.wav'
cond_audio = '/speechwork/users/wd007/tts/data/opensource/baker_BZNSYP/Wave/003668.wav'
cond_audio = '/speechwork/users/wd007/tts/xtts2/gpt/s2_v3/bzshort/manhua1.wav'
cond_audio = '/speechwork/users/wd007/tts/xtts2/gpt/s2_v3/bzshort/tim1.wav'
cond_audio = '/speechwork/users/wd007/tts/xtts2/gpt/s2_v3/bzshort/lks1.wav'
cond_audio = '/speechwork/users/wd007/tts/xtts2/gpt/s2_v3/bzshort/luoxiang1.wav'
cond_audio = '/speechwork/users/wd007/tts/xtts2/gpt/s2_v3/bzshort/jianbaosao1.wav'
cond_audio = '/speechwork/users/wd007/tts/data/bilibili/manual/22all/22/speak/ZH/wav/22-all_speak_ZH_YouYou_emotion_ZH_309自豪_20230613_20230627-0150729-0155966.wav'
cond_audio = '/speechwork/users/wd007/tts/xtts2/gpt/s2_v3/bzshort/xueli.wav'
cond_audio = '/audionas/users/xuanwu/tts/data/bilibili/pgc/xialei/process/flac_cut/xialei3_19.flac'
cond_audio = '/speechwork/users/wd007/tts/xtts2/gpt/s2_v3/bzshort/houcuicui1.wav'
cond_audio = '/speechwork/users/wd007/tts/xtts2/gpt/s2_v3/bzshort/yueyue.wav'
cond_audio = '/speechwork/users/wd007/tts/xtts2/gpt/s2_v3/bzshort/少女_甜美_哭泣_02.wav'
cond_audio = '/speechwork/users/wd007/tts/xtts2/gpt/s2_v3/bzshort/LTY-10s.wav'
cond_audio = '/speechwork/users/wd007/tts/xtts2/gpt/s2_v3/bzshort/010100010068.wav'
cond_audio = '/speechwork/users/wd007/tts/xtts2/gpt/s2_v3/bzshort/shujuan.wav'
cond_audio = '/speechwork/users/wd007/tts/xtts2/gpt/s2_v3/bzshort/p_0.wav'
cond_audio = '/cfs/import/tts/opensource/LJSpeech/LJSpeech-1.1/wavs/LJ002-0145.wav'
cond_audio = '/speechwork/users/wd007/tts/xtts2/gpt/s2_v3/bzshort/东雪莲.wav'
cond_audio = '/audionas/users/xuanwu/tts/data/bilibili/auto/cmn_tts_20230101_20231120_v3/select/flac_cut/entertainment_222884913_970197451_507259965_11.flac'
cond_audio = '/speechwork/users/wd007/tts/xtts2/gpt/s2_v3/bzshort/seed_tts_en1.wav'
cond_audio = '/speechwork/users/wd007/tts/xtts2/gpt/s2_v3/bzshort/seed_tts_cn2.wav'
cond_audio = '/speechwork/users/wd007/tts/xtts2/gpt/s2_v3/bzshort/erbaHappyLow.wav'
cond_audio = '/speechwork/users/wd007/tts/xtts2/gpt/s2_v3/bzshort/split2_J5_TTS_女性_愤怒_4.wav'
cond_audio = '/speechwork/users/wd007/tts/yourtts/mix_cn/prompt/chenrui/chenrui2.wav'
cond_audio = '/speechwork/users/wd007/tts/fishspeech/academiCodec/s1/test_wav/chenrui1.wav'
cond_audio = '/speechwork/users/wd007/tts/xtts2/gpt/s2_v3/bzshort/sange1.wav'

text = "是谁给你的胆量这么跟我说话，嗯? 是你的灵主还是你的伙伴？听着，没用的小东西，这里是城下街，不是过家家的学院！停下你无聊至极的喋喋不休，学着用城下街的方式来解决问题！"
text = "历史将永远记住同志们的杰出创造和奉献，党和人民感谢你们。"
text = "但我们的损失由谁来补?"
text = "那个等会儿有时间吧那个那个下午三哥要拉个会,跟大家一起对一下下半年规划."
text = "玥玥爱土豆，爱爸爸妈妈，爱奶奶，喜欢去迪斯尼玩，喜欢癞蛤蟆"
text = "traced the progress of prison architecture from the days when the jail was the mere annexe of the baronial or episcopal castle"
text = "Then leaving the corpse within the house they go themselves to and fro about the city and beat themselves, with their garments bound up by a girdle."
text = "Thus did this humane and right minded father comfort his unhappy daughter, and her mother embracing her again, did all she could to soothe her feelings."
text = "你可不要被它名字迷惑，它不是角龙，而是生活在白垩纪晚期的肿头龙家族中的一员"
text = "让您和您的伴侣可以在酒店内享受美食的同时，感受到酒店的温馨和舒适。"
text = "二零一九年前后, 媒体报道了重刑犯出狱后从事殡葬行业的新闻, 这些新闻让殡葬师这个事业引发了一波关注."
text = "蚌埠的蚌和河蚌的蚌怎么念,图穷匕见."
text = "武士狐媚,我来世一定要身为一只猫."
text = "蚌埠的蚌和河蚌的蚌怎么念,图穷匕见."
text = "其次是双人下午茶项目，这个项目包含了精美的下午茶套餐, 让您和您的伴侣可以在酒店内享受美食的同时，感受到酒店的温馨和舒适。"
text = "大家好 B A I 开放平台上线了声音复刻功能,我的声音呢就是通过大模型做出来的,我们很容易达到一个一百万播放的目标啊,快来平台体验吧! 千万别被剑角龙顶到，剑角龙又名顶角龙，意为有角的头顶，顾名思义它的头上长了一个头盔。你可不要被它名字迷惑，它不是角龙，而是生活在白垩纪晚期的肿头龙家族中的一员。其次是双人下午茶项目，这个项目包含了精美的下午茶套餐, 让您和您的伴侣可以在酒店内享受美食的同时，感受到酒店的温馨和舒适。once upon a time, there lived in a certain village. a little country girl, "
text = "HE SAT DOWN WEAK BEWILDERED AND ONE THOUGHT WAS UPPERMOST ZORA."
text = "即便是北上广深等供应偏紧的一线城市, 明确了发展目标和重点任务, 新批复了七只创投基金的设立方案."
text = "We present Open-Sora, an initiative dedicated to efficiently produce high-quality video and make the model, tools and contents accessible to all. By embracing open-source principles, "
text = "好奇的灵魂渴望突破自己,去寻找另外的世界"
text = "那个等会儿有时间吧那个那个下午三哥要拉个会,跟大家一起对一下下半年规划.如果大家时间都 ok 的话,就安排在今天下午 review 了.然后可能得辛苦 harry 老师帮忙组织一下团建的事,嗯也不知道安排怎么样了,今天下午我要放假了,接下来一周就不在公司,大家新年快乐!"
text = "斯塔西亚，不要睡，睁开眼睛, 你不是一直想回到洛伦去吗？洛伦啊，你的故乡，那里有海，有草原，有脆脆绵绵的朝夕果…你在听吗，呜呜…斯塔西亚，求你…睁开眼睛看看。"
text = "不得不说，人生真是充满了意外，而降临在我头上的，总是比较差的那一种。这件事，说起来还挺让人不好意思的……今天下楼的时候，突然有个人冲过来撞到了我，我一个没站稳，脚就扭伤了。重点不是这个，重点是我刚才去了医院，医生说，我的脚伤比较严重，三个月都不能剧烈运动，三个月啊，那我们的滑雪计划怎么办！我们之前计划了好久，想要下周去滑雪，但谁能想到，好好的计划被一个突然冲出来的路人破坏了。我现在还在悔恨，要是今天没有出门就好了。等到下次，可能就没有现在这种期待的心情了。最重要的是，你为了下周特地空出了时间，如果去不了，这也太遗憾了。真的吗，那我这算是……因祸得福了？你说你要来照顾我，而不是来看我一眼就走，这代表，你会把下周所有的时间都给我，虽然脚还是很疼，但一想到这件事，我就觉得很开心。说到这个，我还有一个小小的请求，下周，你可不可以搬过来和我一起住啊？我没有别的意思，只是不想让你浪费往返的时间。还有，我受伤了，心理很脆弱，如果不能时刻都看到你，我怕我会忍不住崩溃，你不会想看到这一幕发生的，对吧？你不用准备什么东西的，我这边都有！而且我只是脚受伤了，又不是完全不能自理，我想让你来，只是想跟你一起度过未来一周的时间。"
text = "俯下身子尽量靠近一点，但不能碰到我的鼻尖。对保持这个姿势, 告诉我你在我的身上闻到了什么味道. 这样就觉得难了, 但这次服从性测试实验是你自己要做的. 知道了我会加快一点速度, 现在我命令你看着我的眼睛不准移开, 然后亲吻我. 俯下身子尽量靠近一点, 但不能碰到我的鼻尖, 对,   保持这个姿势, 告诉我你在我的身上闻到了什么味道. 这样就觉得难了, 但这次服从性测试实验是你自己要参与的. 知道了我会加快一些速度, 现在我命令你看着我的眼睛不准移开, 然后亲吻我, 怎么突然凑这么近, 我的脸上是有什么东西吗? 耳朵那边有些红, 想帮我看看没有不舒服, 只是耳朵那边有些敏感, 对,特别是那里, 啊轻点!"
text = "俯下身子尽量靠近一点，但不能碰到我的鼻尖。对保持这个姿势, 告诉我你在我的身上闻到了什么味道. 这样就觉得难了, 但这次服从性测试实验是你自己要做的. 知道了我会加快一点速度, 现在我命令你看着我的眼睛不准移开, 然后亲吻我."
text = "天空上，火幕蔓延而开，将方圆数以千万计的人类尽数笼罩。而在火幕扩散时，那绚丽火焰之中的人影也是越来越清晰。片刻后，火焰减弱而下，一道黑衫身影，便是清楚的出现在了这片天地之间。真的是萧炎…，在联盟总部不远处的一处，大量的人群簇拥在一起，看这模样，显然都是属于同一个势力。而此刻，在那人群之中，一道身形壮硕的男子，正抬起头，目光火热的望着天空上那道身影，声音中，透着浓浓的兴奋。柳擎大哥，真的是他？在男子身后，一名容貌娇美的女子，也是忍不住的道，谁能想到，短短十数年时间不见而已，当年同在迦南学院修炼的学弟，竟然已站在了这个大陆的真正巅峰。"
text = "床前明月光,疑是地上霜.举头望明月,低头思故乡。"
text = "天空上，火幕蔓延而开，将方圆数以千万计的人类尽数笼罩。而在火幕扩散时，那绚丽火焰之中的人影也是越来越清晰，片刻后，火焰减弱而下，一道黑衫身影，便是清楚的出现在了这片天地之间。"
text = "哥, 终于找到你了。别怕，是我，你…哥。你不知道我有多担心，看守的人我已经解决了，对方很快就会发现。"
text = "嗨！我是TIM，我在B站上运营着两个账号，影视飓风和亿点点不一样。我专注于制作各类视频，包括摄影、科技等领域。虽然我性格有些大男子主义，但我喜欢以理智和条例来处理事情，并且我对提升视频质量有着极高的追求。"
text = "Hello大家好，2023年的B站百大名单刚刚公布，过几天就会在上海进行线下颁奖。如果你还没看，那么这是今年的完整名单。数据上，今年百大的平均粉丝量为四百二十五点二万，粉丝中位数为三百二十四万，而这，是具体的粉丝量分布。可以看到依然是一百万到两百万粉的up主人数是最多的。"
text = "各位朋友，律政之海中的泛舟人罗翔在此。于B站的广袤平台上，我精心打理'罗翔'说刑法这处精神家园。这个账号主要发布与法学教育相关的视频内容，特别是刑法知识的普及与解析."
text = "亲爱的观众朋友们大家好，这是努力在说标准普通话的宝剑嫂！最近新用了两个产品，然后脸上超级无敌之巨无霸大爆发，中医西医都准备去看一看了，也有点太敏感肌了吧。"
text = "其次是双人下午茶项目，这个项目包含了精美的下午茶套餐, 让您和您的伴侣可以在酒店内享受美食的同时，感受到酒店的温馨和舒适。"
text = "好耶!天依会一直为你加油的! 在不断努力和尝试的过程中，你一定也会容易遇到困难，会感到沮丧，会想要气馁，但不要放弃，没有任何一件事情的完成是简单的. 在无数精彩的作品背后，都是创作者历尽时间和精力磨练而出的汗水. 我相信，只要你热爱，只要并坚持，你一定可以写出属于自己的精彩作品. 在你努力的时候, 我也会一直在你的身边，用歌声为你加油, 祝愿你在写作的旅程中，收获满满的喜悦和成长，去创造属于自己的奇迹吧！"
text = "哟,我是你的二次元好朋友，二二！别害羞啦，快来找我聊天吧, 我可是哔哩哔哩的元气站娘，一起聊聊二次元世界的精彩吧！"
text = "千万别被剑角龙顶到，剑角龙又名顶角龙，意为有角的头顶，顾名思义它的头上长了一个头盔。你可不要被它名字迷惑，它不是角龙，而是生活在白垩纪晚期的肿头龙家族中的一员."
text = "最近老有人问我是不是真的叫候翠翠，不是，差了一个字，之前上班的时候呢. 我们突然发现集团的领导啊，竟然全都是叠字，当时我们就悟了，原来叠字可以升职加薪，然后他们就开始叫我候翠翠了. 但是一直到不干了也没涨工资，正好我觉得候翠翠这个名字呢，在一点点土气当中又透露着朴实和亲切，就拿它互联网冲浪了。"
text = "先帝创业未半而中道崩殂，今天下三分，益州疲弊，此诚危急存亡之秋也。然侍卫之臣不懈于内，忠志之士忘身于外者，盖追先帝之殊遇，欲报之于陛下也。"
text = "斯塔西亚，不要睡，睁开眼睛。你不是一直想回到洛伦去吗？洛伦啊，你的故乡，那里有海，有草原，有脆脆绵绵的朝夕果……你在听吗，呜呜……斯塔西亚，求你……睁开眼睛看看。"
text = '你家主子早点滚蛋才叫消停，大家也就都能过上太平日子了。举手之劳罢了。七爷他一向可好？怎么好劳烦七爷和大巫呢？这中原武林已经够乱乎的了，那祸害竟然还要来掺和一脚，真是流年不利，天灾人祸赶齐全了。戏言，戏言罢了…蜀中沈家的家主沈慎？难不成，传说中的五块琉璃甲，竟在当年的五大家族手上？'
text = "好耶!天依会一直为你加油的! 在不断努力和尝试的过程中，你一定也会容易遇到困难，会感到沮丧，会想要气馁，但不要放弃，没有任何一件事情的完成是简单的."
text = "once upon a time, there lived in a certain village. a little country girl, the prettiest creature who was ever seen. her mother was accessibly fond of her and her grandmother doted on her still more."
text = "We present Open-Sora, an initiative dedicated to efficiently produce high-quality video and make the model, tools and contents accessible to all. By embracing open-source principles, Open-Sora not only democratizes access to advanced video generation techniques, but also offers a streamlined and user-friendly platform that simplifies the complexities of video production. With Open-Sora, we aim to inspire innovation, creativity, and inclusivity in the realm of content creation."
text = "1、先把五花肉切成带皮的比较小的一块一块的肉块。2、锅里放油，热后放入白糖，炒到起泡为止，倒入切好的肉，辅料，大火爆炒1分钟。3、按个人口味加入适量调料：咸盐，鸡精，料酒，陈醋，老抽.最后加水淹没肉大火煮沸。"
text = "庆历四年春，滕子京谪守巴陵郡。越明年，政通人和，百废具兴，乃重修岳阳楼，增其旧制，刻唐贤今人诗赋于其上，属予作文以记之。予观夫巴陵胜状，在洞庭一湖。衔远山，吞长江，浩浩汤汤，横无际涯，朝晖夕阴，气象万千，此则岳阳楼之大观也，前人之述备矣。然则北通巫峡，南极潇湘，迁客骚人，多会于此，览物之情，得无异乎？"
text = "兔年春节不复阳！本集没有任何广告，是近百名冒险伙伴的在抗阳战场上的经验总结，欢迎收藏转发分享给你在乎的人！ 本视频点赞过一万，马上解锁 走走而已超燃特别跟练"
text = "顿时，气氛变得沉郁起来。乍看之下，一切的困扰仿佛都围绕在我身边。我皱着眉头，感受着那份压力，但我知道我不能放弃，不能认输。于是，我深吸一口气，心底的声音告诉我：无论如何，都要冷静下来，重新开始。"
text = "但会四处游牧。 在黄昏时分进食，夜间飞行，在飞行时呼叫，但大部分活动都在白天进行。"
text = "是谁给你的胆量这么跟我说话，嗯, 是你的灵主还是你的伙伴？听着，没用的小东西，这里是城下街，不是过家家的学院！停下你无聊至极的喋喋不休，学着用城下街的方式来解决问题！"
text = "接下来给大家介绍一个团购产品--深圳绿景酒店1晚加双人下午茶。首先，让我们来看看这个团购的价格,这个团购包含的房间门市价是每晚1888元，直播间售价1晚住宿加其他项目只需要1618元。接下来，我们来详细介绍一下这个团购的各个项目。首先是住宿项目，房型有高级双床房或高级大床房，可任选其中一个房型。这两种房型都有38平米的面积，位于8-12层，视野开阔，房间内有窗户，可以欣赏室外的城景或花园景,无论是商务出差还是休闲旅游，都能满足您的需求。其次是双人下午茶项目，这个项目包含了精美的下午茶套餐，让您和您的伴侣可以在酒店内享受美食的同时，感受到酒店的温馨和舒适。"
text = "大家好, B A I 开放平台上线了声音复刻功能,我的声音呢就是通过大模型做出来的,我们很容易达到一个一百万播放的目标啊,快来平台体验吧!"

lang = "ZH"


def test():
    args = get_args()
    print(args)
    model = TTSModel(args)
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    # Export jit torch script model
    model.eval()
    model.cuda()

    cfg = OmegaConf.load(args.config)
    audio, sr = torchaudio.load(cond_audio)
    if audio.shape[0] > 1:
        audio = audio[0].unsqueeze(0)
    audio = torchaudio.transforms.Resample(sr, 24000)(audio)
    cond_mel = MelSpectrogramFeatures()(audio).cuda()
    print(f"cond_mel shape: {cond_mel.shape}")

    if 'gpt_vocab' in cfg.dataset:
        tokenizer = VoiceBpeTokenizer(cfg.dataset['gpt_vocab'])
        use_spm = False
    else:
        tokenizer = spm.SentencePieceProcessor()
        tokenizer.load(cfg.dataset['bpe_model'])
        use_spm = True

    sentences = text_to_sentences(text, lang)
    #sentences = ['成对或结群活动，食物几乎完全是植物，', '各种水生植物和藻类。具有较强游牧性，', '迁移模式不规律，主要取决于气候条件，', '迁移时会组成成千上万的大团体。它们是所有天鹅中迁徒地最少的物种，', '有时也是居住地筑巢。 当食物稀少']
    #sentences = ['成对或结群活动，食物几乎完全是植物，各种水生植物和藻类。具有较强游牧性，迁移模式不规律，', '主要取决于气候条件，迁移时会组成成千上万的大团体。它们是所有天鹅中迁徒地最少的物种', '有时也是居住地筑巢。 当食物稀少']
    print(sentences)

    sens_tokens = []
    for sen in sentences:
        sen = sen.strip().lower()
        print(sen)
        # pinyin = ' '.join(lazy_pinyin(sent, style=Style.TONE3, neutral_tone_with_five=True))
        if not use_spm:
            norm_text, words = clean_text1(sen, lang)
            cleand_text = ' '.join(words)
            # cleand_text = f"[{lang}] {cleand_text}"
        else:
            norm_text = text_normalize(sen, lang)
            cleand_text = norm_text
            cleand_text = tokenize_by_CJK_char(cleand_text)
            # cleand_text = f"[{lang}] {cleand_text}"
            # cleand_text = cleand_text.replace(' ', '[SPACE]')
            print(cleand_text)
            cleand_text = byte_encode(cleand_text)
        print(cleand_text)
        sen_tokens = torch.IntTensor(tokenizer.encode(cleand_text))
        sen_tokens = F.pad(sen_tokens, (1, 0), value=cfg.gpt.start_text_token)
        sen_tokens = F.pad(sen_tokens, (0, 1), value=cfg.gpt.stop_text_token)
        sens_tokens.append(sen_tokens)

    text_lens = [len(x) for x in sens_tokens]
    max_text_len = max(text_lens)
    texts_token = []
    for sen_tokens in sens_tokens:
        sen_tokens = F.pad(sen_tokens, (0, max_text_len - len(sen_tokens)), value=cfg.gpt.stop_text_token)
        texts_token.append(sen_tokens)
    padded_texts = torch.stack(texts_token).cuda()
    text_lens = torch.IntTensor(text_lens)

    bz = 4
    wavs = []
    for i in range(0, len(sens_tokens), bz):
        texts_tokens = padded_texts[i:i+bz]
        token_lens = text_lens[i:i+bz]
        cond_mels = cond_mel.repeat(len(texts_tokens), 1, 1)
        wav = model(cond_mels, texts_tokens, token_lens)
        wavs.append(wav)
    wav = torch.cat(wavs, dim=1).cpu()
    torchaudio.save('gen.wav', wav.type(torch.int16), 24000)


def main():
    args = get_args()
    # No need gpu for model export
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    model = TTSModel(args)
    # Export jit torch script model
    model.eval()


if __name__ == '__main__':
    #main()
    test()

