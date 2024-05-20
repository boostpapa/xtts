from pypinyin import lazy_pinyin, Style
import torch
import re
import numpy as np
from omegaconf import OmegaConf
from ttts.gpt.model import UnifiedVoice
from ttts.vqvae.xtts_dvae import DiscreteVAE
from ttts.gpt.text.cleaner import clean_text1, text_normalize

    #'diffusion.pth': '/speechwork/users/wd007/tts/xtts2/diffusion/s2_v3/exp/baseline_mrte1_nolangid/epoch_5.pth',
    #'diffusion.pth': '/speechwork/users/wd007/tts/xtts2/diffusion/s2_v4/exp/baseline_nolangid_mrte/epoch_4.pth',
    #'diffusion.pth': '/speechwork/users/wd007/tts/xtts2/diffusion/s3_v2/exp/baseline_mrte1_nolangid_bf16_2/checkpoint_165000.pth',
    #'diffusion.pth': '/speechwork/users/wd007/tts/xtts2/diffusion/dingzhi/xialei/exp/baseline_finetune/epoch_3.pth',
    #'diffusion.pth': '/speechwork/users/wd007/tts/xtts2/diffusion/s3_v2/exp/baseline_mrte1_nolangid_big_bf16/checkpoint_75000.pth',
MODELS = {
    'vqvae.pth':'/speechwork/users/wd007/tts/xtts2/vqvae/s4/exp/baseline_lossl1_ssim1/epoch_19.pth',
    'gpt.pth': '/speechwork/users/wd007/tts/xtts2/gpt/s2_v2/exp/baseline_noe_ignore/epoch_0.pth',
    'diffusion.pth': '/speechwork/users/wd007/tts/xtts2/diffusion/s3_v2/exp/baseline_mrte1_nolangid_big_bf16/checkpoint_75000.pth',
    'clvp2.pth': '',
    'vocoder.pth': 'model/pytorch_model.bin',
    'rlg_auto.pth': '',
    'rlg_diffuser.pth': '',
}
diffusion_ckpt='/speechwork/users/wd007/tts/xtts2/diffusion/s3_v2/exp/baseline_mrte1_nolangid_big_bf16/checkpoint_120000.pth'
diffusion_ckpt='/speechwork/users/wd007/tts/xtts2/diffusion/s3_v2/exp/baseline_mrte1_nolangid_bf16_2/checkpoint_165000.pth'

device = 'cuda:3'



from ttts.utils.infer_utils import load_model
from ttts.vocoder.feature_extractors import MelSpectrogramFeatures
import torchaudio

config='/speechwork/users/wd007/tts/xtts2/diffusion/s2_bpe_v1/configs/config_test_v2.yaml'
config='/speechwork/users/wd007/tts/xtts2/diffusion/s2_v3/configs/config_test_v2.yaml'
config='/speechwork/users/wd007/tts/xtts2/diffusion/s3_v1/configs/config_test_v2.yaml'
config='/speechwork/users/wd007/tts/xtts2/diffusion/s3_bpe_v1/configs/config_test_v2.yaml'
config='/speechwork/users/wd007/tts/xtts2/diffusion/s3_v4/configs/config_test_v2.yaml'
config='/speechwork/users/wd007/tts/xtts2/diffusion/s3_v2/configs/config_test.yaml'
config='/speechwork/users/wd007/tts/xtts2/diffusion/s3_v2/configs/config_test_v2.yaml'
cfg = OmegaConf.load(config)

## load gpt model ##
gpt = UnifiedVoice(**cfg.gpt)
gpt_path = cfg.gpt_checkpoint
gpt_checkpoint = torch.load(gpt_path, map_location=device)
gpt_checkpoint = gpt_checkpoint['model'] if 'model' in gpt_checkpoint else gpt_checkpoint
gpt.load_state_dict(gpt_checkpoint, strict=True)
gpt = gpt.to(device)
gpt.eval()
print(">> GPT weights restored from:", gpt_path)
gpt.post_init_gpt2_config(use_deepspeed=False, kv_cache=False, half=False)

## load vqvae model ##
dvae = DiscreteVAE(**cfg.vqvae)
dvae_path = cfg.dvae_checkpoint
dvae_checkpoint = torch.load(dvae_path, map_location=device)
if 'model' in dvae_checkpoint:
    dvae_checkpoint = dvae_checkpoint['model']
dvae.load_state_dict(dvae_checkpoint, strict=True)
dvae = dvae.to(device)
dvae.eval()
print(">> vqvae weights restored from:", dvae_path)

cond_audio = '/speechwork/users/wd007/tts/xtts2/gpt/s2_v3/bzshort/guzong.wav'
cond_audio = '/speechwork/users/wd007/tts/xtts2/gpt/s2_v3/bzshort/5639-40744-0020.wav'
cond_audio = '/speechwork/users/wd007/tts/data/bilibili/manual/22all/22/speak/ZH/wav/22-all_speak_ZH_YouYou_emotion_ZH_309自豪_20230613_20230627-0150729-0155966.wav'
cond_audio = '/cfs/import/tts/opensource/baker_BZNSYP/BZNSYP/Wave_22k/003261.wav'
cond_audio = '/speechwork/users/wd007/tts/xtts2/gpt/s2_v3/bzshort/xialiu-chuanpu.wav'
cond_audio = '/speechwork/users/wd007/tts/fishspeech/academiCodec/s1/test_wav/live_audio2_57.wav'
cond_audio = '/speechwork/users/wd007/tts/xtts2/gpt/s2_v3/bzshort/split2_J5_TTS_女性_愤怒_4.wav'
cond_audio = '/cfs/import/tts/opensource/baker_BZNSYP/BZNSYP/Wave_22k/008669.wav'
cond_audio = '/speechwork/users/wd007/tts/xtts2/gpt/s2_v3/bzshort/jincheng_dongbei.wav'
cond_audio = '/speechwork/users/wd007/tts/xtts2/gpt/s2_v3/bzshort/bjincheng.wav'
cond_audio = '/speechwork/users/wd007/tts/yourtts/zhibo/live_audio2/wavs/live_audio2_741.wav'
cond_audio = '/cfs/import/tts/opensource/LJSpeech/LJSpeech-1.1/wavs/LJ002-0145.wav'
cond_audio = '/speechwork/users/wd007/tts/xtts2/gpt/s2_v3/bzshort/p_0.wav'
cond_audio = '/speechwork/users/wd007/tts/xtts2/gpt/s2_v3/bzshort/zhoujielun.wav'
cond_audio = '/speechwork/users/wd007/tts/xtts2/gpt/s2_v3/bzshort/xuyuanshen.wav'
cond_audio = '/speechwork/users/wd007/tts/xtts2/gpt/s2_v3/bzshort/guanguan.wav'
cond_audio = '/speechwork/users/wd007/tts/fishspeech/academiCodec/s1/test_wav/dengwei.wav'
cond_audio = '/speechwork/users/wd007/tts/xtts2/gpt/s2_v3/bzshort/ham_male1.wav'
cond_audio = '/speechwork/users/wd007/tts/xtts2/gpt/s2_v3/bzshort/magi.wav'
cond_audio = '/speechwork/users/wd007/tts/xtts2/gpt/s2_v3/bzshort/shujuan.wav'
cond_audio = '/speechwork/users/wd007/tts/xtts2/gpt/s2_v3/bzshort/010100010068.wav'
cond_audio = '/speechwork/users/wd007/tts/fishspeech/academiCodec/s1/test_wav/taylor1.wav'
cond_audio = '/speechwork/users/wd007/tts/xtts2/gpt/s2_v3/bzshort/duyujiao.wav'
cond_audio = '/speechwork/users/wd007/tts/fishspeech/academiCodec/s1/test_wav/dengwei1.wav'
cond_audio = '/speechwork/users/wd007/tts/data/opensource/baker_BZNSYP/Wave/003668.wav'
cond_audio = '/speechwork/users/wd007/tts/xtts2/gpt/s2_v3/bzshort/yueyue.wav'
cond_audio = '/speechwork/users/wd007/tts/data/bilibili/manual/MeiHuo/MeiHuo/speak/ZH/wav/002266.wav'
cond_audio = '/speechwork/users/wd007/tts/xtts2/gpt/s2_v3/bzshort/xueli.wav'
cond_audio = '/speechwork/users/wd007/tts/xtts2/gpt/s2_v3/bzshort/erba.wav'
cond_audio = '/speechwork/users/wd007/tts/xtts2/gpt/s2_v3/bzshort/格恩猫-demo.wav'
cond_audio = '/speechwork/users/wd007/tts/xtts2/gpt/s2_v3/bzshort/格恩猫.wav'
cond_audio = '/speechwork/users/wd007/tts/data/bilibili/manual/22all/22/speak/ZH/wav/22-all_speak_ZH_YouYou_emotion_ZH_309自豪_20230613_20230627-0150729-0155966.wav'
cond_audio = '/speechwork/users/wd007/tts/xtts2/gpt/s2_v3/bzshort/孙笑川.wav'
cond_audio = '/speechwork/users/wd007/tts/xtts2/gpt/s2_v3/bzshort/永雏塔菲.wav'
cond_audio = '/audionas/users/xuanwu/tts/data/bilibili/pgc/xialei/process/flac_cut/xialei3_262.flac'
cond_audio = '/speechwork/users/wd007/tts/data/bilibili/manual/jiachun/jiachun/speak/ZH/wav/00000001_000019.wav'
cond_audio = '/speechwork/users/wd007/tts/xtts2/gpt/s2_v3/bzshort/少女_甜美_哭泣_02.wav'
cond_audio = '/speechwork/users/wd007/tts/xtts2/gpt/s2_v3/bzshort/东雪莲.wav'
cond_audio = '/speechwork/users/wd007/tts/xtts2/gpt/s2_v3/bzshort/sange1.wav'
cond_audio = '/speechwork/users/wd007/tts/fishspeech/academiCodec/s1/test_wav/chenrui1.wav'
cond_audio = '/speechwork/users/wd007/tts/xtts2/gpt/s2_v3/bzshort/j5_angry_2.wav'
cond_audio = '/speechwork/users/wd007/tts/xtts2/gpt/s2_v3/bzshort/siyi.wav'
cond_audio = '/audionas/users/xuanwu/tts/data/bilibili/pgc/xialei/process/flac_cut/xialei3_19.flac'

audio,sr = torchaudio.load(cond_audio)
if audio.shape[0]>1:
    audio = audio[0].unsqueeze(0)
audio = torchaudio.transforms.Resample(sr, 24000)(audio)
cond_mel = MelSpectrogramFeatures()(audio).to(device)
print(f"cond_mel shape: {cond_mel.shape}")


auto_conditioning = cond_mel
settings = {'temperature': .8, 'length_penalty': 1.0, 'repetition_penalty': 2.0,
                    'top_p': .8,
                    'cond_free_k': 2.0, 'diffusion_temperature': 1.0}

from ttts.utils.diffusion import SpacedDiffusion, space_timesteps, get_named_beta_schedule
from ttts.diffusion.aa_model import do_spectrogram_diffusion, normalize_tacotron_mel

from vocos import Vocos
vocos = Vocos.from_pretrained("/speechwork/users/wd007/tts/xtts2/model/charactr/vocos-mel-24khz")

from ttts.gpt.voice_tokenizer import VoiceBpeTokenizer
import sentencepiece as spm
from ttts.utils.byte_utils import byte_encode
from ttts.utils.utils import tokenize_by_CJK_char
import torch.nn.functional as F
if 'gpt_vocab' in cfg.dataset:
    tokenizer = VoiceBpeTokenizer(cfg.dataset['gpt_vocab'])
    use_spm = False
else:
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.load(cfg.dataset['bpe_model'])
    use_spm = True

diffusion = load_model('diffusion', diffusion_ckpt, config, device)
diffusion.eval()
print(">> diffusion weights restored from:", diffusion_ckpt)
diffuser = SpacedDiffusion(use_timesteps=space_timesteps(1000, [50]), model_mean_type='epsilon',
                           model_var_type='learned_range', loss_type='mse', betas=get_named_beta_schedule('linear', 1000),
                           conditioning_free=True, conditioning_free_k=2., sampler='dpm++2m')
diffusion_conditioning = normalize_tacotron_mel(cond_mel)


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
text="人要是行，干一行，行一行，一行行，行行行，行行行，干哪行都行"
text = "蚌埠的蚌和河蚌的蚌怎么念,图穷匕见."
text = "其次是双人下午茶项目，这个项目包含了精美的下午茶套餐, 让您和您的伴侣可以在酒店内享受美食的同时，感受到酒店的温馨和舒适。"
text = "大家好 B A I 开放平台上线了声音复刻功能,我的声音呢就是通过大模型做出来的,我们很容易达到一个一百万播放的目标啊,快来平台体验吧! 千万别被剑角龙顶到，剑角龙又名顶角龙，意为有角的头顶，顾名思义它的头上长了一个头盔。你可不要被它名字迷惑，它不是角龙，而是生活在白垩纪晚期的肿头龙家族中的一员。其次是双人下午茶项目，这个项目包含了精美的下午茶套餐, 让您和您的伴侣可以在酒店内享受美食的同时，感受到酒店的温馨和舒适。once upon a time, there lived in a certain village. a little country girl, "
text = "HE SAT DOWN WEAK BEWILDERED AND ONE THOUGHT WAS UPPERMOST ZORA."
text = "即便是北上广深等供应偏紧的一线城市, 明确了发展目标和重点任务, 新批复了七只创投基金的设立方案."
text = "天空上，火幕蔓延而开，将方圆数以千万计的人类尽数笼罩。而在火幕扩散时，那绚丽火焰之中的人影也是越来越清晰，片刻后，火焰减弱而下，一道黑衫身影，便是清楚的出现在了这片天地之间。"
text = "We present Open-Sora, an initiative dedicated to efficiently produce high-quality video and make the model, tools and contents accessible to all. By embracing open-source principles, Open-Sora not only democratizes access to advanced video generation techniques, but also offers a streamlined and user-friendly platform that simplifies the complexities of video production. With Open-Sora, we aim to inspire innovation, creativity, and inclusivity in the realm of content creation."
text = "We present Open-Sora, an initiative dedicated to efficiently produce high-quality video and make the model, tools and contents accessible to all. By embracing open-source principles, "
text = "once upon a time, there lived in a certain village. a little country girl, the prettiest creature who was ever seen. her mother was accessibly fond of her and her grandmother doted on her still more."
text = "天空上，火幕蔓延而开，将方圆数以千万计的人类尽数笼罩。而在火幕扩散时，那绚丽火焰之中的人影也是越来越清晰。片刻后，火焰减弱而下，一道黑衫身影，便是清楚的出现在了这片天地之间。真的是萧炎…，在联盟总部不远处的一处，大量的人群簇拥在一起，看这模样，显然都是属于同一个势力。而此刻，在那人群之中，一道身形壮硕的男子，正抬起头，目光火热的望着天空上那道身影，声音中，透着浓浓的兴奋。柳擎大哥，真的是他？在男子身后，一名容貌娇美的女子，也是忍不住的道，谁能想到，短短十数年时间不见而已，当年同在迦南学院修炼的学弟，竟然已站在了这个大陆的真正巅峰。"
text = "床前明月光,疑是地上霜.举头望明月,低头思故乡。"
text = "庆历四年春，滕子京谪守巴陵郡。越明年，政通人和，百废具兴，乃重修岳阳楼，增其旧制，刻唐贤今人诗赋于其上，属予作文以记之。予观夫巴陵胜状，在洞庭一湖。衔远山，吞长江，浩浩汤汤，横无际涯，朝晖夕阴，气象万千，此则岳阳楼之大观也，前人之述备矣。然则北通巫峡，南极潇湘，迁客骚人，多会于此，览物之情，得无异乎？"
text = "好奇的灵魂渴望突破自己,去寻找另外的世界"
text = "哟,我是你的二次元好朋友，二二！别害羞啦，快来找我聊天吧, 我可是哔哩哔哩的元气站娘，一起聊聊二次元世界的精彩吧！"
text = "斯塔西亚，不要睡，睁开眼睛。你不是一直想回到洛伦去吗？洛伦啊，你的故乡，那里有海，有草原，有脆脆绵绵的朝夕果……你在听吗，呜呜……斯塔西亚，求你……睁开眼睛看看。"
text = "那个等会儿有时间吧那个那个下午三哥要拉个会,跟大家一起对一下下半年规划.如果大家时间都 ok 的话,就安排在今天下午 review 了.然后可能得辛苦 harry 老师帮忙组织一下团建的事,嗯也不知道安排怎么样了,今天下午我要放假了,接下来一周就不在公司,大家新年快乐!"
text = "斯塔西亚，不要睡，睁开眼睛, 你不是一直想回到洛伦去吗？洛伦啊，你的故乡，那里有海，有草原，有脆脆绵绵的朝夕果…你在听吗，呜呜…斯塔西亚，求你…睁开眼睛看看。"
text = "其次是双人下午茶项目，这个项目包含了精美的下午茶套餐, 让您和您的伴侣可以在酒店内享受美食的同时，感受到酒店的温馨和舒适。"
text = "先帝创业未半而中道崩殂，今天下三分，益州疲弊，此诚危急存亡之秋也。然侍卫之臣不懈于内，忠志之士忘身于外者，盖追先帝之殊遇，欲报之于陛下也。"
text = "大家好 B A I 开放平台上线了声音复刻功能,我的声音呢就是通过大模型做出来的,我们很容易达到一个一百万播放的目标啊,快来平台体验吧!"
text = "接下来给大家介绍一个团购产品--深圳绿景酒店1晚加双人下午茶。首先，让我们来看看这个团购的价格,这个团购包含的房间门市价是每晚1888元，直播间售价1晚住宿加其他项目只需要1618元。接下来，我们来详细介绍一下这个团购的各个项目。首先是住宿项目，房型有高级双床房或高级大床房，可任选其中一个房型。这两种房型都有38平米的面积，位于8-12层，视野开阔，房间内有窗户，可以欣赏室外的城景或花园景,无论是商务出差还是休闲旅游，都能满足您的需求。其次是双人下午茶项目，这个项目包含了精美的下午茶套餐，让您和您的伴侣可以在酒店内享受美食的同时，感受到酒店的温馨和舒适。"
text = "是谁给你的胆量这么跟我说话，嗯, 是你的灵主还是你的伙伴？听着，没用的小东西，这里是城下街，不是过家家的学院！停下你无聊至极的喋喋不休，学着用城下街的方式来解决问题！"
text = "千万别被剑角龙顶到，剑角龙又名顶角龙，意为有角的头顶，顾名思义它的头上长了一个头盔。你可不要被它名字迷惑，它不是角龙，而是生活在白垩纪晚期的肿头龙家族中的一员."
text = "不得不说，人生真是充满了意外，而降临在我头上的，总是比较差的那一种。这件事，说起来还挺让人不好意思的……今天下楼的时候，突然有个人冲过来撞到了我，我一个没站稳，脚就扭伤了。重点不是这个，重点是我刚才去了医院，医生说，我的脚伤比较严重，三个月都不能剧烈运动，三个月啊，那我们的滑雪计划怎么办！我们之前计划了好久，想要下周去滑雪，但谁能想到，好好的计划被一个突然冲出来的路人破坏了。我现在还在悔恨，要是今天没有出门就好了。等到下次，可能就没有现在这种期待的心情了。最重要的是，你为了下周特地空出了时间，如果去不了，这也太遗憾了。真的吗，那我这算是……因祸得福了？你说你要来照顾我，而不是来看我一眼就走，这代表，你会把下周所有的时间都给我，虽然脚还是很疼，但一想到这件事，我就觉得很开心。说到这个，我还有一个小小的请求，下周，你可不可以搬过来和我一起住啊？我没有别的意思，只是不想让你浪费往返的时间。还有，我受伤了，心理很脆弱，如果不能时刻都看到你，我怕我会忍不住崩溃，你不会想看到这一幕发生的，对吧？你不用准备什么东西的，我这边都有！而且我只是脚受伤了，又不是完全不能自理，我想让你来，只是想跟你一起度过未来一周的时间。"

'''
pinyin = ' '.join(lazy_pinyin(text, style=Style.TONE3, neutral_tone_with_five=True))
tokenizer = VoiceBpeTokenizer('ttts/gpt/gpt_tts_tokenizer.json')
text_tokens = torch.IntTensor(tokenizer.encode(pinyin)).unsqueeze(0).to(device)
text_tokens = F.pad(text_tokens, (0, 1))  # This may not be necessary.
text_tokens = text_tokens.to(device)
print(pinyin)
print(text_tokens)
'''

punctuation = ["!", "?", "…", ".", ";", "！", "？", "...", "。", "；"]
punctuation = ["!", "?", ".", ";", "！", "？", "。", "；"]
pattern = r"(?<=[{0}])\s*".format("".join(punctuation))
sentences = ['好。怎么，平安认不得了？不曾。', '你家主子早点滚蛋才叫消停，大家也就都能过上太平日子了。', '举手之劳罢了。七爷他一向可好？', '怎么好劳烦七爷和大巫呢？这中原武林已经够乱乎的了，那祸害竟然还要来掺和一脚，真是流年不利，天灾人祸赶齐全了。', '戏言，戏言罢了…', '咳咳…没有,怎么会莫名其妙想起这老流氓？', '蜀中沈家的家主沈慎？难不成，传说中的五块琉璃甲，竟在当年的五大家族手上？']
sentences = ['斯塔西亚，不要睡，睁开眼睛。你不是一直想回到洛伦去吗？', '洛伦啊，你的故乡，那里有海，有草原，有脆脆绵绵的朝夕果.', '你在听吗，呜呜, 斯塔西亚，求你,,睁开眼睛看看。' ]
sentences = ['你家主子早点滚蛋才叫消停，大家也就都能过上太平日子了。举手之劳罢了。七爷他一向可好？', '怎么好劳烦七爷和大巫呢？这中原武林已经够乱乎的了，那祸害竟然还要来掺和一脚，真是流年不利，天灾人祸赶齐全了。戏言，戏言罢了…', '蜀中沈家的家主沈慎？难不成，传说中的五块琉璃甲，竟在当年的五大家族手上？'] 
sentences = [i for i in re.split(pattern, text) if i.strip() != ""]
print(sentences)

top_p = .8
top_k = 30
temperature = 0.8
autoregressive_batch_size = 1
length_penalty = 0.0
num_beams = 3
repetition_penalty = 10.0
max_mel_tokens = 600
sampling_rate = 24000
lang = "EN"
lang = "ZH"
# text_tokens = F.pad(text_tokens,(0,400-text_tokens.shape[1]),value=0)
wavs = []
wavs1 = []
zero_wav = torch.zeros(1, int(sampling_rate*0.2))
for sent in sentences:
    sent = sent.strip().lower()
    print(sent)
    #pinyin = ' '.join(lazy_pinyin(sent, style=Style.TONE3, neutral_tone_with_five=True))
    if not use_spm:
        norm_text, words = clean_text1(sent, lang)
        cleand_text = ' '.join(words)
        #cleand_text = f"[{lang}] {cleand_text}"
    else:
        norm_text = text_normalize(sent, lang)
        cleand_text = norm_text
        cleand_text = tokenize_by_CJK_char(cleand_text)
        #cleand_text = f"[{lang}] {cleand_text}"
        #cleand_text = cleand_text.replace(' ', '[SPACE]')
        print(cleand_text)
        cleand_text = byte_encode(cleand_text)
        
    print(cleand_text)
    text_tokens = torch.IntTensor(tokenizer.encode(cleand_text)).unsqueeze(0).to(device)
    #text_tokens = F.pad(text_tokens, (0, 1))  # This may not be necessary.
    text_tokens = F.pad(text_tokens, (1,0), value=0)
    text_tokens = F.pad(text_tokens, (0,1), value=1)
    text_tokens = text_tokens.to(device)
    print(text_tokens)
    print(text_tokens.shape)
    with torch.no_grad():
        codes = gpt.inference_speech(auto_conditioning, text_tokens,
                                cond_mel_lengths=torch.tensor([auto_conditioning.shape[-1]], device=text_tokens.device),
                                do_sample=True,
                                top_p=top_p,
                                top_k=top_k,
                                temperature=temperature,
                                num_return_sequences=autoregressive_batch_size,
                                length_penalty=length_penalty,
                                num_beams=num_beams,
                                repetition_penalty=repetition_penalty,
                                max_generate_length=max_mel_tokens)
        print(codes)
        print(codes.shape)
        codes = codes[:, :-2]
        print(codes.shape)
        mel1, _ = dvae.decode(codes)
        wav1 = vocos.decode(mel1.detach().cpu())
        #torchaudio.save('gen1.wav',wav1.detach().cpu(), 24000)
        wav1 = 32767 / max(0.01, torch.max(torch.abs(wav1))) * 1.0 * wav1.detach()
        torch.clip(wav1, -32767.0, 32767.0)
        wavs1.append(wav1)

        latent = gpt(auto_conditioning, text_tokens,
                    torch.tensor([text_tokens.shape[-1]], device=text_tokens.device), codes,
                    torch.tensor([codes.shape[-1]*gpt.mel_length_compression], device=text_tokens.device),
                    cond_mel_lengths=torch.tensor([auto_conditioning.shape[-1]], device=text_tokens.device),
                    return_latent=True, clip_inputs=False).transpose(1,2)
        print(latent.shape)

        upstride = gpt.mel_length_compression/256
        mel = do_spectrogram_diffusion(diffusion, diffuser, latent, diffusion_conditioning, upstride, temperature=1.0).detach().cpu()
        wav = vocos.decode(mel)
        wav = 32767 / max(0.01, torch.max(torch.abs(wav))) * 1.0 * wav.detach()
        torch.clip(wav, -32767.0, 32767.0)
        wavs.append(wav)
        #wavs.append(zero_wav)


#from IPython.display import Audio
'''
wav = torch.cat(wavs, dim=1)[0].detach().cpu().numpy()
wav *= 32767 / max(0.01, np.max(np.abs(wav))) * 0.6
np.clip(wav, -32767.0, 32767.0)
#torchaudio.save('gen.wav', wav.astype(np.int16), 24000)
wavfile.write('gen.wav', 24000, wav.astype(np.int16))
'''

wav1 = torch.cat(wavs1, dim=1)
torchaudio.save('gen1.wav', wav1.type(torch.int16), 24000)
#torchaudio.save('gen1.wav', wav1, 24000)

wav = torch.cat(wavs, dim=1)
torchaudio.save('gen.wav', wav.type(torch.int16), 24000)

#Audio(wav, rate=sampling_rate)
