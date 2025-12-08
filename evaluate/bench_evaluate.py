import os
import json
from collections import defaultdict, OrderedDict
from functools import partial
from tqdm import tqdm
from pprint import pprint


from qwen25_vl_infer import load_qwen25_vl_3b, load_qwen25_vl_7b, load_qwen25_vl_32b, load_qwen25_vl_72b, qwen25_vl_infer
from qwen3_vl_infer import load_qwen3_vl_4b_instruct, load_qwen3_vl_4b_thinking, load_qwen3_vl_8b_instruct, load_qwen3_vl_8b_thinking, load_qwen3_vl_30b_a3b_instruct, load_qwen3_vl_30b_a3b_thinking, load_qwen3_vl_30b_a3b_thinking, load_qwen3_vl_235b_a22b_instruct, load_qwen3_vl_235b_a22b_thinking, qwen3_vl_infer
from internvl_3_infer import load_internvl3_1b, load_internvl3_2b, load_internvl3_8b, load_internvl3_14b, load_internvl3_38b, load_internvl3_78b, internvl3_infer
from internvl_35_infer import load_internvl35_1b, load_internvl35_2b, load_internvl35_4b, load_internvl35_8b, load_internvl35_14b, load_internvl35_38b, load_internvl35_20b_a4b, load_internvl35_30b_a3b, load_internvl35_241b_a28b, internvl35_infer_no_think, internvl35_infer_think
from glm_v_infer import load_glm_41v_9b_thinking, load_glm_45v_106b_a12b_thinking, glm_v_infer
from llava_infer import load_llava_onevision_15_4b_instruct, load_llava_onevision_15_8b_instruct, llava_onevision_15_infer
from keye_vl_infer import load_keye_vl15_8b, keye_vl15_infer_no_think, keye_vl15_infer_think, keye_vl15_infer_auto
from qwen3_infer import llm_check_infer, load_qwen3_7b
from seed_vision_infer import load_seed_vision_client, seed_16_infer, seed_vision_16_infer

from argparse import ArgumentParser



evaluate_prompt_normal_en = """
    You are a professional multimodal large language model assistant. You will be given a single-choice question that includes:
    1. One infrared image.
    2. One question related to the image.
    3. Four answer options (A, B, C, D).

    Your task:
    Carefully analyze the image and the question, evaluate all answer choices, and select the most appropriate one. Please output only a single uppercase letter (A, B, C, or D) as your final answer. Do not include any explanations, reasoning, or additional text.

    Evaluation Guidelines:
    1. Each question has only one correct answer.
    2. Random guessing is not allowed; answers must be based on accurate analysis of the image and the question.
    3. The output format must be a single uppercase letter: A, B, C, or D.

    The input question is: <input_question>.
    The input options are: <input_options>.
"""

evaluate_prompt_normal_cn = """
    你是一个专业的的多模态大语言模型助手。你将收到一道单项选择题，题目内容包括：
    1. 一张红外图像；
    2. 一个与图像相关的问题；
    3. 四个候选答案（A、B、C、D）。

    你的任务是：
    认真理解图像和问题内容，分析所有选项，从中选择一个最合适的答案。请仅输出一个大写字母（A、B、C 或 D）作为最终答案，不要添加任何解释、理由或额外内容。

    评测要求：
    1. 每道题仅有一个正确答案；
    2. 禁止随意作答，必须基于图像和问题进行准确判断；
    3. 输出格式必须是单个大写英文字母：A、B、C 或 D。

    输入问题是: <input_question>。
    输入选项是: <input_options>。
"""

evaluate_prompt_translated_en = """
    You are a professional multimodal large language model assistant. You will be given a single-choice question that includes:
    1. One RGB image translated from the corresponding infrared image by an image translation model.
    2. One question related to the image.
    3. Four answer options (A, B, C, D).

    Your task:
    Carefully analyze the image and the question, evaluate all answer choices, and select the most appropriate one. Please output only a single uppercase letter (A, B, C, or D) as your final answer. Do not include any explanations, reasoning, or additional text.

    Evaluation Guidelines:
    1. Each question has only one correct answer.
    2. Random guessing is not allowed; answers must be based on accurate analysis of the image and the question.
    3. The output format must be a single uppercase letter: A, B, C, or D.

    The input question is: <input_question>.
    The input options are: <input_options>.
"""

evaluate_prompt_translated_cn = """
    你是一个专业的的多模态大语言模型助手。你将收到一道单项选择题，题目内容包括：
    1. 一张可见光图像，使用一个图像翻译模型从对应的红外图像转换而来；
    2. 一个与图像相关的问题；
    3. 四个候选答案（A、B、C、D）。

    你的任务是：
    认真理解图像和问题内容，分析所有选项，从中选择一个最合适的答案。请仅输出一个大写字母（A、B、C 或 D）作为最终答案，不要添加任何解释、理由或额外内容。

    评测要求：
    1. 每道题仅有一个正确答案；
    2. 禁止随意作答，必须基于图像和问题进行准确判断；
    3. 输出格式必须是单个大写英文字母：A、B、C 或 D。

    输入问题是: <input_question>。
    输入选项是: <input_options>。
"""

evaluate_prompt_prior_en = """
    You are a professional multimodal large language model assistant. You will be given a single-choice question that includes:
    1. One infrared image.
    2. One question related to the image.
    3. Four answer options (A, B, C, D).

    Your task:
    Carefully analyze the image and the question, evaluate all answer choices, and select the most appropriate one. Please output only a single uppercase letter (A, B, C, or D) as your final answer. Do not include any explanations, reasoning, or additional text.

    When completing the above tasks, please refer to the following prior knowledge about infrared images:
    1. Imaging Mechanism: Infrared imaging does not rely on visible light reflected from objects but instead captures the infrared radiation (thermal radiation) emitted by the objects themselves or their environment. The higher the temperature of an object, the stronger its infrared radiation; therefore, infrared images usually reflect temperature distribution rather than surface color. Infrared imaging is insensitive to lighting conditions and can function even in complete darkness.
    2. Image Characteristics: Infrared images are usually presented in grayscale, where brightness corresponds to temperature. The resolution of infrared images is generally lower, resulting in less detail and poorer edge sharpness. Due to environmental interference (e.g., atmospheric absorption, sensor noise), infrared images often contain more noise. Moreover, different materials have different infrared emissivities at the same temperature, which may cause brightness differences in the resulting images.

    Evaluation Guidelines:
    1. Each question has only one correct answer.
    2. Random guessing is not allowed; answers must be based on accurate analysis of the image and the question.
    3. The output format must be a single uppercase letter: A, B, C, or D.

    The input question is: <input_question>.
    The input options are: <input_options>.
"""

evaluate_prompt_prior_cn = """
    你是一个专业的的多模态大语言模型助手。你将收到一道单项选择题，题目内容包括：
    1. 一张红外图像；
    2. 一个与图像相关的问题；
    3. 四个候选答案（A、B、C、D）。

    你的任务是：
    认真理解图像和问题内容，分析所有选项，从中选择一个最合适的答案。请仅输出一个大写字母（A、B、C 或 D）作为最终答案，不要添加任何解释、理由或额外内容。

    在完成上述时，请参考以下关于红外图像的先验知识：
    1. 成像机理：红外成像不是依靠物体反射的可见光，而是捕捉物体自身或环境发出的红外辐射（热辐射）。温度越高的物体，红外辐射强度越强，因此红外图像常常反映温度分布而不是表面颜色；红外成像对光照条件不敏感，即使在黑暗环境中也能成像；
    2. 图像特征：红外图像通常以灰度形式呈现，亮度对应温度；红外图像的分辨率一般较低，细节和边缘清晰度较差；受环境干扰（如大气吸收、传感器噪声）影响，红外图像常含有较多噪声；不同材料在相同温度下的红外辐射率不同，可能导致成像后的亮度差异。

    评测要求：
    1. 每道题仅有一个正确答案；
    2. 禁止随意作答，必须基于图像和问题进行准确判断；
    3. 输出格式必须是单个大写英文字母：A、B、C 或 D。

    输入问题是: <input_question>。
    输入选项是: <input_options>。
"""

evaluate_prompt_pair_en = """
    You are a professional multimodal large language model assistant. You will be given a single-choice question that includes:
    1. One infrared image and one corresponding RGB image. The RGB image is translated from the corresponding infrared image by an image translation model.
    2. One question related to the image.
    3. Four answer options (A, B, C, D).

    Your task:
    Carefully analyze the image and the question, evaluate all answer choices, and select the most appropriate one. Please output only a single uppercase letter (A, B, C, or D) as your final answer. Do not include any explanations, reasoning, or additional text.

    Evaluation Guidelines:
    1. Each question has only one correct answer.
    2. Random guessing is not allowed; answers must be based on accurate analysis of the image and the question.
    3. The output format must be a single uppercase letter: A, B, C, or D.

    The input question is: <input_question>.
    The input options are: <input_options>.
"""

evaluate_prompt_pair_cn = """
    你是一个专业的的多模态大语言模型助手。你将收到一道单项选择题，题目内容包括：
    1. 一张红外图像与一张对应的可见光图像，可见光图像使用一个图像翻译模型从对应的红外图像转换而来；
    2. 一个与图像相关的问题；
    3. 四个候选答案（A、B、C、D）。

    你的任务是：
    认真理解图像和问题内容，分析所有选项，从中选择一个最合适的答案。请仅输出一个大写字母（A、B、C 或 D）作为最终答案，不要添加任何解释、理由或额外内容。

    评测要求：
    1. 每道题仅有一个正确答案；
    2. 禁止随意作答，必须基于图像和问题进行准确判断；
    3. 输出格式必须是单个大写英文字母：A、B、C 或 D。

    输入问题是: <input_question>。
    输入选项是: <input_options>。
"""

evaluate_prompt_prior_pair_en = """
    You are a professional multimodal large language model assistant. You will be given a single-choice question that includes:
    1. One infrared image and one corresponding RGB image. The RGB image is translated from the corresponding infrared image by an image translation model.
    2. One question related to the image.
    3. Four answer options (A, B, C, D).

    Your task:
    Carefully analyze the image and the question, evaluate all answer choices, and select the most appropriate one. Please output only a single uppercase letter (A, B, C, or D) as your final answer. Do not include any explanations, reasoning, or additional text.

    When completing the above tasks, please refer to the following prior knowledge about infrared images:
    1. Imaging Mechanism: Infrared imaging does not rely on visible light reflected from objects but instead captures the infrared radiation (thermal radiation) emitted by the objects themselves or their environment. The higher the temperature of an object, the stronger its infrared radiation; therefore, infrared images usually reflect temperature distribution rather than surface color. Infrared imaging is insensitive to lighting conditions and can function even in complete darkness.
    2. Image Characteristics: Infrared images are usually presented in grayscale, where brightness corresponds to temperature. The resolution of infrared images is generally lower, resulting in less detail and poorer edge sharpness. Due to environmental interference (e.g., atmospheric absorption, sensor noise), infrared images often contain more noise. Moreover, different materials have different infrared emissivities at the same temperature, which may cause brightness differences in the resulting images.

    Evaluation Guidelines:
    1. Each question has only one correct answer.
    2. Random guessing is not allowed; answers must be based on accurate analysis of the image and the question.
    3. The output format must be a single uppercase letter: A, B, C, or D.

    The input question is: <input_question>.
    The input options are: <input_options>.
"""

evaluate_prompt_prior_pair_cn = """
    你是一个专业的的多模态大语言模型助手。你将收到一道单项选择题，题目内容包括：
    1. 一张红外图像与一张对应的可见光图像，可见光图像使用一个图像翻译模型从对应的红外图像转换而来；
    2. 一个与图像相关的问题；
    3. 四个候选答案（A、B、C、D）。

    你的任务是：
    认真理解图像和问题内容，分析所有选项，从中选择一个最合适的答案。请仅输出一个大写字母（A、B、C 或 D）作为最终答案，不要添加任何解释、理由或额外内容。

    在完成上述时，请参考以下关于红外图像的先验知识：
    1. 成像机理：红外成像不是依靠物体反射的可见光，而是捕捉物体自身或环境发出的红外辐射（热辐射）。温度越高的物体，红外辐射强度越强，因此红外图像常常反映温度分布而不是表面颜色；红外成像对光照条件不敏感，即使在黑暗环境中也能成像。
    2. 图像特征：红外图像通常以灰度形式呈现，亮度对应温度；红外图像的分辨率一般较低，细节和边缘清晰度较差；受环境干扰（如大气吸收、传感器噪声）影响，红外图像常含有较多噪声；不同材料在相同温度下的红外辐射率不同，可能导致成像后的亮度差异。

    评测要求：
    1. 每道题仅有一个正确答案；
    2. 禁止随意作答，必须基于图像和问题进行准确判断；
    3. 输出格式必须是单个大写英文字母：A、B、C 或 D。

    输入问题是: <input_question>。
    输入选项是: <input_options>。
"""


def bench_evaluate(img_dir_base_path, bench_file, infer_func, save_path, recycle_test=True, transed_img_path=None, rgbt_pair=False, use_prior=False):
    system_prompt_cn = evaluate_prompt_normal_cn
    system_prompt_en = evaluate_prompt_normal_en

    if use_prior and rgbt_pair:
        system_prompt_cn = evaluate_prompt_prior_pair_cn
        system_prompt_en = evaluate_prompt_prior_pair_en

    elif use_prior:
        system_prompt_cn = evaluate_prompt_prior_cn
        system_prompt_en = evaluate_prompt_prior_en

    elif rgbt_pair:
        system_prompt_cn = evaluate_prompt_pair_cn
        system_prompt_en = evaluate_prompt_pair_en

    if transed_img_path and not rgbt_pair:
        system_prompt_cn = evaluate_prompt_translated_cn
        system_prompt_en = evaluate_prompt_translated_en

    dim_cnt_dict = defaultdict(int)
    dim_score_cn_dict = defaultdict(float)
    dim_score_en_dict = defaultdict(float)

    with open(bench_file, 'r') as f:
        bench_list = json.load(f)
    
    check_llm_model, check_llm_tokenizer = load_qwen3_7b()

    cn_output_list = []
    en_output_list = []


    debug = False
    for dim in bench_list.keys():
        for example in tqdm(bench_list[dim], desc=f"evaluate {dim}"):
            dim_cnt_dict[dim] += 1

            cn_question = example["question"]["cn_question"]
            cn_options = example["question"]["cn_options"]
            en_question = example["question"]["en_question"]
            en_options = example["question"]["en_options"]
            if rgbt_pair:
                assert transed_img_path, "rgbt_pair=True, but transed_img_path is None"
                img_path1 = os.path.join(img_dir_base_path, example["dst_thermal_path"])
                img_path2 = os.path.join(transed_img_path, example["dst_thermal_path"].split("/")[-1])
                img_path = [img_path1, img_path2]

            else:
                if transed_img_path:
                    img_path = os.path.join(transed_img_path, example["dst_thermal_path"].split("/")[-1])
                else:
                    img_path = os.path.join(img_dir_base_path, example["dst_thermal_path"])
            answer = example["question"]["answer"]

            option_head_list = ['A', 'B', 'C', 'D']
            answer_idx = option_head_list.index(answer)
            cn_option_list = [cn_options['A'], cn_options['B'], cn_options['C'], cn_options['D']]
            en_option_list = [en_options['A'], en_options['B'], en_options['C'], en_options['D']]
            n_option = len(option_head_list) if recycle_test else 1

            cn_correct_cnt = 0
            en_correct_cnt = 0
            
            for i in range(n_option):
                # circular evaluation
                cn_options = {
                    "A": cn_option_list[(4 - i) % 4],
                    "B": cn_option_list[(4 - i + 1) % 4],
                    "C": cn_option_list[(4 - i + 2) % 4],
                    "D": cn_option_list[(4 - i + 3) % 4],
                }
                en_options = {
                    "A": en_option_list[(4 - i) % 4],
                    "B": en_option_list[(4 - i + 1) % 4],
                    "C": en_option_list[(4 - i + 2) % 4],
                    "D": en_option_list[(4 - i + 3) % 4],
                }
                answer = option_head_list[(i + answer_idx) % 4]


                cn_prompt = system_prompt_cn.replace("<input_question>", str(cn_question)).replace("<input_options>", str(cn_options))
                en_prompt = system_prompt_en.replace("<input_question>", str(en_question)).replace("<input_options>", str(en_options))

                cn_output = infer_func(cn_prompt, img_path)
                en_output = infer_func(en_prompt, img_path)

                cn_output = cn_output.strip()
                en_output = en_output.strip()

                cn_output_list.append(cn_output)
                en_output_list.append(en_output)

                if debug:
                    print(cn_prompt)
                    print(en_prompt)
                    print(answer)
                    print(cn_output)
                    print(en_output)
                    print("=" * 100)

                if "</think>" in cn_output:
                    cn_output = cn_output.split("</think>")[-1].strip()
                if "</think>" in en_output:
                    en_output = en_output.split("</think>")[-1].strip()

                if len(cn_output) >=1 and cn_output[0] not in ["A", "B", "C", "D"]:
                    extracted_cn_output = llm_check_infer(check_llm_model, check_llm_tokenizer, cn_output)
                    if len(extracted_cn_output) == 0 or extracted_cn_output[0] not in ["A", "B", "C", "D"]:
                        print(f"error cn_output: {cn_output}")
                        print(f"error extracted_cn_output: {extracted_cn_output}")
                    else:
                        # print(f"origin cn_output: {cn_output}")
                        # print(f"extracted cn_output: {extracted_cn_output}")
                        cn_output = extracted_cn_output
                
                if len(en_output) >=1 and en_output[0] not in ["A", "B", "C", "D"]:
                    extracted_en_output = llm_check_infer(check_llm_model, check_llm_tokenizer, en_output)
                    if len(extracted_en_output) == 0 or extracted_en_output[0] not in ["A", "B", "C", "D"]:
                        print(f"error en_output: {en_output}")
                        print(f"error extracted_en_output: {extracted_en_output}")
                    else:
                        # print(f"origin en_output: {en_output}")
                        # print(f"extracted en_output: {extracted_en_output}")
                        en_output = extracted_en_output
                
                if len(cn_output) == 0:
                    print(f"error empty cn_output: {cn_output}")
                if len(en_output) == 0:
                    print(f"error empty en_output: {en_output}")

                if len(cn_output) >= 1:
                    cn_output = cn_output[0]
                if len(en_output) >= 1:
                    en_output = en_output[0]

                if cn_output == answer:
                    cn_correct_cnt += 1
                if en_output == answer:
                    en_correct_cnt += 1
                    
            debug = False
            dim_score_cn_dict[dim] += cn_correct_cnt / n_option
            dim_score_en_dict[dim] += en_correct_cnt / n_option


    for dim in dim_cnt_dict.keys():
        dim_score_cn_dict[dim] /= dim_cnt_dict[dim]
        dim_score_en_dict[dim] /= dim_cnt_dict[dim]


    dimension_list = [
        "Scene Understanding",
        "Image Theme",
        "Viewpoint of Capture",
        "Target Localization",
        "Spatial Relationship Understanding",
        "Object Counting",
        "Thermal Feature Understanding",
        "Action Recognition",
        "Thermal Feature Reasoning",
        "Commonsense Reasoning",
    ]

    score_dict = dict()
    print("="*100)
    for dim in dimension_list:
        print(f"{dim} \t cnt: {dim_cnt_dict[dim]} \t cn_score: {dim_score_cn_dict[dim]} \t en_score: {dim_score_en_dict[dim]}")
        score_dict[dim] = {
            "cnt": dim_cnt_dict[dim],
            "cn_score": dim_score_cn_dict[dim],
            "en_score": dim_score_en_dict[dim],
            "avg_score": (dim_score_cn_dict[dim] + dim_score_en_dict[dim]) / 2,
        }
    score_dict["avg"] = {
        "cnt": sum(dim_cnt_dict.values()),
        "cn_score": sum(dim_score_cn_dict.values()) / len(dim_cnt_dict),
        "en_score": sum(dim_score_en_dict.values()) / len(dim_cnt_dict),
    }
    score_dict["avg"]["avg_score"] = (score_dict["avg"]["cn_score"] + score_dict["avg"]["en_score"]) / 2
    print(f"avg \t cnt: {score_dict['avg']['cnt']} \t cn_score: {score_dict['avg']['cn_score']} \t en_score: {score_dict['avg']['en_score']} \t avg_score: {score_dict['avg']['avg_score']}")
    print("="*100)

    with open(save_path, 'w') as f:
        json.dump(score_dict, f, indent=4)

    with open(os.path.join(os.path.dirname(save_path), f"{os.path.basename(save_path).split('.')[0]}_cn_output.json"), 'w') as f:
        json.dump(cn_output_list, f, indent=4)
    with open(os.path.join(os.path.dirname(save_path), f"{os.path.basename(save_path).split('.')[0]}_en_output.json"), 'w') as f:
        json.dump(en_output_list, f, indent=4)


load_func_dict = {
    "qwen25_vl_3b": load_qwen25_vl_3b,
    "qwen25_vl_7b": load_qwen25_vl_7b,
    "qwen25_vl_32b": load_qwen25_vl_32b,
    "qwen25_vl_72b": load_qwen25_vl_72b,
    "qwen3_vl_4b_instruct": load_qwen3_vl_4b_instruct,
    "qwen3_vl_4b_thinking": load_qwen3_vl_4b_thinking,
    "qwen3_vl_8b_instruct": load_qwen3_vl_8b_instruct,
    "qwen3_vl_8b_thinking": load_qwen3_vl_8b_thinking,
    "qwen3_vl_30b_a3b_instruct": load_qwen3_vl_30b_a3b_instruct,
    "qwen3_vl_30b_a3b_thinking": load_qwen3_vl_30b_a3b_thinking,
    "qwen3_vl_235b_a22b_instruct": load_qwen3_vl_235b_a22b_instruct,
    "qwen3_vl_235b_a22b_thinking": load_qwen3_vl_235b_a22b_thinking,
    "internvl3_1b": load_internvl3_1b,
    "internvl3_2b": load_internvl3_2b,
    "internvl3_8b": load_internvl3_8b,
    "internvl3_14b": load_internvl3_14b,
    "internvl3_38b": load_internvl3_38b,
    "internvl3_78b": load_internvl3_78b,
    "internvl35_1b": load_internvl35_1b,
    "internvl35_2b": load_internvl35_2b,
    "internvl35_4b": load_internvl35_4b,
    "internvl35_8b": load_internvl35_8b,
    "internvl35_14b": load_internvl35_14b,
    "internvl35_38b": load_internvl35_38b,
    "internvl35_20b_a4b": load_internvl35_20b_a4b,
    "internvl35_30b_a3b": load_internvl35_30b_a3b,
    "internvl35_241b_a28b": load_internvl35_241b_a28b,
    "glm_v_41v_9b_thinking": load_glm_41v_9b_thinking,
    "glm_v_45v_106b_a12b_thinking": load_glm_45v_106b_a12b_thinking,
    "llava_onevision_15_4b_instruct": load_llava_onevision_15_4b_instruct,
    "llava_onevision_15_8b_instruct": load_llava_onevision_15_8b_instruct,
    "keye_vl15_8b": load_keye_vl15_8b,
    "seed_16_251015": load_seed_vision_client,
    "seed_vision16_250815": load_seed_vision_client,
}


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--img_dir_base_path", type=str)
    parser.add_argument("--bench_file", type=str)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--model_name", type=str, default="qwen25_vl_7b")
    parser.add_argument("--save_prefix", type=str, default=None)
    parser.add_argument("--recycle_test", action="store_true")
    parser.add_argument("--rgbt_transed_img_path", type=str, default=None)
    parser.add_argument("--rgbt_pair", action="store_true")
    parser.add_argument("--use_prior", action="store_true")
    parser.add_argument("--think_mode", type=str, default="no_think")
    args = parser.parse_args()
    
    model_name = args.model_name
    if args.save_prefix is None:
        args.save_prefix = model_name
    save_prefix = args.save_prefix
    bench_file = args.bench_file
    save_dir = args.save_dir
    rgbt_pair = args.rgbt_pair
    use_prior = args.use_prior
    rgbt_transed_img_path = args.rgbt_transed_img_path
    think_mode = args.think_mode
    if save_dir is None:
        save_dir = "./bench_score"
        save_dir = os.path.join(save_dir, os.path.basename(bench_file).split(".")[0])
    recycle_test = args.recycle_test
    os.makedirs(save_dir, exist_ok=True)
    if recycle_test:
        save_prefix += "_recycle"
    else:
        save_prefix += "_single"
    if rgbt_pair:
        save_prefix += "_rgbt"
    if use_prior:
        save_prefix += "_prior"
    
    print(f"loading {model_name}")
    load_func = load_func_dict[model_name]
    if load_func:
        if "seed" in model_name:
            client = load_func()
        else:
            model, processor = load_func()
    if 'qwen25_vl' in model_name:
        infer_func = partial(qwen25_vl_infer, model, processor)
    elif 'qwen3_vl' in model_name:
        infer_func = partial(qwen3_vl_infer, model, processor)
    elif 'internvl3' in model_name and not 'internvl35' in model_name:
        infer_func = partial(internvl3_infer, model, processor)
    elif 'internvl35' in model_name:
        if think_mode == "no_think":
            infer_func = partial(internvl35_infer_no_think, model, processor)
        elif think_mode == "think":
            save_prefix += "_thinking"
            infer_func = partial(internvl35_infer_think, model, processor)
    elif 'glm_v' in model_name:
        infer_func = partial(glm_v_infer, model, processor)
    elif 'llava_onevision_15' in model_name:
        infer_func = partial(llava_onevision_15_infer, model, processor)
    elif 'keye' in model_name:
        if think_mode == "no_think":
            infer_func = partial(keye_vl15_infer_no_think, model, processor)
        elif think_mode == "think":
            save_prefix += "_thinking"
            infer_func = partial(keye_vl15_infer_think, model, processor)
        elif think_mode == "auto":
            save_prefix += "_auto_thinking"
            infer_func = partial(keye_vl15_infer_auto, model, processor)
    elif "seed_16" in model_name:
        infer_func = partial(seed_16_infer, client)
    elif "seed_vision16" in model_name:
        infer_func = partial(seed_vision_16_infer, client)
    else:
        raise ValueError(f"not support for {model_name}")

    save_path = os.path.join(save_dir, f"{save_prefix}.json")
    print(f"Save_prefix: {save_prefix}")
    print(f"Saving path: {save_path}")
    print(f"evaluating {model_name}")
    bench_evaluate(
        img_dir_base_path=args.img_dir_base_path,
        bench_file=bench_file,
        infer_func=infer_func,
        save_path=save_path,
        recycle_test=recycle_test,
        transed_img_path=rgbt_transed_img_path,
        rgbt_pair=rgbt_pair,
        use_prior=use_prior,
    )
