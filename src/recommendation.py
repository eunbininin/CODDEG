from typing import List

from PIL import Image, ImageDraw, ImageFont
import numpy as np
import pandas as pd


def recommend_celebrity(
    avg_evoked_list: List[np.ndarray],
    times_list: List[np.ndarray],
    channels: List[str],
    sex: str,
    image_folder: str,
    result_dir: str,
    screen_width: int,
    screen_height: int,
) -> None:
    male_celebrities = [
        "공유",
        "송중기",
        "박서준",
        "유연석",
        "이종석",
        "김선호",
        "정해인",
        "이제훈",
        "이동욱",
        "뷔",
        "차은우"
    ]
    female_celebrities = [
        "김유정",
        "전소미",
        "한소희",
        "수지",
        "안유진",
        "카리나",
        "윈터",
        "미연",
        "태연",
        "아이유",
        "윤아"
    ]
    # 각 채널의 최대값을 저장할 리스트 초기화
    max_values_per_channel = []
    # 각 채널에 대해
    for channel_idx in range(len(channels)):
        max_values = []
        # 각 이미지에 대해
        for num_images in range(len(times_list)):
            # 0.1초~0.5초 사이의 시간 인덱스 추출
            selected_indices = [
                index for index, value in enumerate(times_list[num_images]) if 0.1 <= value <= 0.5
            ]
            start_index = selected_indices[0]
            end_index = selected_indices[-1]

            # 최대값 추출하여 리스트에 추가
            max_value = max(avg_evoked_list[num_images][channel_idx][start_index: end_index + 1])
            max_values.append(max_value)
        max_values_per_channel.append(max_values)

    # 각 채널의 최대값 상위 3개 인덱스 추출
    indices_of_largest_values_per_channel = []
    for channel in range(len(max_values_per_channel)):
        indices_of_largest_values = sorted(
            range(len(max_values_per_channel[channel])),
            key=lambda i: max_values_per_channel[channel][i],
            reverse=True,
        )[:3]
        largest_values = [max_values_per_channel[channel][i] for i in indices_of_largest_values]
        top_values_and_indices = [
            (value, index) for value, index in zip(largest_values, indices_of_largest_values)
        ]
        indices_of_largest_values_per_channel.append(top_values_and_indices)

    # 상위 3개 값을 기준으로 최종 인덱스 결정
    top_values_and_indices = sum(indices_of_largest_values_per_channel, [])
    sorted_top_values_and_indices = sorted(
        top_values_and_indices, key=lambda i: i[0], reverse=True
    )
    top_index = sorted_top_values_and_indices[0][1]

    # EEG 이미지 파일 경로 정의
    erp_fp1_path = f"{result_dir}/{sex}_{top_index + 1}_electrode_average_EEG_Fp1.png"
    erp_fp2_path = f"{result_dir}/{sex}_{top_index + 1}_electrode_average_EEG_Fp2.png"
    erp_fp1_plot = Image.open(erp_fp1_path)
    erp_fp2_plot = Image.open(erp_fp2_path)

    # 두 EEG 이미지를 수직으로 결합
    vertical_combined_height = erp_fp1_plot.height + erp_fp2_plot.height
    vertical_combined_width = max(erp_fp1_plot.width, erp_fp2_plot.width)
    vertical_combined_image = Image.new("RGB", (vertical_combined_width, vertical_combined_height), "black")
    vertical_combined_image.paste(erp_fp1_plot, (0, 0))
    vertical_combined_image.paste(erp_fp2_plot, (0, erp_fp1_plot.height))
    vertical_combined_image_path = f"{result_dir}/erp_combined.png"
    vertical_combined_image.save(vertical_combined_image_path)

    # 사용자의 성별에 따라 연예인 이미지 및 텍스트 결정
    if sex == "males":
        celebrity_path = f"{image_folder}/M{top_index + 1}.jpg"
        text = f"당신이 끌리는 연예인은 {male_celebrities[top_index]}입니다."
    elif sex == "females":
        celebrity_path = f"{image_folder}/F{top_index + 1}.jpg"
        text = f"당신이 끌리는 연예인은 {female_celebrities[top_index]}입니다."
    else:
        raise ValueError("Invalid sex")

    # 최종 이미지 생성 (연예인 이미지와 EEG 결합 이미지)
    celebrity_image = Image.open(celebrity_path)
    erp_combined_plot = Image.open(erp_combined_path)
    new_width = max(celebrity_image.width + erp_combined_plot.width, screen_width)
    new_height = max(celebrity_image.height, erp_combined_plot.height, screen_height)
    combined_image = Image.new("RGB", (new_width, new_height), color="black")

    # 좌측에 연예인 이미지, 우측에 EEG 결합 이미지 배치
    x_offset = int((new_width - (celebrity_image.width + erp_combined_plot.width)) / 2)
    y_offset = int((new_height - celebrity_image.height) / 2)
    combined_image.paste(celebrity_image, (x_offset, y_offset))
    x_offset += celebrity_image.width
    combined_image.paste(erp_combined_plot, (x_offset, y_offset))

    # 텍스트 추가
    draw = ImageDraw.Draw(combined_image)
    font_size = 50
    font_path = "C:/Windows/Fonts/batang.ttc"
    font = ImageFont.truetype(font_path, font_size)
    text_width, text_height = draw.textsize(text, font=font)
    while text_width > new_width and font_size > 10:
        font_size -= 1
        font = ImageFont.truetype(font_path, font_size)
        text_width, text_height = draw.textsize(text, font=font)
    text_x = (new_width - text_width) / 2
    text_y = 10
    draw.text((text_x, text_y), text, font=font, fill="white")

    # 최종 이미지 저장 및 표시
    combined_image_path = f"{result_dir}/recommendation.png"
    combined_image.save(combined_image_path)
    combined_image.show()


# def recommend_combination(
#     avg_evoked_list: List[np.ndarray], 
#     times_list: List[np.ndarray], 
#     channels: List[str], 
#     image_folder: str, 
#     clothes_type: str,
# ) -> None:
#     max_values_per_channel = []

#     top_recommendations = []

#     top_indices = [t[1] + 1 for t in top_recommendations]
#     if clothes_type == "bottoms":
#         for index in top_indices:
#             print(f"당신이 끌리는 하의 조합은 {index}번 하의입니다.")
#             image_filename = f"{image_folder}/B{index}.jpg"
#             image = Image.open(image_filename)
#             image.show()
#     elif clothes_type == "shoes":
#         for index in top_indices:
#             print(f"당신이 끌리는 신발의 조합은 {index}번 신발입니다.")
#             image_filename = f"{image_folder}/S{index}.jpg"
#             image = Image.open(image_filename)
#             image.show()
#     else:
#         raise ValueError("Invalid clothes type")

def recommend_combination(
    avg_evoked_list: List[np.ndarray], 
    times_list: List[np.ndarray], 
    channels: List[str], 
    image_folder: str, 
    clothes_type: str,
) -> None:
    max_values_per_channel = []
    # 각 채널에 대해
    for channel_idx in range(len(channels)):
        max_values = []
        # 각 이미지에 대해
        for num_images in range(len(times_list)):
            # 0.1초~0.5초 사이의 시간 인덱스 추출
            selected_indices = [
                index for index, value in enumerate(times_list[num_images]) if 0.1 <= value <= 0.5
            ]
            start_index = selected_indices[0]
            end_index = selected_indices[-1]

            # 최대값 추출하여 리스트에 추가
            max_value = max(avg_evoked_list[num_images][channel_idx][start_index: end_index + 1])
            max_values.append(max_value)
        max_values_per_channel.append(max_values)

        # 각 채널의 최대값 상위 3개 인덱스 추출
    indices_of_largest_values_per_channel = []
    for channel in range(len(max_values_per_channel)):
        indices_of_largest_values = sorted(
            range(len(max_values_per_channel[channel])),
            key=lambda i: max_values_per_channel[channel][i],
            reverse=True,
        )[:3]
        largest_values = [max_values_per_channel[channel][i] for i in indices_of_largest_values]
        top_values_and_indices = [
            (value, index) for value, index in zip(largest_values, indices_of_largest_values)
        ]
        indices_of_largest_values_per_channel.append(top_values_and_indices)

    # 상위 3개 값을 기준으로 최종 인덱스 결정
    top_values_and_indices = sum(indices_of_largest_values_per_channel, [])
    sorted_top_values_and_indices = sorted(
        top_values_and_indices, key=lambda i: i[0], reverse=True
    )

    # 중복되지 않는 상위 3개 인덱스 결정
    seen_indices = set()
    top_indices = []
    for _, index in sorted_top_values_and_indices:
        if index not in seen_indices:
            top_indices.append(index)
            seen_indices.add(index)
        if len(top_indices) == 3:
            break
        
    # 옷 종류에 따라 이미지 출력
    if clothes_type == "bottoms":
        for index in top_indices:
            print(f"당신이 끌리는 하의 조합은 {index + 1}번 하의입니다.")
            image_filename = f"{image_folder}/B{index + 1}.jpg"
            image = Image.open(image_filename)
            image.show()
    elif clothes_type == "shoes":
        for index in top_indices:
            print(f"당신이 끌리는 신발의 조합은 {index + 1}번 신발입니다.")
            image_filename = f"{image_folder}/S{index + 1}.jpg"
            image = Image.open(image_filename)
            image.show()
    else:
        raise ValueError("Invalid clothes type")

def recommend_direction_and_moment(
        avg_evoked_list: List[np.ndarray],
        times_list: List[np.ndarray],
        channels: List[str],
        result_dir: str,
) -> None:
    erd_peak_index_per_channel = []
    for channel_idx in range(len(channels)):
        for num_images in range(len(times_list)):
            erd_selected_indices = [
                index
                for index, value in enumerate(times_list[num_images])
                if 0.0 <= value <= 0.5
            ]
            erd_start_index = erd_selected_indices[0]
            erd_end_index = erd_selected_indices[-1]
            erd_peak_point = np.argmin(
                avg_evoked_list[num_images][channel_idx][erd_start_index: erd_end_index + 1]
            )
            erd_peak_index = erd_selected_indices[erd_peak_point]
        erd_peak_index_per_channel.append(erd_peak_index)

    ers_peak_index_per_channel = []
    ers_summation_per_channel = []
    for channel_idx in range(len(channels)):
        for num_images in range(len(times_list)):
            ers_selected_indices = [
                index
                for index, value in enumerate(times_list[num_images])
                if times_list[num_images][erd_peak_index_per_channel[channel_idx]] <= value <= times_list[num_images][
                    erd_peak_index_per_channel[channel_idx]] + 0.5
            ]
            ers_start_index = ers_selected_indices[0]
            ers_end_index = ers_selected_indices[-1]
            ers_peak_point = np.argmax(
                avg_evoked_list[num_images][channel_idx][ers_start_index: ers_end_index + 1]
            )
            ers_peak_index = ers_selected_indices[ers_peak_point]
            ers_summation = avg_evoked_list[num_images][channel_idx][ers_start_index: ers_end_index + 1].sum()
        ers_peak_index_per_channel.append(ers_peak_index)
        ers_summation_per_channel.append(ers_summation)

    dominant_channel_index = ers_summation_per_channel.index(max(ers_summation_per_channel))
    point_of_operation_index = int(
        erd_peak_index_per_channel[dominant_channel_index] * 0.25 + ers_peak_index_per_channel[
            dominant_channel_index] * 0.75)
    point_of_operation = times_list[0][point_of_operation_index]
    if dominant_channel_index == 0:
        direction = "right"
    elif dominant_channel_index == 1:
        direction = "left"
    else:
        raise ValueError("Invalid channel index")
    moment = f"{float(point_of_operation):.2f}"
    direction_and_moment = {"direction": [direction], "moment(s)": [moment]}
    result_df = pd.DataFrame(direction_and_moment)
    result_df.to_csv(f"{result_dir}/result.csv", index=False)


def recommend_answer(
    fp1_df:pd.DataFrame, 
    fp2_df:pd.DataFrame, 
    screen_width: int, 
    screen_height: int, 
    frequencies: List[int], 
    image_folder: str, 
    correct_num: int, 
    result_dir: str,
) -> None:
    combined_df = pd.concat([fp1_df, fp2_df], axis=1)


def recommend_select(
    avg_evoked_list: List[np.ndarray], 
    times_list: List[np.ndarray], 
    channels: List[str], 
    image_folder: str,
) -> None:
    max_values_per_channel = []
    for channel_idx in range(len(channels)):
        max_values = []
        for num_images in range(len(times_list)):
            selected_indices = [
                index
                for index, value in enumerate(times_list[num_images])
                if 0.1 <= value <= 0.5
            ]
            start_index = selected_indices[0]
            end_index = selected_indices[-1]

            max_value = max(
                avg_evoked_list[num_images][channel_idx][start_index : end_index + 1]
            )
            max_values.append(max_value)
        max_values_per_channel.append(max_values)

    indices_of_largest_values_per_channel = []
    for channel in range(len(max_values_per_channel)):
        indices_of_largest_values = sorted(
            range(len(max_values_per_channel[channel])),
            key=lambda i: max_values_per_channel[channel][i],
            reverse=True,
        )[:3]
        largest_values = [
            max_values_per_channel[channel][i] for i in indices_of_largest_values
        ]
        top_values_and_indices = [
            (value, index)
            for value, index in zip(largest_values, indices_of_largest_values)
        ]
        indices_of_largest_values_per_channel.append(top_values_and_indices)

    top_values_and_indices = sum(indices_of_largest_values_per_channel, [])
    sorted_top_values_and_indices = sorted(
        top_values_and_indices, key=lambda i: i[0], reverse=True
    )
    top_index = sorted_top_values_and_indices[0][1]
    print(f"your selection is {top_index*2+1}")
    image_filename = f"{image_folder}/{top_index*2+1}.png"
    image = Image.open(image_filename)
    image.show()


def recommend_speller(
    avg_evoked_list: List[np.ndarray], 
    times_list: List[np.ndarray], 
    channels: List[str], 
    fp1_df:pd.DataFrame, 
    fp2_df:pd.DataFrame, 
    frequencies: List[float], 
    image_folder: str, 
    result_dir: str, 
    threshold: float = 1.5,
) -> None:
    max_values_per_channel = []